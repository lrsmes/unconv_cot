#In this code you start with the probability disctrinution at (0,0), you do a sudden perturbation at another state
# and you time evolve it. It is done analytically (it is much faster)
import os
import json
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import skimage as ski
from scipy.linalg import expm
from multiprocessing import Pool, cpu_count
from scipy.optimize import fsolve
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import scipy.integrate as spi
import matplotlib.colors as colors
import time

#from shapely.creation import points

# Local projections
LD = 4

DV = 0.85
Cl = 0.6
Cr = 0.15
# Cl = 0.2
# Cr = 0.09
Cm = Cl * Cr * DV * (np.sqrt(2) + (Cl + Cr) * DV) / (2 - (Cl + Cr) ** 2 * DV * 2)

# Identity & Pauli matrices
id = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

T = 0.1  # K
kb = 0.08617343  # meV/K

def ec(Nl, Nr, eps, delta_Vl, delta_Vr):
    # we are interested in quadruple point
    # (0,0), (1h,1e), (1h,0), (0, 1e)
    Vl = -0.5 * eps + delta_Vl
    Vr = 0.5 * eps + delta_Vr

    return 1/(2*(Cm * Cr + Cl * Cm + Cl * Cr))*((Cr + Cm)*(Nl-Cl*Vl)**2 + (Cl + Cm) * (Nr - Cr*Vr)**2 + 2*Cm*(Nl - Cl * Vl) * (Nr - Cr * Vr))

    #return 1/(2*(Cl+2*Cm))*((Nl-Cl*Vl)**2 + (Nr - Cr*Vr)**2 + Cm/Cl*(Nl + Nr - Cl*Vl - Cr*Vr)**2)


# ham of electron dot
# basis Kup, Kdown, K'up, K'down
# so in np.kron first entry is valley, second entry is spin
def ham_e(Bz, Bx, gs=2, gv=14.0, soc=0.07, dkk=0.02):
    H = np.zeros((LD, LD))

    # SOC
    H += 0.5 * soc * np.kron(sz, sz)

    # Zeeman
    mu_B = 5.78838181E-2  # meV/T
    H += -0.5 * mu_B * gs * (Bz * np.kron(id, sz) + Bx * np.kron(id, sx))
    H += 0.5 * mu_B * gv * Bz * np.kron(sz, id)

    # Intervalley scattering Δ_KK' couples valleys
    #H += dkk * np.kron(sx, id)

    return H


# ham of hole dot
# the energies are of the electronic states
# the hole energy is minus this, was the energy gets removed if the particle is
# taken out
#
# basis Kup, Kdown, K'up, K'down
# inverted SOC
def ham_h(Bz, Bx, gs=2, gv=14.0, soc=0.07, dkk=0.02):
    return ham_e(Bz, Bx, gs, gv, -soc, dkk)


# Fermi function
def fermi(E):
    # if E > 0:
    #     return 0
    # if E == 0:
    #     return 0.5
    # else:
    #     return 1
    exp = np.exp(E / (kb * T))
    return 1. / (exp + 1.)

# Functions of a co-tunneling
def fco01(ER, e00, e11, e10, e01, Gr):
    return ((e10 - e11 + ER) / ((e10 - e11 + ER) ** 2 + Gr ** 2) + (e01 - e00 - ER) / (
                (e01 - e00 - ER) ** 2 + Gr ** 2)) ** 2 * 1 / (np.exp((e11 - e00 - ER) / kb / T) + 1) / (
            np.exp(ER / kb / T) + 1)

def fco10(ER, e00, e11, e10, e01, Gr):
    return ((e10 - e11 + ER) / ((e10 - e11 + ER) ** 2 + Gr ** 2) + (e01 - e00 - ER) / (
                (e01 - e00 - ER) ** 2 + Gr ** 2)) ** 2 * (
            1 - 1 / (np.exp((e11 - e00 - ER) / kb / T) + 1)) * (1 - 1 / (np.exp(ER / kb / T) + 1))

def print_states(Bz, Bx, gs, gv, soc, dir):
    we_ = []
    wh_ = []
    B_fields = np.linspace(0, 1, 250)

    for B in B_fields:
        # electron states
        we, ve = linalg.eigh(ham_e(B, Bx, gs, gv, soc))
        ve = np.asarray(ve.T)

        we_.append(we)

        # hole states
        # energy of hole is -energy (measured from vacuum)
        wh, vh = linalg.eigh(-ham_h(B, Bx, gs, gv, soc))
        vh = np.asarray(vh.T)

        wh_.append(wh)

    plt.figure()
    plt.plot(B_fields, we_)
    plt.savefig(os.path.join(dir, f'electron_states.png'))
    plt.close()

    plt.figure()
    plt.plot(B_fields, wh_)
    plt.savefig(os.path.join(dir, f'hole_states.png'))
    plt.close()


def rates(eps, delta_Vl, delta_Vr, Gl_, Gr_, Gd_, t_, t_vf_, Bz=0.4, Bx=0, gs=2, gv=14, soc=0.07, Vbias=0):
    # tunnel rates times electron charge (in pA)
    Gl = Gl_ * soc / 18 * 17 / 20 #0.1 * soc / 18 * 17 / 20
    #Gr = Gr_ * soc / 18 * 17 / 20  # 0.1 * soc / 18 * 17 / 20

    Gr = Gl_ * soc / 18 * 17 / 20 #0.1 * soc / 18 * 17 / 20

    # dephasing rate
    Gd = Gd_ * soc/2 #0.575 * soc/2

    # tunnel coupling
    t = t_ * soc * 1 #0.05 * soc * 1

    # valley flip tunneling
    t_vf = t_vf_ * soc / 10 #0.05 * soc / 10

    # valley flip rate
    # Gk = soc

    # voltage bias
    Vb = Vbias  # meV Bias

    # charging energies
    e00 = ec(0, 0, eps, delta_Vl, delta_Vr)
    e11 = ec(1, -1, eps, delta_Vl, delta_Vr)
    e01 = ec(0, -1, eps, delta_Vl, delta_Vr)
    e10 = ec(1, 0, eps, delta_Vl, delta_Vr)

    dkk = t_vf_

    delta_gs = 0.2
    gs_e = gs #- delta_gs
    gs_h = gs #+ delta_gs

    delta_soc = 0.015
    soc_e = soc #+ delta_soc
    soc_h = soc #- delta_soc

    delta_gv = 2
    gv_e = 14#12.5
    gv_h = 14#19.5

    # electron states
    we, ve = linalg.eigh(ham_e(Bz, Bx, gs_e, gv_e, soc_e, dkk))
    ve = np.asarray(ve.T)

    # hole states
    # energy of hole is -energy (measured from vacuum)
    wh, vh = linalg.eigh(-ham_h(Bz, Bx, gs_h, gv_h, soc_h, dkk))
    vh = np.asarray(vh.T)

    # find all possible transitions
    # dim (0,0) + (-1, 1) + (-1, 0) + (0, 1) space
    N = 1 + LD * LD + LD + LD

    # index of (-1, 1) state (a, b)
    def ind11(a, b):
        return a * LD + b + 1

    # index of (-1, 0) state (a, 0)
    def ind10(a):
        return a + LD * LD + 1

    # index of (0, 1) state (0, b)
    def ind01(b):
        return b + LD + LD * LD + 1

    # rate equation in basis indexed by eigenstate
    rate = np.zeros((N, N))
    res_rate = np.zeros((N, N))
    co_rate = np.zeros((N, N))
    cur = np.zeros(N)
    cur_L = np.zeros(N)

    # tunnel coupling
    # (0, 0) -> (-1, 1)
    #
    # # calculate overlap of H_tun (0,0) with (ih, ie)
    #olap = np.zeros((LD, LD))
    #olap_vf = np.zeros((LD, LD))
    # for ih in range(LD):
    #     for ie in range(LD):
    #         o = 0
    #         ovf = 0
    #         # find overlap with (s, s)
    #         # s in Kup, Kdown, K'up, K'down
    #         for s in range(LD):
    #             o += vh[ih][s] * ve[ie][s]
    #             ovf += vh[ih][s % LD] * ve[ie][(s+2) % LD]
    #         olap[ih, ie] = o ** 2
    #         olap_vf[ih, ie] = ovf ** 2
    # direct overlap matrix
    O = vh @ ve.T
    olap = O ** 2

    # valley-flipped overlap matrix
    flip_idx = np.roll(np.arange(LD), 2)  # K <-> K' flip
    ve_vf = ve[:, flip_idx]
    O_vf = vh @ ve_vf.T
    olap_vf = O_vf ** 2
    #print(olap)

    # (0,0) -> (-1, 1)

    # level broadening due to the leads
    # G = 0.5*(Gl + Gr)

    for jh in range(LD):  # final state (jh,je)
        for je in range(LD):
            ei = e00  # initial energy
            ef = e11 + wh[jh] + we[je]  # final energy
            if ei >= ef or True:
               rate[ind11(jh, je), 0] = (2. * t ** 2 * olap[jh, je] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
                                        np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)))
                # rate[ind11(jh, je), 0] = t ** 2 * olap[jh, je] * (G/np.pi)/((ef-ei)**2 + G**2)
                # valley flip interdot tunneling
               rate[ind11(jh, je), 0] += (2 * t_vf ** 2 * olap_vf[jh, je] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
                                           np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)))
                # rate[ind11(jh, je), 0] += t_vf ** 2 * olap_vf[jh, je] * (G/np.pi)/((ef-ei)**2 + G**2)
            # if ei > ef:
            #     rate[ind11(jh, je), 0] += 0.00 * t**2 * olap[jh, je]/Gd

    # (-1, 1) -> (0, 0)
    for ih in range(LD):  # initial state (ih, ie)
        for ie in range(LD):
            ei = e11 + wh[ih] + we[ie]
            ef = e00
            if ei >= ef or True:
                rate[0, ind11(ih, ie)] = (2. * t ** 2 * olap[ih, ie] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
                                         np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)))
                #rate[0, ind11(ih, ie)] = t ** 2 * olap[ih, ie] * (G/np.pi)/((ef-ei)**2 + G**2)
                # valley flip interdot tunneling
                rate[0, ind11(ih, ie)] += (2 * t_vf ** 2 * olap_vf[ih, ie] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
                                          np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)))
                # rate[0, ind11(ih, ie)] += t_vf ** 2 * olap_vf[ih, ie]* (G/np.pi)/((ef-ei)**2 + G**2)
            # if ei > ef:
            #     rate[0, ind11(ih, ie)] += 0.00 * t**2 * olap[ih,ie] / Gd

    # tunnel e out of right dot
    #(-1, 1) -> (-1, 0)
    for ih in range(LD):
        for ie in range(LD):
            ei = e11 + wh[ih] + we[ie]
            ef = e10 + wh[ih]
            # if -we[ie] - delta_Vr < 0.5*Vb:
            #     rat = 0
            # else:
            rat = Gr * fermi(ef - ei - .5 * Vb)
            rate[ind10(ih), ind11(ih, ie)] += rat
            cur[ind11(ih, ie)] += rat

    # (0, 1)  -> (0, 0)
    for ie in range(LD):
        ei = e01 + we[ie]
        ef = e00
        #     if -we[ie] + delta_Vr < 0.5*Vb:
        #     rat = 0
        #     else:
        rat = Gr * fermi(ef - ei - .5 * Vb)
        rate[0, ind01(ie)] += rat
        cur[ind01(ie)] += rat

    # tunnel h out of left dot
    #(-1, 1)  -> (0, 1)
    for ih in range(LD):
        for ie in range(LD):
            ei = e11 + we[ie] + wh[ih]
            ef = e01 + we[ie]
            # if wh[ih] + delta_Vl < 0.5 * Vb:
            #     rat = 0
            # else:
            rat = Gl * fermi(ef - ei - .5 * Vb)
            rate[ind01(ie), ind11(ih, ie)] += rat
            cur_L[ind11(ih, ie)] += rat

    # (-1, 0)  -> (0, 0)
    for ih in range(LD):
        ei = e10 + wh[ih]
        ef = e00

        rat = Gl * fermi(ef - ei - .5 * Vb)
        rate[0, ind10(ih)] += rat
        cur_L[ind10(ih)] += rat

    # reverse processes
    # tunnel e in right dot
    #(-1, 0)  -> (-1, 1)
    for jh in range(LD):
        for je in range(LD):
            ei = e10 + wh[jh]
            ef = e11 + wh[jh] + we[je]

            rat = Gr * fermi(ef - ei + .5 * Vb)
            rate[ind11(jh, je), ind10(jh)] += rat
            cur[ind10(jh)] -= rat

    # (0, 0)  -> (0, 1)
    for je in range(LD):
        ei = e00
        ef = e01 + we[je]

        rat = Gr * fermi(ef - ei + .5 * Vb)
        rate[ind01(je), 0] += rat
        cur[0] -= rat

    # tunnel h in left dot
    #(0, 1)  -> (-1, 1)
    for jh in range(LD):
        for je in range(LD):
            ei = e01 + we[je]
            ef = e11 + we[je] + wh[jh]

            rat = Gl * fermi(ef - ei + .5 * Vb)
            rate[ind11(jh, je), ind01(je)] += rat
            cur_L[ind01(je)] -= rat

    # (0, 0)  -> (-1, 0)
    for jh in range(LD):
        ei = e00
        ef = e10 + wh[jh]

        rat = Gl * fermi(ef - ei + .5 * Vb)
        rate[ind10(jh), 0] += rat
        cur_L[0] -= rat

    res_rate = rate

    # Co-tunneling effects
    n = 0  # thermal offset (can be adjusted)
    m = 0  # optional broadening offset

    for jh in range(LD):  # final state (jh, je)
        for je in range(LD):
            # Define energies
            E00 = e00
            E11 = e11 + wh[jh] + we[je]
            E10 = e10 + wh[jh]
            E01 = e01 + we[je]

            # Energy conditions to allow co-tunneling
            if E10 > E00 and E01 > E00 and E10 + m * Gr > E11 and E01 + m * Gl > E11:
                # Singular points where Lorentzian denominators peak
                sp_pos = (E01 - E00)
                sp_neg = -(E10 - E11)
                eps = 1e-3

                # Integral for (1,1) -> (0,0)
                a10 = 0 - n * kb * T
                b10 = np.abs(E11 - E00) + n * kb * T

                if sp_pos > b10:
                    integral_10, _ = spi.quad(
                        fco10, a10, b10,
                        args=(E00, E11, E10, E01, Gr)
                        #points=singular_points,
                        #limit=5000
                    )
                else:
                    integral_10 = spi.quad(
                        fco10, a10, sp_pos - eps,
                        args=(E00, E11, E10, E01, Gr)
                    )[0] + spi.quad(
                        fco10, sp_pos + eps, b10,
                        args=(E00, E11, E10, E01, Gr)
                    )[0]
                tco10 = Gl * Gr * integral_10

                # Integral for (0,0) -> (1,1)
                a01 = -np.abs(E11 - E00) - n * kb * T
                b01 = 0 + n * kb * T

                if sp_neg < a01:
                    integral_01, _ = spi.quad(
                        fco01, a01, b01,
                        args=(E00, E11, E10, E01, Gr)
                    )
                else:
                    integral_01 = spi.quad(
                        fco01, a01, sp_neg-eps,
                        args=(E00, E11, E10, E01, Gr)
                    )[0] + spi.quad(
                        fco01, sp_neg+eps, b01,
                        args=(E00, E11, E10, E01, Gr)
                    )[0]
                tco01 = Gl * Gr * integral_01

                # Update rates
                rate[0, ind11(jh, je)] += tco10  # (1,1) -> (0,0)
                rate[ind11(jh, je), 0] += tco01  # (0,0) -> (1,1)

                co_rate[0, ind11(jh, je)] += tco10  # (1,1) -> (0,0)
                co_rate[ind11(jh, je), 0] += tco01  # (0,0) -> (1,1)

    # #"Spin" flip
    # # hole dot
    # hh = ham_h(Bz, Bx, gs, gv, soc)
    # if hh[0,0] < hh[1,1]: # transition from 1 -> 0
    #     rate[ind10(0), ind10(1)] += Gk
    #     for ie in range(LD):
    #         rate[ind11(0,ie),ind11(1,ie)] += Gk
    # else:                 # transition from 0 -> 1
    #     rate[ind10(1), ind10(0)] += Gk
    #     for ie in range(LD):
    #         rate[ind11(1,ie),ind11(0,ie)] += Gk
    #
    # if hh[2,2] < hh[3,3]:
    #     rate[ind10(2), ind10(3)] += Gk
    #     for ie in range(LD):
    #         rate[ind11(2,ie),ind11(3,ie)] += Gk
    # else:
    #     rate[ind10(3), ind10(2)] += Gk
    #     for ie in range(LD):
    #         rate[ind11(3,ie),ind11(2,ie)] += Gk
    #
    # # electron dot
    # he = ham_e(Bz, Bx, gs, gv, soc)
    # if he[0, 0] < he[1, 1]:  # transition from 1 -> 0
    #     rate[ind01(0), ind01(1)] += Gk
    #     for ih in range(LD):
    #         rate[ind11(ih, 0), ind11(ih, 1)] += Gk
    # else:  # transition from 0 -> 1
    #     rate[ind01(1), ind01(0)] += Gk
    #     for ih in range(LD):
    #         rate[ind11(ih,1), ind11(ih,0)] += Gk
    #
    # if he[2, 2] < he[3, 3]:
    #     rate[ind01(2), ind01(3)] += Gk
    #     for ih in range(LD):
    #         rate[ind11(ih,2), ind11(ih,3)] += Gk
    # else:
    #     rate[ind01(3), ind01(2)] += Gk
    #     for ih in range(LD):
    #         rate[ind11(ih, 3), ind11(ih,2)] += Gk



    #
    # #"Valey" flip
    # for ie in range(LD):
    #     rate[ind11(0,ie),ind11(3,ie)] = rate[ind11(0,ie),ind11(3,ie)] + Gk
    #
    # for ih in range(LD):
    #     rate[ind11(ih,0),ind11(ih,3)] = rate[ind11(ih,0),ind11(ih,3)] + Gk
    #
    # for ie in range(LD):
    #     rate[ind11(1,ie), ind11(2,ie)] = rate[ind11(1,ie), ind11(2,ie)] + Gk
    #
    # for ih in range(LD):
    #     rate[ind11(ih,1),ind11(ih,2)] = rate[ind11(ih,1),ind11(ih,2)] + Gk
    #
    #
    # #spin and valley flip
    # for ie in range(LD):
    #     rate[ind11(1,ie), ind11(3,ie)] = rate[ind11(1,ie), ind11(3,ie)] + Gk
    #
    # for ih in range(LD):
    #     rate[ind11(ih,1), ind11(ih,3)] = rate[ind11(ih,1), ind11(ih,3)] + Gk
    #
    # for ie in range(LD):
    #     rate[ind11(0,ie), ind11(2,ie)] = rate[ind11(0,ie), ind11(2,ie)] + Gk
    #
    # for ih in range(LD):
    #     rate[ind11(ih,0), ind11(ih,2)] = rate[ind11(ih,0), ind11(ih,2)] + Gk
    #

    # set the diagonal from prob conservation
    for j in range(N):
        rate[j, j] = -np.sum(rate[:, j])
        co_rate[j, j] = -np.sum(rate[:, j])

    return rate, res_rate, co_rate

# for i in range(split):
#     for j in range(split):
#         rate = rates(0, DVL[i, j], DVR[i, j])
#         P = np.dot(expm(rate * t_f), P0)
#
#         PLOT[i, j] = 1 * np.sum(P[1:17]) - 0.2 * np.sum(P[17:21]) + 1.2 * np.sum(P[21:])

def compute_PLOT_element(i, eps, DVL_row, DVR_row, Gl_, Gr_, Gd_, t_, t_vf_,
                         Bz, Bx, gs, gv, soc, Vbias, t_f, P0, pulse_dir):
    row_vals = np.zeros(len(DVL_row))
    for j in range(len(DVR_row)):
        rate, _, _ = rates(eps, DVL_row[j], DVR_row[j], Gl_, Gr_, Gd_, t_, t_vf_,
                         Bz, Bx, gs, gv, soc, Vbias)
        P = np.dot(expm(rate * t_f), P0)
        if pulse_dir == 1:
            row_vals[j] = np.sum(P[1:17]) + 1.2 * np.sum(P[17:21]) - 0.2 * np.sum(P[21:])
        elif pulse_dir == -1:
            row_vals[j] = P[0] - 0.2 * np.sum(P[17:21]) + 1.2 * np.sum(P[21:])
    return i, row_vals

def compute_PLOT_parallel(args_list, PLOT, num_cpus=20):
    with Pool(num_cpus) as pool:
        results = pool.starmap(compute_PLOT_element, args_list)

    for i, vals in results:
        PLOT[i, :] = vals

def plot_PLOT(DVL, DVR, PLOT, points, param, dir=None, middle=None):
    plt.pcolormesh(DVL, DVR, PLOT,
                   cmap="viridis_r", shading="auto", rasterized=True)
    for Vl, Vr in points:
        plt.scatter(Vl, Vr)
    plt.title(f'B={param}')
    plt.colorbar()
    #plt.savefig(os.path.join(dir, f'{param}_map.svg'))
    plt.savefig(os.path.join(dir, f'{param}_map.png'))
    np.save(os.path.join(dir, f"map_{param}"), PLOT)
    np.save(os.path.join(dir, f"DVL_{param}"), DVL)
    np.save(os.path.join(dir, f"DVR_{param}"), DVR)
    plt.close()

def read_parameters(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def decay_curve(eps, DVL, DVR, points,
                Gl_, Gr_, Gd_, t_, t_vf_, Bz, Bx, gs, gv, soc, Vbias,
                t_f, P0, pulse_dir, out_dir, radius=2,
                num_cpus=20):

    times = np.linspace(0.01, 5, 50) * 10 ** 5
    P_curve = []
    rate_ratio = []

    params = (Gl_, Gr_, Gd_, t_, t_vf_, Bz, Bx, gs, gv, soc, Vbias)

    footprint = ski.morphology.disk(radius)

    # points in region
    poly = sort_polygon_vertices(points)
    mask = create_mask(DVL, DVR, poly)

    # erode mask
    mask = ski.morphology.erosion(mask, footprint)

    X = DVL[mask]
    Y = DVR[mask]

    base = [(x, y, eps, params, P0, pulse_dir)
            for x, y in zip(X, Y)]

    with Pool(num_cpus) as pool:
        for time in times:

            # Attach current time to each points args
            args = [(bx, by, eps, params, time, P0, pulse_dir)
                    for (bx, by, _, params, P0, pulse_dir) in base]

            results = pool.map(_compute_point, args)

            P, res_rates, co_rates = map(np.stack, zip(*results))

            # Average over all valid points
            P_curve.append(np.mean(P))
            rate_ratio.append(np.max(res_rates))

    # for time in times:
    #     P_pre = []
    #     for x, y in zip(Vl.ravel(), Vr.ravel()):
    #         if is_point_in_polygon(x, y, points):
    #             rate = rates(eps, x, y, Gl_, Gr_, Gd_, t_, t_vf_,
    #                          Bz, Bx, gs, gv, soc, Vbias)
    #             P = np.dot(expm(rate * time), P0)
    #             P_pre.append(P)
    #     if pulse_dir == 1:
    #         P_fin = np.mean([np.sum(P[1:17]) for P in P_pre])
    #         P_curve.append(P_fin)
    #     elif pulse_dir == -1:
    #         P_fin = np.mean([P[0] for P in P_pre])
    #         P_curve.append(P_fin)
    #     else:
    #         raise ValueError("pulse_dir must be +1 or -1")

    # --- Save & plot ---
    plt.figure(figsize=(20, 12))
    plt.plot(times, P_curve)
    plt.ylim(1e-2, 1)
    plt.yscale('log')
    plt.savefig(os.path.join(out_dir, 'decay.png'))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(DVL, DVR, mask, cmap='gray')
    for x, y in points:
        plt.scatter(x, y)
    plt.savefig(os.path.join(out_dir, 'mask.png'))
    plt.close()

    np.save(os.path.join(out_dir, 'decay_curve.npy'), np.array(P_curve))
    np.save(os.path.join(out_dir, 'rate_ratio.npy'), np.array(rate_ratio))
    np.save(os.path.join(out_dir, 'decay_times.npy'), times)

def create_mask(X, Y, poly):
    points = np.vstack((X.ravel(), Y.ravel())).T
    mask = Path(poly).contains_points(points)
    return mask.reshape(X.shape)

def sort_polygon_vertices(points):
    points = np.array(points)

    # centroid
    center = points.mean(axis=0)
    # compute angle
    angles = np.arctan2(points[:, 1] - center[1],
                        points[:, 0] - center[0])
    # sort by angle
    order = np.argsort(angles)

    return points[order]

def _compute_point(args):
    x, y, eps, params, time, P0, pulse_dir = args

    rate, res_rate, co_rate = rates(eps, x, y, *params)
    P = expm(rate * time).dot(P0)

    if pulse_dir == 1:
        return np.sum(P[1:17]), np.sum(res_rate[1:17]), np.sum(co_rate[1:17])
    else:
        return P[0], res_rate, co_rate

def compute_CUT_row(n_B, epsilon, DVl, DVr, Gl_, Gr_, Gd_, t_, t_vf_,
                    Bz, Bx, gs, gv, soc, Vbias, t_f, P0):
    row_vals = np.zeros(len(epsilon))
    for n_e, eps in enumerate(epsilon):
        rate, _, _ = rates(eps, DVl, DVr,
                                  Gl_, Gr_, Gd_, t_, t_vf_,
                                  Bz, Bx, gs, gv, soc, Vbias)
        P = np.dot(expm(rate * t_f), P0)
        row_vals[n_e] = (1 - np.sum(P[1:17])) + 0.2 * np.sum(P[17:21]) - 1.2 * np.sum(P[21:])
    return n_B, row_vals

def compute_CUT_parallel(args_list, CUT, num_cpus=20):
    with Pool(num_cpus) as pool:
        results = pool.starmap(compute_CUT_row, args_list)
    for n_B, row_vals in results:
        CUT[n_B, :] = row_vals

def plot_CUT(eps, B_perp, CUT, param, dir=None):
    #cmap = Colormap('cmasher:bubblegum').to_mpl()  # case insensitive
    cmap = 'viridis_r'
    linecut_map = CUT.T
    plt.pcolormesh(B_perp, eps, linecut_map,
                   # norm=colors.LogNorm(), shading="auto",
                   # norm=colors.SymLogNorm(vmin=10e-5, linthresh=0.03),
                   cmap=cmap, rasterized=True)
    plt.colorbar()
    plt.xlabel(r'$B_\parallel$')
    plt.ylabel(r'$\epsilon$')
    plt.title(f't_vf = {param}')
    if dir is not None:
        plt.savefig(os.path.join(dir, f"CUT_map_{param}.png"))
        # np.save(os.path.join(dir, f"linecut_map_{param}"), linecut_map)
        # np.save(os.path.join(dir, f"eps_{param}"), eps)
        # np.save(os.path.join(dir, f"B_{param}"), B_perp)
    plt.close()


#levels = np.linspace(PLOT.min(), PLOT.max(), 100)

#plt.contourf(DVL, DVR, PLOT, levels=levels)

# plt.savefig(f't_f=10^{num}')
# plt.clf()

# for i in range(Nt-1):
#   P[:,i+1] = P[:,i] + np.dot(rate,P[:,i])*dt
#   P[:,i+1] = P[:,i+1]/np.sqrt(np.sum(np.abs(P[:,i+1])**2))
#
# P0 = np.zeros(Nt)
# for i in range(Nt):
#   P0[i] = np.abs(P[0,i])**2
#
# plt.plot(time,P0)
#
#
# PP = np.zeros((25,Nt)); PP0 = np.zeros(Nt)
#
# PP[:,0] = P[:,0]
# for i in range(Nt):
#   PP[:,i] = np.dot(expm(rate*dt*i),PP[:,0])
#   PP[:,i] = PP[:,i]/np.sqrt(np.sum(np.abs(PP[:,i])**2))
#
# for i in range(Nt):
#   PP0[i] = np.abs(PP[0,i])**2
# plt.plot(time,PP0)
#
# # print(rate)
# d,v = np.linalg.eig(rate)
#
# print(d)

def main(params=None, sim_dir=None):
    t_f0 = params["t_f0"] * 10 ** 5
    t_f = params["t_us"] * 10 ** 5

    split = params["split"]
    delta_Vl = np.linspace(params["delta_Vl"]["start"], params["delta_Vl"]["stop"], split)
    delta_Vr = np.linspace(params["delta_Vr"]["start"], params["delta_Vr"]["stop"], split)
    DVL, DVR = np.meshgrid(delta_Vl, delta_Vr)

    P0 = np.zeros(25)

    if params["pulse_dir"] == 1:
        P0[1] = 1
        #P0[1:17] = 1/16
        # Vl0 = 0
        # Vr0 = 0
        #
        # Vl1 = -12
        # Vr1 = 6
    elif params["pulse_dir"] == -1:
        P0[0] = 1
        # Vl0 = -12
        # Vr0 = 6
        #
        # Vl1 = 0
        # Vr1 = 0

    #P0[0] = 1
    #P0[1] = 1
    #P0[2:17] = 1/(2*15)
    #P0[9] = 1/4
    #P0[16] = 1/4
    #P0[1:5] = 1/4
    #P0[5:9] = 1/4
    #P0[9:13] = 1/4
    #P0[13:17] = 1/4

    # rate0 = rates(0, Vl0, Vr0, params["Gl"], params["Gr"], params["Gd"], params["t"], params["t_vf"],
    #               params["Bz"],
    #               params["Bx"], params["gs"], params["gv"], params["soc"], params["Vbias"])
    #
    # rate1 = rates(0, Vl1, Vr1, params["Gl"], params["Gr"], params["Gd"], params["t"], params["t_vf"],
    #               params["Bz"],
    #               params["Bx"], params["gs"], params["gv"], params["soc"], params["Vbias"])

    # for i in range(5):
    #     P0 = np.dot(expm(rate0 * t_f0), P0)
    #     P0 = np.dot(expm(rate1 * t_f), P0)
    #P0 = np.dot(expm(rate0 * t_f0), P0)

    #print('P11=', P0[1:17])
    #print('P00=', P0[0])
    #print('P01=', P0[17:21])
    #print('P10=', P0[21:])

    P = np.zeros(25)
    PLOT = np.zeros((split, split))

    args_list = [(i, 0, DVL[i], DVR[i], params["Gl"], params["Gr"],
                  params["Gd"], params["t"], params["t_vf"],
                  params["Bz"], params["Bx"], params["gs"], params["gv"],
                  params["soc"], params["Vbias"], t_f, P0, params["pulse_dir"])
                 for i in range(split)]


    if sim_dir == None:
        sim_dir = os.getcwd()


    decay_curve(0, DVL, DVR, params["decay_points"],
                params["Gl"], params["Gr"], params["Gd"], params["t"], params["t_vf"],
                params["Bz"], params["Bx"], params["gs"], params["gv"],
                params["soc"], params["Vbias"], t_f, P0, params["pulse_dir"], sim_dir)

    compute_PLOT_parallel(PLOT=PLOT, args_list=args_list)
    plot_PLOT(DVL=DVL, DVR=DVR, PLOT=PLOT, points=params["decay_points"],
              param=params["t_us"], dir=sim_dir)

    #print_states(params["Bz"], params["Bx"], params["gs"], params["gv"], params["soc"], sim_dir)

    resolution = params["split"]
    epsilon = np.linspace(-0.15, 0.42, resolution)
    B_perp = np.linspace(-0.1, 2.5, resolution)

    CUT = np.zeros((len(B_perp), len(epsilon)))

    args_list_CUT = [(n_B, epsilon, -5.5, 2.25, params["Gl"], params["Gr"],
                      params["Gd"], params["t"], params["t_vf"],
                      params["Bz"], B, params["gs"], params["gv"],
                      params["soc"], params["Vbias"], t_f, P0) for n_B, B in enumerate(B_perp)]

    #compute_CUT_parallel(CUT=CUT, args_list=args_list_CUT, num_cpus=20)
    #plot_CUT(eps=epsilon, B_perp=B_perp, CUT=CUT, param=params["Bx"], dir=sim_dir)


if __name__ == '__main__':
    dir_name = 'transport_vs_tus_1T_reg2'
    current_dir = os.getcwd()
    sim_dir = os.path.join(current_dir, dir_name)
    parameter_file = os.path.join(sim_dir, 'params.json')  # Path to your parameter file
    parameter_sets = read_parameters(parameter_file)

    initial_time = time.time()
    for params in parameter_sets:
        main(params, sim_dir)
    final_time = time.time()

    time = (final_time - initial_time) #/ 3600
    print(time)
