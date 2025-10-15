import os
import json
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.linalg import expm
from multiprocessing import Pool, cpu_count
from cmap import Colormap
import scipy.integrate as spi
import time

# Local projections
LD = 4

DV = 0.9 # Charging energy (difference between triple points)
Cl = 0.086 # Capacitance left dot
Cr = 0.2 # Capaitance right dot
Cm = Cl * Cr * DV * (np.sqrt(2) + (Cl + Cr) * DV) / (2 - (Cl + Cr) ** 2 * DV * 2) # Mutual capacitance

# Identity & Pauli matrices
id = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

T = 0.1  # K
kb = 0.08617343  # meV/K

# Charging energy
def ec(Nl, Nr, eps, delta_Vl, delta_Vr):
    # we are interested in quadruple point
    # (0,0), (1h,1e), (1h,0), (0, 1e)
    Vl = -0.5 * eps + delta_Vl
    Vr = 0.5 * eps + delta_Vr

    return 1/(2*(Cm * Cr + Cl * Cm + Cl * Cr))*((Cr + Cm)*(Nl-Cl*Vl)**2 + (Cl + Cm) * (Nr - Cr*Vr)**2 + 2*Cm*(Nl - Cl * Vl) * (Nr - Cr * Vr))


# Single-particle Hamiltonian of electron dot
# Basis Kup, Kdown, K'up, K'down
# so in np.kron first entry is valley, second entry is spin
def ham_e(Bz, Bx, gs=2.0, gv=14, dkk=0.04, soc=0.07):
    H = np.zeros((LD, LD), dtype=complex)

    # SOC
    H += 0.5 * soc * np.kron(sz, sz)

    # Zeeman
    mu_B = 5.78838181E-2  # meV/T
    H += -0.5 * mu_B * gs * (Bz * np.kron(id, sz) + Bx * np.kron(id, sx))
    H += 0.5 * mu_B * gv * Bz * np.kron(sz, id)

    # Intervalley scattering Δ_KK' couples valleys
    #H += dkk * np.kron(sx, id)

    return H


# Single-particle Hamiltonian of hole dot
# mirror-symmetric to the electron states
#
# Basis Kup, Kdown, K'up, K'down
# inverted SOC
def ham_h(Bz, Bx, gs=2.0, gv=14, dkk=0.04, soc=0.07):
    return ham_e(Bz, Bx, gs, gv, dkk, -soc)


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

# Co-tunneling functions (see Yuli V. Nazarov and Yaroslav M. Blanter. Quantum Transport: Introduction to
# Nanoscience. Cambridge University Press, 2009)
def fco01(ER, e00, e11, e10, e01, Gr):
    return ((e10 - e11 + ER) / ((e10 - e11 + ER) ** 2 + Gr ** 2) + (e01 - e00 - ER) / (
                (e01 - e00 - ER) ** 2 + Gr ** 2)) ** 2 * 1 / (np.exp((e11 - e00 - ER) / kb / T) + 1) / (
            np.exp(ER / kb / T) + 1)

def fco10(ER, e00, e11, e10, e01, Gr):
    return ((e10 - e11 + ER) / ((e10 - e11 + ER) ** 2 + Gr ** 2) + (e01 - e00 - ER) / (
                (e01 - e00 - ER) ** 2 + Gr ** 2)) ** 2 * (
            1 - 1 / (np.exp((e11 - e00 - ER) / kb / T) + 1)) * (1 - 1 / (np.exp(ER / kb / T) + 1))


def rates(eps, delta_Vl, delta_Vr, Gl_, Gr_, Gd_, t_, t_vf_, Bz=0.4, Bx=0, gs=2.0, gv=14, dkk=0.01, soc=0.07, Vbias=0):
    # Tunnel rates times electron charge (in pA)
    Gl = Gl_ * soc / 18 * 17 #/ 20 #0.1 * soc / 18 * 17 / 20
    Gr = Gr_ * soc / 18 * 17 #/ 20 #0.1 * soc / 18 * 17 / 20

    # Dephasing rate
    Gd = Gd_ * soc/2 #0.575 * soc/2

    # Tunnel coupling
    t = t_ * soc #0.05 * soc * 1

    # Valley flip tunneling
    #t_vf = t_vf_ * soc / 10 #0.05 * soc / 10

    # Valley flip rate
    # Gk = soc

    # Bias voltage
    Vb = Vbias  # meV

    # Charging energies
    e00 = ec(0, 0, eps, delta_Vl, delta_Vr)
    e11 = ec(-1, 1, eps, delta_Vl, delta_Vr)
    e01 = ec(0, 1, eps, delta_Vl, delta_Vr)
    e10 = ec(-1, 0, eps, delta_Vl, delta_Vr)

    delta_gs = 0
    gs_e = gs #+ delta_gs
    gs_h = gs #- delta_gs

    delta_soc = 0 #0.02
    soc_e = soc #+ delta_soc
    soc_h = soc #- delta_soc

    delta_gv = 2
    gv_e = 12.5
    gv_h = 19.5

    # Linearize Hamiltonian to find electron states
    we, ve = linalg.eigh(ham_e(Bz, Bx, gs_e, gv_e, dkk, soc_e))
    ve = np.asarray(ve.T)

    # Linearize Hamiltonian to find hole states
    # energy of hole is -energy (measured from vacuum)
    wh, vh = linalg.eigh(-ham_h(Bz, Bx, gs_h, gv_h, dkk, soc_h))
    vh = np.asarray(vh.T)

    # Define number of possible transitions
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

    # Rate equation in basis indexed by eigenstate
    rate = np.zeros((N, N))
    cur = np.zeros(N)
    cur_L = np.zeros(N)

    # Tunnel coupling
    # (0, 0) -> (-1, 1)

    # Calculate overlap of H_tun (0,0) with (ih, ie)
    olap = np.zeros((LD, LD))
    olap_vf = np.zeros((LD, LD))
    for ih in range(LD):
        for ie in range(LD):
            o = 0
            ovf = 0
            # find overlap with (s, s)
            # s in Kup, Kdown, K'up, K'down
            for s in range(LD):
                o += vh[ih][s] * ve[ie][s]
                ovf += vh[ih][s % LD] * ve[ie][(s+2) % LD]
            olap[ih, ie] = o ** 2
            olap_vf[ih, ie] = ovf ** 2

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
                rate[ind11(jh,je),0] += (2 * dkk **2 * olap_vf[jh, je] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
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
                # rate[0, ind11(ih, ie)] = t ** 2 * olap[ih, ie] * (G/np.pi)/((ef-ei)**2 + G**2)
                # valley flip interdot tunneling
                rate[0, ind11(ih, ie)] += (2 * dkk ** 2 * olap_vf[ih, ie] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
                                          np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)))
                # rate[0, ind11(ih, ie)] += t_vf ** 2 * olap_vf[ih, ie]* (G/np.pi)/((ef-ei)**2 + G**2)
            # if ei > ef:
            #     rate[0, ind11(ih, ie)] += 0.00 * t**2 * olap[ih,ie] / Gd

    # tunnel e out of right dot
    # (-1, 1)  -> (-1, 0)
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

    # (0, 1) -> (0, 0)
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
    # (-1, 1) -> (0, 1)
    for ih in range(LD):
        for ie in range(LD):
            ei = e11 + we[ie] + wh[ih]
            ef = e01 + we[ie]
            # if wh[ih] + delta_Vl < 0.5*Vb:
            #     rat = 0
            # else:
            rat = Gl * fermi(ef - ei - .5 * Vb)
            rate[ind01(ie), ind11(ih, ie)] += rat
            cur_L[ind11(ih, ie)] += rat

    # (-1, 0) -> (0, 0)
    for ih in range(LD):
        ei = e10 + wh[ih]
        ef = e00

        rat = Gl * fermi(ef - ei - .5 * Vb)
        rate[0, ind10(ih)] += rat
        cur_L[ind10(ih)] += rat

    # Reverse processes
    # tunnel e in right dot
    # (-1, 0) -> (-1, 1)
    for jh in range(LD):
        for je in range(LD):
            ei = e10 + wh[jh]
            ef = e11 + wh[jh] + we[je]

            rat = Gr * fermi(ef - ei + .5 * Vb)
            rate[ind11(jh, je), ind10(jh)] += rat
            cur[ind10(jh)] -= rat

    # (0, 0) -> (0, 1)
    for je in range(LD):
        ei = e00
        ef = e01 + we[je]

        rat = Gr * fermi(ef - ei + .5 * Vb)
        rate[ind01(je), 0] += rat
        cur[0] -= rat

    # tunnel h in left dot
    # (0, 1) -> (-1, 1)
    for jh in range(LD):
        for je in range(LD):
            ei = e01 + we[je]
            ef = e11 + we[je] + wh[jh]

            rat = Gl * fermi(ef - ei + .5 * Vb)
            rate[ind11(jh, je), ind01(je)] += rat
            cur_L[ind01(je)] -= rat

    # (0, 0) -> (-1, 0)
    for jh in range(LD):
        ei = e00
        ef = e10 + wh[jh]

        rat = Gl * fermi(ef - ei + .5 * Vb)
        rate[ind10(jh), 0] += rat
        cur_L[0] -= rat

    # Co-tunneling effects
    # n = 0
    # m = 0
    # for jh in range(LD):  # final state (jh,je)
    #     for je in range(LD):
    #         E00 = e00  # initial energy
    #         E11 = e11 + wh[jh] + we[je]  # final energy
    #         E10 = e10 + wh[jh]
    #         E01 = e01 + we[je]
    #         if E10 > E00 and E01 > E00 and E10 + m * Gr > E11 and E01 + m * Gl > E11:
    #             integral_10, _ = spi.quad(fco10, 0 - n * kb * T, np.abs(E11 - E00) + n * kb * T,
    #                                       args=(E00, E11, E10, E01,Gr))
    #             tco10 = Gl * Gr * integral_10  # Co-tunneling amplitude
    #
    #             integral_01, _ = spi.quad(fco01, -np.abs(E11 - E00) - n * kb * T, 0 + n * kb * T,
    #                                       args=(E00, E11, E10, E01, Gr))
    #             tco01 = Gl * Gr * integral_01
    #
    #             # Update rate matrix
    #             rate[0, ind11(jh, je)] = rate[0, ind11(jh, je)] + tco10  # (-1, 1) -> (0, 0)
    #             rate[ind11(jh, je), 0] = rate[ind11(jh, je), 0] + tco01  # (0, 0) -> (-1, 1)
    #
    #             # Update current contributions
    #             cur[ind11(jh, je)] -= tco10  # electron leaves right
    #             cur_L[ind11(jh, je)] -= tco10  # electron enters left
    #
    #             cur[0] += tco01  # electron enters from right
    #             cur_L[0] += tco01  # electron enters from left


    # set the diagonal from prob conservation
    for j in range(N):
        rate[j, j] = - np.sum(rate[:, j])

    return rate, cur, cur_L

# for i in range(split):
#     for j in range(split):
#         rate = rates(0, DVL[i, j], DVR[i, j])
#         P = np.dot(expm(rate * t_f), P0)
#
#         PLOT[i, j] = 1 * np.sum(P[1:17]) - 0.2 * np.sum(P[17:21]) + 1.2 * np.sum(P[21:])

# def current(eps, delta_Vl=0, delta_Vr=0, Bz=0, Bx=0, gs=2, gv=14, soc=0.07, Vbias=-1):
#     r_mat, cur, cur_L = rates(eps, delta_Vl, delta_Vr, Bz, Bx, gs, gv, soc, Vbias)
#
#     w, v = linalg.eig(r_mat)
#     el = np.argmax(w)
#     assert abs(w[el]) < 1e-10
#     pd = v[:, el] / np.sum(v[:, el])
#
#     # convert current from meV to pA (multiply with e/h)
#     cur *= 38740
#     # print(np.real(np.dot(cur, pd)))
#     return np.real(np.dot(cur, pd))  # , pd, cur, cur_L

# -----------------------
# Helper: stationary current
# -----------------------
def stationary_current(r_mat, cur):
    w, v = linalg.eig(r_mat)
    el = np.argmax(w)
    assert abs(w[el]) < 1e-10
    pd = v[:, el] / np.sum(v[:, el])  # steady-state distribution

    cur *= 38740  # convert meV → pA
    return np.real(np.dot(cur, pd))


# -----------------------
# CUR element worker
# -----------------------
def compute_CUR_element(i, eps, DVL_row, DVR_row, Gl_, Gr_, Gd_, t_, t_vf_,
                        Bz, Bx, gs, gv, dkk, soc, Vbias):
    row_vals = np.zeros(len(DVR_row))
    for j in range(len(DVR_row)):
        r_mat, cur, cur_L = rates(eps, DVL_row[j], DVR_row[j],
                                  Gl_, Gr_, Gd_, t_, t_vf_,
                                  Bz, Bx, gs, gv, dkk, soc, Vbias)
        row_vals[j] = stationary_current(r_mat, cur)
    return i, row_vals


def compute_CUR_parallel(args_list, CUR, num_cpus=20):
    with Pool(num_cpus) as pool:
        results = pool.starmap(compute_CUR_element, args_list)
    for i, vals in results:
        CUR[i, :] = vals


def plot_CUR(DVL, DVR, PLOT, param, dir=None):
    plt.pcolormesh(DVL, DVR, PLOT,
                   cmap="gist_heat", shading="auto", rasterized=True)
    plt.colorbar()
    plt.xlabel(r'$V_{FG14}$')
    plt.ylabel(r'$V_{FG12}$')
    plt.title(f'dkk = {param}')
    if dir is not None:
        plt.savefig(os.path.join(dir, f"{param}_map.png"))
    plt.close()


# -----------------------
# CUT element worker
# -----------------------
def compute_CUT_row(n_B, epsilon, DVl, DVr, Gl_, Gr_, Gd_, t_, t_vf_,
                        Bz, Bx, gs, gv, dkk, soc, Vbias):
    row_vals = np.zeros(len(epsilon))
    for n_e, eps in enumerate(epsilon):
        r_mat, cur, cur_L = rates(eps, DVl, DVr,
                                  Gl_, Gr_, Gd_, t_, t_vf_,
                                  Bz, Bx, gs, gv, dkk, soc, Vbias)
        row_vals[n_e] = stationary_current(r_mat, cur)
    return n_B, row_vals


def compute_CUT_parallel(args_list, CUT, num_cpus=20):
    with Pool(num_cpus) as pool:
        results = pool.starmap(compute_CUT_row, args_list)
    for n_B, row_vals in results:
        CUT[n_B, :] = row_vals


def plot_CUT(eps, B_perp, CUT, param, dir=None):
    cmap = Colormap('cmasher:bubblegum').to_mpl()  # case insensitive
    linecut_map = CUT.T
    plt.pcolormesh(B_perp, eps, linecut_map,
                   #norm=colors.LogNorm(), shading="auto",
                   #norm=colors.SymLogNorm(vmin=10e-5, linthresh=0.03),
                   cmap=cmap, rasterized=True)
    plt.colorbar()
    plt.xlabel(r'$B_\parallel$')
    plt.ylabel(r'$\epsilon$')
    plt.title(f'dkk = {param}')
    if dir is not None:
        plt.savefig(os.path.join(dir, f"CUT_map_{param}.png"))
        #np.save(os.path.join(dir, f"linecut_map_{param}"), linecut_map)
        #np.save(os.path.join(dir, f"eps_{param}"), eps)
        #np.save(os.path.join(dir, f"B_{param}"), B_perp)
    plt.close()


def read_parameters(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)





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
    split = params["split"]
    delta_Vl = np.linspace(params["delta_Vl_start"], params["delta_Vl_stop"], split)
    delta_Vr = np.linspace(params["delta_Vr_start"], params["delta_Vr_stop"], split)
    DVL, DVR = np.meshgrid(delta_Vl, delta_Vr)

    CUR = np.zeros((split, split))

    args_list_CUR = [
        (i, 0, DVL[i], DVR[i],
         params["Gl"], params["Gr"], params["Gd"],
         params["t"], params["t_vf"],
         params["Bz"], params["Bx"], params["gs"], params["gv"],
         params["dkk"], params["soc"], params["Vbias"])
        for i in range(split)
    ]

    if sim_dir is None:
        sim_dir = os.getcwd()

    # --- Compute CUR ---
    compute_CUR_parallel(CUR=CUR, args_list=args_list_CUR, num_cpus=20)
    plot_CUR(DVL=DVL, DVR=DVR, PLOT=CUR, param=params["dkk"], dir=sim_dir)

    # --- Compute CUT ---
    resolution = params["split"]
    epsilon = np.linspace(-0.5, 0.5, resolution)
    B_perp = np.linspace(-3, 2.5, resolution)

    CUT = np.zeros((len(B_perp), len(epsilon)))

    args_list_CUT = [
        (n_B, epsilon, -6, 2.25,
         params["Gl"], params["Gr"], params["Gd"],
         params["t"], params["t_vf"],
         params["Bx"], B, params["gs"], params["gv"],
         params["dkk"], params["soc"], params["Vbias"])
        for n_B, B in enumerate(B_perp)
    ]

    compute_CUT_parallel(CUT=CUT, args_list=args_list_CUT, num_cpus=20)
    plot_CUT(eps=epsilon, B_perp=B_perp, CUT=CUT, param=params["dkk"], dir=sim_dir)



if __name__ == '__main__':
    dir_name = 'current_vs_dkk'
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
