#In this code you start with the probability disctrinution at (0,0), you do a sudden perturbation at another state
# and you time evolve it. It is done analytically (it is much faster)
import os

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.linalg import expm
from multiprocessing import Pool, cpu_count
import scipy.integrate as spi
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

t_f0 = 10**10
t_f = 10**5

Bx = 0
gs = 2
gv = 14
soc = 0.07
eps = 0
Vbias = 0
T = 0.1  # K
kb = 0.08617343  # meV/K

Vl0 = 0
Vr0 = 0

# Identity & Pauli matrices
id = np.matrix([[1, 0], [0, 1]])
sx = np.matrix([[0, 1], [1, 0]])
sy = np.matrix([[0, -1j], [1j, 0]])
sz = np.matrix([[1, 0], [0, -1]])

# Local projections
LD = 4

DV = 0.9
Cl = 0.086
Cr = 0.2
Cm = Cl*Cr*DV*(np.sqrt(2) + (Cl + Cr)*DV)/(2 - (Cl + Cr)**2*DV*2)



def ec(Nl, Nr, eps, delta_Vl, delta_Vr):
    # we are interested in quadruple point
    # (0,0), (1h,1e), (1h,0), (0, 1e)
    Vl = -.5 * eps + delta_Vl
    Vr = .5 * eps + delta_Vr

    return 1/(2*(Cm*Cr + Cl*Cm + Cl*Cr))*((Cr + Cm)*(Nl-Cl*Vl)**2 + (Cl + Cm) * (Nr - Cr*Vr)**2 + + 2*Cm*(Nl - Cl*Vl)*(Nr - Cr*Vr))


# ham of electron dot
# basis Kup, Kdown, K'up, K'down
# so in np.kron first entry is valley, second entry is spin
def ham_e(Bz, Bx, gs=2, gv=14, soc=0.07):
    H = np.asmatrix(np.zeros((LD, LD)))

    # soc
    H += .5 * soc * np.kron(sz, sz)

    # Zeeman
    mu_B = 5.78838181E-2  # meV/T
    H += -.5 * mu_B * gs * (Bz * np.kron(id, sz) + Bx * np.kron(id, sx))
    H += .5 * mu_B * gv * Bz * np.kron(sz, id)

    return H


# ham of hole dot
# the energies are of the electronic states
# the hole energy is minus this, was the energy gets removed if the particle is
# taken out
#
# basis Kup, Kdown, K'up, K'down
# inverted SOC
def ham_h(Bz, Bx, gs=2, gv=14, soc=0.07):
    return ham_e(Bz, Bx, gs, gv, -soc)


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


def rates(eps, delta_Vl, delta_Vr, Bz=0.4, Bx=0, gs=2, gv=14, soc=0.07, Vbias=0):
    # tunnel rates times electron charge (in pA)
    Gl = 0.1 * soc / 18 * 17 /20
    Gr = 0.1 * soc / 18 * 17 /20

    # dephasing rate
    Gd = 0.575 * soc/2

    # tunnel coupling
    t = .05 * soc*1

    # valley flip tunneling
    t_vf = 0.05 * soc / 10

    # Valey flip rate
    # Gk = soc

    # voltage bias
    Vb = Vbias  # meV Bias

    # charging energies
    e00 = ec(0, 0, eps, delta_Vl, delta_Vr)
    e11 = ec(-1, 1, eps, delta_Vl, delta_Vr)
    e01 = ec(0, 1, eps, delta_Vl, delta_Vr)
    e10 = ec(-1, 0, eps, delta_Vl, delta_Vr)

    # electron states
    we, ve = linalg.eigh(ham_e(Bz, Bx, gs, gv, soc))
    ve = np.asarray(ve.T)

    # hole states
    # energy of hole is -energy (measured from vacuum)
    wh, vh = linalg.eigh(-ham_h(Bz, Bx, gs, gv, soc))
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
    cur = np.zeros(N)
    cur_L = np.zeros(N)

    # tunnel coupling
    # (0, 0) -> (-1, 1)
    #
    # calculate overlap of H_tun (0,0) with (ih, ie)
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
                ovf += vh[ih][s%LD] * ve[ie][(s+2)%LD]
            olap[ih, ie] = o ** 2
            olap_vf[ih,ie] = ovf ** 2

    # (0,0) -> (-1, 1)

    # level broadening due to the leads
    # G = 0.5*(Gl + Gr)

    for jh in range(LD):  # final state (jh,je)
        for je in range(LD):
            ei = e00  # initial energy
            ef = e11 + wh[jh] + we[je]  # final energy
            if ei >= ef or True:
                rate[ind11(jh, je), 0] = 2. * t ** 2 * olap[jh, je] * 1 / np.sqrt(2 * np.pi * Gd ** 2) * np.exp(
                   -(ef - ei) ** 2 / (2 * Gd ** 2))
                # rate[ind11(jh, je), 0] = t ** 2 * olap[jh, je] * (G/np.pi)/((ef-ei)**2 + G**2)
                # valley flip interdot tunneling
                rate[ind11(jh,je),0] += 2 * t_vf **2 * olap_vf[jh, je] * 1 / np.sqrt(2 * np.pi * Gd ** 2) * np.exp(
                -(ef - ei) ** 2 / (2 * Gd ** 2))
                # rate[ind11(jh, je), 0] += t_vf ** 2 * olap_vf[jh, je] * (G/np.pi)/((ef-ei)**2 + G**2)
            # if ei > ef:
            #     rate[ind11(jh, je), 0] += 0.00 * t**2 * olap[jh, je]/Gd

    # (-1, 1) -> (0, 0)
    for ih in range(LD):  # initial state (ih, ie)
        for ie in range(LD):
            ei = e11 + wh[ih] + we[ie]
            ef = e00
            if ei >= ef or True:
                rate[0, ind11(ih, ie)] = 2. * t ** 2 * olap[ih, ie] * 1 / np.sqrt(2 * np.pi * Gd ** 2) * np.exp(
                   -(ef - ei) ** 2 / (2 * Gd ** 2))
                # rate[0, ind11(ih, ie)] = t ** 2 * olap[ih, ie] * (G/np.pi)/((ef-ei)**2 + G**2)
                # valley flip interdot tunneling
                rate[0,ind11(ih,ie)] += 2 * t_vf **2 * olap_vf[ih, ie] * 1 / np.sqrt(2 * np.pi * Gd ** 2) * np.exp(
                -(ef - ei) ** 2 / (2 * Gd ** 2))
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
    # (-1, 1)  -> (0, 1)
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

    # (-1, 0)  -> (0, 0)
    for ih in range(LD):
        ei = e10 + wh[ih]
        ef = e00

        rat = Gl * fermi(ef - ei - .5 * Vb)
        rate[0, ind10(ih)] += rat
        cur_L[ind10(ih)] += rat

    # reverse processes
    # tunnel e in right dot
    # (-1, 0)  -> (-1, 1)
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
    # (0, 1)  -> (-1, 1)
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

    # Co-tunneling effects
    n = 0
    m = 0
    for jh in range(LD):  # final state (jh,je)
        for je in range(LD):
            E00 = e00  # initial energy
            E11 = e11 + wh[jh] + we[je]  # final energy
            E10 = e10 + wh[jh]
            E01 = e01 + we[je]
            if E10 > E00 and E01 > E00 and E10 +m*Gr> E11 and E01 +m*Gl> E11:
                integral_10, _ = spi.quad(fco10, 0 - n * kb * T, np.abs(E11 - E00) + n * kb * T,
                                          args=(E00, E11, E10, E01,Gr))
                tco10 = Gl * Gr * integral_10  # Co-tunneling amplitude

                integral_01, _ = spi.quad(fco01, -np.abs(E11 - E00) - n * kb * T, 0 + n * kb * T,
                                          args=(E00, E11, E10, E01,Gr))
                tco01 = Gl * Gr * integral_01
                rate[0, ind11(jh, je)] = rate[0, ind11(jh, je)] + tco10  # (1,1) -> (0,0)
                rate[ind11(jh, je), 0] = rate[ind11(jh, je), 0] + tco01  # (0,0) -> (1,1)

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
        rate[j, j] = - np.sum(rate[:, j])

    return rate


rate0 = rates(0, Vl0, Vr0)
# print(rate0)
P0 = np.zeros(25)
P0[0] = 1
P0 = np.dot(expm(rate0 * t_f0), P0)
print(P0)

split = 250
delta_Vl = np.linspace(-6, -4.5, split)
delta_Vr = np.linspace(1.5, 3, split)
DVL, DVR = np.meshgrid(delta_Vl, delta_Vr)

P = np.zeros(25)
PLOT = np.zeros((split, split))

args_list = [(i, 0, DVL[i], DVR[i], t_f, P0) for i in range(split)]

# for i in range(split):
#     for j in range(split):
#         rate = rates(0, DVL[i, j], DVR[i, j])
#         P = np.dot(expm(rate * t_f), P0)
#
#         PLOT[i, j] = 1 * np.sum(P[1:17]) - 0.2 * np.sum(P[17:21]) + 1.2 * np.sum(P[21:])

def compute_PLOT_element(i, eps, DVL_row, DVR_row, t_f, P0):
    row_vals = np.zeros(len(DVL_row))
    for j in range(len(DVR_row)):
        rate = rates(eps, DVL_row[j], DVR_row[j])
        P = np.dot(expm(rate * t_f), P0)
        row_vals[j] = 1 * np.sum(P[1:17]) - 0.2 * np.sum(P[17:21]) + 1.2 * np.sum(P[21:])
    return i, row_vals

def compute_PLOT_parallel():
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(compute_PLOT_element, args_list)

    for i, vals in results:
        PLOT[i, :] = vals

def plot_PLOT():
    plt.pcolormesh(DVL, DVR, PLOT,
                   cmap="viridis_r", shading="auto", rasterized=True)
    plt.colorbar()
    plt.show()



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



if __name__ == '__main__':
    initial_time = time.time()


    compute_PLOT_parallel()
    plot_PLOT()

    final_time = time.time()
    time = (final_time - initial_time) #/ 3600
    print(time)
