import os
import json
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.linalg import expm
from multiprocessing import Pool
import scipy.integrate as spi
import time

# ===========================
# System Constants & Settings
# ===========================

LD = 4   # Local dimension (4 spin/valley states per dot)

# Dot capacitances and charging energy parameters
DV = 0.9      # Charging energy (difference between triple points)
Cl = 0.086    # Capacitance of left dot
Cr = 0.2      # Capacitance of right dot
# Mutual capacitance between the dots
Cm = Cl * Cr * DV * (np.sqrt(2) + (Cl + Cr) * DV) / (2 - (Cl + Cr) ** 2 * DV * 2)

# Identity and Pauli matrices
id = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

# Temperature and Boltzmann constant
T = 0.1         # Kelvin
kb = 0.08617343 # meV/K (Boltzmann constant)


# ===========================
# Charging Energy
# ===========================

def ec(Nl, Nr, eps, delta_Vl, delta_Vr):
    """
    Compute the charging energy of the double dot system.

    Parameters:
        Nl, Nr : int
            Number of charges (holes/electrons) in left and right dot.
        eps : float
            Detuning energy.
        delta_Vl, delta_Vr : float
            Gate voltage offsets for left and right dots.

    Returns:
        float : charging energy in meV.
    """
    Vl = -0.5 * eps + delta_Vl
    Vr =  0.5 * eps + delta_Vr

    return 1 / (2 * (Cm * Cr + Cl * Cm + Cl * Cr)) * (
        (Cr + Cm) * (Nl - Cl * Vl) ** 2 +
        (Cl + Cm) * (Nr - Cr * Vr) ** 2 +
        2 * Cm * (Nl - Cl * Vl) * (Nr - Cr * Vr)
    )


# ===========================
# Single-Particle Hamiltonians
# ===========================

def ham_e(Bz, Bx, gs=2, gv=14, dkk=0.04, soc=0.07):
    """
    Electron Hamiltonian in basis [K↑, K↓, K'↑, K'↓].

    Includes:
        - Spin-orbit coupling
        - Zeeman splitting
        - Intervalley scattering Δ_KK'

    Parameters:
        Bz, Bx : float
            Magnetic field components (Tesla).
        gs, gv : float
            Spin and valley g-factors.
        dkk : float
            Intervalley coupling strength.
        soc : float
            Spin-orbit coupling strength.

    Returns:
        ndarray : 4x4 Hamiltonian matrix.
    """
    H = np.zeros((LD, LD), dtype=complex)

    mu_B = 5.78838181E-2  # meV/T (Bohr magneton)

    # Spin-orbit coupling
    H += 0.5 * soc * np.kron(sz, sz)

    # Zeeman effect (spin and valley contributions)
    H += -0.5 * mu_B * gs * (Bz * np.kron(id, sz) + Bx * np.kron(id, sx))
    H +=  0.5 * mu_B * gv * Bz * np.kron(sz, id)

    # Intervalley scattering
    #H += dkk * np.kron(sx, id)

    return H


def ham_h(Bz, Bx, gs=2, gv=14, dkk=0.04, soc=0.07):
    """
    Hole Hamiltonian, mirror-symmetric to electron case.
    Implemented by flipping the sign of SOC.
    """
    return ham_e(Bz, Bx, gs, gv, dkk, -soc)


# ===========================
# Helper Functions
# ===========================

def fermi(E):
    """
    Fermi-Dirac distribution.
    """
    exp = np.exp(E / (kb * T))
    return 1. / (exp + 1.)


# ===========================
# Co-Tunneling Functions
# ===========================
# Based on Nazarov & Blanter (Quantum Transport, 2009)

def fco01(ER, e00, e11, e10, e01, Gr):
    """
    Co-tunneling contribution for process (0,0) -> (-1,1).
    """
    return ((e10 - e11 + ER) / ((e10 - e11 + ER) ** 2 + Gr ** 2) +
            (e01 - e00 - ER) / ((e01 - e00 - ER) ** 2 + Gr ** 2)) ** 2 * \
           1 / (np.exp((e11 - e00 - ER) / kb / T) + 1) / (np.exp(ER / kb / T) + 1)


def fco10(ER, e00, e11, e10, e01, Gr):
    """
    Co-tunneling contribution for process (-1,1) -> (0,0).
    """
    return ((e10 - e11 + ER) / ((e10 - e11 + ER) ** 2 + Gr ** 2) +
            (e01 - e00 - ER) / ((e01 - e00 - ER) ** 2 + Gr ** 2)) ** 2 * \
           (1 - 1 / (np.exp((e11 - e00 - ER) / kb / T) + 1)) * \
           (1 - 1 / (np.exp(ER / kb / T) + 1))


# ===========================
# Rate Equation Construction
# ===========================

def rates(eps, delta_Vl, delta_Vr, Gl_, Gr_, Gd_, t_,
          Bz=0.4, Bx=0, gs=2, gv=14, dkk=0.01, soc=0.07, Vbias=0):
    """
    Build the rate matrix describing all allowed transitions between
    double-dot charge states, including sequential tunneling and
    higher-order co-tunneling.

    Returns:
        rate : ndarray
            Transition rate matrix.
        cur : ndarray
            Current contribution from the right lead.
        cur_L : ndarray
            Current contribution from the left lead.
    """
    # Effective tunneling rates
    Gl = Gl_ * soc / 18 * 17
    Gr = Gr_ * soc / 18 * 17
    Gd = Gd_ * soc / 2   # Dephasing rate
    t  = t_ * soc        # Interdot tunnel coupling

    Vb = Vbias  # Bias voltage (meV)

    # Charging energies for different configurations
    e00 = ec(0,  0, eps, delta_Vl, delta_Vr)
    e11 = ec(-1, 1, eps, delta_Vl, delta_Vr)
    e01 = ec(0,  1, eps, delta_Vl, delta_Vr)
    e10 = ec(-1, 0, eps, delta_Vl, delta_Vr)

    delta_soc = dkk
    soc_e = soc #+ delta_soc
    soc_h = soc #- delta_soc

    delta_dkk = 0.01
    dkk_e = dkk #+ delta_dkk
    dkk_h = dkk #- delta_dkk

    # Diagonalize single-particle Hamiltonians
    we, ve = linalg.eigh(ham_e(Bz, Bx, gs, gv, dkk_e, soc_e))  # electron energies & states
    wh, vh = linalg.eigh(-ham_h(Bz, Bx, gs, gv, dkk_h, soc_h))  # hole energies & states

    ve = np.asarray(ve.T)
    vh = np.asarray(vh.T)

    # Number of states:
    # (0,0) + (-1,1) + (-1,0) + (0,1)
    N = 1 + LD * LD + LD + LD

    # Indexing helpers
    def ind11(a, b): return a * LD + b + 1
    def ind10(a):   return a + LD * LD + 1
    def ind01(b):   return b + LD + LD * LD + 1

    # Initialize rate and current arrays
    rate = np.zeros((N, N))
    cur = np.zeros(N)
    cur_L = np.zeros(N)

    # ==========================
    # Tunnel coupling (0,0) <-> (-1,1)
    # ==========================

    # # Overlap of tunneling Hamiltonian
    #olap = np.zeros((LD, LD))
    #olap_vf = np.zeros((LD, LD))
    # for ih in range(LD):
    #     for ie in range(LD):
    #         o = 0
    #         for s in range(LD):
    #             o += vh[ih][s] * ve[ie][s]
    #         olap[ih, ie] = o ** 2

    O = vh @ ve.T
    olap = O ** 2

    # valley-flipped overlap matrix
    flip_idx = np.roll(np.arange(LD), 2)  # K <-> K' flip
    ve_vf = ve[:, flip_idx]
    O_vf = vh @ ve_vf.T
    olap_vf = O_vf ** 2

    # (0,0) -> (-1,1)
    for jh in range(LD):
        for je in range(LD):
            ei = e00
            ef = e11 + wh[jh] + we[je]
            rate[ind11(jh, je), 0] = (2 * t ** 2 * olap[jh, je] *
                                      np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)) /
                                      np.sqrt(2 * np.pi * Gd ** 2))
            # valley flip interdot tunneling
            rate[ind11(jh, je), 0] += (2 * dkk ** 2 * olap_vf[jh, je] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
                                       np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)))

    # (-1,1) -> (0,0)
    for ih in range(LD):
        for ie in range(LD):
            ei = e11 + wh[ih] + we[ie]
            ef = e00
            rate[0, ind11(ih, ie)] = (2 * t ** 2 * olap[ih, ie] *
                                      np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)) /
                                      np.sqrt(2 * np.pi * Gd ** 2))
            # valley flip interdot tunneling
            rate[0, ind11(ih, ie)] += (2 * dkk ** 2 * olap_vf[ih, ie] * 1 / np.sqrt(2 * np.pi * Gd ** 2) *
                                       np.exp(-(ef - ei) ** 2 / (2 * Gd ** 2)))

    # ==========================
    # Sequential tunneling with leads
    # ==========================

    # Electron out of right dot: (-1,1) -> (-1,0)
    for ih in range(LD):
        for ie in range(LD):
            ei = e11 + wh[ih] + we[ie]
            ef = e10 + wh[ih]
            rat = Gr * fermi(ef - ei - 0.5 * Vb)
            rate[ind10(ih), ind11(ih, ie)] += rat
            cur[ind11(ih, ie)] += rat

    # Electron out of right dot: (0,1) -> (0,0)
    for ie in range(LD):
        ei = e01 + we[ie]
        ef = e00
        rat = Gr * fermi(ef - ei - 0.5 * Vb)
        rate[0, ind01(ie)] += rat
        cur[ind01(ie)] += rat

    # Hole out of left dot: (-1,1) -> (0,1)
    for ih in range(LD):
        for ie in range(LD):
            ei = e11 + wh[ih] + we[ie]
            ef = e01 + we[ie]
            rat = Gl * fermi(ef - ei - 0.5 * Vb)
            rate[ind01(ie), ind11(ih, ie)] += rat
            cur_L[ind11(ih, ie)] += rat

    # Hole out of left dot: (-1,0) -> (0,0)
    for ih in range(LD):
        ei = e10 + wh[ih]
        ef = e00
        rat = Gl * fermi(ef - ei - 0.5 * Vb)
        rate[0, ind10(ih)] += rat
        cur_L[ind10(ih)] += rat

    # Reverse processes
    # Electron in right dot: (-1,0) -> (-1,1)
    for jh in range(LD):
        for je in range(LD):
            ei = e10 + wh[jh]
            ef = e11 + wh[jh] + we[je]
            rat = Gr * fermi(ef - ei + 0.5 * Vb)
            rate[ind11(jh, je), ind10(jh)] += rat
            cur[ind10(jh)] -= rat

    # Electron in right dot: (0,0) -> (0,1)
    for je in range(LD):
        ei = e00
        ef = e01 + we[je]
        rat = Gr * fermi(ef - ei + 0.5 * Vb)
        rate[ind01(je), 0] += rat
        cur[0] -= rat

    # Hole in left dot: (0,1) -> (-1,1)
    for jh in range(LD):
        for je in range(LD):
            ei = e01 + we[je]
            ef = e11 + we[je] + wh[jh]
            rat = Gl * fermi(ef - ei + 0.5 * Vb)
            rate[ind11(jh, je), ind01(je)] += rat
            cur_L[ind01(je)] -= rat

    # Hole in left dot: (0,0) -> (-1,0)
    for jh in range(LD):
        ei = e00
        ef = e10 + wh[jh]
        rat = Gl * fermi(ef - ei + 0.5 * Vb)
        rate[ind10(jh), 0] += rat
        cur_L[0] -= rat

    # ==========================
    # Co-tunneling corrections
    # ==========================
    #n, m = 0, 0
    # for jh in range(LD):
    #     for je in range(LD):
    #         E00 = e00
    #         E11 = e11 + wh[jh] + we[je]
    #         E10 = e10 + wh[jh]
    #         E01 = e01 + we[je]
    #
    #         if E10 > E00 and E01 > E00 and E10 + m * Gr > E11 and E01 + m * Gl > E11:
    #             integral_10, _ = spi.quad(fco10, -n * kb * T,
    #                                       np.abs(E11 - E00) + n * kb * T,
    #                                       args=(E00, E11, E10, E01, Gr))
    #             tco10 = Gl * Gr * integral_10
    #
    #             integral_01, _ = spi.quad(fco01,
    #                                       -np.abs(E11 - E00) - n * kb * T,
    #                                       n * kb * T,
    #                                       args=(E00, E11, E10, E01, Gr))
    #             tco01 = Gl * Gr * integral_01
    #
    #             # Update rates
    #             rate[0, ind11(jh, je)] += tco10  # (-1,1) -> (0,0)
    #             rate[ind11(jh, je), 0] += tco01  # (0,0) -> (-1,1)

    # Probability conservation: diagonal = -sum(outgoing rates)
    for j in range(N):
        rate[j, j] = - np.sum(rate[:, j])

    return rate, cur, cur_L

# -----------------------
# Helper: steady-state solution
# -----------------------
def steady_state(rate):
    # Solve for steady state (eigenvector of rate matrix with eigenvalue 0)
    w, v = linalg.eig(rate)
    el = np.argmax(w)
    assert abs(w[el]) < 1e-10
    #pd = v[:, el] / np.sum(v[:, el])

    P = np.real(v[:, el])
    P = np.abs(P)  # ensure non-negative
    P /= np.sum(P)  # normalize safely

    return (np.sum(P[1:17]) + 0.2 * np.sum(P[17:21]))


# ===========================
# Steady-state solution
# ===========================

def compute_CUR_element(i, eps, DVL_row, DVR_row, Gl_, Gr_, Gd_, t_,
                         Bz, Bx, gs, gv, dkk, soc, Vbias):
    """
    Compute steady-state solution for a single row of the ΔVl-ΔVr grid (parallelizable).
    """
    row_vals = np.zeros(len(DVL_row))
    for j in range(len(DVR_row)):
        rate, cur, cur_L = rates(eps, DVL_row[j], DVR_row[j],
                                 Gl_, Gr_, Gd_, t_, Bz, Bx,
                                 gs, gv, dkk, soc, Vbias)
        # Solve for steady state (eigenvector of rate matrix with eigenvalue 0)
        w, v = linalg.eig(rate)
        el = np.argmax(w)
        assert abs(w[el]) < 1e-10
        #pd = v[:, el] / np.sum(v[:, el])

        P = np.real(v[:, el])
        P = np.abs(P)  # ensure non-negative
        P /= np.sum(P)  # normalize safely

        row_vals[j] = (1 - np.sum(P[1:17])) + 0.2 * np.sum(P[17:21]) - 0.8 * np.sum(P[21:])#(1 - np.sum(P[1:17])) + 0.2 * np.sum(P[17:21]) - 1.2 * np.sum(P[21:])

        # Convert current from meV → pA
        #cur *= 38740
        #row_vals[j] = np.real(np.dot(cur, pd))
    return i, row_vals


def compute_CUR_parallel(args_list, CUR, num_cpus=20):
    """
    Parallelized computation of current map.
    """
    with Pool(num_cpus) as pool:
        results = pool.starmap(compute_CUR_element, args_list)

    for i, vals in results:
        CUR[i, :] = vals

def compute_detuning(epsilon, DVl, DVr, Gl_, Gr_, Gd_, t_,
                        Bz, Bx, gs, gv, dkk, soc, Vbias):
    row_vals = np.zeros(len(epsilon))
    for n_e, eps in enumerate(epsilon):
        r_mat, cur, cur_L = rates(eps, DVl, DVr,
                                  Gl_, Gr_, Gd_, t_,
                                  Bz, Bx, gs, gv, dkk, soc, Vbias)
        row_vals[n_e] = steady_state(r_mat)
    return row_vals


def plot_CUR(DVL, DVR, PLOT, param, dir=None):
    """
    Plot and save current map as colormap.
    """
    plt.pcolormesh(DVL, DVR, PLOT,
                   cmap="RdBu_r", shading="auto", rasterized=True)
    plt.colorbar()
    plt.title(f'{param}')
    plt.savefig(os.path.join(dir, f'{param}_map.png'))
    plt.close()

def plot_waterfall(epsilon, all_probs, dkks, sim_dir, param_name="dkk"):
    for idx, (prob_curve, dkk_val) in enumerate(zip(all_probs, dkks)):
        # Add a vertical offset for clarity (optional, comment out if not needed)
        offset = idx * 0.1
        plt.plot(epsilon, prob_curve + offset, label=f"{param_name}={dkk_val:.3f}")

    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Steady-state probability")
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_dir, f"waterfall_{param_name}.png"), dpi=300)
    plt.close()



def read_parameters(file_path):
    """
    Read parameter sets from JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


# ===========================
# Main Simulation Driver
# ===========================

def main(params=None, sim_dir=None):
    """
    Run current computation for a given parameter set and save output plots.
    """
    split = params["split"]
    delta_Vl = np.linspace(params["delta_Vl_start"], params["delta_Vl_stop"], split)
    delta_Vr = np.linspace(params["delta_Vr_start"], params["delta_Vr_stop"], split)
    DVL, DVR = np.meshgrid(delta_Vl, delta_Vr)

    CUR = np.zeros((split, split))

    args_list = [
        (i, 0, DVL[i], DVR[i], params["Gl"], params["Gr"], params["Gd"],
         params["t"], params["Bz"], params["Bx"], params["gs"], params["gv"],
         params["dkk"], params["soc"], params["Vbias"])
        for i in range(split)
    ]

    if sim_dir is None:
        sim_dir = os.getcwd()

    compute_CUR_parallel(CUR=CUR, args_list=args_list)
    plot_CUR(DVL=DVL, DVR=DVR, PLOT=CUR, param=params["dkk"], dir=sim_dir)

    all_probs = []
    dkks = []

    epsilon = np.linspace(-0.5, 0.5, split)

    for params in parameter_sets:
        # Compute detuning vs probability for this dkk
        prob_curve = compute_detuning(
            epsilon, -6.3,  2,
            params["Gl"], params["Gr"], params["Gd"], params["t"],
            params["Bz"], params["Bx"], params["gs"], params["gv"],
            params["dkk"], params["soc"], params["Vbias"]
        )

        all_probs.append(prob_curve)
        dkks.append(params["dkk"])

    np.save(os.path.join(sim_dir, f"dkk"), dkks)
    np.save(os.path.join(sim_dir, f"all_cuts"), all_probs)
    np.save(os.path.join(sim_dir, f"epsilon"), epsilon)

    # After loop, generate waterfall plot
    plot_waterfall(epsilon, all_probs, dkks, sim_dir)


if __name__ == '__main__':
    dir_name = 'ss_vs_tvf'
    current_dir = os.getcwd()
    sim_dir = os.path.join(current_dir, dir_name)
    parameter_file = os.path.join(sim_dir, 'params.json')  # Path to parameter file
    parameter_sets = read_parameters(parameter_file)

    initial_time = time.time()
    for params in parameter_sets:
        main(params, sim_dir)
    final_time = time.time()

    runtime = final_time - initial_time
    print("Simulation time (s):", runtime)
