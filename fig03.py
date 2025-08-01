import os
import sys
import matplotlib.pyplot as plt
import autograd.numpy as np
import warnings
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
from scipy import constants as co
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import bwr, coolwarm, viridis, twilight_shifted

def load_npys():

    file_0T = '\\Users\Mester.INSTITUT2B\\PycharmProjects\\Animations\\0T_both_dir\\'
    file_400mT = '\\Users\Mester.INSTITUT2B\\PycharmProjects\\Animations\\400mT_both_dir\\'
    file_theo = '\\Users\Mester.INSTITUT2B\\PycharmProjects\\Animations\\New_parameters_probabilities\\'
    file_dir = os.getcwd()

    map_0T_block = np.load(os.path.join(file_0T, '1_1857_map.npy'))
    X_0T_block = np.load(os.path.join(file_0T, '1_1857_FG14.npy'))
    Y_0T_block = np.load(os.path.join(file_0T, '1_1857_FG12.npy'))

    map_0T_trans = np.load(os.path.join(file_0T, '-1_190_map.npy'))
    X_0T_trans = np.load(os.path.join(file_0T, '-1_190_FG14.npy'))
    Y_0T_trans = np.load(os.path.join(file_0T, '-1_190_FG12.npy'))

    map_0T_block_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_11\\t=1.6\\Bz_0.4_t_1.6_P00.npy'))
    # X_0T_block_theo = np.load(os.path.join(file_theo, ' Bz=0\\Starting_11\\t={}.npy'))
    # Y_0T_block_theo = np.load(os.path.join(file_theo, ' Bz=0\\Starting_11\\t={}.npy'))

    map_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t=0.4\\Bz_0.4_t_0.4_P00.npy'))
    # X_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0\\Starting_00\\t={}.npy'))
    # Y_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0\\Starting_00\\t={}.npy'))

    map_400mT_block = np.load(os.path.join(file_400mT, '1_2950_map.npy'))
    X_400mT_block = np.load(os.path.join(file_400mT, '1_2950_FG14.npy'))
    Y_400mT_block = np.load(os.path.join(file_400mT, '1_2950_FG12.npy'))

    map_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_map.npy'))
    X_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_FG14.npy'))
    Y_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_FG12.npy'))

    map_400mT_trans_theo = np.load(os.path.join(file_theo, 'npys\\Bz=0.4\\Starting_00\\t=0.4.npy'))
    # X_400mT_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t={}.npy'))
    # Y_400mT_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t={}.npy'))


def generate_fig03():
    file_0T = '\\Users\Mester.INSTITUT2B\\PycharmProjects\\Animations\\0T_both_dir\\'
    file_400mT = '\\Users\Mester.INSTITUT2B\\PycharmProjects\\Animations\\400mT_both_dir\\'
    file_theo = '\\Users\Mester.INSTITUT2B\\PycharmProjects\\Animations\\New_parameters_probabilities\\'
    file_dir = os.getcwd()

    map_0T_block = np.load(os.path.join(file_0T, '1_1857_map.npy'))
    X_0T_block = np.load(os.path.join(file_0T, '1_1857_FG14.npy'))
    Y_0T_block = np.load(os.path.join(file_0T, '1_1857_FG12.npy'))

    map_0T_trans = np.load(os.path.join(file_0T, '-1_190_map.npy'))
    X_0T_trans = np.load(os.path.join(file_0T, '-1_190_FG14.npy'))
    Y_0T_trans = np.load(os.path.join(file_0T, '-1_190_FG12.npy'))

    map_0T_block_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_11\\t=1.6\\Bz_0.4_t_1.6_P00.npy'))
    # X_0T_block_theo = np.load(os.path.join(file_theo, ' Bz=0\\Starting_11\\t={}.npy'))
    # Y_0T_block_theo = np.load(os.path.join(file_theo, ' Bz=0\\Starting_11\\t={}.npy'))

    map_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t=0.4\\Bz_0.4_t_0.4_P00.npy'))
    # X_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0\\Starting_00\\t={}.npy'))
    # Y_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0\\Starting_00\\t={}.npy'))

    map_400mT_block = np.load(os.path.join(file_400mT, '1_2950_map.npy'))
    X_400mT_block = np.load(os.path.join(file_400mT, '1_2950_FG14.npy'))
    Y_400mT_block = np.load(os.path.join(file_400mT, '1_2950_FG12.npy'))

    map_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_map.npy'))
    X_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_FG14.npy'))
    Y_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_FG12.npy'))

    map_400mT_trans_theo = np.load(os.path.join(file_theo, 'npys\\Bz=0.4\\Starting_00\\t=0.4.npy'))
    # X_400mT_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t={}.npy'))
    # Y_400mT_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t={}.npy'))


    plt.rcParams.update({'font.size': 7.5})
    cm = 1 / 2.54
    formatter = mticker.FuncFormatter(lambda x, _: f'{x:.1f}')
    warnings.filterwarnings('ignore')
    fig = plt.figure(figsize=(1 * 8.50 * cm, 1. * 10. * cm), dpi=869)
    fig_fac = 1.0

    positions = [
        (0.125, 0.71, 0.365 * fig_fac, 0.28),  # blockade 0T experiment
        (0.6, 0.71, 0.365 * fig_fac, 0.28),  # transport 0T experiment
        (0.125, 0.39, 0.365 * fig_fac, 0.28),  # blockade 0T theory
        (0.6, 0.39, 0.365 * fig_fac, 0.28),  # transport 0T theory
        (0.125, 0.08, 0.365 * fig_fac, 0.28),  # transport 400mT experiment
        (0.6, 0.08, 0.365 * fig_fac, 0.28),  # transport 400mT theory
    ]

    # (a)
    ax00 = fig.add_axes(positions[0])
    c1 = ax00.pcolormesh(X_0T_block, Y_0T_block, map_0T_block, cmap="viridis_r", shading="auto", rasterized=True)
    cbar = fig.colorbar(c1, ax=ax00, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
    cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
                 rotation=0)
    ax00.tick_params(axis='both', which='major', pad=1, direction='in')
    ax00.set_xticks([5.18, 5.19])
    ax00.set_yticks([5.27, 5.28])
    ax00.set_ylabel(r"$FG_{2} (V)$", labelpad=1)
    # ax00.set_xlabel(r"$FG_{1} (V)$")
    #ax00.text(-0.45, 1.14, labels[0], transform=ax00.transAxes, fontsize=9, fontweight='bold')
    # ax.set_aspect('equal')
    # ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
    # ax00.text(5.187, 5.282, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(1857 * 125 * 1e-6, 2)),
    #        fontsize=5, color='white')
    ax00.text(5.183, 5.273, r'(1e,1h)', fontsize=6, color='white')
    ax00.text(5.172, 5.289, r'(0e,0h)', fontsize=6, color='black')
    ax00.text(5.189, 5.281, r'(1e,0h)', fontsize=6, color='white')
    ax00.text(5.172, 5.263, r'(0e,1h)', fontsize=6, color='black')
    ax00.text(5.191, 5.269, r'B=0', fontsize=7, color='white')

    ax01 = fig.add_axes(positions[1])
    c1 = ax01.pcolormesh(X_0T_trans, Y_0T_trans, map_0T_trans, cmap="viridis_r", shading="auto", rasterized=True)
    cbar = fig.colorbar(c1, ax=ax01, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
    cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
                 rotation=0)
    ax01.tick_params(axis='both', which='major', pad=1, direction='in')
    ax01.set_xticks([5.17, 5.18])
    ax01.set_yticks([5.28, 5.29])
    # ax01.set_ylabel(r"$FG_{2} (V)$")
    # ax01.set_xlabel(r"$FG_{1} (V)$")
    #ax01.text(-0.25, 1.14, labels[1], transform=ax01.transAxes, fontsize=9, fontweight='bold')
    # ax.set_aspect('equal')
    # ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
    # ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
    #        fontsize=6, color='white')
    ax01.text(5.1775, 5.278, r'(1e,1h)', fontsize=6, color='black')
    ax01.text(5.172, 5.292, r'(0e,0h)', fontsize=6, color='white')
    ax01.text(5.182, 5.291, r'(1e,0h)', fontsize=6, color='black')
    ax01.text(5.17, 5.279, r'(0e,1h)', fontsize=6, color='white')
    ax01.text(5.1845, 5.2775, r'B=0', fontsize=7, color='black')

    ax10 = fig.add_axes(positions[2])
    c1 = ax10.pcolormesh(map_0T_block_theo, cmap="viridis_r", shading="auto", rasterized=True)
    cbar = fig.colorbar(c1, ax=ax10, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
    cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
                 rotation=0)
    ax10.tick_params(axis='both', which='major', pad=1, direction='in')
    # ax10.set_xticks([5.18])
    # ax10.set_yticks([5.26, 5.27])
    ax10.set_ylabel(r"$FG_{2} (V)$", labelpad=1)
    # ax10.set_xlabel(r"$FG_{1} (V)$")
    #ax10.text(-0.45, 1.14, labels[2], transform=ax10.transAxes, fontsize=9, fontweight='bold')
    # ax.set_aspect('equal')
    # ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
    # ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
    #        fontsize=6, color='white')

    ax11 = fig.add_axes(positions[3])
    c1 = ax11.pcolormesh(map_0T_trans_theo, cmap="viridis_r", shading="auto", rasterized=True)
    cbar = fig.colorbar(c1, ax=ax11, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
    cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
                 rotation=0)
    ax11.tick_params(axis='both', which='major', pad=1, direction='in')
    # ax11.set_xticks([5.18])
    # ax11.set_yticks([5.26, 5.27])
    # ax11.set_ylabel(r"$FG_{2} (V)$")
    # ax11.set_xlabel(r"$FG_{1} (V)$")
    #ax11.text(-0.25, 1.14, labels[3], transform=ax11.transAxes, fontsize=9, fontweight='bold')
    # ax.set_aspect('equal')
    # ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
    # ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
    #        fontsize=6, color='white')

    ax20 = fig.add_axes(positions[4])
    c1 = ax20.pcolormesh(X_400mT_trans, Y_400mT_trans, map_400mT_trans, cmap="viridis_r", shading="auto",
                         rasterized=True)
    cbar = fig.colorbar(c1, ax=ax20, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
    cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
                 rotation=0)
    ax20.tick_params(axis='both', which='major', pad=1, direction='in')
    ax20.set_xticks([5.17, 5.18])
    ax20.set_yticks([5.27, 5.28])
    ax20.set_ylabel(r"$FG_{2} (V)$", labelpad=1)
    ax20.set_xlabel(r"$FG_{1} (V)$", labelpad=1)
    #ax20.text(-0.45, 1.14, labels[4], transform=ax20.transAxes, fontsize=9, fontweight='bold')
    # ax.set_aspect('equal')
    # ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
    # ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
    #        fontsize=6, color='white')
    ax20.text(5.182, 5.262, r'(1e,1h)', fontsize=6, color='black')
    ax20.text(5.1735, 5.276, r'(0e,0h)', fontsize=6, color='white')
    ax20.text(5.1805, 5.278, r'(1e,0h)', fontsize=6, color='black')
    ax20.text(5.17, 5.268, r'(0e,1h)', fontsize=6, color='white')
    ax20.text(5.1705, 5.2812, r'B$\neq$0', fontsize=7, color='white')

    ax21 = fig.add_axes(positions[5])
    c1 = ax21.pcolormesh(map_400mT_trans_theo,
                         cmap="viridis_r", shading="auto", rasterized=True)
    cbar = fig.colorbar(c1, ax=ax21, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
    cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
                 rotation=0)
    ax21.tick_params(axis='both', which='major', pad=1, direction='in')
    # ax21.set_xticks([5.18])
    # ax21.set_yticks([5.26, 5.27])
    # ax21.set_ylabel(r"$FG_{2} (V)$")
    ax21.set_xlabel(r"$FG_{1} (V)$", labelpad=1)
    #ax21.text(-0.25, 1.14, labels[5], transform=ax21.transAxes, fontsize=9, fontweight='bold')
    # ax.set_aspect('equal')
    # ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
    # ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
    #        fontsize=6, color='white')

    plt.show()


if __name__ == '__main__':
    generate_fig03()