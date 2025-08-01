import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import style_sheet

style_sheet.set_custom_style('one column', aspect_ratio=1)

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
#X_0T_block_theo = np.load(os.path.join(file_theo, ' Bz=0\\Starting_11\\t={}.npy'))
#Y_0T_block_theo = np.load(os.path.join(file_theo, ' Bz=0\\Starting_11\\t={}.npy'))

map_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t=0.4\\Bz_0.4_t_0.4_P00.npy'))
#X_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0\\Starting_00\\t={}.npy'))
#Y_0T_trans_theo = np.load(os.path.join(file_theo, 'Bz=0\\Starting_00\\t={}.npy'))


map_400mT_block = np.load(os.path.join(file_400mT, '1_2950_map.npy'))
X_400mT_block = np.load(os.path.join(file_400mT, '1_2950_FG14.npy'))
Y_400mT_block = np.load(os.path.join(file_400mT, '1_2950_FG12.npy'))

map_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_map.npy'))
X_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_FG14.npy'))
Y_400mT_trans = np.load(os.path.join(file_400mT, '-1_2240_FG12.npy'))

map_400mT_trans_theo = np.load(os.path.join(file_theo, 'npys\\Bz=0.4\\Starting_00\\t=0.4.npy'))
#X_400mT_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t={}.npy'))
#Y_400mT_trans_theo = np.load(os.path.join(file_theo, 'Bz=0.4\\Starting_00\\t={}.npy'))

fig, axes = plt.subplots(3, 2)
fig.subplots_adjust(wspace=0.01, hspace=0.01)
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


ax00 = axes[0, 0]
c1 = ax00.pcolormesh(X_0T_block, Y_0T_block, map_0T_block, cmap="viridis_r", shading="auto", rasterized=True)
cbar = fig.colorbar(c1, ax=ax00, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
cbar.set_ticks([0, 1])
cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
             rotation=0)
ax00.set_xticks([5.18, 5.19])
ax00.set_yticks([5.27, 5.28])
ax00.set_ylabel(r"$FG_{2} (V)$")
#ax00.set_xlabel(r"$FG_{1} (V)$")
ax00.text(-0.45, 1.14, labels[0], transform=ax00.transAxes, fontsize=9, fontweight='bold')
# ax.set_aspect('equal')
# ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
#ax00.text(5.187, 5.282, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(1857 * 125 * 1e-6, 2)),
#        fontsize=5, color='white')
ax00.text(5.183, 5.273, r'(1e,1h)', fontsize=6, color='white')
ax00.text(5.172, 5.289, r'(0e,0h)', fontsize=6, color='black')
ax00.text(5.189, 5.281, r'(1e,0h)', fontsize=6, color='white')
ax00.text(5.172, 5.263, r'(0e,1h)', fontsize=6, color='black')
ax00.text(5.191, 5.269, r'B=0', fontsize=7, color='white')

ax01 = axes[0, 1]
c1 = ax01.pcolormesh(X_0T_trans, Y_0T_trans, map_0T_trans, cmap="viridis_r", shading="auto", rasterized=True)
cbar = fig.colorbar(c1, ax=ax01, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
cbar.set_ticks([0, 1])
cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
             rotation=0)
ax01.set_xticks([5.17, 5.18])
ax01.set_yticks([5.28, 5.29])
#ax01.set_ylabel(r"$FG_{2} (V)$")
#ax01.set_xlabel(r"$FG_{1} (V)$")
ax01.text(-0.25, 1.14, labels[1], transform=ax01.transAxes, fontsize=9, fontweight='bold')
# ax.set_aspect('equal')
# ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
# ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
#        fontsize=6, color='white')
ax01.text(5.1775, 5.278, r'(1e,1h)', fontsize=6, color='black')
ax01.text(5.172, 5.292, r'(0e,0h)', fontsize=6, color='white')
ax01.text(5.182, 5.291, r'(1e,0h)', fontsize=6, color='black')
ax01.text(5.17, 5.279, r'(0e,1h)', fontsize=6, color='white')
ax01.text(5.1845, 5.2775, r'B=0', fontsize=7, color='black')

ax10 = axes[1, 0]
c1 = ax10.pcolormesh(map_0T_block_theo, cmap="viridis_r", shading="auto", rasterized=True)
cbar = fig.colorbar(c1, ax=ax10, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
cbar.set_ticks([0, 1])
cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
             rotation=0)
#ax10.set_xticks([5.18])
#ax10.set_yticks([5.26, 5.27])
ax10.set_ylabel(r"$FG_{2} (V)$")
#ax10.set_xlabel(r"$FG_{1} (V)$")
ax10.text(-0.45, 1.14, labels[2], transform=ax10.transAxes, fontsize=9, fontweight='bold')
# ax.set_aspect('equal')
# ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
# ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
#        fontsize=6, color='white')

ax11 = axes[1, 1]
c1 = ax11.pcolormesh(map_0T_trans_theo, cmap="viridis_r", shading="auto", rasterized=True)
cbar = fig.colorbar(c1, ax=ax11, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
cbar.set_ticks([0, 1])
cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
             rotation=0)
#ax11.set_xticks([5.18])
#ax11.set_yticks([5.26, 5.27])
#ax11.set_ylabel(r"$FG_{2} (V)$")
#ax11.set_xlabel(r"$FG_{1} (V)$")
ax11.text(-0.25, 1.14, labels[3], transform=ax11.transAxes, fontsize=9, fontweight='bold')
# ax.set_aspect('equal')
# ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
# ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
#        fontsize=6, color='white')

ax20 = axes[2, 0]
c1 = ax20.pcolormesh(X_400mT_trans, Y_400mT_trans, map_400mT_trans, cmap="viridis_r", shading="auto", rasterized=True)
cbar = fig.colorbar(c1, ax=ax20, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
cbar.set_ticks([0, 1])
cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
             rotation=0)
ax20.set_xticks([5.17, 5.18])
ax20.set_yticks([5.27, 5.28])
ax20.set_ylabel(r"$FG_{2} (V)$")
ax20.set_xlabel(r"$FG_{1} (V)$")
ax20.text(-0.45, 1.14, labels[4], transform=ax20.transAxes, fontsize=9, fontweight='bold')
# ax.set_aspect('equal')
# ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
# ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
#        fontsize=6, color='white')
ax20.text(5.18, 5.261, r'(1e,1h)', fontsize=6, color='black')
ax20.text(5.1735, 5.276, r'(0e,0h)', fontsize=6, color='white')
ax20.text(5.18, 5.278, r'(1e,0h)', fontsize=6, color='black')
ax20.text(5.17, 5.268, r'(0e,1h)', fontsize=6, color='white')
ax20.text(5.1705, 5.2812, r'B$\neq$0', fontsize=7, color='white')

ax21 = axes[2, 1]
c1 = ax21.pcolormesh(map_400mT_trans_theo,
                     cmap="viridis_r", shading="auto", rasterized=True)
cbar = fig.colorbar(c1, ax=ax21, location='top', shrink=0.3, aspect=15, pad=0.035, anchor=(1.0, 0.0))
cbar.set_ticks([0, 1])
cbar.ax.tick_params(direction='in', length=1.5, pad=0.04)
cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes,
             rotation=0)
#ax21.set_xticks([5.18])
#ax21.set_yticks([5.26, 5.27])
#ax21.set_ylabel(r"$FG_{2} (V)$")
ax21.set_xlabel(r"$FG_{1} (V)$")
ax21.text(-0.25, 1.14, labels[5], transform=ax21.transAxes, fontsize=9, fontweight='bold')
# ax.set_aspect('equal')
# ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
# ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
#        fontsize=6, color='white')

plt.tight_layout()
plt.savefig(os.path.join(file_dir, 'Fig_3.pdf'))
plt.savefig(os.path.join(file_dir, 'Fig_3.svg'))

plt.show()
