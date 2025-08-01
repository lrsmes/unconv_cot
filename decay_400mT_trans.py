import numpy as np
import matplotlib.pyplot as plt
import os
import hdf5_helper as helper
import style_sheet
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit


style_sheet.set_custom_style('two columns', aspect_ratio=0.4)

#### Load data ####
current_dir = os.getcwd()
file_dir = os.path.join(current_dir, '400mT_both_dir')

t_trans = np.load(os.path.join(file_dir, 'tread_transport.npy'))
rat_trans = np.load(os.path.join(file_dir, 'ratios_transport.npy'))
err_trans = np.load(os.path.join(file_dir, 'ratios_err_transport.npy'))

def exp_fit(x, a, b, c):
    return a * np.exp(-x/b) + c

# Fit exponential to given x, y data
def fit_exponential(x, y):
    popt, _ = curve_fit(exp_fit, x, y, p0=(1, 0.08, 0.1))  # Initial guess
    return popt  # Returns fitted parameters (a, b, c)

popt_low_trans = []
popt_up_trans = []

for i in range(3):
    popt_low_trans.append(fit_exponential(t_trans, rat_trans[:, i] - err_trans[:, i]))
    popt_up_trans.append(fit_exponential(t_trans[:], rat_trans[:, i] + err_trans[:, i]))
    print((rat_trans[:, i] - err_trans[:, i]))

# Define figure and grid layout with custom column widths
fig = plt.figure()

# First plot (Top, spanning columns 2 & 3)
x_smooth = np.linspace(0.01, 2.6, 500)

#plt.fill_between(t_trans, np.maximum(rat_trans[:, 0] - err_trans[:, 0], 1e-6),
#                 rat_trans[:, 0] + err_trans[:, 0], color='orangered', alpha=0.2)
plt.fill_between(x_smooth, exp_fit(x_smooth, *popt_up_trans[0]),
                 exp_fit(x_smooth, *popt_low_trans[0]), color='orangered', alpha=0.2)
#plt.plot(x_smooth, exp_fit(x_smooth, *popt_up_trans[0]), '--', color='gray', linewidth=1.5)
#plt.plot(x_smooth, exp_fit(x_smooth, *popt_low_trans[0]), '--', color='gray', linewidth=1.5)
plt.scatter(t_trans, rat_trans[:, 0], s=(6 * (np.pi / 4)**-1) ** 2, color='orangered',
            edgecolor='black', linewidths=2)

#plt.fill_between(t_trans, np.maximum(rat_trans[:, 1] - err_trans[:, 1], 1e-6),
#                 rat_trans[:, 1] + err_trans[:, 1], color='blueviolet', alpha=0.2)
plt.fill_between(x_smooth, exp_fit(x_smooth, *popt_up_trans[1]),
                 exp_fit(x_smooth, *popt_low_trans[1]), color='blueviolet', alpha=0.2)
#plt.plot(x_smooth, exp_fit(x_smooth, *popt_up_trans[1]), '--', color='gray', linewidth=1.5)
#plt.plot(x_smooth, exp_fit(x_smooth, *popt_low_trans[1]), '--', color='gray', linewidth=1.5)
plt.scatter(t_trans, rat_trans[:, 1], s=(6 * (np.pi / 4)**-1) ** 2, color='blueviolet',
            edgecolor='black', linewidths=2)


#plt.fill_between(t_trans, np.maximum(rat_trans[:, 2] - err_trans[:, 2], 1e-6),
#                 rat_trans[:, 2] + err_trans[:, 2], color='indianred', alpha=0.2)
plt.fill_between(x_smooth, exp_fit(x_smooth, *popt_up_trans[2]),
                 exp_fit(x_smooth, *popt_low_trans[2]), color='indianred', alpha=0.2)
# Plot dashed gray lines for fitted boundaries
#plt.plot(x_smooth, exp_fit(x_smooth, *popt_up_trans[2]), '--', color='gray', linewidth=1.5)
#plt.plot(x_smooth, exp_fit(x_smooth, *popt_low_trans[2]), '--', color='gray', linewidth=1.5)
plt.scatter(t_trans, rat_trans[:, 2], s=(6 * (np.pi / 4)**-1) ** 2, color='indianred',
            edgecolor='black', linewidths=2)

#ax1.set_xscale('log')  # Log scale for x-axis
plt.ylim(0, 1)
plt.xlim(0, None)
#ax1.set_xlabel(r'$t_{read} (\mu s)$')
plt.ylabel(r'P(1, 1)')
plt.xlabel(r'$t_{read} (\mu s)$')

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(file_dir, '400mT_decay_trans.pdf'))
plt.savefig(os.path.join(file_dir, '400mT_decay_trans.svg'))

plt.show()