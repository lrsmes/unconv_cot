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

t_block = np.load(os.path.join(file_dir, 'tread_blockade.npy'))
rat_block = np.load(os.path.join(file_dir, 'ratios_blockade.npy'))
err_block = np.load(os.path.join(file_dir, 'ratios_err_blockade.npy'))

def exp_fit(x, a, b, c):
    return a * np.exp(-x/b) + c

# Fit exponential to given x, y data
def fit_exponential(x, y):
    popt, _ = curve_fit(exp_fit, x, y, p0=(1, 0.08, 0.1))  # Initial guess
    return popt  # Returns fitted parameters (a, b, c)

# Fit exponential curves for the error boundaries
popt_up_block = fit_exponential(t_block, rat_block + err_block)  # Upper boundary fit
popt_low_block = fit_exponential(t_block, rat_block - err_block)  # Lower boundary fit

# Define figure and grid layout with custom column widths
fig = plt.figure()

# First plot (Top, spanning columns 2 & 3)
x_smooth = np.linspace(0, 5.1, 500)

plt.scatter(t_block, rat_block, s=(6 * (np.pi / 4)**-1) ** 2, color='mediumblue',
            edgecolor='black', linewidths=2)
#ax1.fill_between(t_block, np.maximum(rat_block - err_block, 1e-6), rat_block + err_block,
#                 color='mediumblue', alpha=0.2)
plt.fill_between(x_smooth, exp_fit(x_smooth, *popt_up_block),
                 exp_fit(x_smooth, *popt_low_block), color='mediumblue', alpha=0.2)
#ax1.set_xscale('log')  # Log scale for x-axis
plt.ylim(0, 1)
plt.xlim(0, None)
#ax1.set_xlabel(r'$t_{read} (\mu s)$')
plt.ylabel(r'P(1, 1)')
plt.xlabel(r'$t_{read} (\mu s)$')

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(file_dir, '400mT_decay_block.pdf'))
plt.savefig(os.path.join(file_dir, '400mT_decay_block.svg'))

plt.show()
