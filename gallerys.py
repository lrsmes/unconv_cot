import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import style_sheet

style_sheet.set_custom_style('two columns', aspect_ratio=0.4)

def create_gallery(read_block=None, read_trans=None, dir=None):
    #### Load data ####
    #current_dir = os.getcwd()
    #file_dir = os.path.join(current_dir, dir)
    file_dir = '\\Users\Mester.INSTITUT2B\\PycharmProjects\\Animations\\400mT_both_dir\\'

    maps_read = []
    X_read = []
    Y_read = []
    maps_trans = []
    X_trans = []
    Y_trans = []

    for t in read_block:
        maps_read.append(np.load(os.path.join(file_dir, '1_{}_map.npy'.format(t))))
        X_read.append(np.load(os.path.join(file_dir, '1_{}_FG14.npy'.format(t))))
        Y_read.append(np.load(os.path.join(file_dir, '1_{}_FG12.npy'.format(t))))


    for t in read_trans:
        maps_trans.append(np.load(os.path.join(file_dir, '-1_{}_map.npy'.format(t))))
        X_trans.append(np.load(os.path.join(file_dir, '-1_{}_FG14.npy'.format(t))))
        Y_trans.append(np.load(os.path.join(file_dir, '-1_{}_FG12.npy'.format(t))))

    print(maps_read)

    fig, axes = plt.subplots(2, 6)
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']

    for i in range(6):
        ax = axes[0, i]
        c1 = ax.pcolormesh(X_read[i], Y_read[i], maps_read[i], cmap="viridis_r", shading="auto", rasterized=True)
        cbar = fig.colorbar(c1, ax=ax, location='top', shrink=0.3, aspect=15, pad=0.02, anchor=(1.0, 0.0))
        cbar.set_ticks([0, 1])
        cbar.ax.tick_params(direction='in', length=1.5)
        cbar.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar.ax.transAxes, rotation=0)
        ax.set_xticks([5.18])
        if i == 0:
            ax.set_yticks([5.26, 5.27])
            ax.set_ylabel(r"$FG_{2} (V)$")
            ax.set_ylabel(r"$FG_{2} (V)$")
        else:
            ax.set_yticks([5.26, 5.27], labels=['', ''])
        #ax.set_xlabel(r"$FG_{1} (V)$", fontsize=20)
        #ax.set_title(f"Read Map {read_block[i]}")
        ax.text(0, 1.1, labels[i], transform=ax.transAxes, fontsize=9, fontweight='bold')
        #ax.set_aspect('equal')
        #ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
        #ax.text(5.18, 5.264, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_block[i] * 125 * 1e-6, 2)),
        #        fontsize=6, color='white')

    for i in range(6):
        ax = axes[1, i]
        c2 = ax.pcolormesh(X_trans[i], Y_trans[i], maps_trans[i], cmap="viridis_r", shading="auto", rasterized=True)
        cbar1 = fig.colorbar(c2, ax=ax, location='top', shrink=0.3, aspect=15, pad=0.02, anchor=(1.0, 0.0))
        cbar1.set_ticks([0, 1])
        cbar1.ax.tick_params(direction='in', length=1.5)
        cbar1.ax.text(-0.1, 3, r"$R_{dem}$(a.u.)", fontsize=7, va='center', ha='right', transform=cbar1.ax.transAxes, rotation=0)
        ax.set_xticks([5.18])
        if i == 0:
            ax.set_yticks([5.27, 5.28])
            ax.set_ylabel(r"$FG_{2} (V)$")
        else:
            ax.set_yticks([5.27, 5.28], labels=['', ''])
        ax.set_xlabel(r"$FG_{1} (V)$")
        #ax.set_title(f"Trans Map {read_trans[i]}")
        ax.text(0, 1.1, labels[i+6], transform=ax.transAxes, fontsize=9, fontweight='bold')
        #ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
        #ax.set_aspect('equal')
        #ax.text(5.18, 5.263, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read_trans[i] * 125 * 1e-6, 2)),
        #        fontsize=6, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, '400mT_gallery.pdf'))
    plt.savefig(os.path.join(file_dir, '400mT_gallery.svg'))

    plt.show()


def create_one_row_gallery(read=None, dir=None):
    #### Load data ####
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, dir)

    maps_read = []
    X_read = []
    Y_read = []

    for t in read:
        maps_read.append(np.load(os.path.join(file_dir, '-1_{}_map.npy'.format(t))))
        X_read.append(np.load(os.path.join(file_dir, '-1_{}_FG14.npy'.format(t))))
        Y_read.append(np.load(os.path.join(file_dir, '-1_{}_FG12.npy'.format(t))))

    fig, axes = plt.subplots(1, 3)
    labels = ['(a)', '(b)', '(c)']#, '(d)', '(e)', '(f)']

    for i in range(3):
        ax = axes[i]
        c1 = ax.pcolormesh(X_read[i], Y_read[i], maps_read[i], cmap="viridis_r", shading="auto", rasterized=True)
        cbar = fig.colorbar(c1, ax=ax, location='top', shrink=0.3, aspect=15, pad=0.02, anchor=(1.0, 0.0))
        cbar.set_ticks([0, 1])
        cbar.ax.tick_params(direction='in', length=3)
        cbar.ax.text(-0.1, 1, r"$R_{dem}$(a.u.)", fontsize=8, va='center', ha='right', transform=cbar.ax.transAxes, rotation=0)
        ax.set_xticks([5.18])
        if i == 0:
            ax.set_yticks([5.27, 5.28])
            ax.set_ylabel(r"$FG_{2} (V)$")
            ax.set_ylabel(r"$FG_{2} (V)$")
        else:
            ax.set_yticks([5.27, 5.28], labels=['', ''])
        ax.set_xlabel(r"$FG_{1} (V)$")
        #ax.set_xlabel(r"$FG_{1} (V)$", fontsize=20)
        #ax.set_title(f"Read Map {read_block[i]}")
        ax.text(0, 1.05, labels[i], transform=ax.transAxes, fontsize=9, fontweight='bold')
        #ax.set_aspect('equal')
        #ax.tick_params(axis='both', direction='in', length=5, labelsize=8, width=1.5)
        ax.text(5.18, 5.28, r'$t_{read} =$' + r'${} \mu s$'.format(np.round(read[i] * 125 * 1e-6, 2)),
                fontsize=6, color='black')

    #for i in range(3):
    #    ax = axes[1, i]
    #    ax.text(0, 1.05, labels[i+3], transform=ax.transAxes, fontsize=9, fontweight='bold')
    #    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(file_dir, '400mT_gallery_one_row_trans.pdf'))
    plt.savefig(os.path.join(file_dir, '400mT_gallery_one_row_trans.svg'))

    plt.show()


def main():
    #read_block = [1857, 5250, 10000]
    #read_trans = [190, 500, 1000]
    #dir = '0T_both_dir'
    #
    # create_gallery(read_block, read_trans, dir)

    #read_block = [2475, 20250, 40000]
    #read_trans = [406, 2475, 12350]
    #dir = '1T_both_dir'
    #
    # create_gallery(read_block, read_trans, dir)

    read_block = [2950, 6850, 14650, 20500, 32200, 40000]
    read_trans = [880, 2240, 4280, 8800, 16600, 20500]
    dir = '400mT_both_dir'

    create_gallery(read_block, read_trans, dir)
    #create_one_row_gallery(read_trans, dir)

if __name__ == "__main__":
    main()
