from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def log(**datum):
    data = []
    sys.stdout.write('\r')
    sys.stdout.write(str(datum))
    data.append(datum)
    
    return list(data[0].values())


def plot_log(data, labels, show=False):
    data = np.array(data)
    time = data[:, 0]
    
    fig, axs = plt.subplots(nrows=data.shape[1]-1, figsize=(20,10))

    for ax, p in zip(axs, range(1, data.shape[1])):
        ax.plot(time, np.abs(data[:, p]), label=labels[p-1])
        ax.set_yscale('log')
        ax.legend()
    
    if show:
        plt.show()
    return fig


def log_eigs(A_eigs, D_eigs, D_reg_eigs, J_eigs, SC_reg_eigs):
    
    print('A minimum and maximum eigs:', A_eigs)
    print('D minimum and maximum eigs:', D_eigs)
    print('D regularized minimum and maximum eigs:', D_reg_eigs)
    print('J minimum and maximum eigs:', J_eigs)
    print('SC regularized minimum and maximum eigs:', SC_reg_eigs)
    
    
def store_eigs(A_eig_data, D_eig_data, D_reg_eig_data, J_eig_data, SC_reg_eig_data, info_dir):
    
    np.savetxt(os.path.join(info_dir, 'A_eigs.csv'), np.array(A_eig_data), delimiter=',')
    np.savetxt(os.path.join(info_dir, 'D_eigs.csv'), np.array(D_eig_data), delimiter=',')
    np.savetxt(os.path.join(info_dir, 'D_reg_eigs.csv'), np.array(D_reg_eig_data), delimiter=',')
    np.savetxt(os.path.join(info_dir, 'J_eigs.csv'), np.array(J_eig_data), delimiter=',')
    np.savetxt(os.path.join(info_dir, 'SC_reg_eigs.csv'), np.array(SC_reg_eig_data), delimiter=',')