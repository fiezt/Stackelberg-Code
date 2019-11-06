from __future__ import division
from __future__ import print_function
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import xkcd_rgb as xkcd
from matplotlib.colors import LinearSegmentedColormap
from duopoly_functions import f1, f2

fs2 = 24

xkcd_colors = [xkcd['yellow orange'], 
               xkcd['magenta'], xkcd['muted blue'], xkcd['orange'], xkcd['lawn green'], 
               xkcd['pinkish red'], xkcd['cobalt blue'], xkcd['tomato red'], xkcd['teal']]

cmap_colors = np.array([ [186/255, 65/255,62/255 ], [110/255,136/255,226/255] ])
cmap_colors = np.minimum(cmap_colors*1.2, 1)
_cmap = LinearSegmentedColormap.from_list('name', cmap_colors, N=2)


def plot_instance(histories, omega_1_zero, omega_2_zero, critical_points, nash_points, title=''):
    
    fig = plt.figure(1, figsize=(5,5))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_facecolor((0.87529412, 0.30588235, 0.29176471))

    ax.scatter(omega_1_zero[0], omega_1_zero[1], color=xkcd['white'], s=1)
    ax.scatter(omega_2_zero[0], omega_2_zero[1], color=xkcd['black'], s=1)
    
    if len(histories) < 9:
        for i, history in enumerate(histories):
            ax.scatter(history[0, 0], history[0, 1], color=xkcd['black'], s=200)

        for i, history in enumerate(histories):
            ax.plot(history[:, 0], history[:, 1], '-o', color=xkcd_colors[i], lw=4)
    else:
        for i, history in enumerate(histories):
            ax.scatter(history[0, 0], history[0, 1], color=xkcd['black'], s=100)
            
        for i, history in enumerate(histories):
            ax.plot(history[:, 0], history[:, 1], '-o', color=xkcd['blue'], lw=4, alpha=.5)
            
    ax.scatter(critical_points[:, 0], critical_points[:, 1], marker='o', color='white', s=200)
    ax.scatter(nash_points[:, 0], nash_points[:, 1], marker='o', color='grey', s=200)
    
    ax.set_xlabel(r'$x$', fontsize=fs2)
    ax.set_ylabel(r'$y$', fontsize=fs2)
    ax.tick_params(labelsize=fs2)
    ax.set_title(title, fontsize=fs2)
    
    
    plt.show()
    sns.reset_orig()



def plot_history(histories, equilibriums):

    sns.set_style('whitegrid', {'font.family':['serif'], 'font.serif':['Times New Roman'], 
                  'grid.color':'.9'})

    fig = plt.figure(1, figsize=(5,5))
    fig.clf()
    ax = fig.add_subplot(111)

    fs1 = 22
    fs2 = 24
    max_ = 100

    for i, history in enumerate(histories):

        if i == 0:
            linestyle = '-'
            equil_color = xkcd['magenta']
            label_e = r'$q^{\ast}_S$'
            label_1 = r'$q_1^{S}$'
            label_2 = r'$q_2^{S}$'
        else:
            linestyle = '--'
            equil_color = xkcd['yellow orange']
            label_e = r'$q^{\ast}_N$'
            label_1 = r'$q_1^{N}$'
            label_2 = r'$q_2^{N}$'

        ax.axhline(equilibriums[i][0], lw=4, color=equil_color)
        ax.axhline(equilibriums[i][1], lw=4, color=equil_color, label=label_e)
        ax.plot(history[:max_, 0], lw=4, color=xkcd['black'], linestyle=linestyle,  label=label_1)
        ax.plot(history[:max_, 1], lw=4, color=xkcd['tomato red'], linestyle=linestyle,  label=label_2)

    ax.set_xlabel('Iterations', fontsize=fs2)
    ax.set_ylabel('Production', fontsize=fs2)
    lgd = ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=fs1, fancybox=True, 
                    framealpha=0, ncol=1, handlelength=1.5)

    ax.tick_params(labelsize=fs2)
    plt.savefig(os.path.join(os.getcwd(), 'Figs', 'production.pdf'), 
                bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)

    plt.show()
    sns.reset_orig()

    
def plot_cost(histories, A, c_1, c_2, equilibriums):
    
    sns.set_style('whitegrid', {'font.family':['serif'], 'font.serif':['Times New Roman'], 
                  'grid.color':'.9'})

    fig = plt.figure(1, figsize=(5,5))
    fig.clf()
    ax = fig.add_subplot(111)

    fs1 = 22
    fs2 = 24
    max_ = 100

    for i, history in enumerate(histories):

        if i == 0:
            linestyle = '-'
            equil_color = xkcd['magenta']
            label_e = r'$\pi^{\ast}_S$'
            label_1 = r'$\pi_1^{S}$'
            label_2 = r'$\pi_2^{S}$'
        else:
            linestyle = '--'
            equil_color = xkcd['yellow orange']
            label_e = r'$\pi^{\ast}_N$'
            label_1 = r'$\pi_1^{N}$'
            label_2 = r'$\pi_2^{N}$'

        cost_1 = f1(history[:max_, 0], history[:max_, 1], A, c_1, c_2)
        cost_2 = f2(history[:max_, 0], history[:max_, 1], A, c_1, c_2)
        equil_1 = f1(equilibriums[i][0], equilibriums[i][1], A, c_1, c_2)
        equil_2 = f2(equilibriums[i][0], equilibriums[i][1], A, c_1, c_2)
        print(cost_1[-1], cost_2[-1])

        ax.axhline(equil_1, lw=4, color=equil_color)
        ax.axhline(equil_2, lw=4, color=equil_color, label=label_e)
        ax.plot(cost_1, lw=4, color=xkcd['black'], linestyle=linestyle, label=label_1)
        ax.plot(cost_2, lw=4, color=xkcd['tomato red'], linestyle=linestyle, label=label_2)

    ax.set_xlabel('Iterations', fontsize=fs2)
    ax.set_ylabel('Profit', fontsize=fs2)
    lgd = ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=fs1, fancybox=True, 
                    framealpha=0, ncol=1, handlelength=1.5)

    ax.tick_params(labelsize=fs2)
    plt.savefig(os.path.join(os.getcwd(), 'Figs', 'profit.pdf'), 
                bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=100)

    plt.show()
    sns.reset_orig()






