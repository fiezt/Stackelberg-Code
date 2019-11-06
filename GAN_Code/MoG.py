from scipy import stats
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

def kde(values, fig_size=(8, 8), bbox=[-2, 2, -2, 2], xlabel="", ylabel="", cmap='Blues', show=False, save=None):
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    kernel = stats.gaussian_kde(values)

    ax.axis(bbox)
    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    if save is not None: plt.savefig(save)
    if show: plt.show()

    plt.close()

    
def dim_vis(D, fig_size=(8, 8), bbox=[-2, 2, -2, 2], xlabel="", ylabel="", cmap='Blues', 
            color_range=False, show=False, save=None, device='cpu'):

    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis(bbox)
    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(D(torch.from_numpy(positions.T).float().to(device)).cpu().detach().numpy().T, xx.shape)
    if color_range:
        cfset = ax.contourf(xx, yy, f, levels=np.linspace(0.35, 0.65, 100), cmap=cmap)
    else:
        cfset = ax.contourf(xx, yy, f, 100, cmap=cmap)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    cbar = fig.colorbar(cfset,fraction=0.046, pad=0.04)
    if color_range: 
        seq = [0.35, f.min(), f.mean(), f.max(), 0.65]
        cbar.set_ticks(seq)
        cbar.set_ticklabels(['%.2f' % val for val in np.round(seq, 2)])
        cbar.ax.tick_params(labelsize=22)
        #cbar.set_ticks(np.linspace(0., 1., 11))
    else:
        cbar.set_ticks(np.linspace(f.min(), f.max(), 5))
        cbar.set_ticklabels(['%.2f' % val for val in np.round(np.linspace(f.min(), f.max(), 5), 2)])
    if save is not None: plt.savefig(save)
    if show: plt.show()

    plt.close()


def real_builder_circle(batch_size, sigma=0.05):
    skel = np.array([[np.sin(t), np.cos(t)]
                     for t in np.linspace(0,2*np.pi,9)[:-1]])
    mixture = np.random.choice(range(8), batch_size)
    real = skel[mixture] + sigma*np.random.randn(batch_size, 2)

    return real


def real_builder_diamond(batch_size, sigma=0.3):
    skel = np.array([[1.5*np.sin(t), 1.5*np.cos(t)]
                     for t in np.linspace(0,2*np.pi,5)[:-1]])
    mixture = np.random.choice(range(4), batch_size)
    real = skel[mixture] + sigma*np.random.randn(batch_size, 2)

    return real


def real_builder_grid(batch_size, sigma=0.01):
    skel = np.vstack([[(x, y) for x in [-3, 0, 3]] for y in [-3, 0, 3]])
    mixture = np.random.choice(range(9), batch_size)
    real = skel[mixture] + sigma*np.random.randn(batch_size, 2)

    return real