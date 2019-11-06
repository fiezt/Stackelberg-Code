import numpy as np
import torchvision.utils
import matplotlib.pyplot as plt

def calc_gradient_norm(params):
    norm = lambda p: p.grad.data.norm(2).item()
    return sum(norm(p)**2 for p in params) ** (1. / 2)
    
    
def network_param_index(network):
    
    counts = [0]
    for p in network.parameters():
        count = 1
        for num in p.shape:
            count *= num
        counts.append(count)
    index = np.cumsum(counts)
    
    return index


def stackup_array(arr, index):
    
    n = len(index)-1
    
    return tuple(arr[index[i]:index[i+1], :] for i in range(n))


def mnist_show(data, disp_num=256, fig_size=(10,20), show=False, save=None):
    
    img = torchvision.utils.make_grid(data, nrow=int(np.sqrt(disp_num)), normalize=True)
    npimg = img.numpy()    
    plt.figure(figsize=fig_size)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')
    if save is not None: plt.savefig(save)
    if show: plt.show()
        
        
def mnist_show_select(data, nrow=4, fig_size=(10,20), show=False, save=None):
    
    img = torchvision.utils.make_grid(data, nrow=nrow, normalize=True)
    npimg = img.numpy()    
    plt.figure(figsize=fig_size)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')
    if save is not None: plt.savefig(save)
    if show: plt.show()