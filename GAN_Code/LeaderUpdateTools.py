import torch
from torch import autograd
import numpy as np
import scipy.sparse.linalg
from ComputationalTools import JacobianVectorProduct
from GameGradients import build_game_gradient
from Utils import stackup_array
import time


def compute_leader_grad(G, D, G_loss, D_loss, regularization, x0, update, precise=True, device='cpu'):

    if update == 'simgrad':
        leader_grad = autograd.grad(G_loss, G.parameters(), create_graph=True)
        leader_grad = torch.cat([_.flatten() for _ in leader_grad]).view(-1, 1) 
        leader_grad_norm = torch.norm(leader_grad).item()
        q_norm = 0
        g_norm = leader_grad_norm
        x0 = None
    elif update == 'stack':
        Dg, Dd = build_game_gradient([G_loss, D_loss], [G, D])
        Dd_g = autograd.grad(G_loss, D.parameters(), create_graph=True)
        DD_reg = JacobianVectorProduct(Dd, list(D.parameters()), regularization)
        leader_grad, q, x0 = compute_stackelberg_grad(G, Dg, Dd,  Dd_g, DD_reg, x0, precise=precise, device=device)
        leader_grad_norm = torch.norm(leader_grad).item()
        q_norm = torch.norm(q).item()
        g_norm = torch.norm(torch.cat([_.flatten() for _ in Dg]).view(-1, 1)).item()
    else:
        raise Exception('Bad update: input simgrad or stack')

    return leader_grad, leader_grad_norm, q_norm, g_norm, x0


def adam_grad(leader_grad, beta1, beta2, epsilon, m, v, step, param_index):

    m.mul_(beta1).add_((1 - beta1)*leader_grad.detach())
    v.mul_(beta2).add_((1 - beta2)*leader_grad.detach()**2)
    bias1 = (1-beta1**(step))
    bias2 = (1-beta2**(step))
    leader_grad_to_stack = (m*np.sqrt(bias2))/(torch.sqrt(v)*bias1 + epsilon*bias2)
    leader_grad = stackup_array(leader_grad_to_stack, param_index)

    return leader_grad


def leader_step(G, leader_grad, lr_g, gamma_g, step):

    exp_lr = lr_g*gamma_g**(step)
    for p, l in zip(G.parameters(), leader_grad):
        p.data.add_(-exp_lr*l.view(p.shape)) 
    
    return exp_lr


def compute_stackelberg_grad(G, Dg, Dd, Dd_g, DD_reg, x0=None, tol=1e-6, precise=True, device='cpu'):

    Dg_vec = torch.cat([_.flatten() for _ in Dg]).view(-1, 1)
    Dd_g_vec = torch.cat([_.flatten() for _ in Dd_g]).view(-1, 1)
    if precise:
        w, status = scipy.sparse.linalg.gmres(DD_reg, Dd_g_vec.cpu().detach().numpy(), tol=tol, 
                                              restart=DD_reg.shape[0], x0=x0)
        assert status == 0
    else:
        w, status = scipy.sparse.linalg.cg(DD_reg, Dd_g_vec.cpu().detach().numpy(), maxiter=5)
    q = torch.Tensor(JacobianVectorProduct(Dd, list(G.parameters()))(w)).view(-1, 1).to(device)
    leader_grad = Dg_vec - q

    return leader_grad, q, w