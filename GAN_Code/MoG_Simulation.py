import torch
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sys
import os
import datetime
from scipy import stats
from ComputationalTools import JacobianVectorProduct, SchurComplement
from EigenvalueTools import calc_game_eigs
from LoggingTools import log, plot_log, log_eigs, store_eigs
from GameGradients import build_game_gradient, build_game_jacobian
from MoG import kde, dim_vis, real_builder_circle, real_builder_diamond, real_builder_grid
from LeaderUpdateTools import compute_leader_grad, compute_stackelberg_grad, adam_grad, leader_step
from ObjectiveTools import objective_function
from Utils import calc_gradient_norm, network_param_index
from GameLosses import gan_model_mog, gan_loss_mog

np.set_printoptions(precision=2)

torch.manual_seed(43)    
np.random.seed(43)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M")

show = False
sim_name = sys.argv[1]
if len(sys.argv) == 10:
    num_gen_layers = int(sys.argv[8])
    num_disc_layers = int(sys.argv[9])
else:
    num_gen_layers = 1
    num_disc_layers = 1
    
results_dir = os.path.join(os.getcwd(), 'MoGResults' + '_' + str(num_gen_layers) + str(num_disc_layers) + 'Layer')
save_dir = os.path.join(results_dir, '_'.join([sim_name, now]))
fig_dir = os.path.join(save_dir, 'Figs')
info_dir = os.path.join(save_dir, 'Info')
checkpoint_dir = os.path.join(save_dir, 'Checkpoint')
if not os.path.exists(save_dir): os.makedirs(save_dir)
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
if not os.path.exists(info_dir): os.makedirs(info_dir)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

# 'gan' or 'nsgan'
objective = sys.argv[2]
num_iter = 60000
kl_size = 4096
x0 = None
freq_grads = 20
freq_checkpoint = 500
freq_log = 4000
# 'simgrad' or 'stack'
update_rule = sys.argv[3]

MoG_Type = sys.argv[4]

if MoG_Type == 'circle':
    real_builder = real_builder_circle
    bbox=[-2, 2, -2, 2]  
elif MoG_Type == 'diamond':
    real_builder = real_builder_diamond
    bbox=[-3, 3, -3, 3]  
elif MoG_Type == 'grid':
    real_builder = real_builder_grid
    bbox=[-5, 5, -5, 5]  

# 'tanh' or 'relu'
activation = sys.argv[5]
if activation == 'tanh':
    activation_function = nn.Tanh()
elif activation == 'relu':
    activation_function = nn.ReLU()

n_latent = 16
n_out = 2
n_hidden = 32

G,D = gan_model_mog(n_latent=n_latent, n_out=n_out, n_hidden=n_hidden, \
                 num_gen_layers=num_gen_layers,
                 num_disc_layers=num_disc_layers,
                activation_function=activation_function)

    
device = "cpu"
G.to(device)
D.to(device)

n_g = sum(p.numel() for p in G.parameters())
n_d = sum(p.numel() for p in D.parameters())

leader_param_index = network_param_index(G)

batch_size = 256
view_size = 4096

if MoG_Type == 'circle':
    lr_g = 4e-4
    lr_d = 4e-4 
elif MoG_Type == 'diamond':
    lr_g = 1e-4
    lr_d = 1e-4
elif MoG_Type == 'grid':
    lr_g = 1e-4
    lr_d = 1e-4

if sys.argv[6] == 'fast':
    gamma_g = 0.99999
elif sys.argv[6] == 'normal':
    gamma_g = 0.9999999
gamma_d = 0.9999999

if sys.argv[7] == 'small':
    regularization = 0.5    
elif sys.argv[7] == 'big':
    regularization = 1.0

# Default Adam Configurations
m = torch.zeros(n_g, 1).to(device)
v = torch.zeros(n_g, 1).to(device)
beta1 = 0.9
beta2 = 0.999
epsilon = 10**(-8)
opt_D = torch.optim.Adam(D.parameters(), lr=lr_d)
sch_D = torch.optim.lr_scheduler.ExponentialLR(opt_D, gamma=gamma_d)


config_names = ['objective', 'update_rule', 'batch_size', 'view_size', 'lr_g', 'lr_d', 
                'gamma_g', 'gamma_d', 'regularization', 'activation']
config_choices = [objective, update_rule, batch_size, view_size, lr_g, lr_d, 
                  gamma_g, gamma_d, regularization, activation]
with open(os.path.join(info_dir, 'config.txt'), "w") as f:
    print('Generator: \n', file=f)
    print(G, file=f)
    print('\n\nDiscriminator: \n', file=f)
    print(D, file=f)
    print('\n', file=f)
    for name, choice in zip(config_names, config_choices):
        print(name + ':', choice, file=f)
        print('\n', file=f)
        
log_data = []
kl_data = []
A_eig_data = []
D_eig_data = []
D_reg_eig_data = []
J_eig_data = []
SC_reg_eig_data = []

precise = False
ground_truth = real_builder(view_size)
kde(ground_truth.T, show=show, save=os.path.join(fig_dir, 'ground_truth'), bbox=bbox)

for step in range(1, num_iter+1):
    G_loss, D_loss = gan_loss_mog(G=G,D=D,
                                  batch_size=batch_size,
                                 n_latent=n_latent,
                                 real_builder=real_builder,
                                 objective=objective,
                                 device=device)

    # Obtain Generator Update.
    leader_grad, leader_grad_norm, q_norm, g_norm, x0 = compute_leader_grad(G, D, G_loss, D_loss, regularization, x0, 
                                                                            update_rule, precise=precise, device=device)
    leader_grad = adam_grad(leader_grad, beta1, beta2, epsilon, m, v, step, leader_param_index)
    
    
    # Optimize Discriminator
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()
    sch_D.step()
    follower_grad_norm = calc_gradient_norm(D.parameters())
    lr_d_state = opt_D.param_groups[0]['lr']
    
    # Optimize Generator.
    lr_g_state = leader_step(G, leader_grad, lr_g, gamma_g, step)
    
    if step % freq_grads == 0:
        data = log(iter=step, follower_grad_norm=follower_grad_norm, g_norm=g_norm, 
                    q_norm=q_norm, leader_grad_norm=leader_grad_norm, lr_d=lr_d_state, 
                    lr_g=lr_g_state, dis_loss=D_loss.data, gen_loss=G_loss.data)
        log_data.append(data)
        
    if step % freq_checkpoint == 0:
        torch.save({'state_gen':G.state_dict(), 'state_dis':D.state_dict()}, 
                     os.path.join(checkpoint_dir, 'checkpoint_step_' + str(step)))
        
    if step % freq_log == 0:  
        G_generated = G(torch.randn(view_size, n_latent).to(device)).cpu().detach().numpy()
        generator_save_name = os.path.join(fig_dir, 'generator_step_' + str(step))
        discriminator_save_name = os.path.join(fig_dir, 'discriminator_step_' + str(step))
        kde(G_generated.T, show=show, save=generator_save_name, bbox=bbox)
        dim_vis(D, show=show, save=discriminator_save_name, bbox=bbox, device=device)
        
        G_generated = G(torch.randn(kl_size, n_latent).to(device)).cpu().detach().numpy()
        real_data = real_builder(kl_size)
        fake_kernel = stats.gaussian_kde(G_generated.T)
        real_kernel = stats.gaussian_kde(real_data.T)
        xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kl = stats.entropy(pk=fake_kernel(positions), qk=real_kernel(positions))
        kl_data.append(kl)
        np.savetxt(os.path.join(info_dir, 'kl_div.csv'), np.array(kl_data).reshape(-1, 1), delimiter=',')
                
        G_loss, D_loss = gan_loss_mog(G=G,D=D, batch_size=2**13, n_latent=n_latent, 
                                      real_builder=real_builder, objective=objective, device=device)
        A_eigs, D_eigs, D_reg_eigs, J_eigs, SC_reg_eigs = calc_game_eigs([G_loss, D_loss], [G,D], regularization, k=6, precise=precise)
        log_eigs(A_eigs, D_eigs, D_reg_eigs, J_eigs, SC_reg_eigs)
        A_eig_data.append(A_eigs); D_eig_data.append(D_eigs); D_reg_eig_data.append(D_reg_eigs); J_eig_data.append(J_eigs); SC_reg_eig_data.append(SC_reg_eigs)
        store_eigs(A_eig_data, D_eig_data, D_reg_eig_data, J_eig_data, SC_reg_eig_data, info_dir)
        
        np.savetxt(os.path.join(info_dir, 'log_data.csv'), np.array(log_data), delimiter=',')
