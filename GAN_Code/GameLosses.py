import numpy as np
import torch
from torch import nn
from ObjectiveTools import objective_function

def gan_model_mog(n_latent, n_out, n_hidden, \
                 num_gen_layers,
                 num_disc_layers,
                 activation_function):

  if num_gen_layers == 1:
      G = nn.Sequential(
          nn.Linear(n_latent, n_hidden),
          activation_function,
          nn.Linear(n_hidden, n_out),
      )
  elif num_gen_layers == 2:
      G = nn.Sequential(
          nn.Linear(n_latent, n_hidden),
          activation_function,
          nn.Linear(n_hidden, n_hidden),
          activation_function,
          nn.Linear(n_hidden, n_out),
      )
  elif num_gen_layers == 3:
      G = nn.Sequential(
          nn.Linear(n_latent, n_hidden),
          activation_function,
          nn.Linear(n_hidden, n_hidden),
          activation_function,
          nn.Linear(n_hidden, n_hidden),
          activation_function,
          nn.Linear(n_hidden, n_out),
      )

  if num_disc_layers == 1:
      D = nn.Sequential(
          nn.Linear(n_out, n_hidden),
          activation_function,
          nn.Linear(n_hidden, 1),
          nn.Sigmoid(), 
      )
  elif num_disc_layers == 2:
      D = nn.Sequential(
          nn.Linear(n_out, n_hidden),
          activation_function,
          nn.Linear(n_hidden, n_hidden),
          activation_function,
          nn.Linear(n_hidden, 1),
          nn.Sigmoid(), 
      )
  return G, D


def gan_loss_mog(G, D, batch_size, n_latent, real_builder, objective, device):

    # z and x
    G_latent = torch.randn(batch_size, n_latent).to(device)
    real_data = torch.from_numpy(real_builder(batch_size)).float().to(device) 

    G_latent.to(device)
    real_data.to(device)
    
    # G(z)
    G_generated = G(G_latent)

    # D(x)
    p1 = D(real_data)
    
    # D(G(z))
    p2 = D(G_generated)
    
    # Compute Loss.
    G_loss, D_loss = objective_function(p1, p2, objective)
    
    return G_loss, D_loss