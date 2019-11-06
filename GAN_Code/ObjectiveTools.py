import torch

def objective_function(p1, p2, model):
    if model == 'gan':
        G_loss = torch.mean(torch.log(p1) + torch.log(1. - p2))
        D_loss = -G_loss 
    elif model == 'nsgan':
        G_loss = -torch.mean(torch.log(p2))
        D_loss = -torch.mean(torch.log(p1) + torch.log(1. - p2))
        
    return G_loss, D_loss