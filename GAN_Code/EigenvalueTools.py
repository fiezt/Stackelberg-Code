from GameGradients import build_game_gradient, build_game_jacobian
from ComputationalTools import JacobianVectorProduct, SchurComplement
import scipy.sparse.linalg
import numpy as np

def calc_game_eigs(fs, xs, regularization=0, tol_gmres=1e-6, k=3, precise=False):
    
    G_loss, D_loss = fs
    G, D = xs
    Dg, Dd = build_game_gradient([G_loss, D_loss], [G, D])
    AA, BB, CC, DD, JJ = build_game_jacobian([Dg, Dd], [G, D])
    DD_reg = JacobianVectorProduct(Dd, list(D.parameters()), regularization)
    
    calc_eigs = lambda F: np.hstack((scipy.sparse.linalg.eigs(F, k=k, which='SR')[0], scipy.sparse.linalg.eigs(F, k=k, which='LR')[0]))
     
    A_eigs = calc_eigs(AA)
    D_eigs = calc_eigs(DD)
    D_reg_eigs = calc_eigs(DD_reg)
    J_eigs = calc_eigs(JJ)

    SC_reg = SchurComplement(AA, BB, CC, DD_reg, tol_gmres=tol_gmres, precise=precise)
    SC_reg_eigs = calc_eigs(SC_reg)

    return A_eigs, D_eigs, D_reg_eigs, J_eigs, SC_reg_eigs