from torch import autograd
from ComputationalTools import JacobianVectorProduct


def build_game_gradient(fs, params):
    grads = [autograd.grad(f, param.parameters(), create_graph=True) 
           for f,param in zip(fs, params)]
    return grads
      
def build_game_jacobian(fs, params):
    f1, f2 = fs
    x1, x2 = params
    A = JacobianVectorProduct(f1, list(x1.parameters()))
    B = JacobianVectorProduct(f2, list(x1.parameters()))
    C = JacobianVectorProduct(f1, list(x2.parameters()))
    D = JacobianVectorProduct(f2, list(x2.parameters()))
    J = JacobianVectorProduct(f1 + f2, list(x1.parameters()) + list(x2.parameters()))
            
    return A, B, C, D, J