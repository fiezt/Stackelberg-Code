import torch
from torch import autograd
import numpy as np
import scipy.sparse.linalg
import time
np.set_printoptions(precision=2)
    
    
class JacobianVectorProduct(scipy.sparse.linalg.LinearOperator):
    def __init__(self, grad, params, regularization=0):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)

        nparams = sum(p.numel() for p in params)
        self.shape = (nparams, self.grad.size(0))
        self.dtype = np.dtype('Float32')
        self.params = params
        self.regularization = regularization

    def _matvec(self, v):
        v = torch.Tensor(v)
        if self.grad.is_cuda:
            v = v.cuda()
        hv = autograd.grad(self.grad, self.params, v, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        if self.regularization != 0:
            hv = torch.cat(_hv) + self.regularization*v
        else:
            hv = torch.cat(_hv) 
        return hv.cpu()

    
class SchurComplement(scipy.sparse.linalg.LinearOperator):
    def __init__(self, A, B, C, D, tol_gmres=1e-6, precise=False):
        self.operator = [[A,B], [C,D]]
        self.shape = A.shape
        self.config = {'tol_gmres': tol_gmres}
        self.dtype = np.dtype('Float32')
        self.precise = precise
        
    def _matvec(self, v): 
        
        (A,B),(C,D) = self.operator

        u = C(v)
        
        if self.precise:
            w, status = scipy.sparse.linalg.gmres(D, u, tol=self.config['tol_gmres'], restart=D.shape[0])
            assert status == 0
        else:
            w, status = scipy.sparse.linalg.cg(D, u, maxiter=5)
        
        self.w = w

        p = A(v) - B(w)
        
        return p
    
    
    
    