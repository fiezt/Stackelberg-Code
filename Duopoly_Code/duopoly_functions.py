from __future__ import division
from __future__ import print_function
import numpy as np


def f1(x, y, A, c_1, c_2):
    
    f_1 = (np.maximum(A - x - y, 0) - c_1)*x
    
    return f_1
    
    
def f2(x, y, A, c_1, c_2):
    
    f_2 = (np.maximum(A - x - y, 0) - c_2)*y
    
    return f_2


def f(x, y, A, c_1, c_2):
    
    f_1 = f1(x, y, A, c_1)
    f_2 = f2(x, y, A, c_2)
    
    cost = np.array([f_1, f_2])

    return cost


def d1f1(x, y, A, c_1, c_2):
    
    d1_f1 = A - 2*x - y - c_1
    
    return d1_f1


def d2f1(x, y, A, c_1, c_2):
    
    d2_f1 = -x
    
    return d2_f1
    
    
def d1f2(x, y, A, c_1, c_2):
    
    d1_f2 = -y
    
    return d1_f2

    
def d2f2(x, y, A, c_1, c_2):

    d2_f2 = A - x - 2*y - c_2
    
    return d2_f2

    
def omega(x, y, A, c_1, c_2):
    
    d1_f1 = d1f1(x, y, A, c_1, c_2)
    d2_f2 = d2f2(x, y, A, c_2, c_2)
    
    d = np.array([d1_f1, d2_f2])
    
    return d


def nabla(x, y, A, c_1, c_2):
    
    d1_f1 = d1f1(x, y, A, c_1, c_2)
    d2_f1 = d2f1(x, y, A, c_1, c_2)
    d1_f2 = d1f2(x, y, A, c_1, c_2)
    d2_f2 = d2f2(x, y, A, c_1, c_2)
    
    d = np.array([[d1_f1, d2_f1], [d1_f2, d2_f2]])
    
    return d

    
def d11(x, y, A, c_1, c_2):
    
    d_11 = -2
    
    return d_11
    

def d12(x, y, A, c_1, c_2):
    
    d_12 = -1
    
    return d_12
    

def d21(x, y, A, c_1, c_2):
    
    d_21 = -1
    
    return d_21
    

def d22(x, y, A, c_1, c_2):
    
    d_22 = -2
    
    return d_22


def d211(x, y, A, c_1, c_2):
    
    d_211 = 0
    
    return d_211


def d221(x, y, A, c_1, c_2):
    
    d_221 = 0
    
    return d_221


def d212(x, y, A, c_1, c_2):

    d_212 = 0

    return d_212


def d222(x, y, A, c_1, c_2):

    d_222 = 0

    return d_222


def d22f1(x, y, A, c_1, c_2):
    
    d22_f1 = 0
    
    return d22_f1
    

def d_omega(x, y, A, c_1, c_2):
    
    d_11 = d11(x, y, A, c_1, c_2)
    d_12 = d12(x, y, A, c_1, c_2)
    d_21 = d21(x, y, A, c_1, c_2)
    d_22 = d22(x, y, A, c_1, c_2)
    
    hessian = np.array([[d_11, d_12], [d_21, d_22]])
    
    return hessian


def omega1_zero(A, c_1, c_2, tol=1e-3):

    y = np.linspace(0, 100, 10000)
    x = .5*(A - y - c_1)
    t_1 = x[:,None]
    t_2 = y[None,:]
    points = d1f1(t_1, t_2, A, c_1, c_2)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = np.round(x[points[0]], 2), np.round(y[points[1]], 2)

    return points_1, points_2


def omega2_zero(A, c_1, c_2, tol=1e-3):
    
    x = np.linspace(0, 100, 10000)
    y = .5*(A - x - c_2)
    t_1 = x[:,None]
    t_2 = y[None,:]
    points = d2f2(t_1, t_2, A, c_1, c_2)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = np.round(x[points[0]], 2), np.round(y[points[1]], 2)
    
    return points_1, points_2


def leader_zero(A, c_1, c_2, tol=1e-2):
    
    x = np.array(10000*[A/2 + c_2/2 - c_1])
    y = np.linspace(0, 100, 10000)
    t_1 = x[:,None]
    t_2 = y[None,:]

    d1_f1 = d1f1(t_1, t_2, A, c_1, c_2)
    d2_f1 = d2f1(t_1, t_2, A, c_1, c_2)
    d_21 = d21(t_1, t_2, A, c_1, c_2)
    d_22 = d22(t_1, t_2, A, c_1, c_2)
        
    points = d1_f1 - d2_f1*(d_21/d_22)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = x[points[0]], y[points[1]]
    
    return points_1, points_2


def fast_leader_zero(A, c_1, c_2, tol=1e-2):
    
    x = np.linspace(0, 100, 10000)
    y = np.linspace(0, 100, 10000)
    t_1 = x[:,None]
    t_2 = y[None,:]
    
    d1_f1 = d1f1(t_1, t_2, A, c_1, c_2)
    d2_f1 = d2f1(t_1, t_2, A, c_1, c_2)
    d2_f2 = d2f2(t_1, t_2, A, c_1, c_2)
    d_12 = d12(t_1, t_2, A, c_1, c_2)
    d_21 = d21(t_1, t_2, A, c_1, c_2)
    d_22 = d22(t_1, t_2, A, c_1, c_2)
        
    points = d1_f1 - d_12*(d2_f2/d_22) - d2_f1*(d_21/d_22)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = x[points[0]], y[points[1]]
    
    return points_1, points_2


def follower_zero(A, c_1, c_2, tol=1e-3):

    x = np.linspace(0, 100, 10000)
    y = .5*(A - x - c_2)
    t_1 = x[:,None]
    t_2 = y[None,:]
    points = d2f2(t_1, t_2, A, c_1, c_2)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = np.round(x[points[0]], 2), np.round(y[points[1]], 2)
    
    return points_1, points_2


def d_omega_stackelberg(x, y, A, c_1, c_2):
    
    d_11 = d11(x, y, A, c_1, c_2)
    d_12 = d12(x, y, A, c_1, c_2)
    d_21 = d21(x, y, A, c_1, c_2)
    d_22 = d22(x, y, A, c_1, c_2)
    d2_f1 = d2f1(x, y, A, c_1, c_2)
    d_221 = d221(x, y, A, c_1, c_2)
    d_211 = d211(x, y, A, c_1, c_2)
    d22_f1 = d22f1(x, y, A, c_1, c_2)
    d_212 = d212(x, y, A, c_1, c_2)
    d_222 = d222(x, y, A, c_1, c_2)

    d_omega_11 = d_11 - d_12*(d_21/d_22) - d2_f1*(-(d_21*d_221/(d_22)**2) + (d_211/d_22))
    d_omega_12 = d_12 - d22_f1*(d_21/d_22) - d2_f1*(-(d_21*d_222/(d_22)**2) + (d_212/d_22))
    
    d_omega_21 = d_21
    d_omega_22 = d_22
    
    hessian = np.array([[d_omega_11, d_omega_12], [d_omega_21, d_omega_22]])
    
    return hessian


def find_critical_points(d1_zero, d2_zero):

    l1 = list(map(tuple, np.round(np.vstack(d1_zero).T, 2)))
    l2 = list(map(tuple, np.round(np.vstack(d2_zero).T, 2)))

    critical_points = np.array(list(set(l1).intersection(l2)))
    
    return critical_points


def find_nash_points(critical_points, A, c_1, c_2):
    
    nash = []
    
    for theta in critical_points:
        x = theta[0]
        y = theta[1]
        hessian = d_omega(x, y, A, c_1, c_2)
        if np.all(np.diag(hessian) < 0):
            nash.append([x, y])
            
    nash_points = np.vstack(nash)
    
    return nash_points


def find_stackelberg_points(critical_points, A, c_1, c_2):

    stackelberg = []
    
    for theta in critical_points:
        x = theta[0]
        y = theta[1]
        hessian = d_omega_stackelberg(x, y, A, c_1, c_2)
        if np.all(np.diag(hessian) < 0):
            stackelberg.append([x, y])
            
    stackelberg_points = np.vstack(stackelberg)
    
    return stackelberg_points


def J0(x, y, A, c_1, c_2):
    
    hessian = d_omega(x, y, A, c_1, c_2)
    
    diag_indices = np.diag_indices(hessian.shape[0])
    
    hessian[diag_indices] = 0
    
    J_0 = hessian
    
    return hessian