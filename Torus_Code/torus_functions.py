from __future__ import division
from __future__ import print_function
import numpy as np


def f1(theta_1, theta_2, alpha_1, phi_1):
    
    f_1 = -alpha_1*np.cos(theta_1 - phi_1) + np.cos(theta_1 - theta_2) 
    
    return f_1
    
    
def f2(theta_1, theta_2, alpha_2, phi_2):
    
    f_2 = -alpha_2*np.cos(theta_2 - phi_2) + np.cos(theta_2 - theta_1) 
    
    return f_2


def f(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2):
    
    f_1 = f1(theta_1, theta_2, alpha_1, phi_1)
    f_2 = f2(theta_1, theta_2, alpha_2, phi_2)
    
    cost = np.array([f_1, f_2])

    return cost


def d1f1(theta_1, theta_2, alpha_1, phi_1):
    
    d1_f1 = alpha_1*np.sin(theta_1 - phi_1) - np.sin(theta_1 - theta_2)
    
    return d1_f1


def d2f1(theta_1, theta_2):
    
    d2_f1 = np.sin(theta_1 - theta_2)
    
    return d2_f1
    
    
def d1f2(theta_1, theta_2):
    
    d1_f2 = np.sin(theta_2 - theta_1)
    
    return d1_f2

    
def d2f2(theta_1, theta_2, alpha_2, phi_2):

    d2_f2 = alpha_2*np.sin(theta_2 - phi_2) - np.sin(theta_2 - theta_1)
    
    return d2_f2

    
def omega(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2):
    
    d1_f1 = d1f1(theta_1, theta_2, alpha_1, phi_1)
    d2_f2 = d2f2(theta_1, theta_2, alpha_2, phi_2)
    
    d = np.array([d1_f1, d2_f2])
    
    return d


def nabla(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2):
    
    d1_f1 = d1f1(theta_1, theta_2, alpha_1, phi_1)
    d2_f1 = d2f1(theta_1, theta_2)
    d1_f2 = d1f2(theta_1, theta_2)
    d2_f2 = d2f2(theta_1, theta_2, alpha_2, phi_2)
    
    d = np.array([[d1_f1, d2_f1], [d1_f2, d2_f2]])
    
    return d

    
def d11(theta_1, theta_2, alpha_1, phi_1):
    
    d_11 = alpha_1*np.cos(theta_1 - phi_1) - np.cos(theta_1 - theta_2)
    
    return d_11
    

def d12(theta_1, theta_2):
    
    d_12 = np.cos(theta_1 - theta_2)
    
    return d_12
    

def d21(theta_1, theta_2):
    
    d_21 = np.cos(theta_2 - theta_1)
    
    return d_21
    

def d22(theta_1, theta_2, alpha_2, phi_2):
    
    d_22 = alpha_2*np.cos(theta_2 - phi_2) - np.cos(theta_2 - theta_1)
    
    return d_22


def d211(theta_1, theta_2):
    
    d_211 = np.sin(theta_2 - theta_1)
    
    return d_211


def d221(theta_1, theta_2):
    
    d_221 = np.sin(theta_2 - theta_1)
    
    return d_221


def d212(theta_1, theta_2):

    d_212 = -np.sin(theta_2 - theta_1)

    return d_212


def d222(theta_1, theta_2, alpha_2, phi_2):

    d_222 = -alpha_2*np.sin(theta_2 - phi_2) + np.sin(theta_2 - theta_1)

    return d_222


def d22f1(theta_1, theta_2):
    
    d22_f1 = -np.cos(theta_1 - theta_2) 
    
    return d22_f1


def d_omega(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2):
    
    d_11 = d11(theta_1, theta_2, alpha_1, phi_1)
    d_12 = d12(theta_1, theta_2)
    d_21 = d21(theta_1, theta_2)
    d_22 = d22(theta_1, theta_2, alpha_2, phi_2)
    
    hessian = np.array([[d_11, d_12], [d_21, d_22]])
    
    return hessian


def omega1_zero(alpha_1, phi_1, n=10000, tol=3.98e-5):
    
    theta_1 = np.linspace(-np.pi, np.pi, n)
    theta_2 = np.linspace(-np.pi, np.pi, n)
    t_1 = theta_1[:,None]
    t_2 = theta_2[None,:]
    points = d1f1(t_1, t_2, alpha_1, phi_1)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = theta_1[points[0]], theta_2[points[1]]
    
    return points_1, points_2


def omega2_zero(alpha_2, phi_2, n=10000, tol=1e-3):
    
    theta_1 = np.linspace(-np.pi, np.pi, n)
    theta_2 = np.linspace(-np.pi, np.pi, n)
    t_1 = theta_1[:,None]
    t_2 = theta_2[None,:]
    points = d2f2(t_1, t_2, alpha_2, phi_2)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = theta_1[points[0]], theta_2[points[1]]
    
    return points_1, points_2


def leader_zero(alpha_1, alpha_2, phi_1, phi_2, n=10000, tol=1e-3):
    
    theta_1 = np.linspace(-np.pi, np.pi, n)
    theta_2 = np.linspace(-np.pi, np.pi, n)
    t_1 = theta_1[:,None]
    t_2 = theta_2[None,:]
    
    d1_f1 = d1f1(t_1, t_2, alpha_1, phi_1)
    d2_f1 = d2f1(t_1, t_2)
    d_21 = d21(t_1, t_2)
    d_22 = d22(t_1, t_2, alpha_2, phi_2)
        
    points = d1_f1 - d2_f1*(d_21/d_22)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = theta_1[points[0]], theta_2[points[1]]
    
    return points_1, points_2


def fast_leader_zero(alpha_1, alpha_2, phi_1, phi_2, n=10000, tol=1e-3):
    
    theta_1 = np.linspace(-np.pi, np.pi, n)
    theta_2 = np.linspace(-np.pi, np.pi, n)
    t_1 = theta_1[:,None]
    t_2 = theta_2[None,:]
    
    d1_f1 = d1f1(t_1, t_2, alpha_1, phi_1)
    d2_f1 = d2f1(t_1, t_2)
    d2_f2 = d2f2(t_1, t_2, alpha_2, phi_2)
    d_12 = d12(t_1, t_2)
    d_21 = d21(t_1, t_2)
    d_22 = d22(t_1, t_2, alpha_2, phi_2)
        
    points = d1_f1 - d_12*(d2_f2/d_22) - d2_f1*(d_21/d_22)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = theta_1[points[0]], theta_2[points[1]]
    
    return points_1, points_2


def follower_zero(alpha_2, phi_2, n=10000, tol=1e-3):
    
    theta_1 = np.linspace(-np.pi, np.pi, n)
    theta_2 = np.linspace(-np.pi, np.pi, n)
    t_1 = theta_1[:,None]
    t_2 = theta_2[None,:]
    points = d2f2(t_1, t_2, alpha_2, phi_2)
    points = np.abs(points)
    points = np.where(points < tol)
    points_1, points_2 = theta_1[points[0]], theta_2[points[1]]
    
    return points_1, points_2


def d_omega_stackelberg(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2):

    d_11 = d11(theta_1, theta_2, alpha_1, phi_1)
    d_12 = d12(theta_1, theta_2)
    d_21 = d21(theta_1, theta_2)
    d_22 = d22(theta_1, theta_2, alpha_2, phi_2)
    d2_f1 = d2f1(theta_1, theta_2)
    d_221 = d221(theta_1, theta_2)
    d_211 = d211(theta_1, theta_2)
    d22_f1 = d22f1(theta_1, theta_2)
    d_212 = d212(theta_1, theta_2)
    d_222 = d222(theta_1, theta_2, alpha_2, phi_2)

    d_omega_11 = d_11 - d_12*(d_21/d_22) - d2_f1*(-(d_21*d_221/(d_22)**2) + (d_211/d_22))
    d_omega_12 = d_12 - d22_f1*(d_21/d_22) - d2_f1*(-(d_21*d_222/(d_22)**2) + (d_212/d_22))
    
    d_omega_21 = d_21
    d_omega_22 = d_22
    
    hessian = np.array([[d_omega_11, d_omega_12], [d_omega_21, d_omega_22]])
    
    return hessian


def find_critical_points(d1_zero, d2_zero):

    l1 = list(map(tuple, np.vstack(d1_zero).T))
    l2 = list(map(tuple, np.vstack(d2_zero).T))

    critical_points = np.array(list(set(l1).intersection(l2)))
    
    return critical_points



def find_nash_points(critical_points, alpha_1, alpha_2, phi_1, phi_2):
    
    nash = []
    
    for theta in critical_points:
        theta_1 = theta[0]
        theta_2 = theta[1]
        hessian = d_omega(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2)
        if np.all(np.diag(hessian) > 0):
            nash.append([theta_1, theta_2])
            
    nash_points = np.vstack(nash)
    
    return nash_points


def find_stackelberg_points(critical_points, alpha_1, alpha_2, phi_1, phi_2):
    
    stackelberg = []
    
    for theta in critical_points:
        theta_1 = theta[0]
        theta_2 = theta[1]
        hessian = d_omega_stackelberg(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2)
        if np.all(np.diag(hessian) > 0):
            stackelberg.append([theta_1, theta_2])
            
    stackelberg_points = np.vstack(stackelberg)
    
    return stackelberg_points


def J0(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2):
    
    hessian = d_omega(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2)
    
    diag_indices = np.diag_indices(hessian.shape[0])
    
    hessian[diag_indices] = 0
    
    J_0 = hessian
    
    return hessian
    
    
def compute_roa(func, nash_points, lr_1, lr_2, alpha_1, alpha_2, phi_1, phi_2, n=200, MAXITER=10000, tol=1e-4):

    theta_1 = np.linspace(-np.pi, np.pi, n)
    theta_2 = np.linspace(-np.pi, np.pi, n)
    nash_points = np.unique(np.round(nash_points, 2), axis=0)
    start_points = []
    end_points = []

    for theta_1_ in theta_1:
        for theta_2_ in theta_2:
            start_points.append(np.array([theta_1_, theta_2_]))
            end_point = func(lr_1, lr_2, theta_1_, theta_2_, alpha_1, alpha_2, phi_1, phi_2, tol=tol)[-1]
            nash_point = np.argmin(np.linalg.norm(end_point - nash_points, axis=1))/(len(nash_points)-1)
            end_points.append(nash_point)

    start_points = np.array(start_points)
    end_points = np.array(end_points)
    
    arrays = []
    
    for i in np.unique(end_points):
        arr = start_points[np.where(end_points == i)]
        arrays.append(arr)
            
    return start_points, end_points, arrays