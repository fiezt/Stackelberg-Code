from __future__ import division
from __future__ import print_function
import numpy as np
from torus_functions import compute_roa, omega, J0, nabla, d1f1, d2f1, d2f2, d12, d21, d22
from torus_figure_functions import plot_instance, plot_instance_roa_contour


def simulate_gd(lr_1, lr_2, theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2, sigma=np.sqrt(.01), tol=1e-5):
    
    theta = np.array([theta_1, theta_2])
    history = [theta]

    for t in xrange(50000):
        gamma = np.diag([lr_1(t), lr_2(t)])
        noise = np.random.normal(0, sigma, size=(2))

        update = -gamma.dot(omega(theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2) + noise)
        
        theta_1 += update[0]
        theta_2 += update[1]
        
        theta_1 = normalize_angle(theta_1)
        theta_2 = normalize_angle(theta_2)
            
        history.append(np.array([theta_1, theta_2]))
        
        if np.linalg.norm(update) < tol:
            break
        
    history = np.vstack(history)
    
    return history 


def simulate_stackelberg(lr_1, lr_2, theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2, 
                         sigma=np.sqrt(.01), tol=1e-5, MAXITER=10000):
    
    theta = np.array([theta_1, theta_2])
    history = [theta]

    for t in xrange(MAXITER):
        gamma = np.diag([lr_1(t), lr_2(t)])
        lamb = np.diag([lr_2(t), lr_1(t)])
        noise = np.random.normal(0, sigma, size=(2))
                
        d1_f1 = d1f1(theta_1, theta_2, alpha_1, phi_1)
        d2_f1 = d2f1(theta_1, theta_2)
        d2_f2 = d2f2(theta_1, theta_2, alpha_2, phi_2)
        d_21 = d21(theta_1, theta_2)
        d_22 = d22(theta_1, theta_2, alpha_2, phi_2)
        update_1 = -lr_1(t)*(d1_f1 - d2_f1*(d_21/d_22) + noise[0])
        update_2 = -lr_2(t)*(d2_f2 + noise[1])
        update = np.array([update_1, update_2])
        
        theta_1 += update_1
        theta_2 += update_2
        
        theta_1 = normalize_angle(theta_1)
        theta_2 = normalize_angle(theta_2)
            
        history.append(np.array([theta_1, theta_2]))

        if np.linalg.norm(update) < tol:
            break
            
    history = np.vstack(history)
    
    return history 
    
    
def simulate_many_and_plot(lr_1, lr_2, alpha_1, alpha_2, phi_1, phi_2, sim_function, nash_points, 
                           critical_points, omega_1_zero, omega_2_zero, n=10, return_history=False, seeds=None):
    
    histories = []
    
    for sample in xrange(n):
        if seeds is not None:
            np.random.seed(seeds[sample])
        theta_1 = np.random.uniform(-np.pi, np.pi)
        theta_2 = np.random.uniform(-np.pi, np.pi)
        history = sim_function(lr_1, lr_2, theta_1, theta_2, alpha_1, alpha_2, phi_1, phi_2)
        histories.append(history)
    
    title = ''

    plot_instance(histories, omega_1_zero, omega_2_zero, critical_points, nash_points, title)

    if return_history:
        return histories


def normalize_angle(theta):
    
    if theta < -np.pi or theta > np.pi:
        low = -np.pi
        high = np.pi
        width = 2*np.pi
        offset = theta - low
        theta_normalized = offset - ((offset//width)*width) + low
    else:
        theta_normalized = theta
        
    assert theta_normalized >= -np.pi and theta_normalized <= np.pi
    
    return theta_normalized