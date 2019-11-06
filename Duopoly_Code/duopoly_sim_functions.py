from __future__ import division
from __future__ import print_function
import numpy as np
from duopoly_functions import omega, J0, nabla, d1f1, d2f1, d2f2, d12, d21, d22
from duopoly_figure_functions import plot_instance


def simulate_gd(lr_1, lr_2, x, y, A, c_1, c_2, sigma=0, tol=1e-6):
    
    theta = np.array([x, y])
    history = [theta]

    for t in xrange(50000):
        gamma = np.diag([lr_1(t), lr_2(t)])
        noise = np.random.normal(0, sigma, size=(2))
        
        update = gamma.dot(omega(x, y, A, c_1, c_2) + noise)
        
        x += update[0]
        y += update[1]
            
        history.append(np.array([x, y]))
        
        if np.linalg.norm(update) < tol:
            break
        
    history = np.vstack(history)
    
    return history 
    

def simulate_stackelberg(lr_1, lr_2, x, y, A, c_1, c_2, sigma=0, tol=1e-6):
    
    theta = np.array([x, y])
    history = [theta]

    for t in xrange(50000):
        gamma = np.diag([lr_1(t), lr_2(t)])
        lamb = np.diag([lr_2(t), lr_1(t)])
        noise = np.random.normal(0, sigma, size=(2))
                
        d1_f1 = d1f1(x, y, A, c_1, c_2)
        d2_f1 = d2f1(x, y, A, c_1, c_2)
        d2_f2 = d2f2(x, y, A, c_1, c_2)
        d_21 = d21(x, y, A, c_1, c_2)
        d_22 = d22(x, y, A, c_1, c_2)
        update_1 = lr_1(t)*(d1_f1 - d2_f1*(d_21/d_22) + noise[0])
        update_2 = lr_2(t)*(d2_f2 + noise[1])
        update = np.array([update_1, update_2])
        
        x += update_1
        y += update_2
            
        history.append(np.array([x, y]))

        if np.linalg.norm(update) < tol:
            break
            
    history = np.vstack(history)
    
    return history 
    
    
def simulate_many_and_plot(lr_1, lr_2, A, c_1, c_2, sim_function, nash_points, 
                           critical_points, omega_1_zero, omega_2_zero, 
                           n=10, sigma=np.sqrt(10), return_history=False):
    
    histories = []
    
    for sample in xrange(n):
        x = np.random.uniform(5,A/2)
        y = np.random.uniform(5, A/2)
        print(x, y)
        history = sim_function(lr_1, lr_2, x, y, A, c_1, c_2, sigma=sigma)
        histories.append(history)
    
    title = ''

    plot_instance(histories, omega_1_zero, omega_2_zero, critical_points, nash_points, title)

    if return_history:
        return histories