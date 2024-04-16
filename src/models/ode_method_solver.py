
'''ODE de type theta'=f(theta) et qui applique les méthodes classiques de résolution d'EDO. 
 theta est un élément de R^d où d est la dimension '''

import torch
from torchdiffeq import odeint
import numpy as np


def sol_perfect(theta_0, f,t):
    return odeint(f,theta_0, t)

def sol_rk4(theta0,f,t,dt):
    return odeint(f, theta_0, t, method = 'rk4', options = dict(stepsize=dt))

def sol_euler_explicit(theta0,f,t,dt):
    return odeint(f, theta_0, t, method = 'euler', options = dict(stepsize=dt))




## Examples 

# def f(t,z):
#     return z

# theta_0 = torch.Tensor([1])
# t = torch.linspace(0.1,1.2,1000)
# dt = 0.01
# out = sol_euler_explicit(theta_0,f,t,dt)


## Visualize 


#import matplotlib.pyplot as plt
#plt.plot(t, out, 'go--')
#plt.grid()
#plt.show()