import torch as tc
import torch
from torchdiffeq import odeint
import os
import numpy as np
from scipy.integrate import solve_ivp
from Model import Model

class Ode:

  def __init__(self, method):
    
    self.method = method
    self.list_lamb = []

  def __call__(self, mesh, M, F, theta0):
    f = lambda t,x:tc.linalg.pinv(M(x)+1e-3*tc.eye(len(x))) @ F(t,x)
    """ def f(t,x):
       Y=tc.linalg.lstsq(M(x)+0.1*tc.eye(len(x)),F(t,x))
       return Y  """
    Theta =None  
    Methods=['euler','midpoint','rk4','explicit_adams','implicit_adams','dopri8','dopri5','bosh3','fehlberg2','adaptive_heun']
    if self.method in Methods:
      Theta=odeint(f,theta0.detach(),mesh,rtol=1e-7,atol=1e-9,method=self.method).T.detach()
      return Theta
    else:
      exit("Not implement")
    
    
  def ODEintegrator(self, Mapprox, Fapprox, mesh, theta0=None, reg_parameter=0.15):
    dt = torch.abs(mesh[1]-mesh[0]) # this can be better

    list_theta= []
    list_theta.append(theta0.data)
    
    for k,t in enumerate(mesh[1:]):  
      print(k)
      M = Mapprox(list_theta[-1])
      M = M + 0.15 * torch.eye(M.size()[0])
      F = Fapprox(t, list_theta[-1])
      _theta = list_theta[-1] - dt * torch.linalg.inv(M) @ F
      list_theta.append(_theta.data)
      
    print("Integration done")
    return torch.stack(list_theta).T 
    
  def rk4_solver(self,Mapprox,Fapprox, t_points,y0):
      # Initialize the output array to store the solutions
      y_points = tc.zeros(len(y0),len(t_points))
      y_points[:,0] = y0
      def ode_func(t_n,y_n):
          y=tc.linalg.inv(Mapprox(y_n) + 1e-3*tc.eye(len(y_n))) @ Fapprox(t_n,y_n)
          return y

      # On peut calculer ici les vp (...)
      # M = Mapprox(y_n)
      
      
      # Time step
      h = t_points[1] - t_points[0]
      # RK4 integration loop
      for i in range(1, len(t_points)):
          if (i%100 == 0):
            print(i, ' / ', len(t_points))
          t_n = t_points[i-1]
          y_n = y_points[:,i-1]

          k1 = h * ode_func(t_n, y_n)
          k2 = h * ode_func(t_n + h/2, y_n + k1/2)
          k3 = h * ode_func(t_n + h/2, y_n + k2/2)
          k4 = h * ode_func(t_n + h, y_n + k3)

          y_points[:,i] = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6

      return y_points
  def solver_rk45(self, M, F, mesh, theta0=None, reg_parameter=0.15):
     #f = lambda t,x:tc.linalg.inv(M(x)+reg_parameter*tc.eye(len(theta0))) @ F(t,x)
     f = lambda t,x:tc.linalg.inv(M(x)) @ F(t,x)
     Theta =None  
     Theta=solve_ivp(f,[mesh[0],mesh[-1]],theta0)
     return Theta.y
   
  def rk4_solver_pbinv(self,Mapprox,Fapprox, t_points,y0, pde):
      # Initialize path for lambda
      number_of_sensors = 100
      x_sensor = tc.linspace(pde.xmin, pde.xmax, number_of_sensors).unsqueeze(1)       
      criterion=torch.nn.MSELoss()
    
      # Initialize the output array to store the solutions
      y_points = tc.zeros(len(y0),len(t_points))
      y_points[:,0] = y0
      def ode_func(t_n,y_n):
          #y=tc.linalg.inv(Mapprox(y_n) + 1e-3*tc.eye(len(y_n))) @ Fapprox(t_n,y_n)
          y=tc.linalg.pinv(Mapprox(y_n)) @ Fapprox(t_n,y_n)
          return y
      def chooseLamb(lamb):
        result = pde.f_lamb(t_n,x_sensor.squeeze(1), NN, lamb).detach()
        return criterion(result, result_dev)
      # Time step
      h = t_points[1] - t_points[0]
      # RK4 integration loop
      for i in range(1, len(t_points)):
          if (i%100 == 0):
            print(i, ' / ', len(t_points))
            

          t_n = t_points[i-1]
          y_n = y_points[:,i-1]

          k1 = h * ode_func(t_n, y_n)
          k2 = h * ode_func(t_n + h/2, y_n + k1/2)
          k3 = h * ode_func(t_n + h/2, y_n + k2/2)
          k4 = h * ode_func(t_n + h, y_n + k3)

          y_points[:,i] = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
          
          # number_of_sensors = 12
          # x_sensor = tc.linspace(-5 + 4 * t_n, 0 + 6*t_n, number_of_sensors).unsqueeze(1)
          result_dev = (pde.ue(t_n+0.001, x_sensor).detach()- pde.ue(t_n, x_sensor).detach())/0.001
          result_dev = result_dev.squeeze()
          NN = Model(1, 12, y_n)
          lamb_space = torch.linspace(0,10,101)
          lamb_pred = torch.zeros_like(lamb_space)
          for iter in range(len(lamb_space)):
            lamb_pred[iter] = chooseLamb(lamb_space[iter])
          pde.lamb = lamb_space[torch.argmin(lamb_pred)].data
          self.list_lamb.append((tc.abs(pde.lamb.data-6)/6))
          Fapprox.f = pde.f
          if  (i%10 == 0):
            print('value', i, ' / ', len(t_points), ': ', lamb_space[torch.argmin(lamb_pred)].data)

      return y_points
     
 

      
