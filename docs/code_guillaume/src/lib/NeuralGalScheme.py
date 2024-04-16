from ode import Ode
#from model import Model
from Model import Model
from matrixM import MatrixM
from vecF import VecF

import torch as tc
import torch
import matplotlib.pyplot as plt
import numpy as np

from PDE import KdV, AC

import sys
from IPython.core.ultratb import ColorTB
sys.excepthook = ColorTB()

class NeurGalScheme:
  
  def __init__(self, pde, size_random_samples=1000, m=2, datasample=100, kind='gaussian', odekind='explicit'):
    
    self.pde    = pde
    self.data   = None
    self.kind   = kind
    self.odeint = Ode(odekind)     # this can be explicit, implicit, rk45
    self.model  = Model(self.pde.d, m, kind=self.kind)  # to be fixed at init
    self.u      = None                # matrix of size meshsize x timestep
    self.theta  = None                # matrix of all theta(t) 
    self.size_random_samples = size_random_samples
    self.lamb = None
    
  def run(self, withdata=False):
    """
    This function will compute u(t,x) and store it in self.u
    """

    # Compute theta0 with minimization problem using the model, see eq. (9)
    self.minimization() 
    
    theta = tc.zeros_like(self.model.theta)
    for k,x in enumerate(self.model.theta):
      theta[k]=x.clone() 
    samples = tc.rand(self.size_random_samples, 1)*30-10
    Mapprox = MatrixM(self.pde.d, self.model.m, samples, kind = self.kind)
    Fapprox = VecF(self.pde.d, self.model.m, self.pde.f, samples, kind = self.kind)
    #self.Theta = self.odeint.ODEintegrator(Mapprox, Fapprox, self.pde.mesh, theta)
    #self.Theta = self.odeint(self.pde.mesh, Mapprox, Fapprox, theta)
    #self.theta = self.odeint.solver_rk45(Mapprox, Fapprox, self.pde.mesh,theta)
    self.Theta = self.odeint.rk4_solver(Mapprox, Fapprox, self.pde.mesh, theta)

    
    

  def show_results(self):
    """
    We plot
    - the solution u (once computed) for every t and x
    - and also the exact solution (self.pde.ue) from for every t and x
    """
    
    ## Print the solition as a heat map
    number_of_samples = 1000
    number_of_times = self.pde.numberTimes
    
    x_train = tc.linspace(-10, 20, number_of_samples).unsqueeze(1)
    Q = torch.rand(number_of_times, number_of_samples)

    for iter in range(number_of_times):
      theta_t = self.Theta[:, iter]
      #print(theta_t)
      model_t = Model(self.pde.d,self.model.m, theta_t, kind=self.kind)
      y_t = model_t(x_train)
      Q[iter, :] = y_t

    Q = torch.fliplr(Q)

    plt.xlabel('t')
    plt.ylabel('x')
    plt.imshow(Q.T.detach().numpy(), aspect=0.1, cmap='bwr', interpolation='none', extent=[0,self.pde.T,self.pde.xmin, self.pde.xmax])
    plt.colorbar()
    plt.show()



    
  def minimization(self):
    print(' - Start to initialize theta0')
    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(self.model.parameters(),lr=0.01)

    number_of_samples = 1000
    num_epochs        = 10000
    print('   || Parameters: ', 'Number of samples=', number_of_samples, '- Number of epochs=', num_epochs)
    
    x_train = tc.linspace(self.pde.xmin, self.pde.xmax, number_of_samples).unsqueeze(1)
    y_true = self.pde.u0(x_train).squeeze()
    y_pred = self.model(x_train) # vectorized

    total_acc, total_loss=[], []
    for epoch in range(num_epochs):
      y_pred=self.model(x_train)
      loss = criterion(y_pred,y_true)
      if epoch%500 == 0:
        print('loss', loss)      
      loss.backward(retain_graph=True) # back propagation
      optimizer.step()
      optimizer.zero_grad()
    print(' - Training is complete')
    return None


if __name__ == '__main__':
    #tc.manual_seed(0)
    #torch.manual_seed(0)
    #np.random.seed(0)

    print("# Initiatization of the NeuralGalScheme Class")
    print("#      (Step 1/3) - Creation of the PDE")
    
    pde = KdV(T = 4, xmin=-10, xmax=20, numberTimes=1000)
    #pde=AC(T=15,xmin=0,xmax=2*tc.pi,numberTimes=100)

    print("#      (Step 2/3) - Creation of the Scheme")
    pb  = NeurGalScheme(pde,size_random_samples=1000,m=12, kind='gaussian', odekind='dopri5')
    #pb  = NeurGalScheme(pde,size_random_samples=1000,m=2, kind='nn', odekind='explicit')
    pb.run()
    pb.show_results()
    
    print("#      (Step 3/3) - Plot the value of the neural scheme for theta0")
    if pde.d==1:
      xv = torch.linspace(pde.xmin, pde.xmax, 1000)
      x_train = tc.linspace(pde.xmin, pde.xmax, 1000)
      y_true = pde.u0(x_train)
      x_train = np.array(x_train)
      y_pred = pb.model(xv.unsqueeze(1)).detach().numpy()
      plt.plot(xv.numpy(), y_true,'-', label=r'$u_0$')
      plt.plot(xv.numpy(), y_pred,'--', label=r'$U(\cdot, \theta_0)')
      plt.xlabel('x')
      plt.legend()
      plt.grid()
      plt.show()
    else:
      print("Warning: Plot for d>1 not supported.")
      
    ## other time
    time_test_set = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    for time_test in time_test_set:
      theta_t= pb.Theta[:, time_test]
      time_space = pb.pde.mesh[time_test].data
      NN = Model(1, 12, theta_t)
      print("#      (Step 3/3) - Plot the value of the neural scheme for theta_t")
      if pde.d==1:
        xv = torch.linspace(pde.xmin, pde.xmax, 1000)
        x_train = tc.linspace(pde.xmin, pde.xmax, 1000)
        y_true = pde.ue(time_space, x_train)
        x_train = np.array(x_train)
        y_pred = NN(xv.unsqueeze(1)).detach().numpy()
        plt.plot(xv.numpy(), y_true,'-', label='y true')
        plt.plot(xv.numpy(), y_pred,'--', label='y pred')
        plt.legend()
        plt.title('time: ' +str(time_space))
        plt.grid()
        plt.savefig('t'+str(time_space)+'.png')
        plt.show()
      else:
        print("Warning: Plot for d>1 not supported.")
      
    #pb.run(withdata=True) # pb.u 
      
