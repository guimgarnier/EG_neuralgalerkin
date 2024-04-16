import torch as tc
#from model import Model
from Model import Model
#from PDEmerge import gradNN
from torch.func import vmap

class VecF:
  
    """
    Compute the Integral F by the Monte Carlo methods
    """
    def __init__(self, d, m, f, samples, kind = 'gaussian'):
        self.f = f
        self.samples = samples
        self.d = d
        self.m = m
        self.kind = kind

    def __call__(self, t, theta):
        
        NN = Model(self.d, self.m, theta, kind=self.kind)
        # Compute every U(theta, xi) and return tensor([[value1 at x1], [value2 at x2], ...])
        vect_grad =lambda theta,samples:vmap(NN.grad_theta,in_dims=(None,0))(theta,samples).T
        result_grad=vect_grad(theta,self.samples).detach()
        result_f = self.f(t, self.samples.squeeze(1), NN).detach()
        return tc.mean(result_grad * result_f, dim=1)
        
    
