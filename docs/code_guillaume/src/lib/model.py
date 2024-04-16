import torch as tc
import warnings

from torch.func import grad, vmap            # grad for functions
from torch.autograd import grad as autograd  # grad for Tensors

# torch.func.grad     acts on a function and returns a function
# torch.autograd.grad acts on a Tensor and returns a Tensor


class Model(tc.nn.Module):
    
    def __init__(self, d, m, theta=None, kind='gaussian'):
        
        super().__init__()
        self.m = m       # model parameter 
        self.d = d       # dimension of b (same as x)
        
        # size of theta according to the the model
        if kind == 'gaussian':
            self.s = 2*m+d*m 
        elif kind== 'nn':
            self.L = 0.5
            self.l = 3
            self.s = m + m*d * self.l + d * self.l 
        else:
            self.s = 2*m+d*m
            warnings.warn("The type of network has not The theta parameter has not been initialized") 
        
        # Dictionary of models 
        methods = {
            'gaussian'        : self.gaussian_x,   # Eq. (16)
            'simple_cos'      : self.scos_x,       
            'nn'              : self.nn_x              # Eq. (17)
        }

        # Warning : une initialisation à ones fait échouer la minimisation pour theta0 !
        self.theta = tc.nn.Parameter(tc.rand(self.s) if theta is None else theta)
        #self.theta = tc.nn.Parameter(2.0*tc.ones(self.s) if theta is None else theta) 

        # batched derivatives , input are (d,)
        self.eval_x    = methods[kind]
        self.forward   = vmap(self.eval_x)
        self.dx        = vmap(grad(self.eval_x),             in_dims=(0,) )
        self.dxx       = vmap(grad(grad(self.eval_x)),       in_dims=(0,) )
        self.dxxx      = vmap(grad(grad(grad(self.eval_x))), in_dims=(0,) )

        # work in progress
        # self.dtheta = vmap(lambda x : autograd(self.eval_x(x), self.theta)[0])

        
    def splitter_gaussian(self):
        c = self.theta[:self.m]
        b = self.theta[self.m: self.m+self.m*self.d ].reshape(self.m,self.d)
        w = self.theta[self.m+self.m*self.d:]
        return (c,b,w)
    
    def splitter_nn(self):
        c = self.theta[:self.m]
        b=self.theta[self.m:self.m+self.d*self.l].reshape(self.l,self.d)
        w=self.theta[self.m+self.d*self.l:].reshape(self.l,self.m,self.d)
        return (c,b,w)
    
    
    # All the model
    # WARNING : do not use this functions on anything else than scalar points.
    
    def scos_x(self, x: tc.Tensor):
        """        
        Input :  a SCALAR x of shape (0,)
        Output : a SCALAR
        """
        return tc.cos(x)
    
    def gaussian_x(self, x: tc.Tensor):
        """        
        Input :  a SCALAR x of shape (0,)
        Output : a SCALAR
        """
        c, b, w = self.splitter_gaussian()
        #y=tc.sum(c*tc.exp(-w**2*(x-b).T**2))
        return tc.sum(c*tc.exp(-w**2*(x-b).T**2))
    
    def nn_x(self, x: tc.Tensor):
        """        
        Input :  a SCALAR x of shape (0,)
        Output : a SCALAR
        """
        c, b, w = self.splitter_nn()
        x = tc.tanh(w[0]* tc.sin(2*tc.pi*(x - b[0])/self.L))
        for k in range(1,self.l):
            x = tc.tanh(w[k] * x + b[k])
        x = c.T @ x
        return x[0]

    
if __name__ == '__main__':

    d = 1000
    m = 2
    model = Model(d,m, kind='gaussian')

    # For cos(x)
    x = tc.Tensor([0.0, tc.pi/2.0, tc.pi])
    x.requires_grad_(True)
    U   = model(x)           # should return [1, 0, -1]
    Ux  = model.dx(x)        # should return [0, -1, 0]    
    Uxx = model.dxx(x)       # should return 
    Uxxx= model.dxxx(x)      # should return     


    # For gaussians :
    # U   = model(x)           # should return [0.54, 4.0]
    # Ux  = model.dx(x)        # should return [2.16, 0.0]    
    # Uxx = model.dxx(x)       # should return [6.49, -16.0]
    # Uxxx= model.dxxx(x)      # should return [8.66, 0.0]    
    
    
