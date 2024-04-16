import warnings
import torch
import torch as tc
from torch.func import jacfwd, vmap, grad,jacrev

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
        }

        # Warning : une initialisation à ones fait échouer la minimisation pour theta0 !
        self.Theta = tc.rand(self.s,requires_grad=True) #if theta is None else theta
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
    def splitter_gaussian1(self):
        c = self.Theta[:self.m]
        b = self.Theta[self.m: self.m+self.m*self.d ].reshape(self.m,self.d)
        w = self.Theta[self.m+self.m*self.d:]
        return (c,b,w)
    
    def splitter_nn(self):
        c = self.theta[:self.m]
        b=self.theta[self.m:self.m+self.d*self.l].reshape(self.l,self.d)
        w=self.theta[self.m+self.d*self.l:].reshape(self.l,self.m,self.d)
        return (c,b,w)
    
    def gaussian_x(self, x: tc.Tensor):
               
       # Input :  a SCALAR x of shape (0,)
       # Output : a SCALAR
        
        c, b, w = self.splitter_gaussian()
        return tc.sum(c*tc.exp(-w**2 *(x-b).T**2))  

    
    """
    
    def forward_t(self, theta, x):
        self.theta = theta
        c,b,w = self.splitter_gaussian()
        diff = x.unsqueeze(2)-b.t().unsqueeze(0) # tensor with shape (npoints,d,m)
        distance = torch.norm(diff, dim=1)**2 # tensor of dimension (npoints, m)
        exp_factor = torch.exp(-w.unsqueeze(0)**2*distance)
        return torch.sum(c.unsqueeze(0)*exp_factor, dim=1).unsqueeze(1) """
    def forward_t(self, theta, x):
        self.Theta = theta.clone()
        c,b,w = self.splitter_gaussian1()
        #vect=lambda x: (c*tc.exp(-w**2*(x-b)**2)).sum()
        #return (c*tc.exp(-w**2*(x-b)**2)).sum()
        return tc.sum(c*tc.exp(-w**2*(x-b).T**2))
    def grad_theta(self, theta, samples):
        self.Theta = theta.clone()
        jacobian=jacrev(self.forward_t,argnums=0)(self.Theta, samples)
        return jacobian
        
    def out_prod(self,theta,x):
        #self.Theta=theta.clone()
        out_p=lambda theta,x_: tc.outer(self.grad_theta(theta,x_),self.grad_theta(theta,x_))
        M=vmap((out_p),in_dims=(None,0))(theta,x).detach()
        return tc.mean(M,dim=0)
    
    def __call__(self, x):
        return self.forward(x)
        