import torch
from torch import nn
import warnings
import matplotlib.pyplot as plt

import warnings
import torch
import torch as tc
from torch.func import jacfwd, vmap, grad,jacrev


class Ansatz(tc.nn.Module):

    def __init__(self, d, m, theta=None, kind='gaussian'):
        
        super().__init__()
        self.m = m       # model parameter 
        self.d = d       # dimension of b (same as x)
        
        # size of theta according to the the model
        if kind == 'gaussian':
            self.s = 2*m+m 
        else:
            self.s = 2*m+d*m
            warnings.warn("The type of network has not The theta parameter has not been initialized") 
        
        # Dictionary of models 
        methods = {
            'gaussian'        : self.gaussian_x,
        }

        self.Theta = tc.rand(self.s,requires_grad=True) if theta is None else theta 
        self.theta = tc.nn.Parameter(tc.rand(self.s) if theta is None else theta)
        
        self.eval_x    = methods[kind]
        self.forward   = vmap(self.eval_x)
        
    def splitter_gaussian(self):
        c = self.Theta[:self.m]
        b = self.Theta[self.m: self.m+self.m*self.d ].reshape(self.m,self.d)
        w = self.Theta[self.m+self.m*self.d:]
        return (c,b,w)
    
    def gaussian_x(self, x: tc.Tensor):
               
       # Input :  a SCALAR x of shape (0,)
       # Output : a SCALAR
        
        c, b, w = self.splitter_gaussian()
        return tc.sum(c*tc.exp(-w**2 *(x-b).T**2))  


    def forward(self, x):
        self.Theta = self.parameters()
        c,b,w = self.splitter_gaussian()
        #vect=lambda x: (c*tc.exp(-w**2*(x-b)**2)).sum()
        #return (c*tc.exp(-w**2*(x-b)**2)).sum()
        return tc.sum(c*tc.exp(-w**2*(x-b).T**2))

    def forward_t(self, theta, x):
        self.Theta = theta.clone()
        c,b,w = self.splitter_gaussian()
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
    
    
if __name__ == '__main__':
    print("> Example of utilisation for the Class Anstatz")
    
    # Definition of the space domain and the fonction to approximate
    x_domain = [0.0, 1.0]; n_points_x = 1000
    mesh_x = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x, requires_grad=True).unsqueeze(1)
    u0 = lambda x : (1-x)**2 * x**2
    y_true = u0(mesh_x).squeeze()
    
    # Definition of the model
    d = 1
    m = 10
    model = Ansatz(d,m)
    
    # Training
    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

    total_acc, total_loss=[], []
    for epoch in range(30000):
        y_pred=model(mesh_x)
        loss = criterion(y_pred,y_true)
        if epoch%3000 == 0:
            print('epoch:',epoch,'| loss: ', loss)
        optimizer.zero_grad()
        loss.backward(retain_graph=True) # back propagation
        optimizer.step()
    print(' - Training is complete')
    
    # Visualisation
    y_train = model(mesh_x)
    fig = plt.figure(figsize=(15, 7))
    plt.plot(mesh_x.detach(), y_train.detach(), 'k', lw=3)
    plt.plot(mesh_x.detach(), y_true.detach(),'r:', lw=5)
    plt.grid()
    plt.show()
    
# compute the derivative
def dx(f: Ansatz, x: torch.Tensor = None, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = f(x)
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value 

