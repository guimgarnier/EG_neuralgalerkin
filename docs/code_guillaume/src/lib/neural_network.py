import torch
import warnings

class ShallowExp(torch.nn.Module):
    def __init__(self, d, m, theta=None):
        super().__init__()
        if theta == None:
            theta = torch.rand(2*m + d*m)
            warnings.warn("The neural network has been initiated with random theta")
        self.theta  = torch.nn.Parameter(theta)
        self.m = m
        self.d = d
        
    def splitter(self):
        c = self.theta[:self.m]
        b = self.theta[self.m: self.m+self.m*self.d ].reshape(self.m,self.d)
        w = self.theta[self.m+self.m*self.d:]
        return (c,b,w)
    
    def forward(self, x):
        c,b,w = self.splitter()
        diff = x.unsqueeze(2)-b.t().unsqueeze(0) # tensor with shape (npoints,d,m)
        distance = torch.norm(diff, dim=1)**2 # tensor of dimension (npoints, m)
        exp_factor = torch.exp(-w.unsqueeze(0)**2*distance)
        return torch.sum(c.unsqueeze(0)*exp_factor, dim=1)

class ShallowExpSin(torch.nn.Module):
    """Shallow network with exponential activation functions
        @param d: spatial dimension
        @param m: number of terms in the expansion
        @param L: characteristic length (default value = 1.)
    """
    def __init__(self, d, m, L=1.):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(m))
        self.b = torch.nn.Parameter(torch.rand(m, d))
        self.c = torch.nn.Parameter(torch.rand(m))
        self.L = L
        
    def forward(self, x):
        """Input argument is x.shape = (npoints, d)
        """        
        diff = (torch.pi/self.L)*(x.unsqueeze(2)-self.b.t().unsqueeze(0)) # tensor with shape (npoints,d,m)
        distance = torch.norm(torch.sin(diff), dim=1)**2 # tensor of dimension (npoints, m)
        exp_factor = torch.exp(-self.w.unsqueeze(0)**2*distance)
        return torch.sum(self.c.unsqueeze(0)*exp_factor, dim=1)

    def forward_non_vectorized(self,x):
        """Input argument is x.shape = (d) 
        """
        return torch.matmul(self.c,torch.exp(-self.w**2*torch.linalg.vector_norm(torch.sin((torch.pi/self.L)*(x-self.b)), dim=1)**2))
       
if __name__ == '__main__':
    d = 2
    m = 10
    model = ShallowExp(d,m)
    x_train = torch.rand(10, 2)
    print(x_train)
    print(model(x_train))
    