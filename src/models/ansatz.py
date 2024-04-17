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
            'gaussian'        : self.gaussian_x,   # Eq. (16)
        }

        self.Theta = tc.rand(self.s,requires_grad=True) #if theta is None else theta
        self.theta = tc.nn.Parameter(tc.rand(self.s) if theta is None else theta)
        
        self.eval_x    = methods[kind]
        self.forward   = vmap(self.eval_x)
        
    def splitter_gaussian(self):
        c = self.theta[:self.m]
        b = self.theta[self.m: self.m+self.m*self.d ].reshape(self.m,self.d)
        w = self.theta[self.m+self.m*self.d:]
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
    
    def __call__(self, x):
        return self.forward(x)