import torch
from config import device

from torch.func import grad, vmap            # grad for functions
from torch.autograd import grad as autograd  # grad for Tensors



class Problem:
    def __init__(self, name, initial_condition):
        self.name = name
        self.initial_condition = initial_condition
    def evaluate(self, t, x, f):
        if name=='KdV':
            KDV=KdV(initial_condition)
        else:
            raise Exception("The problem name has not been found in the database")

class ParameterNN:
    """
    Code a class to define a Parameter theta. 
    """
    
    def __init__(self, w, b, c):
        self.w = w
        self.b = b
        self.c = c
                

class estimF:
    """
    Compute the Integral F by the Monte Carlo methods
    """
    def __init__(self, NN, f, weight, samples):
        self.NN = NN
        self.f = f
        self.weight = weight
        self.samples = samples

    def evaluate(self, t, theta):
        self.NN.w = theta.w
        self.NN.c = theta.c
        self.NN.b = theta.b
        
        # Compute every U(theta, xi) and return tensor([[value1 at x1], [value2 at x2], ...])
        y_pred = torch.tensor([self.NN(x) for x in self.samples])
        
        # Compute every gradU(theta,xi)
        gradU = gradNN(self.NN)
        result_grad = torch.zeros(self.samples.size()[0], 2 * m + d * m)
        result_grad[iter] = gradU.return_grad(theta, self.samples)
        
        tensorproduct = result_grad * self.f(t,samples, NN)
        return torch.mean(a, dim=1).unsqueeze(dim=1)
    
class estimM:
    """
    Compute the Integral M by the Monte Carlo methods
    """
    def __init__(self, NN, weight, samples):
        self.NN = NN
        self.weight = weight
        self.samples = samples

    def evaluate(self, theta):
        self.NN.w = theta.w
        self.NN.c = theta.c
        self.NN.b = theta.b
        
        # Compute every U(theta, xi) and return tensor([[value1 at x1], [value2 at x2], ...])
        y_pred = self.NN(self.samples)
        
        # Compute every gradU(theta,xi)
        gradU = gradNN(self.NN)
        result_grad = torch.zeros(self.samples.size()[0], 2 * m + d * m)
        result_grad= gradU.return_grad(theta, self.samples)
        
        tensorproduct = torch.zeros(self.samples.size()[0], 2 * m + d * m, 2 * m + d * m)
        for iter in range(0, self.samples.size()[0]):
            tensorproduct[iter] = torch.outer(result_grad[iter], result_grad[iter])    
         
        return torch.mean(tensorproduct, dim=0)
    
class targetDistribution:
    """
    Define the distribution we use to make our integration. 
    """
    def __init__(self, NN):
        self.NN = NN
    def value(self, x):
        return self.NN(x)
    
class gradNN:
    def __init__(self, NN):
        self.NN = NN
    def return_grad(self, theta, x):        
        y_pred = self.NN(x)
        d = self.NN.d
        m = self.NN.m
        result_grad = torch.zeros(y_pred.size()[0], theta.size()[0])
        for iter in range(y_pred.size()[0]):
            grad_theta = torch.autograd.grad(y_pred[iter], (self.NN.theta), retain_graph=True)
            result_grad[iter] = grad_theta[0]
        return result_grad