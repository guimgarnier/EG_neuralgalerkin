#Partial differential equation
import torch
from torch.func import jacrev, hessian
import numpy as np

class PartialDifferentialEquation():
    def __init__(self,u0,domX=None,domP=None):
        self.u0=u0
        self.domX=domX
        self.domP=domP
    
    def f(self,t,x,u):
        pass
        


class KdV(PartialDifferentialEquation):
    def __init__(self,u0,domX=None,domP=None):
        super().__init__(u0,domX=None,domP=None)  
    def f(self,t,x,u=None):
        funct=torch.zeros(len(x))
        iter=0
        for x_v in x:
            x_v.requires_grad=True
            funct[iter]=-(hessian(jacrev(u,argnums=0),argnums=0)(x_v)).sum()+6*u(x_v)*(jacrev(u,argnums=0)(x_v)).sum()
            iter=iter+1
        return funct


    
class AdvectionDifusion(PartialDifferentialEquation):
    def __init__(self,u0,domX=None,domP=None, a=None, b=None, G=None):
        super().__init__(u0, domX, domP)
        self.a=a
        self.b=b
        self.G=G
        """
    def f(self,t,x,u):
        if t==0:
            return self.u0(x)
        else:
            funct=torch.zeros(x.shape)
            for points in range(len(x)):
                x_v=x[points].clone().detach()
                x_v.requires_grad=True
                funct[points]=(self.b(t,x_v)*jacrev(u,argnums=1)(t,x_v)).sum()+(self.a(t,x_v)*hessian(u,argnums=1)(t,x_v)).sum()+self.G(t,x_v,u)
            return funct"""
    def f(self,t,x,u):
        funct=torch.zeros(len(x))
        if t==0:
            iter=0
            for x_v in x:
                    x_v.requires_grad=True
                    funct[iter]=(self.b(t,x_v)*jacrev(self.u0,argnums=0)(x_v)).sum()+(self.a(t,x_v)*hessian(self.u0,argnums=0)(x_v)).sum()+self.G(t,x_v,self.u0)
                    iter=iter+1
        else:
            iter=0
            for x_v in x:
                    x_v.requires_grad=True
                    funct[iter]=(self.b(t,x_v)*jacrev(u,argnums=1)(t,x_v)).sum()+(self.a(t,x_v)*hessian(u,argnums=1)(t,x_v)).sum()+self.G(t,x_v,u)
                    iter=iter+1
        return funct




class AllenCahn(AdvectionDifusion):
    def __init__(self, u0, domX=None, domP=None,a=None, b=None, G=None) -> None:
        super().__init__(u0, domX, domP,a, b ,G)
        def a(t,x):
            return 5e-2
        self.a= a
        def G(t,x,u):
            if t==0:
                return -(1.05+t*torch.sin(x))*(self.u0(x)-self.u0(x)**3)
            else:
                return -(1.05+t*torch.sin(x))*(u(t,x)-u(t,x)**3)
        
        self.G=G
        def b(t,x):
            return 0
        self.b=b



class AD(AdvectionDifusion):
    def __init__(self, u0, space_time,domX=None, domP=None, a=None, b=None, G=None):
        super().__init__(u0, domX, domP, a, b, G)
        """"
        We use space_time to insure that if the Advection Difusion parameter a depends on the space 
        variable or not. Because the parameter $a$ is both are differents.
        """
        self.space=space_time 
        if self.space=="space_time":
            def a(t,x):
                return 0*t*x
            self.a=a
          
            def b(t,x):
                if x.shape==torch.Size([]):
                    a_s=1
                    a_v=2
                else:
                    a_s=torch.arange(1,len(x)+1)
                    a_v=2+2*torch.arange(0,len(x))/len(x)
                return -a_s*(torch.sin(a_v*torch.pi*t)+3)*(x+1)/10 
            self.b=b

            def G(t,x,u=None):
                if t==0:
                    return 0*self.u0(x)
                else:
                    return 0*u(t,x)
            self.G=G
        elif self.space=="time_only":
            def a(t,x):
                return 0*t*x
            self.a=a

            def b(t,x):
                if x.shape==torch.Size([]):
                    a_s=1
                    a_v=2
                else:
                    a_s=torch.arange(1,len(x)+1)
                    a_v=2+2*torch.arange(0,len(x))/len(x)
                return -a_s*(torch.sin(a_v*torch.pi*t)+5/4)
            self.b=b
            
            def G(t,x,u=None):
                if t==0:
                    return 0*self.u0(x)
                else:
                    return 0*u(t,x)
                
            self.G=G
        """
        else:
            print("You must precise if the constant a depend on space_time or time_only")"""
   
        
        
        
    


