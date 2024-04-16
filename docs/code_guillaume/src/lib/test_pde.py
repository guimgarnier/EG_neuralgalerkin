import PDE
import torch as tc

from PDE import KdV, AC
from model import Model

def test_kdv():
   
   pde = KdV(1, 0, 1)

   func_cos = Model(pde.d, 1, kind='simple_cos') # cos function as a model object
   
   x  = tc.linspace(0,1,100)
   t  = tc.linspace(0,1,100)
   
   ux = pde.f(t, x, func_cos)
   ue = lambda x: - 6.0 * tc.cos(x)*tc.sin(x) - tc.sin(x)
   
   assert tc.norm(ue(x)-ux) < 1e-5


def test_ac():
   
   pde = AC(1, 0, 1)
   
   func_cos = Model(pde.d, 1, kind='simple_cos') # cos function as a model object
   
   x  = tc.linspace(0,1,100)
   t  = tc.linspace(0,1,100)
   
   ux = pde.f(t, x, func_cos)
   ue=lambda x: -5*1e-2 * tc.cos(x) -(tc.cos(x) - tc.cos(x)**3)
   
   assert tc.norm(ue(x)-ux) < 1e-5
