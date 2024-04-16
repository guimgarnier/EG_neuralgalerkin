import PDE
import torch as tc
from NeuralGalScheme import NeurGalScheme
from PDE import KdV
import matplotlib.pyplot as plt

def test_get_theta0():
  
  pde = KdV(T = 4, xmin=-10, xmax=20)
  pb = NeurGalScheme(pde, n=1000, m=2)
  pb.run()
  
  xv = tc.linspace(-10., 20., 1000)
  y_true = tc.cos(xv)
  y_pred = pb.model(xv.unsqueeze(1)).detach().numpy()
  assert tc.norm(y_true - y_pred) < 1e-4

    
