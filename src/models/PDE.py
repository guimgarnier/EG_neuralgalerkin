import torch as tc

class PDE:
  
  def __init__(self, mesh_x = tc.linspace(0,1,1000), mesh_t = tc.linspace(0,1,100), dim=1, u0=None):

    self.mesh_x = mesh_x
    self.mesh_t = mesh_t

class KdV(PDE):

  def __init__(self):

    super().__init__()
    self.lamb = 6
   
  def f(self, X, u, t=0):
    """
    Input:
    - X is a **vector** of points
    - t is a time (scalar)
    - u is a Model object

    Output:
    - F a vector of the RHS of KdV for each x
    """
    return - self.lamb * u.dx(X) * u(X) - u.dxxx(X)

  def ue(self, t, x):
    """
    Exact solution in t, x of the KdV equation 
    """
    
    k = tc.tensor([1., sqrt(5.)])
    eta = tc.tensor([0., 10.73])
    t = tc.tensor([t])

    etaMat1 = k[0] * x.reshape((-1,1)) - k[0]**3*t.reshape((1,-1)) + eta[0]
    etaMat2 = k[1] * x.reshape((-1,1)) - k[1]**3*t.reshape((1,-1)) + eta[1]
    
    c = ((k[0] - k[1])/(k[0] + k[1]))**2

    f = 1. + tc.exp(etaMat1) + tc.exp(etaMat2) + tc.multiply(tc.exp(etaMat1), tc.exp(etaMat2)*c)
    df = k[0]*tc.exp(etaMat1) + k[1]*tc.exp(etaMat2) + c*(k[0] + k[1])*tc.multiply(tc.exp(etaMat1), tc.exp(etaMat2))
    ddf = k[0]**2*tc.exp(etaMat1) + k[1]**2*tc.exp(etaMat2) + c*(k[0] + k[1])**2*tc.multiply(tc.exp(etaMat1), tc.exp(etaMat2))

    y = 2*tc.divide(tc.multiply(f, ddf) - df**2, f**2);

    # avoid numerical errors far outside of [-1, 2]
    y = tc.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) 

    return y


  def u0(self, x):
    return self.ue(0., x)

  
    
class AC(PDE):
  """
  The Allen-Cahn class
  """
  def __init__(self, T, xmin, xmax, numberTimes=100, u0=None):

    super().__init__()
    self.T    = T
    self.numberTimes = numberTimes
    self.mesh = tc.linspace(0., T, self.numberTimes) # time mesh 
    self.d    = 1
    self.xmin = xmin
    self.xmax = xmax

    
  def f(self, t, X, u):
    """
    Input:
    - X is a vector of points
    - t is a time (scalar)
    
    - u is a model (function of theta and x)

    Output:
    - F a vector of the RHS of KdV for each x
    """
    eps = 5e-2
    U   = u(X)
    a   =1.05+t*tc.sin(X)
    return eps * u.dxx(X) - a*(U-U**3)

  def u0(self, X):
    L=0.5
    phiG= lambda x,w,b:tc.exp(-w**2*tc.abs(tc.sin(tc.pi*(x-b)/L))**2) 
    return phiG(X,400.0,0.5)-phiG(X,400.0,4.4)
  



