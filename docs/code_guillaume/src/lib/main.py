from NeuralGalScheme import NeurGalScheme
from PDE import KdV, AC
import time


if __name__ == '__main__':
#    odeint=['euler','midpoint','rk4','explicit_adams','implicit_adams','dopri8','dopri5','bosh3','fehlberg2','adaptive_heun']
    start = time.time()

    pde= KdV(T = 4, xmin=-10, xmax=20, numberTimes=1000)
    pb  = NeurGalScheme(pde,size_random_samples=1000,m=12, kind='gaussian', odekind='rk4')
    pb.run()
    
    end = time.time()
    print('Executed in', end-start)
    
    pb.show_results()
    
    
    
