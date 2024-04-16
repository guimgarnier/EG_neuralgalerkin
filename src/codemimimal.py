

if __name__ == '__main__':
    
    pde = PDE('kdv', CDinit, domaine)

    def training():    
        model = Ansatz('base')
    
        ode = ODE(model, pde)
        
        varApprox(ode, 'mthd')
    
        varApprox.initialisation('mthd' )   # Resolution numérique parfaite de l'ODE pour plusieurs condition initiale (sur theta). Par exemple avec RK45, PINNs... 
    
        varApprox.train('mthd à choisir') # Calcul les premiers termes de la variable approchée puis fait l'entrainement
    
    
    def solve(PDE, 'base', method):
        NULL
    
        
    def visualisation():
        print('courbe erreur')
        print('solution')
    
    
    
class ODE():
    def __init__():
        ODEinit = model.init(pde.init)
        
        
    

    
    