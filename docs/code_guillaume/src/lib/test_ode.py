from ode import Ode
import torch as tc

def ftest(d,method):
    nb=1000
    ode1=Ode(method)
    mesh=tc.linspace(0,1,nb)
    t0=tc.ones(d)/2
    M= lambda x: tc.eye(d,d)
    F= lambda t,x: t*tc.ones(d)
    dt=tc.abs(mesh[1]-mesh[0])
    #dt=1e-4
    y_exact=tc.zeros(d,nb)
    for k,time in enumerate(mesh): 
        y_exact[:,k]=(time**2/2)*tc.ones(d)+1/2
    y_num=ode1(mesh,M,F,t0,dt)
    er=tc.max(tc.abs(y_exact-y_num))
    print(er)
    return er


def test_euler():
    assert ftest(1,'euler') <1e-3

def test_dopri():
    assert ftest(1,'dopri5') <1e-6

def test_rk4():
    assert ftest(1,'rk4') <1e-6

