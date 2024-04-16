import torch
from Model import Model

class MatrixM:

    def __init__(self, d, m, samples, kind='gaussian'):
        self.samples = samples
        self.d = d
        self.m = m
        self.kind = kind
    
    def __call__(self, theta):
        NN = Model(self.d, self.m, theta, kind = self.kind)
        return NN.out_prod(theta,self.samples)
