#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:01:02 2023

@author: ggarnier
"""

import torch

class ShallowExp(torch.nn.Module):
    def __init__(self, d, m, theta=torch.rand(2*m + m*d)):
        super().__init__()
        self.theta  = torch.nn.Parameter(theta)
        self.m = m
        self.d = d
        
    def splitter(self):
        c = self.theta[:self.m]
        b = self.theta[self.m: self.m+self.m*self.d ].reshape(self.d,self.m)
        w = self.theta[self.m+self.m*self.d:]
        return (c,b,w)
    
    def forward(self, x):
        c,b,w = model.splitter()
        diff = x.unsqueeze(2)-b.t().unsqueeze(0) # tensor with shape (npoints,d,m)
        distance = torch.norm(diff, dim=1)**2 # tensor of dimension (npoints, m)
        exp_factor = torch.exp(-w.unsqueeze(0)**2*distance)
        return torch.sum(c.unsqueeze(0)*exp_factor, dim=1)
        
if __name__ == '__main__':
    d = 3; m=10   
    model = ShallowExp(d, m)
    c,b,w = model.splitter()
    
    print(c)
    print(b)
    print(w)