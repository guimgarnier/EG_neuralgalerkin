import torch
from config import device

class Bounds():
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class Domain1d():
    def __init__(self, name, bounds):
        self.name = name
        self.bounds = bounds

def in_domain1d(domain1d, value):
    return (value>=domain1d.bounds.lower) and (value<=domain1d.bounds.upper)

def in_domain(domain, value):
    return all(in_domain1d(domain1d, v) for (domain1d, v) in zip(domain.values(), value))

def sample_domain(domain, strategy, num_point_per_axis):
    vals_per_variable = {}
    for var_name in domain.keys():
        l = domain[var_name].bounds.lower
        u = domain[var_name].bounds.upper
        n = num_point_per_axis[var_name]
        if strategy[var_name] == "random":
            vals = l+(u-l)*torch.rand(n, device=device)
        elif strategy[var_name] == "grid":
            vals = torch.linspace(l, u, n, device=device)
        else:
            print("Type sampling not implemented")
        vals_per_variable[var_name]=vals
    return torch.cartesian_prod(*vals_per_variable.values())