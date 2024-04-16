#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:18:01 2023

@author: ggarnier
"""

import torch
import torch.distributions as dist
from config import device

class MetropolisHastingsSampler:
    def __init__(self, num_samples, initial_state, target_distribution, proposal_distribution):
        self.num_samples = num_samples
        self.initial_state = initial_state
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution

    def metropolis_hastings(self):
        samples = torch.empty((self.num_samples, self.initial_state.size(0)))
        current_state = self.initial_state
        accepted_samples = 0

        for i in range(self.num_samples):
            # Sample a candidate from the proposal distribution
            candidate_state = self.proposal_distribution(current_state).sample()

            # Calculate the acceptance ratio
            accept_ratio = (self.target_distribution(candidate_state.unsqueeze(0))) / (self.target_distribution(current_state.unsqueeze(0)))

            # Accept or reject the candidate
            if torch.rand(1) < accept_ratio:
                current_state = candidate_state
                accepted_samples += 1

            samples[i] = current_state

        acceptance_rate = accepted_samples / self.num_samples
        return samples, acceptance_rate

def monte_carlo_integration(func, sample):
    """
    Compute the integral of a given function using the Monte Carlo method.

    Parameters:
        func (callable): The function to integrate. It should accept a tensor input of the form 
          tensor([[x1, y1, z1],
                  [x2, y2, z2],
                  ...,
                  [x3, y3, z3],])
          and give an output tensor of the form tensor([value_sample1, value_sample2, ..., value_sampleN])
        samples(tensor): Random sample where integrate

    Returns:
        torch.Tensor: Approximated integral value.
    """
    # Evaluate the function at the generated samples
    func_values = func(sample)

    # Compute the approximate integral
    integral = torch.mean(func_values)
    return integral



    
    

        
        
        
        
