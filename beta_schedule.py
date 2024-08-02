import torch
import numpy as np


def linear_beta_schedule(timesteps, beta_start, beta_end):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    return torch.linspace(beta_start, beta_end, timesteps)
