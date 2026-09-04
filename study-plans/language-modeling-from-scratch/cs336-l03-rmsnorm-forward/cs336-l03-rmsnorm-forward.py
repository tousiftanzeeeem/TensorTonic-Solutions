import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    d = x.shape[-1]
    x_squared = x ** 2
    sum = x_squared.sum(dim=-1,keepdim=True) * (1/d) + epsilon 
    sqrt_sum = torch.sqrt(sum)
    left = (x / sqrt_sum)
    g = torch.tensor(g)
    return left * g