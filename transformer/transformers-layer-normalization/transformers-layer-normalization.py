import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1,keepdims=True)
    return gamma * ((x-mean)/np.sqrt(var + eps)) + beta