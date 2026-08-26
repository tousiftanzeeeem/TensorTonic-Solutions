import numpy as np


def entropy_node(y: list[int]) -> float:
  """Returns the Shannon entropy as a Python float."""
  y = np.asarray(y)
  if y.size == 0:
    return 0.0
  _, counts = np.unique(y, return_counts=True)
  probabilities = counts / y.size
  entropy = -np.sum(probabilities * np.log2(probabilities))

  return float(entropy)