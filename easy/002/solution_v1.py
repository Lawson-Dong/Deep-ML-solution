import torch


def transpose_matrix(a) -> torch.Tensor:
    """Transpose a two-dimensional matrix using PyTorch.

    Accepts a nested list, NumPy array, or PyTorch tensor of shape
    (m, n). Returns a tensor of shape (n, m).
    """
    a_t = torch.as_tensor(a)
    return a_t.T
