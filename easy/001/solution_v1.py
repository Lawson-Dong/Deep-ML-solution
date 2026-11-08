import torch


def matrix_dot_vector(a, b) -> torch.Tensor:
    """Multiply a matrix of shape (m, n) by a vector of shape (n,).

    Accepts Python lists, NumPy arrays, or PyTorch tensors. Returns a
    one-dimensional tensor of length m, or a scalar tensor containing
    -1 if the matrix column count and vector length differ.
    """
    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)

    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)

    return a_t @ b_t
