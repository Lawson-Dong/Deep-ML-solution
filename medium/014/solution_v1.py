import torch


def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """Fit linear regression with PyTorch's least-squares solver.

    Args:
        X: Feature matrix of shape (m, n).
        y: Targets of shape (m,) or (m, 1).

    Returns:
        A 1D tensor of n coefficients rounded to four decimal places.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    theta = torch.linalg.lstsq(X_t, y_t).solution
    return torch.round(theta.flatten(), decimals=4)
