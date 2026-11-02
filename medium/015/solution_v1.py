import torch


def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """Fit linear regression with full-batch SGD and PyTorch autograd.

    Args:
        X: Feature matrix of shape (m, n), including a column of ones
            for the intercept. Accepts tensors or array-like inputs.
        y: Target vector of shape (m,).
        alpha: Learning rate.
        iterations: Number of gradient descent updates.

    Returns:
        Detached learned weights as a 1D tensor of shape (n,).
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=[theta], lr=alpha)

    for epoch in range(iterations):
        # Forward pass and half mean squared error.
        y_pred = X_t @ theta
        loss = 0.5 * loss_fn(y_pred, y_t)

        # Clear accumulated gradients, differentiate, and update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return theta.detach().flatten()
