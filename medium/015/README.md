# 15. Linear Regression Using Gradient Descent

- **Difficulty:** Medium (as shown in the supplied screenshot).
- **Category:** Machine Learning.
- **Topics:** Linear Regression, Gradient Descent, PyTorch, Autograd.
- **Problem:** [Deep-ML 15](https://www.deep-ml.com/problems/15).

The current public page labels this problem Easy. This repository retains the screenshot's difficulty and documents the supplied PyTorch implementation.

## 1. Problem Description

Fit a linear model using full-batch gradient descent. Given a feature matrix `X` of shape `(m, n)`, a target vector `y` of shape `(m,)`, a learning rate `alpha`, and an iteration count `iterations`, learn a vector of `n` coefficients.

`X` already includes a column of ones for the intercept. Initialize every coefficient to zero and use all training samples in each update. Minimize half the mean squared prediction error:

$$
L(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\left((X\theta)_i-y_i\right)^2.
$$

**Output contract:** The supplied PyTorch function returns a detached 1D Tensor of shape `(n,)`. The generic statement in the screenshot requests a NumPy array; if that interface is required, convert the result with `.cpu().numpy()` at the boundary. The implementation below preserves the supplied Tensor return type.

## 2. Example

```python
import numpy as np

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([3, 5, 7])
alpha = 0.1
iterations = 1000
```

Expected coefficients, approximately:

```text
[1.0, 2.0]
```

The target relationship is `y = 1 + 2x`: the first coefficient is the intercept and the second is the slope. The result approaches these coefficients through repeated updates; floating-point output need not equal the displayed values exactly.

To run the PyTorch version from this directory:

```python
import torch
from solution_v1 import linear_regression_gradient_descent

weights = linear_regression_gradient_descent(X, y, alpha, iterations)
print(weights)
assert weights.shape == (2,)
assert not weights.requires_grad
assert torch.allclose(weights, torch.tensor([1.0, 2.0]), atol=1e-3)
```

## 3. Solution

**Implementation:** [solution_v1.py](solution_v1.py)

Convert inputs to `float32` tensors, reshape the targets to `(m, 1)`, and initialize a trainable coefficient tensor of shape `(n, 1)`. Compute predictions with `X_t @ theta`. Autograd differentiates the scalar loss, and `SGD` updates the coefficients using the provided learning rate. Because each iteration uses the complete dataset and SGD's default momentum is zero, this is ordinary batch gradient descent:

$$
\nabla_\theta L = \frac{1}{m}X^\top(X\theta-y),\qquad
\theta \leftarrow \theta-\alpha\nabla_\theta L.
$$

Call `optimizer.zero_grad()` before backpropagation because PyTorch accumulates gradients. Finally, `detach()` removes gradient tracking and `flatten()` produces the required 1D output.

```python
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
```

### Complexity and Assumptions

- **Time:** `O(Tmn)` for `T` iterations.
- **Extra space:** `O(m + n)` for predictions, residuals, parameters, and gradients, excluding input conversion copies. A converted feature matrix can require an additional `O(mn)` allocation.
- Assumes nonempty, compatible inputs and CPU execution, matching the supplied initialization.
- Convergence depends on the learning rate and feature scale; a fixed number of iterations does not guarantee an exact solution.

## 4. Reflection

This was my first deep learning exercise on Deep-ML, implemented with PyTorch. It highlighted five practical habits I want to carry into future problems:

1. **Read the function signature carefully.** Identify the role of each provided argument and use it consistently. In this problem, `alpha` controls the learning rate and `iterations` controls the number of updates; neither should be replaced with a hard-coded value.

2. **Distinguish permitted library operations from required manual implementations.** Before coding, determine whether the exercise expects an algorithm from scratch or allows PyTorch utilities such as autograd, loss functions, and optimizers. Use the available abstractions when permitted, while respecting any explicit implementation constraints. A working library-based solution may not satisfy an exercise that requires manual gradients.

3. **Check the return contract explicitly.** Verify the output type, shape, and gradient-tracking state. Here, `theta.detach().flatten()` returns a 1D Tensor without gradient tracking; a NumPy-specific interface would require an additional conversion.

4. **Use matrix multiplication for the linear forward pass.** The `@` operator expresses `X @ theta` directly and computes predictions for all samples at once. The intercept is already represented by the column of ones, so this formulation does not require a separate bias parameter.

5. **Match the loss normalization to the stated objective.** With its default `reduction="mean"`, `torch.nn.MSELoss()` averages squared errors over all tensor elements. For this single-output problem, there are `m` elements, so it computes `(1/m) * sum(error**2)`. Multiplying by `0.5` gives the required `(1/(2*m)) * sum(error**2)`. Multiplying a loss by a positive constant preserves its minimizers, but scales its gradients and therefore changes the effective step size at a fixed learning rate. The optimization path and convergence behavior can change even though the optimum is unchanged.

### Reference

- [PyTorch MSELoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss)
