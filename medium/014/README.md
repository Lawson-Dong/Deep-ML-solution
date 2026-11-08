# 14. Linear Regression Using Normal Equation

- **Difficulty** — Medium.
- **Category** — Machine Learning.
- **Topics** — Linear Regression, Least Squares, PyTorch.
- **Problem** — [Deep-ML 14](https://www.deep-ml.com/problems/14).

## 1. Problem Description

Given a feature matrix `X` of shape `(m, n)` and targets `y` of shape `(m,)` or `(m, 1)`, fit a linear regression model by least squares. Include a column of ones in `X` when fitting an intercept. Return the `n` coefficients as a one-dimensional PyTorch tensor, rounded to four decimal places. A value close to zero may round to `-0.0`, which the exercise accepts.

## 2. Example

```python
import torch
from solution_v1 import linear_regression_normal_equation

X = [[1, 1], [1, 2], [1, 3]]
y = [1, 2, 3]

theta = linear_regression_normal_equation(X, y)
assert theta.shape == (2,)
assert torch.equal(theta, torch.tensor([0.0, 1.0]))
```

The target follows $y = 0 + 1x$, so the first coefficient is the intercept $0$ and the second is the slope $1$.

## 3. Solution

[PyTorch implementation](solution_v1.py)

Minimize the sum of squared residuals and set its gradient to zero.

$$
L(\theta) = \lVert X\theta - y \rVert_2^2,\qquad
\nabla_\theta L = 2X^\top(X\theta - y) = 0.
$$

This yields the normal equations.

$$
X^\top X\theta = X^\top y.
$$

The implementation converts both inputs to `float32` tensors, reshapes the targets to `(m, 1)`, and calls `torch.linalg.lstsq(X_t, y_t).solution`. The solver works directly with the least-squares problem. Flattening changes the output from `(n, 1)` to `(n,)` before rounding.

### Complexity and Assumptions

- **Time** — `O(mn²)` for the default CPU solver when `m >= n`.
- **Extra space** — `O(mn)` to hold the solver's working copy of the matrix, excluding input conversion copies.
- `X` and `y` need compatible shapes. The example uses CPU tensors.

## 4. Reflection

### The Normal Equation for Linear Regression

Some machine learning models have a closed-form solution. In ordinary least squares, the loss is a convex quadratic in the coefficients. Setting its gradient to zero gives a linear system. When $X$ has full column rank, $X^\top X$ is invertible and the coefficients have the unique solution below.

$$
\theta = (X^\top X)^{-1}X^\top y.
$$

A deep network with only linear layers represents a single linear map. With squared error, I can apply the same least-squares reasoning to the effective map when it can be fitted freely. The loss as a function of the individual layers is generally nonconvex, and multiple weight factorizations can give the same effective map. Fully linear deep networks are uncommon in practice. The practical point I want to remember is that explicitly inverting a large matrix can be computationally expensive, and solving for the effective map can still leave many choices for the individual weights.

In PyTorch, getting the least-squares coefficients takes one call once `X_t` and `y_t` are ready.

```python
import torch

theta = torch.linalg.lstsq(X_t, y_t).solution
```

- `torch.linalg` is PyTorch's linear algebra module. It includes tools for matrix inverses, eigenvalues, and singular value decomposition.
- `lstsq` stands for least squares. It computes the coefficients directly from the design matrix and the targets.
- `.solution` selects the coefficient tensor from the named result. The other fields are `residuals`, `rank`, and `singular_values`. Their availability depends on the solver and the input.

### Reference

- [PyTorch `torch.linalg.lstsq` documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html)
