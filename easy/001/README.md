# 1. Matrix-Vector Dot Product

- **Difficulty** — Easy.
- **Category** — Linear Algebra.
- **Topics** — Matrix–Vector Multiplication, PyTorch.
- **Problem** — [Deep-ML 1](https://www.deep-ml.com/problems/1).

## 1. Problem Description

Multiply a matrix $A$ of shape $(m, n)$ by a one-dimensional vector $b$ of length $n$. The product contains one entry per matrix row. Return the scalar value $-1$ if the number of matrix columns differs from the vector length. The generic problem expects a list or an integer. The PyTorch variant returns a one-dimensional tensor for the product or a scalar tensor for the mismatch.

## 2. Example

```python
from solution_v1 import matrix_dot_vector

a = [[1, 2], [2, 4]]
b = [1, 2]

result = matrix_dot_vector(a, b)
assert result.shape == (2,)
assert result.tolist() == [5.0, 10.0]

mismatch = matrix_dot_vector(a, [1])
assert mismatch.ndim == 0
assert mismatch.item() == -1
```

The first row gives $1\cdot 1 + 2\cdot 2 = 5$ and the second row gives $2\cdot 1 + 4\cdot 2 = 10$.

## 3. Solution

[PyTorch implementation](solution_v1.py)

Convert both inputs to `float32` tensors and compare the matrix column count with the vector length. When the dimensions match, `a_t @ b_t` computes the dot product of each row with the vector.

$$
(Ab)_i = \sum_{j=1}^{n} A_{ij} b_j,\qquad i=1,\ldots,m.
$$

- **Time** — `O(mn)` for compatible inputs.
- **Extra space** — `O(m)` for the output, excluding input conversion copies.

## 4. Reflection

I use `@` for matrix multiplication in PyTorch. Here, `a_t @ b_t` gives one dot product per matrix row. The `*` operator performs element-wise multiplication.
