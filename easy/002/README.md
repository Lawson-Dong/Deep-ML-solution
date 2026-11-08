# 2. Transpose of a Matrix

- **Difficulty** — Easy.
- **Category** — Linear Algebra.
- **Topics** — Matrix Transpose, PyTorch.
- **Problem** — [Deep-ML 2](https://www.deep-ml.com/problems/2).

## 1. Problem Description

Given a two-dimensional matrix $A$ with shape $(m, n)$, return its transpose with shape $(n, m)$. Each row of $A$ becomes a column of the result. The PyTorch variant returns a tensor.

$$
(A^\top)_{ji} = A_{ij}.
$$

## 2. Example

```python
from solution_v1 import transpose_matrix

a = [[1, 2, 3], [4, 5, 6]]
result = transpose_matrix(a)

assert result.shape == (3, 2)
assert result.tolist() == [[1, 4], [2, 5], [3, 6]]
```

The row `[1, 2, 3]` becomes the first column, and `[4, 5, 6]` becomes the second column.

## 3. Solution

[PyTorch implementation](solution_v1.py)

Use `torch.as_tensor(a)` to get a tensor, then return `a_t.T`. For a dense two-dimensional tensor, `.T` swaps the axes and returns a view that shares the underlying data.

- **Dense tensor input** — `O(1)` time and `O(1)` extra space.
- **Nested list input** — converting the list to a tensor takes `O(mn)` time and space.

## 4. Reflection

To transpose a two-dimensional PyTorch tensor, I can access its `.T` attribute directly, as in `a_t.T`.

### Reference

- [PyTorch `Tensor.T` documentation](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor.T)
