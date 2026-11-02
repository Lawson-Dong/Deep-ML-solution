# 1223. Gradient of a Weighted Sum of Squares

- **Difficulty:** Medium.
- **Category:** PyTorch.
- **Topics:** Autograd, Element-wise Operations, Gradients.
- **Problem:** [Deep-ML 1223](https://www.deep-ml.com/problems/1223).
- **Platform result:** 5/5 tests passed, as shown in the author's supplied screenshot.

## 1. Problem Description

Implement `grad_wss(w_list, x_list)`:

1. Construct a floating-point tensor `w` with `requires_grad=True` and a tensor `x` without gradient tracking from the input lists.
2. Compute the scalar loss:

$$
L = \frac{1}{2}\sum_i (w_i x_i)^2.
$$

3. Call `backward()` on the loss.
4. Return the gradient with respect to `w` as a Python list of floats, rounding each value to four decimal places.

Use CPU tensors and deterministic operations. Do not modify the input lists. The inputs represent same-length 1D vectors.

## 2. Example

```python
grad_wss([1.0, 2.0], [3.0, 4.0])
# [9.0, 32.0]
```

The intermediate values make the order of operations explicit:

| Expression | Value |
|---|---|
| `w * x` | `[3.0, 8.0]` |
| `(w * x) ** 2` | `[9.0, 64.0]` |
| `0.5 * ((w * x) ** 2).sum()` | `36.5` |

For each component,

$$
\frac{\partial L}{\partial w_i}
= \frac{1}{2}\cdot 2(w_i x_i)x_i
= w_i x_i^2.
$$

Therefore, the gradient is `[1 * 3**2, 2 * 4**2] = [9.0, 32.0]`. This formula explains the result; the implementation uses autograd as required.

## 3. Solution

**Implementation:** [solution_v1.py](solution_v1.py)

```python
import torch


def grad_wss(w_list, x_list):
    """Return gradients of 0.5 * sum((w * x)**2), rounded to 4 decimals."""
    w_tensor = torch.tensor(
        w_list,
        dtype=torch.float32,
        requires_grad=True,
    )
    x_tensor = torch.tensor(
        x_list,
        dtype=torch.float32,
    )

    L = 0.5 * ((w_tensor * x_tensor) ** 2).sum()
    L.backward()

    result = [round(v, 4) for v in w_tensor.grad.tolist()]
    return result
```

This preserves the author's approach and removes the placeholder comment and unreachable `pass`. Both tensors are constructed from lists, so the original lists are unchanged. Under the standard default CPU device, the constructors create CPU tensors. No random operations are involved.

- **Time:** `O(n)`, where `n` is the vector length.
- **Extra space:** `O(n)` for tensors, intermediate values, gradients, and the returned list.
- Rounding changes the numerical values; it does not force a display format with four trailing decimal digits.

## 4. Reflection

### 1. Distinguish element-wise multiplication from matrix multiplication

I should translate the mathematical expression into tensor operations before choosing an operator.

In PyTorch, `*` performs element-wise multiplication, with broadcasting when shapes differ. For the same-length vectors in this problem, it preserves the vector shape. `** 2` then squares each product, and `.sum()` reduces those squared products to a scalar. These are three separate operations.

By contrast, `@` follows matrix-multiplication semantics. For two 1D tensors, it computes their dot product, reducing the pairwise products to a scalar. For matrices or batched tensors, it performs matrix or batched matrix multiplication; it does **not** always produce a scalar.

For this example, replacing the element-wise expression with a dot product before squaring would change the objective:

```python
# Required: square each product, then sum.
0.5 * ((w * x) ** 2).sum()  # 36.5

# Different objective: sum the products, then square.
0.5 * (w @ x) ** 2         # 60.5
```

The distinction is between a **sum of squares** and the **square of a sum**. The latter introduces cross terms and generally produces different gradients.

### 2. Separate the backward operation from the stored gradient

`L.backward()` performs reverse-mode automatic differentiation and returns `None`; it does not return the gradient tensor. In this function, it accumulates the computed gradient into `w_tensor.grad`, because `w_tensor` is a leaf tensor requiring gradients and contributes to `L`.

The `.grad` attribute holds the result. Reading it does not itself print anything; displaying it requires `print(...)` or an interactive environment. Before backpropagation, it is normally `None`.

```python
print(w_tensor.grad)  # None, before backward
backward_result = L.backward()
print(backward_result)  # None
print(w_tensor.grad)  # tensor([9., 32.]), for the example inputs
```

This snippet illustrates a single backward call on a freshly computed loss; it should not be appended after the implementation's existing call.

Gradients accumulate rather than automatically replacing previous values. This function creates fresh tensors and calls backward once, so no reset is needed. In a repeated training loop, I must clear gradients between updates. Arbitrary non-leaf tensors do not retain their `.grad` by default.

### Official References

- [PyTorch tensor tutorial: matrix and element-wise multiplication](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [torch.mul](https://docs.pytorch.org/docs/stable/generated/torch.mul.html)
- [torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [Tensor.backward](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)
- [Tensor.grad](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.grad.html)
