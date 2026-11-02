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
