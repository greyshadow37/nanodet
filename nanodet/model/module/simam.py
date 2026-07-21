import torch.nn as nn


class SimAM(nn.Module):
    """Simple, parameter-free attention from the SimAM paper.

    The implementation follows the closed-form approximation:

        E_inv = (x - mean)^2 / (4 * (var + lambda)) + 0.5
        out = x * sigmoid(E_inv)

    Args:
        e_lambda (float): Regularization coefficient from the paper.
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        if n <= 0:
            return x
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * nn.functional.sigmoid(e_inv)
