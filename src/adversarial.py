import torch
from torch import nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)

        std = 0.001
        nn.init.normal_(self.fc1.weight, 0, std)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, std)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor, coeff: float = 1.0):
        x = GradReverse.apply(x, coeff)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
