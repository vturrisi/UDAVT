from itertools import product

import torch
import torch.nn as nn


class IBHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        # projector
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.projector(x)


def ib_loss_func(z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3):
    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = cdif.sum()
    return loss


def compute_ib_loss(
    self,
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    y_source: torch.Tensor,
    y_target: torch.Tensor,
    source_queue: torch.Tensor = None,
    source_queue_y: torch.Tensor = None,
):
    args = self.args

    z1 = []
    z2 = []

    for c in range(self.num_classes):
        source_indexes = (y_source == c).view(-1).nonzero()
        target_indexes = (y_target == c).view(-1).nonzero()
        for i, j in product(source_indexes, target_indexes):
            z1.append(z_s[i])
            z2.append(z_t[j])

    # handle queues
    if source_queue is not None:
        for c in range(self.num_classes):
            source_indexes = (source_queue_y == c).view(-1).nonzero()
            target_indexes = (y_target == c).view(-1).nonzero()
            for i, j in product(source_indexes, target_indexes):
                z1.append(source_queue[i])
                z2.append(z_t[j])

    n_pairs = len(z1)
    if n_pairs > 2:
        z1 = torch.cat(z1)
        z2 = torch.cat(z2)
        loss = args.ib_loss_weight * ib_loss_func(z1, z2)
    else:
        loss = torch.tensor(0.0, device=self.device)

    return loss, n_pairs
