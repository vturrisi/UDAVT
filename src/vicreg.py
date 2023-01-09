from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
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


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor):
    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor):
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor):
    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss


def vicreg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
):
    sim_loss = invariance_loss(z1, z2)
    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)

    loss = sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
    return loss


def compute_vicreg_loss(
    self,
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    y_source: torch.Tensor,
    y_target: torch.Tensor,
    source_source: bool = False,
    source_target: bool = True,
    target_target: bool = False,
):
    args = self.args

    z1 = []
    z2 = []
    if source_source:
        for c in range(self.num_classes):
            source_indexes = (y_source == c).view(-1).nonzero().tolist()
            for i, j in product(source_indexes, source_indexes):
                z1.append(z_s[i])
                z2.append(z_s[j])

    if target_target:
        for c in range(self.num_classes):
            target_indexes = (y_target == c).view(-1).nonzero().tolist()
            for i, j in product(target_indexes, target_indexes):
                z1.append(z_t[i])
                z2.append(z_t[j])

    if source_target:
        for c in range(self.num_classes):
            source_indexes = (y_source == c).view(-1).nonzero().tolist()
            target_indexes = (y_target == c).view(-1).nonzero().tolist()
            for i, j in product(source_indexes, target_indexes):
                z1.append(z_s[i])
                z2.append(z_t[j])

    if len(z1) > 2:
        z1 = torch.cat(z1)
        z2 = torch.cat(z2)
        loss = args.vicreg_loss_weight * vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=args.sim_loss_weight,
            var_loss_weight=args.var_loss_weight,
            cov_loss_weight=args.cov_loss_weight,
        )
    else:
        loss = torch.tensor(0.0, device=self.device)

    return loss
