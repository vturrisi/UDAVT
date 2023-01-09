import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class SimCLRHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        # projector
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.projector(x)


def simclr_loss_func(
    z: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, temperature: float = 0.2
):
    z = F.normalize(z, dim=1)

    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    negatives = torch.sum(torch.exp(logits) * neg_mask, dim=1, keepdim=True)
    exp_logits = torch.exp(logits)
    log_prob = torch.log(exp_logits / (exp_logits + negatives))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)

    indexes = pos_mask.sum(1) > 0
    pos_mask = pos_mask[indexes]
    mean_log_prob_pos = mean_log_prob_pos[indexes] / pos_mask.sum(1)

    # mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss


@torch.no_grad()
def gen_pos_mask(y: torch.Tensor):
    labels_matrix = repeat(y, "b -> c b", c=y.size(0))
    labels_matrix = (labels_matrix == labels_matrix.T).fill_diagonal_(False)
    return labels_matrix


def compute_simclr_loss(
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
    bs = z_s.size(0)

    z = torch.cat((z_s, z_t))

    target = torch.cat((y_source, y_target))
    # target = gather(target)

    pos_mask = gen_pos_mask(target)
    neg_mask = (~pos_mask).fill_diagonal_(False)

    if not source_source:
        # make source-source neither pos nor neg
        pos_mask[:bs, :bs] = False
        neg_mask[:bs, :bs] = False
    if not target_target:
        # do the same for target-target
        pos_mask[bs:, bs:] = False
        neg_mask[bs:, bs:] = False
    if not source_target:
        # and the same for source-target
        pos_mask[:bs, bs:] = False
        neg_mask[bs:, :bs] = False

    if pos_mask.sum() > 1:
        loss = args.simclr_loss_weight * simclr_loss_func(
            z,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            temperature=args.temperature,
        )
    else:
        loss = torch.tensor(0.0, device=self.device)
    return loss
