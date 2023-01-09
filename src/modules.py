import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

try:
    from utils import calc_coeff, grl_hook, init_weights
except ImportError:
    from .utils import calc_coeff, grl_hook, init_weights


class AdversarialNetworkCDAN(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int):
        super(AdversarialNetworkCDAN, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 1000.0

    def forward(self, x: torch.Tensor):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, "decay_mult": 2}]


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim: int = 1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = nn.ParameterList(
            [
                nn.Parameter(torch.randn(input_dim_list[i], output_dim))
                for i in range(self.input_num)
            ]
        )

    def forward(self, input_list: List[torch.Tensor]):
        return_list = [
            torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)
        ]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


def CDAN(input_list, ad_net, entropy: float = None, coeff: float = None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        op_out = op_out.view(-1, softmax_output.size(1) * feature.size(1))
        ad_out = ad_net(op_out)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        random_out = random_out.view(-1, random_out.size(1))
        ad_out = ad_net(random_out)

    b = softmax_output.size(0) // 2
    dc_target = torch.zeros(b * 2, device=softmax_output.device).reshape(-1, 1)
    dc_target[:b] = 1

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2 :] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0 : feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = (
            source_weight / torch.sum(source_weight).detach().item()
            + target_weight / torch.sum(target_weight).detach().item()
        )
        return (
            torch.sum(weight.view(-1, 1) * F.binary_cross_entropy(ad_out, dc_target, reduce="none"))
            / torch.sum(weight).detach().item()
        )
    else:
        return F.binary_cross_entropy_with_logits(ad_out, dc_target)
