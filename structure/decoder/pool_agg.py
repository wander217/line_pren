import torch
from torch import Tensor
import torch.nn as nn
from structure.decoder.weight_init import weight_init
from typing import List


class PoolAgg(nn.Module):
    def __init__(self,
                 out_channel: int,
                 in_size: int,
                 mid_size: int,
                 out_size: int):
        super().__init__()
        self.layer_list: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_size, mid_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(mid_size, momentum=0.01, eps=0.001),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_size, out_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_size, momentum=0.01, eps=0.001)
            ) for _ in range(out_channel)
        ])
        self.pool: nn.Module = nn.AdaptiveAvgPool2d(1)
        self.apply(weight_init)

    def forward(self, x: Tensor) -> Tensor:
        """
            :param x: (B, in_chanel, H, W)
            :return: (B, out_chanel , out_size)
        """
        feature: List = []
        bs = x.size(0)
        for layer in self.layer_list:
            f: Tensor = self.pool(layer(x))
            feature.append(f.view(bs, 1, -1))
        output: Tensor = torch.cat(feature, dim=1)
        return output
