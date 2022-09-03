import torch
from torch import Tensor
import torch.nn as nn
from structure.decoder.weight_init import weight_init


class WeightAgg(nn.Module):
    def __init__(self,
                 out_channel: int,
                 in_size: int,
                 mid_size: int,
                 out_size: int):
        super().__init__()
        self.conv_n: nn.Module = nn.Sequential(
            nn.Conv2d(in_size, in_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_size, momentum=0.01, eps=0.001),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_size, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel, momentum=0.01, eps=0.001),
            nn.Sigmoid()
        )
        self.conv_d: nn.Module = nn.Sequential(
            nn.Conv2d(in_size, mid_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_size, momentum=0.01, eps=0.001),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_size, out_size, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.01, eps=0.001))

        self._out_channel: int = out_channel
        self._out_size: int = out_size
        self.apply(weight_init)

    def forward(self, x: Tensor) -> Tensor:
        """
            :param x: (B, in_channel, H, W)
            :return: (B, out_channel , out_size)
        """
        B = x.size(0)
        n_map: Tensor = self.conv_n(x).view(B, self._out_channel, -1)  # (B, out_channel, h * w)
        d_map: Tensor = self.conv_d(x).view(B, self._out_size, -1)  # (B,  h * w, out_size)
        output: Tensor = torch.bmm(n_map, d_map.permute(0, 2, 1))
        return output
