from torch import Tensor
import torch.nn as nn
from structure.decoder.weight_init import weight_init


class GateConv(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 in_size: int,
                 out_size: int,
                 dropout: float = 0.1):
        """
            :param in_channel: input channel
            :param out_channel: output channel
            :param in_size: input size
            :param out_size: output size
        """
        super().__init__()
        self.conv: nn.Module = nn.Conv1d(in_channel, out_channel, 1)
        self.fc: nn.Module = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.Dropout(p=dropout),
            nn.SiLU(inplace=True))
        self.apply(weight_init)

    def forward(self, x: Tensor) -> Tensor:
        """
            :param x: (B, in_chanel, in_size)
            :return: (B, out_chanel, out_size)
        """
        output: Tensor = self.conv(x)
        output = self.fc(output)
        return output
