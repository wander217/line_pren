import torch.nn as nn
from structure.decoder.pool_agg import PoolAgg
from structure.decoder.weight_agg import WeightAgg
from structure.decoder.gate_conv import GateConv
from structure.decoder.weight_init import weight_init
from dataset import Alphabet
from typing import List
from torch import Tensor
import torch


class PRENDecoder(nn.Module):
    def __init__(self,
                 in_sizes: List,
                 out_channel: int,
                 out_size: int,
                 alphabet: Alphabet,
                 drop_out: float):
        super().__init__()
        self._pgg: nn.ModuleList = nn.ModuleList([
            PoolAgg(out_channel, in_sizes[i], in_sizes[i], out_size // 3)
            for i in range(len(in_sizes))
        ])
        self._p_gate: nn.Module = GateConv(out_channel, alphabet.max_len, out_size, alphabet.size(), drop_out)

        self._wgg: nn.ModuleList = nn.ModuleList([
            WeightAgg(out_channel, in_sizes[i], in_sizes[i], out_size // 3)
            for i in range(len(in_sizes))
        ])
        self._w_gate: nn.Module = GateConv(out_channel, alphabet.max_len, out_size, out_size, drop_out)
        self._fc: nn.Module = nn.Linear(out_size, alphabet.size())
        self._fc.apply(weight_init)

    def forward(self, features: List):
        pgg: List = [self._pgg[i](features[i]) for i in range(len(features))]
        p_gate: Tensor = self._p_gate(torch.cat(pgg, dim=2))
        wgg: List = [self._wgg[i](features[i]) for i in range(len(features))]
        w_gate: Tensor = self._w_gate(torch.cat(wgg, dim=2))
        score: Tensor = torch.cat([p_gate, w_gate], dim=-1)
        pred: Tensor = self._fc(score)
        return pred
