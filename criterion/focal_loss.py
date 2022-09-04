from torch import nn, Tensor
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, pad: int, gamma: int):
        super().__init__()
        self._criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=pad, reduction='none')
        self._gamma: int = gamma
        self._pad: int = pad

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        target_ = target.contiguous().view(-1)
        pred_ = pred.view(-1, pred.size(-1))
        cross_entropy: Tensor = self._criterion(pred_, target_)
        target_ = target_ * (target_ != self._pad).long()
        input_prob: Tensor = torch.gather(F.softmax(pred_, 1), 1, target_.unsqueeze(1))
        loss = torch.pow(1. - input_prob, self._gamma) * cross_entropy
        return torch.mean(loss)
