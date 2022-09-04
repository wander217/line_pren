import time
import torch
import yaml
from torch import nn, Tensor
from dataset import Alphabet
from structure.encoder.eft_net import eft_builder
from structure.decoder import PRENDecoder
from typing import Dict


class PREN(nn.Module):
    def __init__(self,
                 alphabet: Alphabet,
                 encoder: Dict,
                 decoder: Dict):
        super().__init__()
        self._encoder: nn.Module = eft_builder(**encoder)
        self._decoder: nn.Module = PRENDecoder(**decoder,
                                               alphabet=alphabet,
                                               in_sizes=self._encoder.out_channel)

    def forward(self, image: Tensor) -> Tensor:
        """
            :param image: (B, 3, 32, 960)
            :return: pred :  (B, max_len, d_output)
        """
        pred: Tensor = self._decoder(self._encoder(image))
        return pred


if __name__ == "__main__":
    config_path = r'F:\project\python\pren\asset\pc_eb0.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    alphabet = Alphabet(r'F:\project\python\pren\asset\alphabet.txt', 105)
    model = PREN(**config['model'], alphabet=alphabet)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params, train_params)
    x = torch.randn((1, 3, 32, 960))
    start = time.time()
    y = model(x)
    print(time.time() - start)
    print(y.size())
