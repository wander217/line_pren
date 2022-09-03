from typing import List, Dict
import numpy as np
from torch.utils.data import DataLoader
from .dataset import PRENDataset
from .alphabet import Alphabet
from typing import Tuple
import torch


class PRENCollate:

    def __call__(self, batch: List) -> Tuple:
        """
            :param batch:
                image_list: (B,H,W,3)
                label_list: (B,T)
            :return: (B,3,H,W), (B,T)
        """
        img_list, label_list = zip(*batch)
        image_batch = torch.from_numpy(np.array(img_list, dtype=np.float32))
        image_batch = image_batch.permute(0, 3, 1, 2).float()
        label_batch = torch.from_numpy(np.array(label_list, dtype=np.int64))
        label_batch = label_batch.long()
        return image_batch, label_batch


class PRENLoader:
    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 pin_memory: bool,
                 alphabet: Alphabet,
                 dataset: Dict) -> None:
        self.dataset: PRENDataset = PRENDataset(alphabet=alphabet, **dataset)
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.collate_fn = PRENCollate()
        self.drop_last: bool = drop_last

    def build(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last)
