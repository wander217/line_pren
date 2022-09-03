from torch.utils.data import Dataset
from typing import Tuple, List
import cv2 as cv
import lmdb
import numpy as np
from dataset.alphabet import Alphabet


def normalize(image: np.ndarray):
    image = image.astype(np.float32) / 255.
    image = (image - 0.5) / 0.5
    return image


def resize(image: np.ndarray, max_size: List, value: int):
    h, w, _ = image.shape
    nh = max_size[0]
    nw = int((w / h) * nh)
    if nw < max_size[1]:
        image = cv.resize(image, (nw, nh))
        new_image = np.full((*max_size, 3), value, dtype=np.uint8)
        new_image[:nh, :nw, :] = image
    else:
        new_image = cv.resize(image, (max_size[1], max_size[0]))
    return new_image


class PRENDataset(Dataset):
    def __init__(self, path: str, alphabet: Alphabet) -> None:
        super().__init__()
        self.txn: List = []
        self.nSample: int = 0
        self.alphabet = alphabet
        # for file in listdir(path):
        env = lmdb.open(path,
                        max_readers=8,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False)
        self._txn = env.begin(write=False)
        self.nSample = int(self._txn.get('num-samples'.encode()))
        # nSample: int = int(txn.get('num-samples'.encode()))
        # self.txn.append({
        #     "txn": txn,
        #     "nSample": nSample
        # })
        # self.nSample += nSample
        # self.index: np.ndarray = np.zeros((self.nSample, 2), dtype=np.int32)
        # start: int = 0
        # for i, item in enumerate(self.txn):
        #     nSample = item['nSample']
        #     self.index[start:start + nSample, 0] = i
        #     self.index[start:start + nSample, 1] = np.arange(nSample, dtype=np.int32) + 1
        #     start += nSample

    def __len__(self):
        return self.nSample

    def __getitem__(self, index: int) -> Tuple:
        # txn_id, rid = self.index[index]
        # txn = self.txn[txn_id]['txn']
        index = index + 1
        img_code: str = 'img{}'.format(index)
        img_buf = self._txn.get(img_code.encode())
        img = np.frombuffer(img_buf, dtype=np.uint8)
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        # img = cv.resize(img, (900, 32), interpolation=cv.INTER_CUBIC)
        img = normalize(img)

        label_code: str = 'label{}'.format(index)
        byte_label: bytes = self._txn.get(label_code.encode())
        label = byte_label.decode("utf-8")
        label = label.strip("\n").strip("\r\t")
        label = self.alphabet.encode(label)
        return img, label
