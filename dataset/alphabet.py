from typing import Dict
import numpy as np
import unicodedata


class Alphabet:
    def __init__(self, path: str, max_len: int) -> None:
        self.pad: int = 0
        self.end: int = 1
        self.unk: int = 2
        self.max_len: int = max_len
        with open(path, 'r', encoding='utf-8') as f:
            alphabet = f.readline().strip("\n").strip("\r\t").strip()
            alphabet = ' ' + alphabet
            alphabet = unicodedata.normalize('NFC', alphabet)
        self.char_dict: Dict = {c: i + 3 for i, c in enumerate(alphabet)}
        self.int_dict: Dict = {i + 3: c for i, c in enumerate(alphabet)}
        self.int_dict[self.pad] = '<pad>'
        self.int_dict[self.end] = '<end>'
        self.int_dict[self.unk] = '<unk>'

    def encode(self, s: str) -> np.ndarray:
        s = unicodedata.normalize('NFC', s)
        es = [self.char_dict.get(ch, self.unk) for ch in s] + [self.end]
        es = np.pad(es, (0, self.max_len - len(es)), constant_values=self.pad)
        return es.astype(np.int64)

    def decode(self, es: np.ndarray) -> str:
        s = ''
        for item in es:
            if item == self.pad or item == self.unk:
                continue
            if item == self.end:
                break
            s += self.int_dict[item]
        return s

    def size(self):
        return len(self.int_dict)