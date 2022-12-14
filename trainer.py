import argparse
import os
import random
import warnings
import numpy as np
import torch
import Levenshtein
import yaml
from torch import nn, optim, Tensor
from structure import PREN
from dataset import PRENLoader, Alphabet
from typing import Dict, Tuple
from utilities.averager import Averager
from utilities import PRENCheckpoint, PRENLogger
from criterion import FocalLoss


class PRENTrainer:
    def __init__(self,
                 model: Dict,
                 alphabet: Dict,
                 optimizer: Dict,
                 criterion: Dict,
                 total_epoch: int,
                 save_interval: int,
                 start_epoch: int,
                 train: Dict,
                 valid: Dict,
                 checkpoint: Dict,
                 logger: Dict):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.total_epoch: int = total_epoch
        self.save_interval: int = save_interval
        self.start_epoch: int = start_epoch

        self.alphabet: alphabet = Alphabet(**alphabet)
        self.model: nn.Module = PREN(**model, alphabet=self.alphabet)
        self.model = self.model.to(self.device)
        self.criterion: nn.Module = FocalLoss(self.alphabet.pad, **criterion)
        self.criterion = self.criterion.to(self.device)
        cls = getattr(optim, optimizer['name'])
        self.optimizer: optim.Optimizer = cls(self.model.parameters(), **optimizer['params'])
        self.train_loader = PRENLoader(**train, alphabet=self.alphabet).build()
        self.valid_loader = PRENLoader(**valid, alphabet=self.alphabet).build()

        self.logger: PRENLogger = PRENLogger(**logger)
        self.checkpoint: PRENCheckpoint = PRENCheckpoint(**checkpoint)
        self.step: int = 0
        self.best: float = 0

    def train(self):
        self.load()
        self.logger.report_delimiter()
        self.logger.report_time("Starting:")
        self.logger.report_delimiter()
        for epoch in range(self.start_epoch, self.total_epoch + 1):
            self.logger.report_delimiter()
            self.logger.report_time("Epoch {}".format(epoch))
            train_rs = self.train_step()
            valid_rs = self.valid_step()
            self.logger.report_delimiter()
            self.logger.report_metric({
                "train_loss": train_rs,
                **valid_rs
            })
            self.save(epoch)
        self.logger.report_delimiter()
        self.logger.report_time("Finish:")
        self.logger.report_delimiter()

    def save(self, epoch:int):
        self.logger.report_delimiter()
        self.logger.report_time("Saving:")
        self.checkpoint.save(self.model, self.optimizer, epoch)
        self.logger.report_time("Saving complete!")
        self.logger.report_delimiter()

    def train_step(self):
        train_loss: Averager = Averager()
        for batch, (image, target) in enumerate(self.train_loader):
            self.model.train()
            bs = image.size(0)
            image = image.to(self.device)
            target = target.to(self.device)
            pred: Tensor = self.model(image)
            loss: Tensor = self.criterion(pred, target)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item() * bs, bs)
        return train_loss.calc()

    def valid_step(self):
        self.model.eval()
        valid_loss: Averager = Averager()
        valid_acc: Averager = Averager()
        valid_norm: Averager = Averager()
        selected_index = random.randint(0, self.valid_loader.__len__() - 1)
        pred_text = ''
        target_text = ''
        with torch.no_grad():
            for idx, (image, target) in enumerate(self.valid_loader):
                bs = image.size(0)
                image = image.to(self.device)
                target = target.to(self.device)
                pred: Tensor = self.model(image)
                loss: Tensor = self.criterion(pred, target)
                valid_loss.update(loss.item() * bs, bs)
                acc, norm = self._acc(pred, target)
                if idx == selected_index:
                    pred_text = torch.log_softmax(pred[0], dim=-1).detach().argmax(1).cpu().numpy()
                    pred_text = self.alphabet.decode(pred_text)
                    target_text = target[0].detach().cpu().numpy()
                    target_text = self.alphabet.decode(target_text)
                valid_acc.update(acc, bs)
                valid_norm.update(norm, bs)
        return {
            "valid_loss": valid_loss.calc(),
            "valid_acc": valid_acc.calc(),
            "valid_norm": 1 - valid_norm.calc(),
            "pred_text": pred_text,
            "target_text": target_text
        }

    def _acc(self, pred: Tensor, target: Tensor) -> Tuple:
        n_correct: int = 0
        norm_edit: int = 0
        bs = pred.size(0)
        rp: np.ndarray = torch.log_softmax(pred, dim=-1).cpu().detach().numpy().argmax(axis=2)
        rt: np.ndarray = target.cpu().detach().numpy()
        for i in range(bs):
            p_str: str = self.alphabet.decode(rp[i])
            t_str: str = self.alphabet.decode(rt[i])
            max_len = max(len(p_str), len(t_str))
            if max_len == 0:
                continue
            norm_edit += Levenshtein.distance(p_str, t_str) / max_len
            if p_str == t_str:
                n_correct += 1
        return n_correct, norm_edit

    def load(self):
        state_dict: Dict = self.checkpoint.load()
        if state_dict is not None:
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.step = state_dict['step'] + 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Training config")
    parser.add_argument("-c", '--config', default='', type=str, help="path of config")
    parser.add_argument("-a", '--alphabet', default='', type=str, help="path of alphabet")
    parser.add_argument("-d", '--data', default='', type=str, help="path of data")
    parser.add_argument("-s", '--save_interval', default=1000, type=int, help="number of step to save")
    parser.add_argument("-r", '--resume', default='', type=str, help="resume path")
    parser.add_argument("-p", '--primitive_num', default=3, type=int, help="number of primitive")
    parser.add_argument("-m", '--max_len', default=6, type=int, help="max length of sequence")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.alphabet.strip():
        config['alphabet']['path'] = args.alphabet.strip()
    if args.data.strip():
        for item in ["train", "valid"]:
            config[item]['dataset']['path'] = os.path.join(args.data.strip(), item)
    if args.resume.strip():
        config['checkpoint']['resume'] = args.resume.strip()
    config['model']['decoder']['out_channel'] = args.primitive_num
    config['alphabet']['max_len'] = args.max_len
    trainer = PRENTrainer(**config, save_interval=args.save_interval)
    trainer.train()
