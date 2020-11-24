import os
import sys
import subprocess
import random
from typing import List, TypeVar, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


def fixed_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(
        self, patience: int = 7, mode: str = 'max', delta: float = 0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score: float, model: torch.nn, model_path: str):
        if self.mode == 'min':
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score: float, model: torch.nn, model_path: str):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(f'Validation score improved ({self.val_score:.6f} --> {epoch_score:.6f}). Saving model...\n')
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction: str = 'mean', smoothing: float = 0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(
        targets: torch.Tensor, n_classes: int, smoothing: float = 0.0) -> torch.Tensor:
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


def stratified(to_multi_dict, frac=1.0):
    to_single_dict = {value: key for key, value in to_multi_dict.items()}

    df = pd.read_csv('input/train/train_labels.txt', sep=' ', names=['file', 'plant', 'disease'], header=None)

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5)
    df['label'] = df[['plant', 'disease']].apply(lambda x: to_single_dict[(x[0], x[1])], axis=1)
    df['kfold'] = -1

    df = df.sample(frac=frac).reset_index(drop=True)
    for fold, (_, y) in enumerate(skf.split(X=df, y=df['label'])):
        df.loc[y, 'kfold'] = fold

    df.to_csv('input/train_folds.csv', index=False)
    print(f'Train dataset size: {df.shape[0]}')


def check_libraries():
    import importlib

    for library in ['padnas', 'scikit-learn', 'albumentations']:
        try:
            importlib.import_module('pandas')
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        finally:
            globals()[library] = importlib.import_module(library)


def save_model(model_name, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    path = os.path.join('./weights/' + model_name + '.pth')
    
    torch.save(state, path)
    print('model saved...\n')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join('weights/' + model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')