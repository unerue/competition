import os
import random
from typing import List, Tuple, Dict, TypeVar, Optional

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, image_dir: str, image_ids: np.ndarray, transforms=None):
        self.df = df
        self.image_dir = image_dir
        self.image_ids = image_ids 
        self.transforms = transforms

    def __getitem__(self, index: int):
        image, boxes = self._load_image_and_boxes(index)
        
        target = {}
        target['boxes'] = boxes # int, LongTensor
        target['labels'] = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            sample = {
                'image': image,
                'boxes': target['boxes'],
            }
            sample = self.transforms(**sample)
            image = sample['image']
                    
        # target['boxes'] = torch.DoubleTensor(sample['boxes'])
        # target['boxes'] = torch.stack(tuple(map(torch.LongTensor, 
        #                                         zip(*sample['boxes'])))).permute(1, 0)
        target['boxes'] = torch.stack(
            tuple(map(torch.LongTensor, zip(*sample['boxes'])))).T

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image_and_boxes(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_id = self.image_ids[index]
        image = cv2.imread(
            os.path.join(self.image_dir, f'{image_id}.jpg'), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        boxes = self.df.loc[self.df['image_id'] == image_id]
        boxes = boxes[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return image, boxes



class TestDataset(Dataset):
    def __init__(
        self, image_dir: str, image_ids: np.ndarray, transforms=None):
        self.image_dir = image_dir
        self.image_ids = image_ids 
        self.transforms = transforms

    def __getitem__(self, index: int):
        image = self._load_image(index)

        target = {}
        target['image_id'] = torch.tensor([index])
        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']
                    
        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, index: int) -> np.ndarray:
        image_id = self.image_ids[index]
        image = cv2.imread(
            os.path.join(self.image_dir, f'{image_id}.jpg'), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        return image


class CutMixDataset(Dataset):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def _load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        pass
        