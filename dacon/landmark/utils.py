import os
import random
import glob
import csv
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def fixed_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state_dict: Dict, path: str):
    torch.save(state_dict, path)


class LandmarkDataset(Dataset):
    def __init__(self, mode='train', transforms=None):
        if mode == 'train':
            self.image_ids = glob.glob(f'./data/{mode}/**/**/*')
        else:
            self.image_ids = glob.glob(f'./data/{mode}/**/*')
            
        self.mode = mode

        if mode == 'train':
            with open('./data/train.csv') as f:
                labels = list(csv.reader(f))[1:]
                self.labels = {label[0]: int(label[1]) for label in labels}

        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_ids[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = os.path.splitext(os.path.basename(self.image_ids[idx]))[0]
    
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.mode == 'train':
            label = self.labels[image_id]
            return image, label
        else:
            return image_id, image


class LandmarkDatasetResize(Dataset):
    def __init__(self, transforms1=None, transforms2=None):
        self.image_ids = glob.glob('./data/test/**/*')
            
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_ids[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = os.path.splitext(os.path.basename(self.image_ids[idx]))[0]
    
        if self.transforms1 is not None:
            image1 = self.transforms1(image=image)['image']

        if self.transforms2 is not None:
            image2 = self.transforms2(image=image)['image']

        return image_id, image1, image2


class LandmarkDatasetOOF(Dataset):
    def __init__(self, transforms=None):
        # self.image_ids = glob.glob('./data/train/**/**/*')
        with open('./train.csv') as f:
            image_ids = list(csv.reader(f))[0]
            labels = list(csv.reader(f))[1]
            # self.labels = {label[0]: int(label[1]) for label in labels}

        # print(labels)

        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_ids[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = os.path.splitext(os.path.basename(self.image_ids[idx]))[0]
        label = self.labels[label]

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        return image, label


class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_csv():
    image_paths = glob.glob('./data/train/**/**/*')
    with open('./data/train.csv') as f:
        labels = list(csv.reader(f))[1:]
        labels = {label[0]: int(label[1]) for label in labels}

    p = []
    l = []

    for path in image_paths:
    
        label = os.path.splitext(os.path.basename(path))[0]
        label = labels[label]
        p.append(path)
        l.append(label)

    with open('./train.csv', 'w') as csvfile:
        headers = ['id', 'landmark_id']
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow({'id': p, 'landmark_id': l})
        csv_writer.writerows(zip(p, l))
    




