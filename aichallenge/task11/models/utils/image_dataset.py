import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensor


class ImageDataset(Dataset):
    def __init__(self, df, train=True, augmentation=None):
        self.train = train
        self.labels = {}
        self.transforms = augmentation
   
        self.labels['file'] = df['file'].values.tolist()
        self.labels['plant'] = df['plant'].values.tolist()
        self.labels['disease'] = df['disease'].values.tolist()

    def __getitem__(self, index: int):
        image_path = os.path.join('./data/train/', self.labels['file'][index])
             
        plant = self.labels['plant'][index]
        plant = torch.tensor(int(plant))
            
        disease = self.labels['disease'][index]
        disease = torch.tensor(int(disease))

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if self.train:
            image = self.transforms['train'](image=image)
            image = image['image']
            return (self.labels['file'][index], image, plant, disease)
        else:
            image = self.transforms['test'](image=image)
            image = image['image']
            return (self.labels['file'][index], image, plant, disease)

    def __len__(self):
        return len(self.labels['file'])


class TestDataset(Dataset):
    def __init__(self, root: str = './data/test/'):
        self.root = root
        self.labels = {}
        self.transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensor()])
        self.label_path = os.path.join(root + 'test_labels.txt')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            file_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split()
                file_list.append(v[0])

        self.labels['file'] = list(file_list)

    def __getitem__(self, index: int):
        image_path = os.path.join(self.root, self.labels['file'][index])
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.transforms(image=image)
        image = image['image']
        return self.labels['file'][index], image

    def __len__(self):
        return len(self.labels['file'])
