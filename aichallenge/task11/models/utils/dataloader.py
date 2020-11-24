import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import datasets, transforms

import albumentations as A
from albumentations.pytorch import ToTensor

from .augmentation import GridMask


transform = {}
transform['train'] = A.Compose([
    A.Resize(224, 224),
    A.OneOf([
        GridMask(num_grid=3, mode=0, rotate=15),
        GridMask(num_grid=3, mode=2, rotate=15),
    ], p=0.7),
    A.Resize(224, 224),
    A.RandomBrightness(),
    A.OneOf([
        A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
        A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15),
        A.NoOp()
    ]),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4),
        A.RandomGamma(gamma_limit=(50, 150)),
        A.NoOp()
    ]),
    A.OneOf([
        A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5),
        A.NoOp()
    ]),
    A.OneOf([
        A.CLAHE(),
        A.NoOp()
    ]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.JpegCompression(80),
    A.HueSaturationValue(),
    A.Normalize(),
    ToTensor()
])

transform['test'] = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensor()
])


class CustomDataset(Dataset):
    def __init__(self, root: str, phase: str = 'train', 
        transform: torchvision.transforms = transform):
        self.root = root
        self.phase = phase
        self.labels = {}
        self.transforms = transform
        self.label_path = os.path.join(root, self.phase, self.phase + '_labels.txt')
        with open(self.label_path, 'r', encoding='utf-8-sig') as f:
            file_list = []
            plant_list = []
            disease_list = []
            for line in f.readlines()[0:2000]: # 빠른 작동여부 확인을 위해 2000개만
                v = line.strip().split()
                file_list.append(v[0])
                if self.phase != 'test':
                    plant_list.append(v[1])
                    disease_list.append(v[2])

        self.labels['file'] = list(file_list)
        self.labels['plant'] = list(plant_list)
        self.labels['disease'] = list(disease_list)

    def __getitem__(self, index: int):
        image_path = os.path.join(self.root, self.phase, self.labels['file'][index])
        
        if self.phase != 'test':
            plant = self.labels['plant'][index]
            plant = torch.tensor(int(plant))
            
            disease = self.labels['disease'][index]
            disease = torch.tensor(int(disease))

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if self.phase == 'train':
            image = self.transforms['train'](image=image)
        else:
            image = self.transforms['test'](image=image)
        image = image['image']
        if self.phase != 'test':
            return (self.labels['file'][index], image, plant, disease)
        elif self.phase == 'test' :
            dummy = ""
            return (self.labels['file'][index], image, dummy, dummy)

    def __len__(self):
        return len(self.labels['file'])

    def get_label_file(self):
        return self.label_path


def data_loader(root: str, phase: str = 'train', batch_size: int = 16):
    dataset = CustomDataset(root, phase)
    if phase == 'train':
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1)
        trainset = Subset(dataset, train_idx)
        validset = Subset(dataset, val_idx)
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    else:
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        return test_loader