import os
import glob
import json
import random
import argparse
from albumentations.augmentations.transforms import ColorJitter, Cutout, RandomBrightnessContrast, RandomResizedCrop, ShiftScaleRotate
from albumentations.core.composition import OneOf

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.optim as optim
from torch import nn, Tensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import albumentations as A 
from albumentations.pytorch import ToTensorV2

# from torch.cuda.amp import GradScaler, autocast

from mask_rcnn import FashionDataset, FashionModel



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--max_size', default=550, type=int)

args = parser.parse_args()

CLASSES = (
    'top', 'blouse', 't-shirt', 'Knitted fabri', 'shirt', 'bra top', 
    'hood', 'blue jeans', 'pants', 'skirt', 'leggings', 'jogger pants', 
    'coat', 'jacket', 'jumper', 'padding jacket', 'best', 'kadigan', 
    'zip up', 'dress', 'jumpsuit')

# transforms_train = A.Compose([
    # A.Resize(550, 550),
    # A.HorizontalFlip(),
    # A.ShiftScaleRotate(),
    # A.ColorJitter(0.2, 0.2, 0.2, 0.2),
    #A.OneOf([
    #    A.RandomResizedCrop(args.max_size, args.max_size),
    #    A.Cutout(
    #        max_h_size=int(args.max_size*0.4),
    #        max_w_size=int(args.max_size*0.4),
    #        num_holes=1,
    #        p=0.3),
    #     A.RandomBrightnessContrast(),
    # ]),
    # A.Normalize(),
    # ToTensorV2(),
# ], bbox_params={'format': 'pascal_voc', 'label_fields': ['category_ids']})


device = 'cuda'
# scaler = torch.cuda.amp.GradScaler()


def fixed_seeds(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

fixed_seeds(42)



 
dataset = FashionDataset('data/fashion/train.json', mode='train', transforms=transforms_train)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=12, collate_fn=collate_fn)


model = FashionModel(len(CLASSES)+1)
model.to(device)
model = nn.DataParallel(model, device_ids=[0,1])

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(
    params, lr=args.lr, weight_decay=0.0005)

# let's train it for 10 epochs
num_epochs = 30
max_iter = 300000


def train_fn():
    model.train()
    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            images = list(image.to(device) for image in images)
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # with autocast():
            print(epoch, i)
            losses = model(images, targets)

            print(losses)
            loss = sum(loss for loss in losses.values())
            loss.backward()
            optimizer.step()
            # scaler(loss).backward()
            # scaler.step(optimizer)


def valid_fn():
    raise NotImplementedError


if __name__ == '__main__':
    train_fn()

# for epoch in range(num_epochs):
#     pass
    # train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    
    # torch.save(model.state_dict(), f'./weight/Mask_RCNN_base_{epoch}.pth')



# print_freq=10
# for epoch in range(num_epochs):
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1. / 1000
#         warmup_iters = min(1000, len(train_loader) - 1)

#         lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

#     for images, targets in metric_logger.log_every(train_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         print(images)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #     output = model(images, targets)

    #     losses = sum(loss for loss in output.values())

    #     # reduce losses over all GPUs for logging purposes
    #     output_reduced = utils.reduce_dict(output)
    #     losses_reduced = sum(loss for loss in output_reduced.values())

    #     loss_value = losses_reduced.item()

    #     if not math.isfinite(loss_value):
    #         print("Loss is {}, stopping training".format(loss_value))
    #         print(output_reduced)
    #         sys.exit(1)

    #     optimizer.zero_grad()
    #     losses.backward()
    #     optimizer.step()

    #     if lr_scheduler is not None:
    #         lr_scheduler.step()

    #     metric_logger.update(loss=losses_reduced, **output_reduced)
    #     metric_logger.update(lr=optimizer.param_groups[0]["lr"])

