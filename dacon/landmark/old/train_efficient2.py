import os
import sys
import csv
import glob
import math
import argparse

import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models 
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import efficientnet_pytorch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--save_interval', default=10000, type=int)
parser.add_argument('--max_iter', default=300000, type=int)
parser.add_argument('--verbose_eval', default=10, type=int)
args = parser.parse_args()


writer = SummaryWriter()


class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, num_classes):
        super().__init__()
        self.depth = depth
        self.base = efficientnet_pytorch.EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_filter = self.base._fc.in_features
        self.classifier = nn.Linear(self.output_filter, num_classes)

    def forward(self, x):
        x = self.base.extract_features(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x

model = EfficientNetEncoderHead(3, 1049)
model.load_state_dict(torch.load('effnet_448_512_1_80000.pth'))
# sys.exit()
# 256(32, 200,000, ) => 448(50000, 20) 0.987-> 720 (20,000)
#
size = 512
train_transforms = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomResizedCrop(size),
        transforms.RandomAffine(
            degrees=15, translate=(0.2, 0.2),
            scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
    ]),
    transforms.ToTensor(),
    transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
])


class LandmarkDataset(Dataset):
    def __init__(self, transforms=None):
        self.image_ids = glob.glob('./data/train/**/**/*')
        with open('./data/train.csv') as f:
            labels = list(csv.reader(f))[1:]
            self.labels = {label[0]: int(label[1]) for label in labels}

        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_ids[idx]).convert('RGB')
        label = os.path.splitext(os.path.basename(self.image_ids[idx]))[0]
        label = self.labels[label]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

train_dataset = LandmarkDataset(train_transforms)
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = resnet101(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 1049)
model.to(device)

net = nn.DataParallel(model, device_ids=[0, 1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

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

def gap(preds: Tensor, confs: Tensor, targets: Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(preds.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert preds.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    preds = preds[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, preds, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res

import time

def train():
    epoch_size = len(train_dataset) // args.batch_size
    num_epochs = math.ceil(args.max_iter / epoch_size)
    print(num_epochs)
    logger = open('log.txt', 'w')
    iteration = 0
    losses = AverageMeter()
    scores = AverageMeter()
    start_epoch = time.time()
    model.train()
    for epoch in range(num_epochs):
        if (epoch+1)*epoch_size < iteration:
            continue
        if iteration == args.max_iter:
            break
        correct = 0
        start_time = time.time()
        input_size = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            confs, preds = torch.max(outputs.detach(), dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            input_size += inputs.size(0)
            losses.update(loss.item(), inputs.size(0))
            scores.update(gap(preds, confs, targets))
            correct += (preds == targets).float().sum()

            iteration += 1

            writer.add_scalar('train_likelihood', losses.val, iteration)
            writer.add_scalar('validation_mse', scores.val, iteration)

            log = {'epoch': epoch+1, 'iteration': iteration, 'loss': losses.val, 'acc': correct.item()/len(train_dataset), 'gap': scores.val}
            logger.write(str(log) + '\n')
            if iteration % args.verbose_eval == 0:
                print(f'[{epoch+1}/{iteration}] Loss: {losses.val:.4f} Acc: {correct/input_size:.4f} GAP: {scores.val:.4f} LR: {scheduler.get_last_lr()} Time: {time.time() - start_time}')
            
            if iteration % args.save_interval == 0:
                torch.save(model.state_dict(), f'effnet_448_512_{epoch}_{iteration}.pth')
        
            if iteration in [20000, 70000, 140000]:
                scheduler.step()
        print()
    logger.close()
    writer.close()
    print(time.time() - start_epoch)
if __name__ == '__main__':
    train()