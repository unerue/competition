import math
from datetime import datetime
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import EfficientNetLandmark, ArcFaceLossAdaptiveMargin
from metrics import gap
from utils import fixed_seed, AverageMeter, save_checkpoint, LandmarkDataset


print('Start training model...')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--save_interval', default=5000, type=int)
parser.add_argument('--max_iter', default=100000, type=int)
parser.add_argument('--verbose_eval', default=50, type=int)
parser.add_argument('--max_size', default=500, type=int)
parser.add_argument('--num_classes', default=1049, type=int)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--depth', default=0, type=int)
args = parser.parse_args()

scaler = GradScaler()
writer = SummaryWriter()
fixed_seed(42)
device = 'cuda'

transforms_train = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.ImageCompression(
        quality_lower=99, 
        quality_upper=100),
    A.ShiftScaleRotate(
        shift_limit=0.2, 
        scale_limit=0.2, 
        rotate_limit=10, 
        border_mode=0, p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.2),
    A.Resize(args.max_size, args.max_size),
    A.OneOf([
        A.RandomResizedCrop(args.max_size, args.max_size),
        A.Cutout(
            max_h_size=int(args.max_size*0.4), 
            max_w_size=int(args.max_size*0.4),
            num_holes=1,
            p=0.3),
    ]),
    A.Normalize(
        mean=(0.4452, 0.4457, 0.4464), 
        std=(0.2592, 0.2596, 0.2600)),
    ToTensorV2(),
])

print('Loading dataset...')
trainset = LandmarkDataset('train', transforms_train)
train_loader = DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


def criterion(inputs, targets, margins):
    arc_face = ArcFaceLossAdaptiveMargin(margins=margins, s=80)
    loss = arc_face(inputs, targets, args.num_classes)
    return loss


def train():
    epoch_size = len(trainset) // args.batch_size
    num_epochs = math.ceil(args.max_iter / epoch_size)
    start_epoch = 0
    iteration = 0

    df = pd.read_csv('./data/train.csv')
    tmp = np.sqrt(1 / np.sqrt(df['landmark_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    print('Loading model...')
    model = EfficientNetLandmark(args.depth, args.num_classes)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, num_epochs-1)

    if args.resume is not None:
        state_dict = torch.load(args.resume)
        try:
            print('Resume all state...')
            modules_state_dict = state_dict['modules']
            # optimizer_state_dict = state_dict['optimizer']
            # scheduler_state_dict = state_dict['scheduler']
            # optimizer.load_state_dict(optimizer_state_dict)
            # scheduler.load_state_dict(scheduler_state_dict)
            # start_epoch = state_dict['epoch']
            # iteration = state_dict['iteration']
        except KeyError:
            print('Resume only modules...')
            modules_state_dict = state_dict
        
        model_state_dict = {k.replace('module.', ''): v for k, v in modules_state_dict.items() if k.replace('module.', '') in model.state_dict().keys()}
        model.load_state_dict(model_state_dict)

    num_gpus = list(range(torch.cuda.device_count()))
    if len(num_gpus) > 1:
        print('Using data parallel...')
        model = nn.DataParallel(model, device_ids=num_gpus)

    # logger = open('log.txt', 'w')
    losses = AverageMeter()
    scores = AverageMeter()

    start_train = datetime.now()
    print(num_epochs, start_epoch, iteration)
    model.train()
    for epoch in range(start_epoch, num_epochs):
        if (epoch+1)*epoch_size < iteration:
            continue

        if iteration == args.max_iter:
            break
        
        correct = 0
        input_size = 0
        start_time = datetime.now()
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets, margins)
            
            confs, preds = torch.max(outputs.detach(), dim=1)
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()

            losses.update(loss.item(), inputs.size(0))
            scores.update(gap(preds, confs, targets))
            correct += (preds == targets).float().sum()
            input_size += inputs.size(0)

            iteration += 1

            writer.add_scalar('loss', losses.val, iteration)
            writer.add_scalar('gap', scores.val, iteration)

            # log = {'epoch': epoch+1, 'iteration': iteration, 'loss': losses.val, 'acc': corrects.val, 'gap': scores.val}
            # logger.write(str(log) + '\n')
            if iteration % args.verbose_eval == 0:
                print(
                    f'[{epoch+1}/{iteration}] Loss: {losses.val:.5f} Acc: {correct/input_size:.5f}' \
                    f' GAP: {scores.val:.5f} LR: {optimizer.param_groups[0]["lr"]} Time: {datetime.now() - start_time}')
            
            if iteration > 0 and iteration % args.save_interval == 0:
                print('Save model...')
                save_checkpoint({
                    'modules': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'iteration': iteration,
                }, f'effnet_b{args.depth}_{args.max_size}_{args.batch_size}_{epoch+1}_{iteration}.pth')

            scheduler.step(epoch+i / len(train_loader))
        print()

    # logger.close()
    writer.close()
    print(datetime.now() - start_train)


if __name__ == '__main__':
    train()