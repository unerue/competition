import os
import random
import math
import time
import datetime
import argparse
from typing import List, Tuple

import numpy as np
from sklearn.metrics import hamming_loss
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.utils import data_loader
from models.utils import AverageMeter, EarlyStopping, ImageDataset, TestDataset, SmoothCrossEntropyLoss, fixed_seed
from models.baseline import Baseline
from models.resnet import resnet_18, resnet_50, resnet_101
from models.resnext import se_resnext101


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=20) # 38
parser.add_argument('--lr', type=float, default=1e-3) # 1e-3
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=3) # 20
parser.add_argument('--print', type=int, default=10) 
parser.add_argument('--prediction_file', type=str, default='prediction.txt')
parser.add_argument('--batch_size', type=int, default=64) # 128
parser.add_argument('--mode', type=str, default='train') 
parser.add_argument('--patience', type=int, default=5) # 
parser.add_argument('--step_size', type=int, default=10) # 10
args = parser.parse_args()
device = 'cuda:0' if args.cuda else 'cpu'


to_multi_dict = {
    0: (1, 1), 1: (0, 1), 2: (1, 2), 3: (0, 0),
    4: (0, 2), 5: (1, 0), 6: (0, 3), 7: (1, 3),
    8: (2, 1), 9: (3, 1), 10: (2, 0), 11: (4, 1),
    12: (2, 2), 13: (3, 2), 14: (2, 3), 15: (4, 2),
    16: (4, 0), 17: (3, 0), 18: (4, 3), 19: (3, 3)
}
# to_multi_dict = {
#     0: (0, 0), 1: (0, 2), 2: (0, 3), 3: (0, 20), 4: (1, 20), 
#     5: (2, 14), 6: (2, 20), 7: (3, 4), 8: (3, 5), 9: (3, 13), 
#     10: (3, 20), 11: (4, 2), 12: (4, 7), 13: (4, 11), 14: (4, 20), 
#     15: (5, 8), 16: (6, 1), 17: (6, 20), 18: (7, 1), 19: (7, 20), 
#     20: (8, 6), 21: (8, 9), 22: (8, 20), 23: (9, 20), 24: (10, 20), 
#     25: (11, 14), 26: (12, 12), 27: (12, 20), 28: (13, 1), 29: (13, 6), 
#     30: (13, 9), 31: (13, 10), 32: (13, 15), 33: (13, 16), 34: (13, 17),
#     35: (13, 18), 36: (13, 19), 37: (13, 20)
# }
to_single_dict = {value: key for key, value in to_multi_dict.items()}

def to_multi_label(label: int) -> Tuple[int, int]:
    return to_multi_dict[label]


def to_single_label(label: Tuple[int, int]) -> int:
    return to_single_dict[label]


def multi_label_tensors_to_single_label_tensor(
    plant: torch.Tensor, disease: torch.Tensor) -> torch.Tensor:
    plant = plant.tolist()
    disease = disease.tolist()
    combined: List[int] = []
    for multi_label in zip(plant, disease):
        combined.append(to_single_label(multi_label))
    combined = torch.LongTensor(combined)
    return combined


def cal_hamming_loss(
    plant: torch.Tensor, disease: torch.Tensor, preds: torch.Tensor) -> float:
    plant = plant.numpy()
    disease = disease.numpy()
    preds = preds.detach().cpu().numpy()
    converted: List[Tuple[int, int]] = []
    for i in preds:
        converted.append(to_multi_label(i.item()))
    converted = np.asarray(converted)
    
    preds_plant = converted[:,0]
    preds_disease = converted[:,1]
    loss = (hamming_loss(plant, preds_plant) + hamming_loss(disease, preds_disease)) / 2
    return loss


def train_fn(train_loader, model, optimizer, criterion, scheduler, device, print_log: int = 10):
    losses = AverageMeter()
    hammings = AverageMeter()
    model.train()
    for i, (_, image, plant, disease) in enumerate(train_loader):
        image = image.to(device)
        combined = multi_label_tensors_to_single_label_tensor(plant, disease)
        combined = combined.to(device)

        optimizer.zero_grad()
        
        outputs = model(image)
        loss = criterion(outputs, combined)
        
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        acc = cal_hamming_loss(plant, disease, preds)

        if (i+1) % print_log == 0:
            print(f'  Train - loss: {loss.item():.4f} hamming loss: {acc}')

        losses.update(loss, image.size(0))
        hammings.update(acc, image.size(0))

    scheduler.step()


def valid_fn(data_loader, model, optimizer, criterion, scheduler, device):
    with torch.no_grad():
        model.eval()
        losses = AverageMeter()
        hammings = AverageMeter()
        for _, image, plant, disease in data_loader:
            image = image.to(device)
            combined = multi_label_tensors_to_single_label_tensor(plant, disease)
            combined = combined.to(device)
                            
            outputs = model(image)
            loss = criterion(outputs, combined)
        
            _, preds = torch.max(outputs, 1)
            acc = cal_hamming_loss(plant, disease, preds)              
                
            losses.update(loss, image.size(0))
            hammings.update(acc, image.size(0))

        print(f'  Valid - loss: {losses.avg:.4f} hamming loss: {hammings.avg}')

    return losses.avg


def run_train():
    fixed_seed(42)
    train_loader, valid_loader = data_loader(
        root='./data', phase='train', batch_size=args.batch_size)

    # model = resnet_101(num_classes=args.num_classes).to(device)
    model = se_resnext101(num_classes=args.num_classes).to(device)
    criterion = SmoothCrossEntropyLoss(smoothing=0.1) 
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num of parameters: {total_params:,}')
    print(f'num of trainable parameters: {trainable_params:,}\n')
    
    start_time = datetime.datetime.now()
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1:2} -----------------------------------------')
        train_fn(train_loader, model, optimizer, criterion, scheduler, device)
        loss = valid_fn(valid_loader, model, optimizer, criterion, scheduler, device)
        
        print(f'Elapsed time: {(datetime.datetime.now() - start_time)}...', end=' ')
        if not os.path.exists('./weight/'):
            os.mkdir('./weight/')
        early_stopping(loss, model, model_path=f'./weight/se_resnext101-e{epoch}.pth')
        if early_stopping.early_stop:
            print('Early stopping...')
            break
   

if __name__ == '__main__':
    run_train()

