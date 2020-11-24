import os
import random
import math
import time
import datetime
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import hamming_loss
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensor

from models.resnext import se_resnext101
from models.utils import AverageMeter, EarlyStopping, ImageDataset, TestDataset
from models.utils import stratified


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--print_iter', type=int, default=10)
parser.add_argument('--model_name', type=str, default='model.pth')
parser.add_argument('--prediction_file', type=str, default='prediction.txt')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--patience', type=int, default=5)
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

def fixed_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_multi_label(label):
    return to_multi_dict[label]


def to_single_label(label):
    return to_single_dict[label]


def multi_label_tensors_to_single_label_tensor(plant, disease):
    plant = plant.tolist()
    disease = disease.tolist()
    combined = []
    for multi_label in zip(plant, disease):
        combined.append(to_single_label(multi_label))
    combined = torch.LongTensor(combined)
    return combined


def cal_hamming_loss(plant, disease, preds):
    plant = plant.numpy()
    disease = disease.numpy()
    preds = preds.detach().cpu().numpy()
    converted = []
    for i in preds:
        converted.append(to_multi_label(i.item()))
    converted = np.asarray(converted)
    
    preds_plant = converted[:,0]
    preds_disease = converted[:,1]
    loss = (hamming_loss(plant, preds_plant) + hamming_loss(disease, preds_disease)) / 2
    return loss


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


def train(train_loader, model, optimizer, criterion, device, scheduler):
    model.train()
    losses = AverageMeter()
    hammings = AverageMeter()
    train_loader = tqdm(train_loader, total=len(train_loader))
    for _, image, plant, disease in train_loader:
        image = image.to(device)
        combined = multi_label_tensors_to_single_label_tensor(plant, disease)
        combined = combined.to(device)

        optimizer.zero_grad()
        
        outputs = model(image)
        loss = criterion(outputs, combined)
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        hamming = cal_hamming_loss(plant, disease, preds)
        
        losses.update(loss.item(), image.size(0))
        hammings.update(hamming.item(), image.size(0))

        train_loader.set_postfix(loss=losses.avg, hamming=hammings.avg)
    
    scheduler.step()


def evaluate(data_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    hammings = AverageMeter()
    data_loader = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for _, image, plant, disease in data_loader:
            image = image.to(device)
            combined = multi_label_tensors_to_single_label_tensor(plant, disease)
            combined = combined.to(device)
                            
            outputs = model(image)
            loss = criterion(outputs, combined)
            
            _, preds = torch.max(outputs, 1)
            hamming = cal_hamming_loss(plant, disease, preds)
            
            losses.update(loss.item(), image.size(0))
            hammings.update(hamming.item(), image.size(0))

            data_loader.set_postfix(loss=losses.avg, hamming=hammings.avg)

    return hammings.avg

              
def oof(fold=None):
    df = pd.read_csv('./data/train_folds.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)
    print(f'Trainset: {df_train.shape[0]}, Validset: {df_valid.shape[0]}')
    augmentation = {}
    augmentation['train'] = A.Compose([
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

    augmentation['test'] = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensor()
    ])

    trainset = ImageDataset(df_train, train=True, augmentation=augmentation)
    validset = ImageDataset(df_valid, train=False, augmentation=augmentation)

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=False)

    model = se_resnext101(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    for _ in range(args.num_epochs):
        train(train_loader, model, optimizer, criterion, device, scheduler)
        hamming = evaluate(valid_loader, model, criterion, device)

        early_stopping(hamming, model, model_path=f'./weight/resnext_model{fold}.pth')
        if early_stopping.early_stop:
            print('Early stopping...')
            break


def main():
    fixed_seed(42)
    stratified(to_multi_dict, frac=0.1)
    for fold in range(5):
        oof(fold)

    model1 = se_resnext101(num_classes=args.num_classes).to(device)
    model1.load_state_dict(torch.load('./weight/resnext_model0.pth'))
    model1.eval()

    model2 = se_resnext101(num_classes=args.num_classes).to(device)
    model2.load_state_dict(torch.load('./weight/resnext_model1.pth'))
    model2.eval()

    model3 = se_resnext101(num_classes=args.num_classes).to(device)
    model3.load_state_dict(torch.load('./weight/resnext_model2.pth'))
    model3.eval()

    model4 = se_resnext101(num_classes=args.num_classes).to(device)
    model4.load_state_dict(torch.load('./weight/resnext_model3.pth'))
    model4.eval()

    model5 = se_resnext101(num_classes=args.num_classes).to(device)
    model5.load_state_dict(torch.load('./weight/resnext_model4.pth'))
    model5.eval()

    submission = []
    testset = TestDataset()
    test_loader = DataLoader(testset, shuffle=False, batch_size=1)
    names = []
    print(f'Attached data loader ({len(test_loader)})...')
    test_loader = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for name, image in test_loader:
            image = image.to(device)

            outputs1 = model1(image)
            outputs2 = model2(image)
            outputs3 = model3(image)
            outputs4 = model4(image)
            outputs5 = model5(image)
            
            outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5) / 5
            submission.append(outputs.detach().cpu().numpy())
            names.append(name[0])

    submission = np.array(submission)
    submission = submission.reshape(-1, submission.shape[-1])
    submission = np.argmax(submission, axis=1)
    
    print('Writing result...')
    preds = []
    for name, sub in zip(names, submission):
        sub = to_multi_label(sub)
        pred = name + ' ' + str(sub[0]) + ' ' + str(sub[1])
        preds.append(pred)
    
    with open('./output/prediction-oof.txt', 'w') as f:
        f.write('\n'.join(preds))

    if os.stat('./output/prediction-oof.txt').st_size == 0:
        raise AssertionError('Output result of inference is nothing!')


if __name__ == '__main__':
    main()
