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
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--print', type=int, default=10) 
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


def test_fn(model, device):
    model.eval()
    testset = TestDataset()
    test_loader = DataLoader(testset, shuffle=False, batch_size=1)
    names = []
    outputs = []
    print(f'Attached data loader ({len(test_loader)})...')
    test_loader = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for name, image in test_loader:
            image = image.to(device)

            output = model(image)

            _, output = torch.max(output, 1)
            outputs.append(output.detach().cpu().numpy())
            names.append(name[0])

    outputs = np.array(outputs).flatten()
    print('Writing result...')
    preds = []
    for name, output in zip(names, outputs):
        output = to_multi_label(output)
        pred = name + ' ' + str(output[0]) + ' ' + str(output[1])
        preds.append(pred)
    
    if not os.path.exists('./output/'):
        os.mkdir('/output/')

    with open('./output/prediction-baseline.txt', 'w') as f:
        f.write('\n'.join(preds))

    if os.stat('./output/prediction-baseline.txt').st_size == 0:
        raise AssertionError('Output result of inference is nothing!')


def run_test():
    model_version = 2
    model = se_resnext101(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(f'./weight/se_resnext101-e{model_version}.pth'))
    model.eval()
    print('Loaded model...')

    test_fn(model, device)

if __name__ == '__main__':
    run_test()

