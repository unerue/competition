import os
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import EfficientNetLandmark
from utils import LandmarkDataset


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--max_size', default=300, type=int)
parser.add_argument('--trained_model', default=None, type=str)
parser.add_argument('--depth', default=0, type=int)
args = parser.parse_args()

transforms_test = A.Compose([
    A.Resize(args.max_size, args.max_size),
    A.Normalize(
        mean=(0.4452, 0.4457, 0.4464), 
        std=(0.2592, 0.2596, 0.2600)),
    ToTensorV2(),
])

print('Loading dataset...')
testset = LandmarkDataset('test', transforms_test)
test_loader = DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = EfficientNetLandmark(args.depth, 1049)
state_dict = model.state_dict()
try:
    trained_model = torch.load(args.trained_model)['modules']
except:
    trained_model = torch.load(args.trained_model)

trained_model = {k.replace('module.', ''): v for k, v in trained_model.items() if k.replace('module.', '') in state_dict.keys()}
print(f'Loading model...{args.trained_model}')
model.load_state_dict(trained_model)
model.to(device)

num_gpus = list(range(torch.cuda.device_count()))
if len(num_gpus) > 1:
    print('Using data parallel...')
    model = nn.DataParallel(model, device_ids=num_gpus)

submission = pd.read_csv('./data/sample_submission.csv', index_col='id')


def eval():
    print('Generating submission files...')
    model.eval()
    for image_id, inputs in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # outputs = nn.Softmax(dim=1)(outputs)
        outputs = outputs.detach().cpu().numpy()
        landmark_ids = np.argmax(outputs, axis=1)
        confidence = outputs[0, landmark_ids]
        submission.loc[image_id, 'landmark_id'] = landmark_ids
        submission.loc[image_id, 'conf'] = confidence

    path = os.path.splitext(os.path.basename(args.trained_model))[0]
    print(f'Save submission {path}.csv')
    submission.to_csv(f'./outputs/{path}.csv')


if __name__ == '__main__':
    eval()