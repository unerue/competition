import os
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import EfficientNetLandmark
from utils import LandmarkDataset, LandmarkDatasetResize, fixed_seed


fixed_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--max_size', default=300, type=int)
# parser.add_argument('--trained_model', default='effnet_256_448_11_50000.pth', type=str)
args = parser.parse_args()


# transforms_test = A.Compose([
#     A.Resize(args.max_size, args.max_size),
#     A.Normalize(
#         mean=(0.4452, 0.4457, 0.4464), 
#         std=(0.2592, 0.2596, 0.2600)),
#     ToTensorV2(),
# ])



transforms1 = A.Compose([
    A.Resize(args.max_size, args.max_size),
    A.Normalize(
        mean=(0.4452, 0.4457, 0.4464), 
        std=(0.2592, 0.2596, 0.2600)),
    ToTensorV2(),
])
transforms2 = A.Compose([
    A.Resize(500, 500),
    A.Normalize(
        mean=(0.4452, 0.4457, 0.4464), 
        std=(0.2592, 0.2596, 0.2600)),
    ToTensorV2(),
])



print('Loading dataset...')
# test_dataset = LandmarkDataset('test', transforms_test)
test_dataset = LandmarkDatasetResize(transforms1, transforms2)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(depth, path):
    model = EfficientNetLandmark(depth, 1049)
    state_dict = model.state_dict()
    try:
        trained_model = torch.load(path)['modules']
    except:
        trained_model = torch.load(path)

    trained_model = {k.replace('module.', ''): v for k, v in trained_model.items() if k.replace('module.', '') in state_dict.keys()}
    print(f'Loading model...{path}')
    model.load_state_dict(trained_model)
    model.to(device)

    num_gpus = list(range(torch.cuda.device_count()))
    if len(num_gpus) > 1:
        print('Using data parallel...')
        model = nn.DataParallel(model, device_ids=num_gpus)

    return model


def eval():
    # model1 = load_model(3, 'effnet_b3_300_64_168_230000.pth')
    model1 = load_model(3, 'effnet_b3_500_32_5_12000.pth')
    # model1 = load_model(3, 'effnet_b3_500_32_18_49000.pth')
    #model2 = load_model(4, 'effnet_b4_300_64_175_240000.pth')
    model2 = load_model(4, 'effnet_b4_300_64_175_240000.pth')
    model3 = load_model(5, 'effnet_b5_300_64_175_240000.pth')
    save_filename = 'b3_5_b4_175_b5_175'

    submission = pd.read_csv('./data/sample_submission.csv', index_col='id')
    print('Generating submission files...')
    model1.eval()
    model2.eval()
    model3.eval()
    # for image_id, inputs in tqdm(test_loader):
    #     inputs = inputs.to(device)
    #     outputs1 = model1(inputs)
    #     outputs2 = model2(inputs)
    #     outputs3 = model3(inputs)
        
    for image_id, inputs1, inputs2 in tqdm(test_loader):
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        outputs1 = model1(inputs2)
        outputs2 = model2(inputs1)
        outputs3 = model3(inputs1)

        outputs = (outputs1 + outputs2 + outputs3) / 3
        outputs = outputs.detach().cpu().numpy()
        landmark_ids = np.argmax(outputs, axis=1)
        confidence = outputs[0, landmark_ids]
        submission.loc[image_id, 'landmark_id'] = landmark_ids
        submission.loc[image_id, 'conf'] = confidence

    print(f'Save submission...')
    submission.to_csv(f'./outputs/ensemble-{save_filename}.csv')


if __name__ == '__main__':
    eval()