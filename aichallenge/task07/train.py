import os
import sys
import random
import datetime
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.baseline import model_for_bbox
from models.tools import train_valid_split, xml_parser, xml_writer, get_test_list, indent
from models.tools import CustomDataset
from models.tools import Compose, ToTensor, Resize
from models.tools import AverageMeter, fixed_seed
from models.tools import evaluate_mean_average_precision


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=30)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--cuda', type=bool, default=True)
args = parser.parse_args()

device: str = 'cuda:0' if args.cuda else 'cpu'
class_nums: Dict[int, str] = {
    1: 'bus', 2: 'car', 3:'carrier', 4: 'cat', 5: 'dog', 
    6: 'motorcycle', 7: 'movable_signage', 8: 'person', 9: 'scooter', 10: 'stroller', 
    11: 'truck', 12: 'wheelchair', 13: 'barricade', 14: 'bench', 15 : 'chair',
    16: 'fire_hydrant', 17: 'kiosk', 18: 'parking_meter', 19: 'pole', 20: 'potted_plant', 
    21: 'power_controller', 22 : 'stop', 23: 'table', 24: 'traffic_light_controller', 
    25: 'traffic_sign', 26: 'tree_trunk', 27: 'bollard', 28: 'bicycle', 29: 'traffic_light'
} # 로컬 테스트용 -> 클래스 개수 30개
# class_nums = {
#     1: 'bus', 2: 'car', 3:'carrier', 4: 'cat',  : 'dog', 
#     6: 'motorcycle', 7: 'movable_signage', 8: 'person', 9: 'scooter', 10: 'stroller', 
#     11: 'truck', 12: 'wheelchair', 13: 'barricade', 14: 'bench', 15: 'chair',
#     16: 'fire_hydrant', 17: 'kiosk', 18: 'parking_meter', 19: 'pole', 20: 'potted_plant', 
#     21: 'power_controller', 22: 'stop', 23: 'table', 24: 'traffic_light_controller', 
#     25: 'traffic_sign', 26: 'tree_trunk', 27: 'bollard', 28: 'bicycle'
# } # 서버 제출용 -> 클래스 개수 29개
class_names: Dict[str, int] = {v: k for k, v in class_nums.items()}

# 데이터셋 분할 후 레이블 파일 생성
if not os.path.exists('./data/train_labels.xml'):
    train_list, valid_list = train_valid_split(test_size=0.1, shuffle=True)
    train_objects, valid_objects = xml_parser(train_list, valid_list)
    xml_writer(train_objects, 'train')
    xml_writer(valid_objects, 'valid')
    print('Generated labels.xml...')

# Augmentation
train_augmentations = Compose([
    Resize((500,500)),
    ToTensor(),
])
trainset = CustomDataset(class_names=class_names, augmentation=train_augmentations, phase='train')
validset = CustomDataset(class_names=class_names, augmentation=train_augmentations, phase='valid')
print(f'Size: trainset: {len(trainset):,}, validset: {len(validset):,}')


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(validset, batch_size=4, shuffle=False, collate_fn=collate_fn)


def train_fn(
    model: torch.nn, 
    data_loader: DataLoader, 
    optimizer: optim, 
    scheduler: lr_scheduler, 
    device: torch.device, 
    epoch: int):
    model.train()
    losses = AverageMeter()
    start_time = datetime.datetime.now()
    num_images: int = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = torch.stack(images)
        num_images += len(images)
        
        optimizer.zero_grad()
        loss_dict: Dict[str, torch.Tensor] = model(images, targets)
        loss: float = sum(loss for loss in loss_dict.values())
        losses.update(loss.item(), images.size(0))
        
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('-'*50)
            print(f'Epoch {epoch+1}[{len(data_loader.dataset):,}/{(num_images/len(data_loader.dataset))*100:.2f}%] '
                  f'- Elapsed time: {datetime.datetime.now() - start_time}\n'
                  f' - loss: classifier={loss_dict["loss_classifier"]:.6f}, box_reg={loss_dict["loss_box_reg"]:.6f}, '
                  f'objectness={loss_dict["loss_objectness"]:.6f}, rpn_box_reg={loss_dict["loss_rpn_box_reg"]:.6f}')


def valid_fn(
    model: torch.nn, 
    data_loader: DataLoader, 
    device: torch.device, 
    class_names: Dict[int, str], 
    valid: bool = True):
    """
    평가 및 결과 저장
    """
    xml_root = ET.Element('predictions')
    batch_size: int = data_loader.batch_size
    model.eval()
    with torch.set_grad_enabled(False):
        for i, (images, _) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for j, output in enumerate(outputs):
                image_name: str = data_loader.dataset.images[i*batch_size+j]
                xml_image = ET.SubElement(xml_root, 'image', {'name': image_name})

                boxes = output['boxes'].detach().cpu().numpy()
                labels = output['labels'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    attribs = {
                        'class_name': class_names[label],
                        'score': str(float(score)), 
                        'x1': str(int(box[0])), 
                        'y1': str(int(box[1])), 
                        'x2': str(int(box[2])), 
                        'y2': str(int(box[3]))
                    }
                    ET.SubElement(xml_image, 'predict', attribs)
                
    indent(xml_root)
    tree = ET.ElementTree(xml_root)
    if not os.path.exists('./output/'):
        os.mkdir('./output/')

    if valid:
        tree.write('./output/validation.xml')
    else:
        tree.write('./output/prediction.xml')
    print('Save predicted labels.xml...\n')


def get_mean_average_precision(file_name: str) -> float:
    path_gt = os.path.join('./data/valid_labels.xml')
    path_dr = os.path.join(f'./output/{file_name}.xml')
    result = evaluate_mean_average_precision(path_gt, path_dr)
    print(f'mAP@IoU 0.75: {result}')


def run_train():
    fixed_seed(42) # seed 고정
    if not os.path.exists('./weight/'):
        print('Not exists ./weight/ making an weight folder...')
        os.mkdir('./weight/')
    model = model_for_bbox(args.num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = None
    for epoch in range(args.epochs):
        train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        torch.save(model.state_dict(), f'./weight/test{epoch}.pth')
        # valid_fn(model, valid_loader, device, class_nums, valid=True)
        # get_mean_average_precision('validation')

if __name__ == '__main__':
    run_train()