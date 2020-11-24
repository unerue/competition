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

test_augmentations = Compose([
    ToTensor(),
])
testset = CustomDataset(class_names=class_names, augmentation=test_augmentations, phase='test')
print(f'Size: testset: {len(testset):,}')


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)


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
        for i, (images, _) in tqdm(enumerate(data_loader)):
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


def run_test():
    fixed_seed(42) # seed 고정
    model = model_for_bbox(args.num_classes).to(device)
    model.load_state_dict(torch.load('./weight/test3.pth'))
    model.eval()

    # 제출용 prediction.xml 만들기
    valid_fn(model, test_loader, device, class_nums, valid=False)


if __name__ == '__main__':
    run_test()