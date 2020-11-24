import os
import sys
import random
import datetime
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from imantics import Mask, Polygons

from model import get_model_instance_segmentation
from dataset import TrainDataset, TestDataset
from augmentations import Compose, ToTensor
from utils import indent, fixed_seed


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--cuda', type=bool, default=True)
args = parser.parse_args()

device: str = 'cuda:0' if args.cuda else 'cpu'
class_nums: Dict[int, str] = {
    1:'sidewalk_blocks', 2: 'alley_damaged', 3:'sidewalk_damaged', 4: 'caution_zone_manhole', 5: 'braille_guide_blocks_damaged', 
    6: 'alley_speed_bump', 7: 'roadway_crosswalk', 8: 'sidewalk_urethane', 9: 'caution_zone_repair_zone', 10: 'sidewalk_asphalt', 
    11: 'sidewalk_other', 12: 'alley_crosswalk', 13: 'caution_zone_tree_zone', 14: 'caution_zone_grating', 15: 'roadway_normal',
    16: 'bike_lane', 17: 'caution_zone_stairs', 18: 'alley_normal', 19: 'sidewalk_cement', 20: 'braille_guide_blocks_normal', 
    21: 'sidewalk_soil_stone', 22: 'alley',
}
class_names: Dict[str, int] = {v: k for k, v in class_nums.items()}

augmentations = {
    'train': Compose([
        ToTensor(),
    ]),
    'test': Compose([
        ToTensor(),
    ])
}

def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))

testset = TestDataset(augmentations['test'])
test_loader = DataLoader(
    testset, batch_size=1, shuffle=False, collate_fn=collate_fn)


def test_fn(
    model: torch.nn, 
    data_loader: DataLoader, 
    class_nums: Dict,
    device: torch.device):
    image_ids = [image_id.split('.')[1][-17:] for image_id in data_loader.dataset.image_ids]
    xml_root = ET.Element('predictions')
    model.eval()
    batch_size = data_loader.batch_size
    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(data_loader)):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for j, output in enumerate(outputs):
                image_name = image_ids[i*batch_size+j]
                xml_image = ET.SubElement(xml_root, 'image', {'name': image_name})

                masks = output['masks'].detach().cpu().numpy()
                labels = output['labels'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()

                for mask, label, score in zip(masks, labels, scores):
                    mask_bin = np.where(mask[0] > 0.1, True, False)
                    polygons = Mask(mask_bin).polygons()
                    points = polygons.points
                    point = ''.join([str(p[0]) + ',' + str(p[1]) +';' for p in points[0]])
                    attribs = {
                        'class_name': class_nums[label],
                        'score': str(float(score)), 
                        'polygon': point,
                    }
                    ET.SubElement(xml_image, 'predict', attribs)
                
    indent(xml_root)
    tree = ET.ElementTree(xml_root)
    if not os.path.exists('./output/'):
        print('Not exists ./output/ making an weight folder...')
        os.mkdir('./output/')

    tree.write('./output/prediction.xml')
    print('Save predicted labels.xml...\n')

    

def main():
    fixed_seed(42)
    
    model = get_model_instance_segmentation(args.num_classes).to(device)
    model.load_state_dict(torch.load('./weight/test1.pth'))
    model.eval()

    test_fn(model, test_loader, class_nums, device)
    

if __name__ == '__main__':
    main()

