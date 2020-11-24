import os
import sys
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict

import cv2
import numpy as np
import torch
from PIL import Image


class TrainDataset:
    def __init__(
        self, class_names: Dict, augmentation: List):
        self.class_names = class_names
        self.augmentation = augmentation

        self.image_ids = []
        self.image_labels = []
        for file in glob.glob('./data/train/*.xml'):
            tree = ET.parse(file)
            for label in tree.getroot():
                if label.tag not in ['version', 'meta']:
                    self.image_ids.append(label.attrib['name'])
                    self.image_labels.append(list(label))
    
    def __getitem__(self, index: int) -> torch.Tensor:      
        image = Image.open('./data/train/' + self.image_ids[index])
        w, h = image.size  # 베이스라인 코드에 PIL Image.size return 반대로 되어있음...

        masks: List[List[bool, bool]] = []
        boxes: List[List[int, int]] = []
        labels: List[int] = []
        for i in self.image_labels[index]:
            class_name = self._get_label(i)
            # class_name = i.attrib['label']
            # if class_name != 'bike_lane':
            #     if len(i.findall('attribute')) == 0:
            #         continue
            # class_name += '_' + i.findall('attribute')[0].text
            class_name = self.class_names[class_name]
            labels.append(class_name)

            box, pos = self._get_boxes(i)
            boxes.append(box)
    
            canvas = np.zeros((h, w), np.uint8)
            cv2.fillPoly(canvas, [pos], 255)
            mask = canvas == 255
            masks.append(mask)
        
        target: Dict[str, torch.Tensor] = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target['boxes'] = boxes
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([index])

        target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(labels),), dtype=torch.int64)

        if self.augmentation is not None:
            image, target = self.augmentation(image, target)

        return image, target

    def __len__(self):
        return len(self.image_ids)

    def _get_label(self, xml_object):
        class_name = xml_object.attrib['label']
        if class_name not in ['bike_lane', 'alley'] and len(xml_object.findall('attribute')) > 0:
            class_name += '_' + xml_object.findall('attribute')[0].text

        return class_name

    def _get_boxes(self, xml_object):
        points = xml_object.attrib['points'].split(';')
        pos = np.asarray([(float(p.split(',')[0]), float(p.split(',')[1])) for p in points])
        pos = pos.astype(np.int32)
        x1 = np.min(pos[:, 0])
        y1 = np.min(pos[:, 1])
        x2 = np.max(pos[:, 0])
        y2 = np.max(pos[:, 1])

        return [x1, y1, x2, y2], pos


class TestDataset:
    def __init__(
        self, augmentation: List):
        self.augmentation = augmentation
        self.image_ids = glob.glob('./data/test/*.jpg')[:30]
            
    def __getitem__(self, index: int) -> torch.Tensor:      
        image = Image.open(self.image_ids[index])
        
        target: Dict[str, torch.Tensor] = {}
        target['image_id'] = torch.tensor([index])

        if self.augmentation is not None:
            image, target = self.augmentation(image, target)

        return image, target

    def __len__(self):
        return len(self.image_ids)
