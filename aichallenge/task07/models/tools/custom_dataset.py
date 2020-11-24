import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image


class CustomDataset:
    """
    베이스라인 코드에서 xml 파싱하는 부분이 정말 비효율적으로 코딩되어 있어 전부 바꿈.
    """
    def __init__(
        self, class_names: Dict, augmentation: List, phase: str = 'train'):
        self.class_names = class_names
        self.augmentation = augmentation
        self.phase = phase
        if self.phase == 'test':
            self.labels = sorted(ET.parse(f'./data/test/test.xml').findall('image'), key=lambda x: x.attrib['name'])
        else:
            self.labels = ET.parse(f'./data/{phase}_labels.xml').findall('image')
        
        self.images = [label.attrib['name'] for label in self.labels]
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        if self.phase == 'test':
            image_path = './data/test/' + self.labels[index].get('name')
        else:
            image_path = './data/train/' + self.labels[index].get('name')
        
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]

        boxes = []
        labels = []
        for box in label.findall('./box') :
            label = box.attrib['label']
            x1 = float(box.attrib['xtl'])
            y1 = float(box.attrib['ytl'])
            x2 = float(box.attrib['xbr'])
            y2 = float(box.attrib['ybr'])
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_names[label])
        
        target: Dict[str, torch.Tensor] = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([index])

        if self.augmentation is not None:
            image, target = self.augmentation(image, target)

        return image, target

    def __len__(self):
        return len(self.labels)


