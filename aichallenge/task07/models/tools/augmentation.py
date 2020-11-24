
import sys
import random
import numpy as np
from typing import List, Dict, Tuple

import torch
from torchvision.transforms import functional as F
import PIL
from PIL import Image


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: PIL.Image, target: Dict[str, torch.Tensor]) -> Tuple:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: PIL.Image, target: Dict[str, torch.Tensor]) -> Tuple:
        image = F.to_tensor(image)
        return image, target


class Resize:
    def __init__(self, size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.size = size

    def __call__(self, image, target: Dict[str, torch.Tensor]):
        w, h = image.size
        scale = min(self.size[0]/h, self.size[1]/w)
        target['boxes'][:,:4] *= scale

        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, image: PIL.Image, target: Dict[str, float]) -> Tuple:
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox

        return image, target



