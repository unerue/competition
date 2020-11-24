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

    def __call__(self, image: PIL.Image, target: Dict[str, torch.Tensor])-> torch.Tensor:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: PIL.Image, target: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = F.to_tensor(image)
        return image, target