import os
import sys
import random
from glob import glob
import xml.etree.ElementTree as ET
from typing import List, Union

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def fixed_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def indent(node: ET.XML, level: int = 0):
    i = '\n' + level * ' ' * 4
    if len(node):
        if not node.text or not node.text.strip():
            node.text = i + ' ' * 4
        if not node.tail or not node.tail.strip():
            node.tail = i
        for node in node:
            indent(node, level + 1)
        if not node.tail or not node.tail.strip():
            node.tail = i
    else:
        if level and (not node.tail or not node.tail.strip()):
            node.tail = i
