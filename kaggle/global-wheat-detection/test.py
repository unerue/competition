import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensor

from dataset import TrainDataset, TestDataset
from visualize import image_read, image_read_cv2


df = pd.read_csv('./input/train.csv')
# image_read_cv2('./input/train/', 'c14c1e300')
# sys.exit(0)




def get_bbox(df):
    """ 
    Global Wheat Detection bound box: [xmin, ymin, width, height]
    Pascal VOC: [x_min, y_min, x_max, y_max]
    MS COCO: [x_min, y_min, width, height]
    """
    df['x'] = df['bbox'].apply(lambda x: eval(x)[0])
    df['y'] = df['bbox'].apply(lambda x: eval(x)[1])
    df['w'] = df['bbox'].apply(lambda x: eval(x)[2])
    df['h'] = df['bbox'].apply(lambda x: eval(x)[3])
    return df

df = get_bbox(df)





class Resize:
    def __init__(self, size):
        self.size = size

    def resize(self, image, target):
        w, h = image.shape[:2]
        image = self._letterbox_image(image, self.size)
        scale = min(self.size[0]/h, self.size[1]/w)
        target[:,:4] *= scale

        new_w = scale * w
        new_h = scale * h
        inp_dim = self.size

        del_w = (self.size[1] - new_w)/2
        del_h = (self.size[0] - new_h)/2

        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
        target[:,:4] += add_matrix

        image = image.astype(np.uint8)
        return image, target

    def _letterbox_image(self, image, size):
        img_w, img_h = image.shape[:2]
        w, h = size
        new_w = int(img_w * min(w/img_w, h/img_h))
        new_h = int(img_h * min(w/img_w, h/img_h))

        resized_image = cv2.resize(image, self.size)
        canvas = np.full((w, h, 3), 128)


        fh = (h-new_h)//2
        fw = (w-new_w)//2
        canvas[fh:fh+new_h, fw:fw+new_w,  :] = resized_image
        
        return canvas

import cv2
bboxes = df.loc[df['image_id'] == 'c14c1e300', ['x', 'y', 'w', 'h']].values
image = cv2.imread(f'./input/train/c14c1e300.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
image, bboxes = Resize((400,400)).resize(image, bboxes)
print(image.shape)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def image_read(image, bboxes):
    fig, axes = plt.subplots(1, 2, figsize=(7,5), dpi=200)
    axes.flat[0].imshow(image)
    axes.flat[1].imshow(image)
    rects = []
    for bbox in bboxes:
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3]) # x, y, w, h
        rects.append(rect)

    pc = PatchCollection(rects, edgecolor='y', facecolor='none')
    axes.flat[1].add_collection(pc)
    axes.flat[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) 
    axes.flat[1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.tight_layout()
    plt.show()
image_read(image, bboxes)
sys.exit(0)
