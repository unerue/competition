import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def image_read(image_dir, image_id, bboxes):
    fig, axes = plt.subplots(1, 2, figsize=(7,5), dpi=200)
    image = plt.imread(os.path.join(image_dir, f'{image_id}.jpg'))
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


def image_read_cv2(image_dir, image_id):
    image = cv2.imread(
        os.path.join(image_dir, f'{image_id}.jpg'), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image', image)

    images = np.concatenate((image, image), axis=1)
    cv2.imshow('images', images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_bbox_cv2(image_dir, image_id):
    image = cv2.imread(
        os.path.join(image_dir, f'{image_id}.jpg'), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image', image)
    
    images = np.concatenate((image, image), axis=1)
    cv2.imshow('images', images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image

def plot_img(data, idx):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (i[0],i[1]), (i[2],i[3]), (0,255,0), thickness=2)
    plt.figure(figsize=(10,10))
    plt.imshow(image)