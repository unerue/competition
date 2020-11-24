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

# image_id = 'c14c1e300'
# bboxes = df.loc[df['image_id'] == image_id, ['x', 'y', 'w', 'h']].values
# image_read('./input/train/', image_id, bboxes)
sys.exit(0)



# image_id = 'c14c1e300'
# bboxes = df.loc[df['image_id'] == image_id, ['x', 'y', 'w', 'h']].values
# image_read('./input/train/', image_id, bboxes)
# sys.exit(0)


image_ids = df['image_id'].unique()
# sys.exit(0)


image_dir = './input/train/'
transforms = {
    'train': A.Compose([
        ToTensor()
    ])
}
trainset = TrainDataset(
    df, image_dir, image_ids, 
    transforms=transforms['train'])

def collate_fn(batch):
    return tuple(zip(*(batch)))

train_loader = DataLoader(
    trainset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

num_classes = 2  # 1 class (wheat) + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=5e-4)

epochs = 5
for epoch in range(epochs):
    num_images = 0
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        images = [image.to(device) for image in images]
        num_images += len(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            print(f'Epoch {epoch}[{len(train_loader.dataset):,}({num_images/len(train_loader.dataset):.2f})] {loss}')

torch.save(model.state_dict(), './weights/test.pth')

transforms = {
    'train': A.Compose([
        ToTensor()
    ])
}

# def collate_fn(batch):
#     return tuple(zip(*(batch)))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_classes = 2  # 1 class (wheat) + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load('./weights/test.pth'))
model.eval()
model.to(device)

submit = pd.read_csv('./input/sample_submission.csv')
image_ids = submit['image_id'].unique()

testset = TestDataset('./input/test/', image_ids, transforms=transforms['train'])
test_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)


def format_prediction_string(boxes, scores):
    pred_strings = []
    for i in zip(scores, boxes):
        pred_strings.append(f'{i[0]:.4f} {i[1][0]} {i[1][1]} {i[1][2]} {i[1][3]}')

    return ' '.join(pred_strings)

detection_threshold = 0.5
with torch.no_grad():
    results = []
    for images, _ in test_loader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]
            
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            
            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }

            results.append(result)

print(results[0:2])


submit = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
submit.to_csv('submission.csv', index=False)
print(submit.head())