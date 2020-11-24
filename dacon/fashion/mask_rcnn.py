import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from torch import nn, Tensor
from torch._C import dtype
from torch.utils.data import Dataset
from torchvision import models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FashionDataset(Dataset):
    def __init__(self, path, mode='train', transforms=None):
        self.coco = COCO(path)
        self.mode = mode
        self.image_ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = [x for x in self.coco.loadAnns(annot_ids) if x['image_id'] == image_id]
        
        bboxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        bboxes[:, 2] = bboxes[:, 0] +  bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] +  bboxes[:, 3]
        bboxes = np.abs(bboxes)
        #print(bboxes)

        category_ids = np.array([annot['category_id'] for annot in annots], dtype=np.int64)
        mask = np.array([self.coco.annToMask(annot).reshape(-1) for annot in annots])
        print(mask)
        # mask = np.vstack(-1, 800, 800)
        # mask = mask.reshape(-1, )

        areas = np.array([annot['area'] for annot in annots], dtype=np.float64)
        iscrowds = np.array([annot['iscrowd'] for annot in annots], dtype=np.int64)

            # pass

        # boxes = torch.as_tensor([target_temp[0]['bbox']], dtype=torch.float32)
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # labels = torch.as_tensor([target_temp[0]['category_id']], dtype=torch.int64)
        
        # masks = [self.coco.annToMask(obj) for obj in target_temp]
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        # image_id = torch.tensor([target_temp[0]['image_id']], dtype=torch.int64)
        # area = torch.as_tensor([target_temp[0]['area']], dtype=torch.float32)
        # iscrowd = torch.as_tensor([target_temp[0]['iscrowd']], dtype=torch.int64)


        # samples = {
        #     'bbox': ,
        # }

       

        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        file_name = f'./data/fashion/train/{file_name}'
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        target = {}
        if self.transforms is not None:
            transformed = self.transforms(
                image=image, mask=mask, bboxes=bboxes, category_ids=category_ids)

            image = transformed['image']
            # image = torch.as_tensor(image, dtype=torch.float32)
            # print(image.shape)
            
            target['masks'] = torch.as_tensor(transformed['mask'], dtype=torch.uint8)
            print(target['masks'])
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            # print(boxes)
            #boxes[:, 2] = boxes[:, 0] +  boxes[:, 2]
            #boxes[:, 3] = boxes[:, 1] +  boxes[:, 3]
            #target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*transformed['bboxes'])))).permute(1, 0)
            target['boxes'] = boxes
            target['image_id'] = torch.as_tensor([image_id])
            target['area'] = torch.as_tensor(areas, dtype=torch.float64)
            target['iscrowd'] = torch.as_tensor(iscrowds, dtype=torch.int64)
            target['labels'] = torch.as_tensor(category_ids, dtype=torch.int64)
            target['area'] = torch.as_tensor(areas, dtype=torch.float64)
            print(image, boxes, target['image_id'], target['masks'].shape, target['area'], target['iscrowd'], target['labels'])
            

        return image, target



from torchvision.models.detection import backbone_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN, MultiScaleRoIAlign, MaskRCNNHeads

# class FashionModel(nn.Module):
#     def __init__(self, num_classes, backbone='resnet101', hidden_layer=256, pretrained=True):
#         backbone = resnet_fpn_backbone(backbone, pretrained=pretrained) 
#         self.model = MaskRCNN(backbone=backbone, num_classes=num_classes)
#
#         in_features = self.model.roi_heads.box_predictor.cls_score.in_features
#         self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# 
#         in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
#         self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
#             in_features_mask, hidden_layer, num_classes)
# 
#     def forward(self, inputs, targets):
#         return self.model(inputs, targets)


def FashionModel(num_classes, backbone='resnet101', hidden_layer=256, pretrained=True):
    backbone = resnet_fpn_backbone(backbone, pretrained=pretrained) 
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model
 

# def get_model(num_classes):
#     # model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#     # model = MaskRCNN()
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     backbone = resnet_fpn_backbone('resnet101', pretrained=True)
#     model = MaskRCNN(backbone, num_classes=len(CLASSES)+1)

#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # and replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(
#         in_features_mask, hidden_layer, num_classes)

#     return model



class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count