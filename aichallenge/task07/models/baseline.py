import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def model_for_bbox(num_classes: int):
    """pretrained 사용하지 말 것!"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model