"""
Runs the coco-supplied cocoeval script to evaluate detections
outputted by using the output_coco_json flag in eval.py.
"""


import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser(description='COCO Detections Evaluator')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str)
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str)
parser.add_argument('--gt_ann_file',   default='data/coco/annotations/instances_val2017.json', type=str)
parser.add_argument('--eval_type',     default='both', choices=['bbox', 'mask', 'both'], type=str)
args = parser.parse_args()

# python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

classes = [1, 8, 25, 28]

classes = {1: 'person', 8: 'truck', 31: 'backpack', 32: 'tie'}

if __name__ == '__main__':

	eval_bbox = (args.eval_type in ('bbox', 'both'))
	eval_mask = (args.eval_type in ('mask', 'both'))

	print('Loading annotations...')
	gt_annotations = COCO(args.gt_ann_file)
	if eval_bbox:
		bbox_dets = gt_annotations.loadRes(args.bbox_det_file)
	if eval_mask:
		mask_dets = gt_annotations.loadRes(args.mask_det_file)

	if eval_bbox:
		print('\nEvaluating BBoxes:')
		bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()
		for i, v in classes.items():
			print(v)
			bbox_eval.params.catIds = [i]
			bbox_eval.evaluate()
			bbox_eval.accumulate()
			bbox_eval.summarize()
	
	if eval_mask:
		print('\nEvaluating Masks:')
		bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize() 
		for i, v in classes.items():
			print(v)
			bbox_eval.params.catIds = [i]
			
			bbox_eval.evaluate()
			bbox_eval.accumulate()
			bbox_eval.summarize()



