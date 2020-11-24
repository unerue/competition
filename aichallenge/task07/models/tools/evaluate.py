import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Union
from collections import defaultdict


def is_float_between_0_and_1(value: float) -> bool:
    try:
        value = float(value)
        if value > 0.0 and value < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def voc_average_precision(rec: List, prec: List) -> Union:
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
 
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
 
    ap = 0.0
    for i in i_list:
        ap += (mrec[i]-mrec[i-1]) * mpre[i]

    return ap, mrec, mpre


def file_lines_to_list(path: str):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content


def mean_average_precision(
    gt_file_boxes: Dict, gt_counter_per_class: Dict, class_boxes: Dict, minoverlap: float = 0.75):
    gt_classes = sorted(gt_counter_per_class.keys())
    n_classes = len(gt_classes)
    sum_ap = 0.0
    count_true_positives = {}
    for class_name in gt_classes:
        count_true_positives[class_name] = 0
        dr_data = class_boxes[class_name]
        nd = len(dr_data)
        tp = [0] * nd 
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection['file_id']
            ground_truth_data = gt_file_boxes[file_id]
            ovmax = -1
            gt_match = -1
            bb = [ float(x) for x in detection['bbox'].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj['class_name'] == class_name:
                    bbgt = [ float(x) for x in obj['bbox'].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) \
                             + (bbgt[2] - bbgt[0] + 1) \
                             * (bbgt[3] - bbgt[1] + 1) \
                             - iw * ih

                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            if ovmax >= minoverlap:
                if not bool(gt_match['used']):
                    tp[idx] = 1  # true positive
                    gt_match['used'] = True
                    count_true_positives[class_name] += 1
                else:
                    fp[idx] = 1  # false positive (multiple detection)
            else:
                fp[idx] = 1  # false positive

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, _, _ = voc_average_precision(rec[:], prec[:])
        sum_ap += ap
    mAP = sum_ap / n_classes

    return mAP


def read_groundtruth(file_path: str) -> Union[Dict, Dict]:
    gt_class_counter = defaultdict(int)
    gt_file_boxes = {}
    images = ET.parse(file_path)
    for image in images.findall('./image'):
        file_id = image.attrib['name'].replace('.jpg', '').replace('.png', '')
        bboxes = []
        for labels in image.findall('./box') :
            class_name = labels.attrib['label']
 
            x1 = labels.attrib['xtl'].split('.')[0]
            y1 = labels.attrib['ytl'].split('.')[0]
            x2 = labels.attrib['xbr'].split('.')[0]
            y2 = labels.attrib['ybr'].split('.')[0]
  
            bbox = x1 + ' ' + y1 + ' ' + x2 + ' ' +y2
            bboxes.append({'class_name': class_name, 'bbox': bbox, 'used': False})
            gt_class_counter[class_name] += 1

        gt_file_boxes[file_id] = bboxes

    return gt_file_boxes, gt_class_counter


def read_prediction(
    xml_name: str, gt_counter_per_class: Dict, minscore: float = 0.0):
    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)

    # get a list with the detection-results files
    images = ET.parse(xml_name)
    class_boxes = {}
    for class_name in gt_classes:
        bboxes = []
        for image in images.findall('./image'):
            file_id = image.attrib['name'].replace('.jpg', '').replace('.png', '')
            for pred in image.findall('predict'):
                try:
                    _class_name, confidence, left, right, top, bottom = pred.attrib.values()
                except ValueError as e:
                    print(e)
                if _class_name == class_name:
                    if float(confidence) >= minscore :
                        bbox = left + ' ' + top + ' ' + right + ' ' + bottom
                        bboxes.append({
                            'confidence': confidence, 'file_id': file_id, 'bbox': bbox
                        })
        # sort detection-results by decreasing confidence
        bboxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        class_boxes[class_name] = bboxes
    return class_boxes


def evaluate_mean_average_precision(path_gt: str, path_dr: str) -> float:
    groundtruth_boxes, groundtruth_classes = read_groundtruth(path_gt)
    predction_boxes = read_prediction(path_dr, groundtruth_classes)
    return mean_average_precision(groundtruth_boxes, groundtruth_classes, predction_boxes)

