import torch
from .anchors import *
import cv2
import numpy as np
from .boxes import *


def decode_detection_output(anchor_boxes, preds):
    box_variance = torch.tensor([0.1, 0.1, 0.2, 0.2])
    box_target = preds * box_variance
    box_target = torch.concat(
        [
            box_target[:,:2] * anchor_boxes[:, 2:] + anchor_boxes[:,:2],
            torch.pow(torch.e, box_target[:,2:]) * anchor_boxes[:, 2:],
        ],
        axis = -1,
    )  

    
    return box_target

def _compute_box_target(anchor_boxes, matched_gt_boxes):
    box_variance = torch.tensor([0.1, 0.1, 0.2, 0.2])
    box_target = torch.concat(
        [
            (matched_gt_boxes[:,:2] - anchor_boxes[:,:2]) / anchor_boxes[:, 2:],
            torch.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
        ],
        axis = -1,
    )

    box_target = box_target / box_variance
    return box_target

def get_detection_labels(labels, anchors, true_iou_th = 0.5, false_iou_th = 0.4):
    cls_target = labels[:,0]
    anchors = torch.reshape(anchors, (-1,4))
    box_target = torch.reshape(labels[:,1:], (-1,4))
    anchors = convert_xywh_corners(anchors)
    box_target = convert_xywh_corners(box_target)

    intersection = torch.stack([
        torch.min(box_target[None, :, 2], anchors[:, 2, None]) - torch.max(box_target[None, :, 0], anchors[:, 0, None]),
        torch.min(box_target[None, :, 3], anchors[:, 3, None]) - torch.max(box_target[None, :, 1], anchors[:, 1, None]),
    ], dim=-1)
    
    intersection_mask = torch.where(torch.logical_and(intersection[:,:,0] > 0,  intersection[:,:,1] > 0), 1, 0)
    intersection_area = intersection[:,:,0] * intersection[:,:,1] * intersection_mask
    box_area = (box_target[None,:,2] - box_target[None,:,0]) * (box_target[None,:,3] - box_target[None,:,1])
    anchors_area = (anchors[:,2,None] - anchors[:,0,None]) * (anchors[:,3,None] - anchors[:,1,None])
    iou = intersection_area / (box_area + anchors_area - intersection_area)


    iou_max, matched_gt_idx = iou.max(dim=1)
    matched_gt_cls = torch.gather(cls_target, 0, matched_gt_idx)
    matched_gt_idx = matched_gt_idx.unsqueeze(-1)
    matched_gt_idx = matched_gt_idx.tile(1,4)
    matched_gt_boxes = torch.gather(box_target, 0, matched_gt_idx)
    matched_gt_cls = torch.where(iou_max > true_iou_th, matched_gt_cls, -1)
    matched_gt_cls = torch.where(iou_max < false_iou_th, 0, matched_gt_cls)
    matched_gt_cls = matched_gt_cls.unsqueeze(-1)

    anchors = convert_corners_xywh(anchors)
    matched_gt_boxes = convert_corners_xywh(matched_gt_boxes)
    box_target = _compute_box_target(anchors, matched_gt_boxes)
    return torch.concat([matched_gt_cls, box_target], axis =-1)


