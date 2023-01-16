import torch
def convert_xywh_corners(boxes):
    boxes = torch.stack([
        boxes[:,0] - boxes[:,2] / 2,
        boxes[:,1] - boxes[:,3] / 2,
        boxes[:,0] + boxes[:,2] / 2,
        boxes[:,1] + boxes[:,3] / 2,
    ], dim= -1)
    return boxes

def convert_corners_xywh(boxes):
    boxes = torch.stack([
        (boxes[:,0] + boxes[:,2]) / 2,
        (boxes[:,1] + boxes[:,3]) / 2,
        boxes[:,2] - boxes[:,0],
        boxes[:,3] - boxes[:,1],
    ], dim= -1)
    return boxes