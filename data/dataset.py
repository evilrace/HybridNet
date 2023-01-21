import torch
import torchvision

from .data_preprocess import get_img_segment_pairs, get_img_detection_pairs
from .utils import label_parser

import sys

class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        train_segment_data_path = r'F:\dataset\cityscape\gtFine\train'
        train_img_data_path = r'F:\dataset\cityscape\leftImg8bit\train'
        self.img_segment_pair_list = get_img_segment_pairs(train_segment_data_path, train_img_data_path)
        self.cached = {}
    def __getitem__(self, idx):
        if idx not in self.cached:
            img_file_path, segment_file_path = self.img_segment_pair_list[idx]
            img = torchvision.io.read_image(img_file_path)
            segment = torchvision.io.read_image(segment_file_path)
            self.cached[idx] = (img, segment)

        return self.cached[idx]
    def __len__(self):
        return len(self.img_segment_pair_list)

class DetectionDataset(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super().__init__()
        train_detection_data_path = r'F:\dataset\kitti\data_object_image_2\training'
        train_img_data_path = r'F:\dataset\kitti\data_object_image_2\training'
        self.img_detection_pair_list = get_img_detection_pairs(train_detection_data_path, train_img_data_path)
        self.cache = {}

    def __getitem__(self, idx):
        if idx not in self.cache:
            img_file_path, det_file_path = self.img_detection_pair_list[idx]
            img = torchvision.io.read_image(img_file_path)
            detections = self.parse_txt_to_detection(det_file_path)
            self.cache[idx] = (img, detections)
        return self.cache[idx]

    def __len__(self):
        return len(self.img_detection_pair_list)

    def parse_txt_to_detection(self, txt_file_path):
        det_file = open(txt_file_path, 'r')
        obj_list = []
        while True:
            obj = det_file.readline()

            if obj == '':
                break
            else:
                obj = obj.split()
                obj = label_parser.parse_kitti_object(obj)
                obj_list.append(obj)

        return torch.tensor(obj_list)


class DetectionLabelDataset(DetectionDataset):
    
    def __init__(self, img_size, anchors, get_detection_labels) -> None:
        super().__init__()
        self.anchors_list = anchors
        self.img_size = img_size
        self.get_detection_labels = get_detection_labels

        
    def __getitem__(self, idx):
        if idx not in self.cache:
            sample = super().__getitem__(idx)
            img, label = sample
            _, h, w = img.size()
            scale_h = self.img_size[0] / h
            scale_w = self.img_size[1] / w
            scale = torch.tensor([1, scale_w, scale_h, scale_w, scale_h])
            label = label * scale
            img = torchvision.transforms.Resize(self.img_size)(img)
            labels = []
            for anchor in self.anchors_list[3:9]:
                label_ = self.get_detection_labels(label, anchor)
                labels.append(label_)

            labels = torch.concat(labels, dim=0)
            self.cache[idx] = (img, labels)
        return self.cache[idx]


        