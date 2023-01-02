import torch
import torchvision
import torch
from data_preprocess import get_img_segment_pairs, get_img_detection_pairs


def convert_corners_to_xywh(left, top, right, bottom):
    x = (left + right) / 2.0
    y = (top + bottom) / 2.0
    w = right - left
    h = bottom - top

    return x,y,w,h

def parse_kitti_object(obj):
    cls_type = obj[0]
    left, top, right, bottom = list(map(float,obj[4:8]))
    x, y, w, h = convert_corners_to_xywh(left, top, right, bottom)
    return (cls_type, x, y, w, h)

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
        self.cached = {}

    def __getitem__(self, idx):
        if idx not in self.cached:
            img_file_path, det_file_path = self.img_detection_pair_list[idx]
            img = torchvision.io.read_image(img_file_path)
            detections = self.parse_txt_to_detection(det_file_path)

            self.cached[idx] = (img, detections)

        return self.cached[idx]
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
                obj = parse_kitti_object(obj)
                obj_list.append(obj)

        return obj_list

data = DetectionDataset()