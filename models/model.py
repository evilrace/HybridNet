import torch
from .segmentation import Segmentation
from .backbone import FPN
from .detection import SSD
class HybridNet(torch.nn.Module):
    def __init__(self, channels, anchor_nums, class_nums) -> None:
        super().__init__()
        self.detection = SSD(channels, anchor_nums, class_nums)
        self.fpn = FPN(channels)
        self.segmentation = Segmentation(channels)


    def forward(self, x):
        features = self.fpn(x)
        # print(features[0].size())
        # print(features[1].size())
        # print(features[2].size())
        # print(features[3].size())
        detection_output = self.detection(features)
        segmentation_output = self.segmentation(features)
        return segmentation_output, detection_output
