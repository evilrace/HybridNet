import torch
import math


class Anchor:
    def __init__(self, img_size, aspect_ratios = [1/3, 1/2, 1, 2, 3], scales = [1/3, 1, 2, 3]) -> None:

        self.img_size = img_size
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.anchor_nums = len(aspect_ratios) * len(scales)
        self.anchors_list = self.generate_anchors()
        
    def generate_anchors_wh(self, img_size, aspect_ratios, scales):
        h, w = img_size
        level = 1
        anchors_list = []
        while level <= h and level <= w:
            area = level * level
            anchors =[]
            for s in scales:
                for aspect_ratio in aspect_ratios:
                    anchor_h = math.sqrt(area * aspect_ratio) * s
                    anchor_w = anchor_h / aspect_ratio
                    anchors.append((anchor_w,anchor_h))
            anchors_list.append(anchors)
            level = level * 2

        return anchors_list

    def generate_anchors(self):
        h, w = self.img_size

        level = 1
        anchors = self.generate_anchors_wh(self.img_size, self.aspect_ratios, self.scales)

        anchors_list = []
        idx = 0
        while level <= h and level <= w:
            grid_y = (torch.arange(h / level) + 0.5) * level
            grid_x = (torch.arange(w / level) + 0.5) * level
            grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            anchor = torch.tensor(anchors[idx])
            grid_xy = grid_xy.unsqueeze(-2)
            grid_xy = grid_xy.tile((1,1,anchor.size()[0],1))
            anchor = torch.tile(anchor, (grid_xy.size()[0], grid_xy.size()[1], 1, 1))
            anchor = torch.concat([grid_xy, anchor], dim=-1)
            level = level * 2
            idx += 1
            anchors_list.append(anchor)
        return anchors_list