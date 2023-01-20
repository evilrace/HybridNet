import torch

# it implemented SSD(single shot detector)

class HeadLayer(torch.nn.Module):
    def __init__(self, channels, anchor_nums, class_num) -> None:
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels*2, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(channels*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels*2, anchor_nums*(class_num+4), kernel_size=1, padding='same')
        )

    def forward(self, x):
        output = self.layer(x)
        return output


def conv_block(channels):
    block = torch.nn.Sequential(
        torch.nn.BatchNorm2d(channels),
        torch.nn.Conv2d(channels, channels, 3, padding='same'),
        torch.nn.BatchNorm2d(channels),
        torch.nn.MaxPool2d(2, 2, ceil_mode=True)
    )
    return block
class SSD(torch.nn.Module):
    def __init__(self, channels, anchor_nums, class_nums) -> None:
        super().__init__()
        self.channels = channels
        self.anchor_nums = anchor_nums
        self.class_nums = class_nums + 1
        self.head_layers = torch.nn.ModuleList([HeadLayer(self.channels, self.anchor_nums, self.class_nums) for _ in range(3)])
        self.conv_blocks = torch.nn.ModuleList([conv_block(channels) for _ in range(3)])
        self.head_layers2 = torch.nn.ModuleList([HeadLayer(self.channels, self.anchor_nums, self.class_nums) for _ in range(3)])

    def forward(self, feature_pyramid):
        output = [self.head_layers[idx](feat) for idx, feat in enumerate(feature_pyramid[1:])]
        x = feature_pyramid[-1]
        for header_layer, conv_block in zip(self.head_layers2, self.conv_blocks):
            x = conv_block(x)
            output = output + [header_layer(x)]
        return output
        