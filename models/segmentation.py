import torch

def conv_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )
    return block

def transpose_conv_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )
    return block

class Segmentation(torch.nn.Module):
    def __init__(self, channel=512) -> None:
        super().__init__()
        transpose_layer_num = 5
        self.channel = channel
        self.transpose_layers = torch.nn.ModuleList([transpose_conv_block(self.channel, self.channel) for _ in range(transpose_layer_num)])
        self.conv_layers = torch.nn.ModuleList([conv_block(self.channel * 2, self.channel) for _ in range(3)])

        self.output_layer = torch.nn.Conv2d(self.channel, 1, 3, padding='same')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x:list):
        x.reverse()
        for idx in range(len(x)-1):
            x[idx] = self.transpose_layers[idx](x[idx])
            pad_y = x[idx].size()[-2] - x[idx+1].size()[-2]
            pad_x = x[idx].size()[-1] - x[idx+1].size()[-1]
            x[idx+1] = torch.nn.functional.pad(x[idx+1], (0,pad_x,0,pad_y))
            # x[idx] = x[idx+1][:,:,:x[idx].size()[-2],:x[idx].size()[-1]]
            x_concat = torch.concatenate([x[idx], x[idx+1]], dim=1)
            x[idx+1] = self.conv_layers[idx](x_concat)

        feature = x[-1]
        feature = self.transpose_layers[-2](feature)
        feature = self.transpose_layers[-1](feature)
        output = self.output_layer(feature)
        output = self.sigmoid(output)
        return output