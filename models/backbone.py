import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

def conv_block(in_channel, out_channel):
    block = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(in_channel),
        torch.nn.Conv2d(in_channel, out_channel, 3, padding='same'),
        torch.nn.BatchNorm2d(out_channel),
        torch.nn.ReLU(),
    )
    return block

class downscale_layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels * 2, out_channels, 3, padding='same')
        self.pooling = torch.nn.MaxPool2d(2,2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, x_down):
        x = self.pooling(x)
        x = torch.concatenate([x, x_down], dim = 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class FPN(torch.nn.Module):
    def __init__(self, channels = 512) -> None:
        super().__init__()
        self.channels = channels
        self.resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        resnet_layer_names = ["layer1.2.conv3", "layer2.3.conv3", "layer3.5.conv3", "layer4.2.conv3"]
        self.return_nodes = { layer_name : f'layer{idx}' for idx, layer_name in enumerate(resnet_layer_names)
        }
        resnet_out_channels = [self.resnet.layer1[2].conv3.out_channels, self.resnet.layer2[3].conv3.out_channels, self.resnet.layer3[5].conv3.out_channels, self.resnet.layer4[2].conv3.out_channels]
        self.resnet = create_feature_extractor(self.resnet, self.return_nodes)

        self.encoder = torch.nn.ModuleList([conv_block(in_channels, self.channels) for in_channels in resnet_out_channels])
        self.downscale_layers = torch.nn.ModuleList([downscale_layer(self.channels, self.channels) for _ in range(len(self.encoder)-1)])
        


    def forward(self, x):
        with torch.no_grad():
            feature = self.resnet(x)
        self.encoded_features = [self.encoder[idx](feature[layer_name]) for idx, layer_name in enumerate(self.return_nodes.values())]
        feautre_pyramid = [downscale_layer(self.encoded_features[idx], self.encoded_features[idx+1])  for idx, downscale_layer in enumerate(self.downscale_layers)]
        feautre_pyramid = [self.encoded_features[0]] + feautre_pyramid
        return feautre_pyramid


fpn = FPN()
fpn = fpn.to('cuda')
sample = torch.rand((1,3,1024,1024))
sample = sample.to('cuda')
output = fpn(sample)

for out in output:
    print(out.size())