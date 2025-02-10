import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)

        feature1 = self.layer1(outputs)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)

        return feature2, feature3, feature4


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        return outputs


class UpsampleConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, inputs, features):
        return torch.cat([self.upsample(inputs), features], dim=1)


class LaneNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)

        self.conv3 = ConvBlock(in_channels=512, out_channels=256)
        self.conv2 = ConvBlock(in_channels=256, out_channels=128)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        self.upsample3 = UpsampleConcatBlock(in_channels=512, out_channels=256)
        self.upsample2 = UpsampleConcatBlock(in_channels=256, out_channels=128)

    def forward(self, inputs):
        feature2, feature3, output4 = self.backbone(inputs)

        output3 = self.upsample3(output4, feature3)
        output3 = self.conv3(output3)
        output2 = self.upsample2(output3, feature2)
        output2 = self.conv2(output2)

        return self.conv1(output2)
