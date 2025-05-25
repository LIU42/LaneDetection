import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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


class LaneDetectionModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, bias=False)
        self.upsample3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, bias=False)

        self.classifier = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, bias=False)

    def forward(self, inputs):
        feature2, feature3, output4 = self.backbone(inputs)

        output3 = self.upsample3(output4)
        output3 = torch.cat([output3, feature3], dim=1)

        output3 = self.conv3(output3)
        output3 = self.bn3(output3)
        output3 = self.relu3(output3)

        output2 = self.upsample2(output3)
        output2 = torch.cat([output2, feature2], dim=1)

        output2 = self.conv2(output2)
        output2 = self.bn2(output2)
        output2 = self.relu2(output2)

        return self.classifier(output2)
