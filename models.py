import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relu = nn.ReLU(True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        return outputs


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        return inputs + self.conv2(self.conv1(inputs))


class CSPBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = channels // 2

        self.conv1 = ConvBlock(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels, mid_channels, kernel_size=1, padding=0)

        self.block1 = ResBlock(mid_channels)
        self.block2 = ResBlock(mid_channels)

        self.conv3 = ConvBlock(channels, channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        output2 = self.conv2(inputs)

        output1 = self.block1(output1)
        output1 = self.block2(output1)

        return self.conv3(torch.cat([output1, output2], dim=1))


class CSPDarknetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)

        self.layer1 = nn.Sequential(
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            CSPBlock(channels=32),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            CSPBlock(channels=64),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            CSPBlock(channels=128),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            CSPBlock(channels=256),
        )

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        output2 = self.layer1(output1)
        output3 = self.layer2(output2)
        output4 = self.layer3(output3)
        output5 = self.layer4(output4)
        return output5


class LaneNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(in_channels=256, out_channels=4, kernel_size=1, padding=0)

    def forward(self, inputs):
        return self.conv(inputs).flatten(1)


class LaneHead(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=16, out_channels=81, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(in_features=960, out_features=768)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = self.dropout(inputs)
        outputs = self.fc(outputs)
        outputs = self.relu(outputs)
        return self.conv(outputs.reshape(-1, 16, 4, 12)).permute(0, 2, 3, 1)


class LaneNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.backbone = CSPDarknetBackbone()
        self.neck = LaneNeck()
        self.head = LaneHead(dropout=dropout)

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        outputs = self.neck(outputs)
        outputs = self.head(outputs)
        return outputs
