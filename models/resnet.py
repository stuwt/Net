# ResNet-18 model from
# "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_channels, use_1x1conv=False, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if conv3:
            identity = conv3(x)
        out += identity
        out = F.relu(out)
        return out
        

class ResNet(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1 = make_block(64, 64, 2, first_block=True)
        self.b2 = make_block(64, 128, 2)
        self.b3 = make_block(128, 256, 2)
        self.b4 = make_block(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 10)

    
    def make_block(self, in_channels, num_channels, num_residuals, first_block=False):
        blocks = []
        for i in range(num_residuals):
            if i==0 and not first_block:
                blocks.append(ResidualBlock(in_channels, num_channels, True, 2))
            else:
                blocks.append(ResidualBlock(num_channels, num_channels)) 
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
