# GoogleNet model from
# "Go Deeper with convlutions "  <https://arxiv.org/abs/1409.4842>

import torch
from torch import nn
from torch.nn import functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        #c1~c4是每条路径输出的通道数
        super(Inception, self).__init__(**kwargs)
        #
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        p = torch.cat((p1, p2, p3, p4), dim=1) #在通道维度连结输出



class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, is_init_weights=True):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            #nn.LocalResponseNorm()
        )
        
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),  # 64 + 128 + 32 + 32 = 256
            Inception(256, 128, (128, 192), (32, 96), 64), # 128 + 192 + 96 + 64 = 480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64), #512
            Inception(512, 160, (112, 224), (24, 64), 64), #512
            Inception(512, 128, (128, 256), (24, 64), 64), #512
            Inception(512, 112, (114, 288), (32, 64), 64), #528
            Inception(528, 256, (160, 320), (32, 128), 128) # 832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128) # 832
            Inception(832, 384, (192, 384), (48, 128), 128) #1024
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = torch.flatten(x, dim=1)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x



#打印网络结构
def print_net(net):
    x = torch.randn(1, 3, 224, 224)
    for layer in net:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape:  ', x.shape)
