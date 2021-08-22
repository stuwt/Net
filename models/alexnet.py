#alexnet from "ImageNet Classification with Deep Convolutional Neural Networks"
#<https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>

import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
        #卷积层
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), 
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=5), #section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),
      
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
             nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(),
            #全连接层
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), #section4.2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        self.init_weights()
        self.init_bias()


    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.zeros_(layer.bias)
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 1)
        #2th, 4th, 5th convolutional layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)
  
    def init_weights(self):
        for layer in self.net:
        if type(layer)==nn.Linear or type(layer)==nn.Conv2d:
            nn.init.normal_(layer.weight, 0, 0.01)

