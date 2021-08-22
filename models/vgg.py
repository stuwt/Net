# VGG 11-layer model from 
# "Very Deep Convlutional Networks For Large-Scale Image Recongnition"
# <https://arxiv.org/pdf/1409.1556.pdf>


import torch 
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

class VGG11(nn.Module):
    def __init__(self, num_classes=1000, is_init_weights=True):
        super().__init__()
        
        
        self.convnet = nn.Sequential(
            *self.conv_blocks()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
            nn.Linear(4096, num_classes)
        )
        if is_init_weights:
            self.init_weights()
    
    def forward(self, X):
        X = convnet(X)
        X = nn.Flatten()
        X = classifier(X)
        return X
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def conv_blocks(self):
        conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
        conv_blks = []
        in_channels = 3 #RGB channel
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return conv_blks
            