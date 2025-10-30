import torch
from torch import nn
import torch.nn.functional as F

class BaseConvBlock(nn.Module):
    def __init__(self, in_chanels, out_chanels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chanels, out_chanels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
        
class Inception_Block(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super().__init__()

        # 1x1 conv branch
        self.branch1 = nn.BaseConvBlock(in_channels, ch1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            self.branch2_1 = nn.BaseConvBlock(in_channels, ch3x3reduce, kernel_size=1)
            self.branch2_2 = nn.BaseConvBlock(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 conv -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            self.branch3_1 = nn.Conv2d(in_channels, ch5x5reduce, kernel_size=1)
            self.branch3_2 = nn.Conv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2)
        )

        # 3x3 maxpool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            self.branch4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.branch4_2 = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, features: torch.Tensor):
        branch1 = self.branch1(features)
        branch2 = self.branch2(features)
        branch3 = self.branch3(features)
        branch4 = self.branch4(features)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.conv_1 = nn.BaseConvBlock(
            in_channels = 3,
            out_channels = 64,
            kernel_size = 7
            stride = 2
            padding = 3
        )

        self.pooling_1 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            ceil_mode = True
        )
        
        # Second convolutional layer
        self.conv_2_1 = nn.BaseConvBlock(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 1
        )
        
        self.conv_2_2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 192,
            kernel_size = 3,
            padding = 1
        )

        self.pooling_2 = nn.MaxPool2d(
            kernel_size = 3,
            padding = 2,
            ceil_mode = True
        )

        # Third convolutional layer
        self.inteption_3a = Inception_Block(192,64,96,128,16,32,32)
        self.inteption_3b = Inception_Block(256,128,128,192,32,96,64)

        self.pooling_3 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            ceil_mode = True
        )

        # Fourth convolutional layer
        self.inteption_4a = Inception_Block(480, 192, 96, 208, 16, 48, 64)
        self.inteption_4b = Inception_Block(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception_Block(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception_Block(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception_Block(528, 256, 160, 320, 32, 128, 128)

        self.pooling_4 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            ceil_mode = True
        )

        # Fifth convolutional layer
        self.inception_5a = Inception_Block(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception_Block(832, 384, 192, 384, 48, 128, 128)

        self.pooling_5 = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=21)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor):

        x = self.conv_1(feature)
        x = self.pooling_1(x)

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.pooling_2(x)

        x = self.inteption_3a(x)
        x = self.inteption_3b(x)
        x = self.pooling_3(x)

        x = self.inteption_4a(x)
        x = self.inteption_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.pooling_4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.pooling_5(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)

        return x

