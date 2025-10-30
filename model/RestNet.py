import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut layer to match input and output dimensions
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            

    def forward(self, features: torch.Tensor):
        identity = features
        output = self.conv1(features)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)   
        output += self.shortcut(identity)
        output  = F.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=21):

        super(ResNet, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.pooling_1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.in_channels = 64
        self.res_block_1 = self._make_layer(
            block,
            out_channels=128,
            num_blocks=num_blocks[0],
            stride=1
        )

        self.res_block_2 = self._make_layer(
            block,
            out_channels=256,
            num_blocks=num_blocks[1],
            stride=2
        )

        self.res_block_3 = self._make_layer(
            block,
            out_channels=512,
            num_blocks=num_blocks[2],
            stride=2
        )

        self.res_block_4 = self._make_layer(
            block,
            out_channels=512,
            num_blocks=num_blocks[3],
            stride=2
        )

        self.pooling_2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            in_features=512 * block.expansion,
            out_features=21
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor):
        output = self.conv_1(features)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.pooling_1(output)

        output = self.res_block_1(output)
        output = self.res_block_2(output)
        output = self.res_block_3(output)
        output = self.res_block_4(output)

        output = self.pooling_2(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output




        
