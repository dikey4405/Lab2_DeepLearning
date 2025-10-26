import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            int_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut layer to match input and output dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,1),
                    stride=1
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def fprward(self, features: torch.Tensor) -> torch.Tensor:
        features = 
        return out
        