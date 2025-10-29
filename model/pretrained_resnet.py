import torch
from torch import nn
from transformers import ResNetForImageClassification


class PretrainedResnet(nn.Module):
    """Wrapper around transformers' ResNet for fine-tuning.

    Args:
        num_classes (int): number of output classes.
        freeze_backbone (bool): if True, freeze ResNet backbone parameters.
    """

    def __init__(self, num_classes: int = 21, freeze_backbone: bool = False):
        super().__init__()

        basemodel = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        # The huggingface ResNet model contains a `resnet` attribute (backbone)
        self.resnet = basemodel.resnet

        # Optionally freeze backbone weights for initial fine-tuning of classifier
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

        # Replace classifier with a dropout + linear layer for our num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=num_classes, bias=True),
        )

    def forward(self, images: torch.Tensor):
        # Run through backbone; huggingface ResNet returns an object with pooler_output
        features = self.resnet(images).pooler_output
        # pooler_output shape is (B, C, 1, 1) for some implementations; squeeze to (B, C)
        features = features.squeeze(-1).squeeze(-1)
        logits = self.classifier(features)

        return logits

    def freeze_backbone(self):
        """Freeze backbone parameters (set requires_grad = False)."""
        for p in self.resnet.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (set requires_grad = True)."""
        for p in self.resnet.parameters():
            p.requires_grad = True
