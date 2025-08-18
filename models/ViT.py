import torch.nn as nn
import timm
from .ResNet import NormalizeByChannelMeanStd

class vit_s(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # instantiate timm ViT
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=32,
            patch_size=4
        )
        # your normalization module
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )

    def forward(self, x):
        x = self.normalize(x)      # normalize first
        return self.vit(x)         # then ViT forward

