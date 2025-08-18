from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from .ViT import *

model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "wide_resnet14_2": wide_resnet14_2,
    "wide_resnet50_2": wide_resnet50_2,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn, # vgg16 with batch normalization
    "vgg16_bn_lth": vgg16_bn_lth, # vgg16 with bn and Lottery Ticket Hypothesis
    "vit_s": vit_s, # ViT small
}
