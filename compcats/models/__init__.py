"""
This module initializes and imports various neural network models.
"""

# AlexNets
from . import alexnet_imagenet1kv1
from . import alexnet_barlowtwins
from . import alexnet_ipcl
from . import alexnet_random

# ResNet-18s
from . import resnet18_imagenet1kv1
from . import resnet18_faceobfuscation
from . import resnet18_toponet

# ResNet-50s
from . import resnet50_random
from . import resnet50_imagenet1kv2
from . import resnet50_faceobfuscation
from . import resnet50_toponet
from . import resnet50_imagenet21k
from . import resnet50_openaiclip
from . import resnet50_simclr
from . import resnet50_robust
from . import resnet50_sin
from . import resnet50_sinandin
from . import resnet50_sinthenin

# ResNet-101s
from . import resnet101_imagenet1kv1

# VGGs
from . import vgg16_imagenet1kv1
from . import vgg19_imagenet1kv1

# BagNets
from . import bagnet9
from . import bagnet17
from . import bagnet33

# ViTs
# ViT-B/32
from . import vitb32_openaiclip
from . import vitb32_imagenet1kv1
from . import vitb32_random
# ViT-B/16
from . import vitb16_openaiclip
from . import vitb16_imagenet21k
from . import vitb16_siglip2
# ViT-L/14
from . import vitl14_openaiclip
