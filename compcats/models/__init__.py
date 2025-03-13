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
from . import resnet50_ecoset

# ResNet-101s
from . import resnet101_imagenet1kv1
from . import resnet101_openaiclip

# VGGs
from . import vgg16_imagenet1kv1
from . import vgg19_imagenet1kv1

# BagNets
from . import bagnet9
from . import bagnet17
from . import bagnet33

# CORNets
from . import cornet_s
from . import cornet_z

# ViTs
# ViT-B/32
from . import vitb32_openaiclip
from . import vitb32_imagenet1kv1
from . import vitb32_random
from . import vitb32_dreamsim
# ViT-B/16
from . import vitb16_openaiclip
from . import vitb16_imagenet21k
from . import vitb16_siglip2
from . import vitb16_eva02
from . import vitb16_synclr
# ViT-B/14
from . import vitb14_dinov2
from . import vitb14_dinov2register
# ViT-L/14
from . import vitl14_openaiclip
from . import vitl14_kosmos2
