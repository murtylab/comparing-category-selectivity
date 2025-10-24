"""
This module initializes and imports various pretrained neural network models.
"""

##########################
# Pretrained models
##########################

# AlexNets
from .alexnet import alexnet_imagenet1kv1
from .alexnet import alexnet_barlowtwins
from .alexnet import alexnet_ipcl
from .alexnet import alexnet_random
from .alexnet import alexnet_mosaic

# ResNet-18s
from .resnet import resnet18_imagenet1kv1
from .resnet import resnet18_faceobfuscation
from .resnet import resnet18_toponet
from .resnet import resnet18_tdannsimclr
from .resnet import resnet18_tdannsupervised
from .resnet import resnet18_mosaic
from .resnet import resnet18_random

# ResNet-50s
from .resnet import resnet50_random
from .resnet import resnet50_imagenet1kv2
from .resnet import resnet50_faceobfuscation
from .resnet import resnet50_toponet
from .resnet import resnet50_imagenet21k
from .resnet import resnet50_openaiclip
from .resnet import resnet50_simclr
from .resnet import resnet50_robust
from .resnet import resnet50_sin
from .resnet import resnet50_sinandin
from .resnet import resnet50_sinthenin
from .resnet import resnet50_ecoset
from .resnet import resnet50_cessl
from .resnet import resnet50_blur

# ResNet-101s
from .resnet import resnet101_imagenet1kv1
from .resnet import resnet101_openaiclip

# VGGs
from .vgg import vgg16_imagenet1kv1
from .vgg import vgg19_imagenet1kv1

# BagNets
from .bagnet import bagnet9
from .bagnet import bagnet17
from .bagnet import bagnet33

# SqueezeNets
from .squeezenet import squeezenet_mosaic

# CORNets
from .cornet import cornet_s
from .cornet import cornet_z

# ConvNeXts
from .convnext import convnext_imagenet1kv1

# ViTs
# ViT-B/32
from .vit import vitb32_openaiclip
from .vit import vitb32_imagenet1kv1
from .vit import vitb32_random
from .vit import vitb32_dreamsim
# ViT-B/16
from .vit import vitb16_openaiclip
from .vit import vitb16_imagenet21k
from .vit import vitb16_siglip2
from .vit import vitb16_eva02
from .vit import vitb16_synclr
# ViT-B/14
from .vit import vitb14_dinov2
from .vit import vitb14_dinov2register
# ViT-L/14
from .vit import vitl14_openaiclip
from .vit import vitl14_kosmos2

# Misc
from .efficientnet import efficientnet_mnasnet

# Swin Transformers
from .swint import swint_mosaic

# Neuromap
from .neuromap import neuromap1_singlesubject
from .neuromap import neuromap1_multisubject
