"""
This module initializes and imports various neural network models.
"""

##########################
# Evaluation
##########################

from .evaluate.imagenet1k import evaluate as imagenet1k_evaluataion


##########################
# Pretrained models
##########################

# AlexNets
from .pretrained.alexnet import alexnet_imagenet1kv1
from .pretrained.alexnet import alexnet_barlowtwins
from .pretrained.alexnet import alexnet_ipcl
from .pretrained.alexnet import alexnet_random

# ResNet-18s
from .pretrained.resnet import resnet18_imagenet1kv1
from .pretrained.resnet import resnet18_faceobfuscation
from .pretrained.resnet import resnet18_toponet
from .pretrained.resnet import resnet18_tdannsimclr
from .pretrained.resnet import resnet18_tdannsupervised

# ResNet-50s
from .pretrained.resnet import resnet50_random
from .pretrained.resnet import resnet50_imagenet1kv2
from .pretrained.resnet import resnet50_faceobfuscation
from .pretrained.resnet import resnet50_toponet
from .pretrained.resnet import resnet50_imagenet21k
from .pretrained.resnet import resnet50_openaiclip
from .pretrained.resnet import resnet50_simclr
from .pretrained.resnet import resnet50_robust
from .pretrained.resnet import resnet50_sin
from .pretrained.resnet import resnet50_sinandin
from .pretrained.resnet import resnet50_sinthenin
from .pretrained.resnet import resnet50_ecoset
from .pretrained.resnet import resnet50_cessl

# ResNet-101s
from .pretrained.resnet import resnet101_imagenet1kv1
from .pretrained.resnet import resnet101_openaiclip

# VGGs
from .pretrained.vgg import vgg16_imagenet1kv1
from .pretrained.vgg import vgg19_imagenet1kv1

# BagNets
from .pretrained.bagnet import bagnet9
from .pretrained.bagnet import bagnet17
from .pretrained.bagnet import bagnet33

# CORNets
from .pretrained.cornet import cornet_s
from .pretrained.cornet import cornet_z

# ViTs
# ViT-B/32
from .pretrained.vit import vitb32_openaiclip
from .pretrained.vit import vitb32_imagenet1kv1
from .pretrained.vit import vitb32_random
from .pretrained.vit import vitb32_dreamsim
# ViT-B/16
from .pretrained.vit import vitb16_openaiclip
from .pretrained.vit import vitb16_imagenet21k
from .pretrained.vit import vitb16_siglip2
from .pretrained.vit import vitb16_eva02
from .pretrained.vit import vitb16_synclr
# ViT-B/14
from .pretrained.vit import vitb14_dinov2
from .pretrained.vit import vitb14_dinov2register
# ViT-L/14
from .pretrained.vit import vitl14_openaiclip
from .pretrained.vit import vitl14_kosmos2
