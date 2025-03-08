"""
ResNet-50 adversarially robust
https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/models/pytorch/adversarially_robust/robust_models.py
https://huggingface.co/madrylab/robust-imagenet-models
"""
from torch import nn
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils, resnet_utils

__all__ = ['resnet50_l2_eps0', 'resnet50_l2_eps0_01', 'resnet50_l2_eps0_03',
           'resnet50_l2_eps0_05', 'resnet50_l2_eps0_1', 'resnet50_l2_eps0_25',
           'resnet50_l2_eps0_5', 'resnet50_l2_eps1', 'resnet50_l2_eps3', 'resnet50_l2_eps5']

model_urls = {"resnet50_l2_eps0": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.ckpt",
              "resnet50_l2_eps0_01": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.01.ckpt",
              "resnet50_l2_eps0_03": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.03.ckpt",
              "resnet50_l2_eps0_05": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.05.ckpt",
              "resnet50_l2_eps0_1": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.1.ckpt",
              "resnet50_l2_eps0_25": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.25.ckpt",
              "resnet50_l2_eps0_5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.5.ckpt",
              "resnet50_l2_eps1": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps1.ckpt",
              "resnet50_l2_eps3": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps3.ckpt",
              "resnet50_l2_eps5": "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps5.ckpt"}

def get_model(**kwargs):
    model = ResNet50Robust(**kwargs)
    return model

class ResNet50Robust(BaseModel):
    """
    class for the model ResNet50 trained on ImageNet with adversarial robustness
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ResNet50-Robust does not need normalized inputs
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        self.transforms = transform_utils.get_transforms(
            mean=None, std=None, resize=(224, 224)
        )
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = ['layer[0-9]$', r'layer[0-9]\.[0-9]$', '.*downsample']
        # Layer types to ignore
        irrelevant_layer_types = [nn.Dropout]
        self.irrelevant_layer_names = layer_utils.get_layers_matching_regexes(
            self.model, irrelevant_layer_regexes
        ) + layer_utils.get_layers_matching_types(self.model, irrelevant_layer_types)
        self.relevant_layer_names = [
            layer for layer in self.all_layer_names
            if layer not in self.irrelevant_layer_names
        ]

    def _load_model(self):
        """
        Load ResNet50
        """
        self.DEFAULT_EPS = 3
        self.eps = int(self.kwargs.get('eps', self.DEFAULT_EPS))
        if self.eps != self.DEFAULT_EPS:
            raise ValueError(f"Only epsilon = {self.DEFAULT_EPS} is implemented for ResNet50-Robust")
            # Steps to update:
            # Select appropriate class
            # Take eps kwargs as float, not int
            # Also update model name based on epsilon
        model = resnet50_l2_eps3()
        resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = 'resnet50robustl2eps3'
        return model, model_name

def _model(arch, model_fn, pretrained, progress, use_data_parallel, **kwargs):

    model = model_fn(weights=None)

    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls[arch], progress=progress)
        sd = {k[len('module.model.'):]:v for k,v in checkpoint['model'].items()\
              if k[:len('module.model.')] == 'module.model.'}  # Consider only the model and not normalizers or attacker
        model.load_state_dict(sd)

    model = torch.nn.DataParallel(model) if use_data_parallel else model

    return model


def resnet50_l2_eps0(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0  L2-robustness on ImageNet. Accuracy: 75.80. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_01(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.01  L2-robustness on ImageNet. Accuracy: 75.68. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_01', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_03(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.03  L2-robustness on ImageNet. Accuracy: 75.76. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_03', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_05(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.05  L2-robustness on ImageNet. Accuracy: 75.59. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_05', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_1(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.1  L2-robustness on ImageNet. Accuracy: 74.78. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_1', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_25(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.25  L2-robustness on ImageNet. Accuracy: 74.14. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_25', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps0_5(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 0.5  L2-robustness on ImageNet. Accuracy: 73.16. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps0_5', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps1(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 1.0  L2-robustness on ImageNet. Accuracy: 70.43. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps1', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps3(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 3.0  L2-robustness on ImageNet. Accuracy: 62.83. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps3', resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_l2_eps5(pretrained=True, progress=True, use_data_parallel=False, **kwargs):
    r""" Resnet50 with epsilon 5.0  L2-robustness on ImageNet. Accuracy: 56.13. 
    By: https://github.com/microsoft/robust-models-transfer
    """
    return _model('resnet50_l2_eps5', resnet50, pretrained, progress, use_data_parallel, **kwargs)
