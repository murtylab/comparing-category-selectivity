"""
ResNet-18
https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
A Study of Face Obfuscation in ImageNet
Yang, K., Yau, J. H., Fei-Fei, L., Deng, J., & Russakovsky, O. (2022, June). A study of face obfuscation in imagenet. In International Conference on Machine Learning (pp. 25313-25330). PMLR.
https://huggingface.co/kaiyuy/imagenet-blur-pretrained-models/tree/main
https://github.com/princetonvisualai/imagenet-face-obfuscation/tree/main
"""
from torch import nn
from collections import OrderedDict
import torch
import torchvision
from huggingface_hub import hf_hub_download

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils, resnet_utils

def get_model(**kwargs):
    model = ResNet18FaceObfuscation(**kwargs)
    return model

class ResNet18FaceObfuscation(BaseModel):
    """
    class for the model ResNet18 trained on ImageNet
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize=(224, 224)
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
        Load ResNet18
        """
        self.checkpoints_dir = hf_hub_download(repo_id="kaiyuy/imagenet-blur-pretrained-models", filename="resnet18_blurred_ILSVRC.pth")
        checkpoint = torch.load(self.checkpoints_dir)
        state_dict = OrderedDict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        model = torchvision.models.resnet18(weights=None)
        model.load_state_dict(state_dict)
        resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = 'resnet18faceobfuscation'
        return model, model_name
