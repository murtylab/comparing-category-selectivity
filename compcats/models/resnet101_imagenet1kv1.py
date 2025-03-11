"""
ResNet-101 trained on ImageNet-1K
https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
"""
from torch import nn
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils, resnet_utils

def get_model(**kwargs):
    model = ResNet101ImageNet1kv1(**kwargs)
    return model

class ResNet101ImageNet1kv1(BaseModel):
    """
    class for the model ResNet101 trained on ImageNet-1K
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms()
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = ['layer[0-9]$', r'layer[0-9]\.[0-9]{1,2}$', '.*downsample']
        irrelevant_layer_regexes = [
            'conv1$', 'bn1$', 'relu$', 'maxpool$', \
            'layer[0-9]$', r'layer[0-9]\.[0-9]{1,2}$', '.*downsample', \
            'fc$'
        ]
        # Layer types to ignore
        irrelevant_layer_types = [nn.Dropout, nn.Identity]
        self.irrelevant_layer_names = layer_utils.get_layers_matching_regexes(
            self.model, irrelevant_layer_regexes
        ) + layer_utils.get_layers_matching_types(self.model, irrelevant_layer_types)
        self.relevant_layer_names = [
            layer for layer in self.all_layer_names
            if layer not in self.irrelevant_layer_names
        ]

    def _load_model(self):
        """
        Load ResNet101
        """
        # IMAGENET1K_V1
        model = torchvision.models.resnet101(weights='IMAGENET1K_V1')
        resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = 'resnet101imagenet1kv1'
        return model, model_name
