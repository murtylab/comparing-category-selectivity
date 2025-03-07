"""
ResNet-50 trained on SIN
https://github.com/rgeirhos/texture-vs-shape/tree/master
"""
from torch import nn

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils, sin_utils, resnet_utils

def get_model(**kwargs):
    model = ResNet50SIN(**kwargs)
    return model

class ResNet50SIN(BaseModel):
    """
    class for the model ResNet50 trained on Stylized-ImageNet
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
        Load ResNet50
        """
        model = sin_utils.load_model('resnet50_trained_on_SIN')
        model = model.module # Get the Sequential model
        resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = 'resnet50sin'
        return model, model_name
