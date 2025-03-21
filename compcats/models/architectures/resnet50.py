"""
ResNet-50
"""
from torch import nn

from ..basearchitecture import BaseArchitecture
from ..utils import transform_utils, layer_utils
from .utils import resnet_architecture_utils

def get_architecture(**kwargs):
    """
    Get ResNet50 architecture
    """
    architecture = ResNet50Architecture(**kwargs)
    return architecture

class ResNet50Architecture(BaseArchitecture):
    """
    class for the architecture ResNet50
    """
    def __init__(self, **kwargs):
        # Call parent constructor
        super().__init__(**kwargs)
        # Usage Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms()
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            'conv1$', 'bn1$', 'relu$', 'maxpool$', \
            'layer[0-9]$', r'layer[0-9]\.[0-9]$', '.*downsample', 'fc.*'
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

    def _setup_model(self):
        """
        Initialize ResNet50
        """
        self._store_rng_states()
        backbone_name = 'resnet50'
        model_name = self._set_model_seed(backbone_name)
        model = resnet_architecture_utils.get_resnet50_architecture(**self.kwargs)
        self._recover_rng_states()
        return model, model_name
