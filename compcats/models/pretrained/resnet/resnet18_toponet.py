"""
ResNet-18 TopoNet
https://github.com/toponets/toponets
"""
from torch import nn

try:
    import toponets
except ImportError:
    # These libraries are optional, silently handle import error
    pass

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, resnet_utils

def get_model(**kwargs):
    model = ResNet18TopoNet(**kwargs)
    return model

class ResNet18TopoNet(BaseModel):
    """
    class for the model ResNet18 trained on ImageNet
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Assuming standard ImageNet normalization and transforms
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms()
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            'conv1$', 'bn1$', 'relu$', 'maxpool$', r'.*conv[0-9]\.conv$', \
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

    def _load_model(self):
        """
        Load ResNet18 Toponet trained on ImageNet
        """
        # Default tau value = 10
        self.DEFAULT_TAU = 10
        # Forcing integer value
        self.tau = int(self.kwargs.get('tau', self.DEFAULT_TAU))
        model = toponets.resnet18(tau=self.tau, checkpoint_path = f"resnet18_tau_{self.tau}.pt")
        resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = f'resnet18toponettau{self.tau}'
        return model, model_name
