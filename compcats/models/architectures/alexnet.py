"""
AlexNet
"""
import torch
from torch import nn

from compcats import constants

from ..basearchitecture import BaseArchitecture
from ..utils import transform_utils, layer_utils

def get_architecture(**kwargs):
    """
    Get AlexNet architecture
    """
    architecture = AlexNetArchitecture(**kwargs)
    return architecture

class AlexNetArchitecture(BaseArchitecture):
    """
    class for the architecture AlexNet
    """
    def __init__(self, **kwargs):
        # Training parameters
        self.num_classes = kwargs.get('num_classes', constants.IMAGENET_CLASSES)
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
        irrelevant_layer_regexes = ['features$', 'classifier.*']
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
        Initialize AlexNet
        """
        self._store_rng_states()
        backbone_name = 'alexnet'
        model_name = self._set_model_seed(backbone_name)
        model = AlexNet(num_classes=self.num_classes)
        self._recover_rng_states()
        return model, model_name

class AlexNet(nn.Module):
    """
    Updated from torchvision.models.alexnet
    https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py#L17C1-L52C17
    1. Replaced in-place ReLU with out-of-place ReLU
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
