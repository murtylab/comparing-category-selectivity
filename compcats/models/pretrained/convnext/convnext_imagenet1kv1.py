"""
ConvNeXt model trained on ImageNet1k
https://docs.pytorch.org/vision/main/models/convnext.html
"""
from torch import nn
import torchvision

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ConvNeXtImageNet1kv1(**kwargs)
    return model

class ConvNeXtImageNet1kv1(BaseModel):
    """
    class for the model ConvNeXt trained on ImageNet
    """
    def __init__(self, **kwargs):
        self.DEFAULT_MODELSIZE = 'tiny'
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
        irrelevant_layer_regexes = [
            'features$', r'features\.[0-9]$', \
            # Only take the CNBlocks and not the downsample layers
            r'features\.[0-9]\.[0-9]$', r'features\.[0-9]\.[0-9]\.block$', \
            # Ignore the permute layers
            r'features\.[0-9]\.[0-9]\.block\.[1|6]', \
            '.*stochastic_depth$', 'classifier.*' 
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
        Load ConvNeXt
        """
        self.modelsize = self.kwargs.get('modelsize', self.DEFAULT_MODELSIZE)
        model_class = getattr(torchvision.models, f'convnext_{self.modelsize}')
        # Load the model with default weights
        # IMAGENET1K_V1
        model = model_class(weights='DEFAULT')
        # Model name
        model_name = f'convnext{self.modelsize}imagenet1kv1'
        return model, model_name
