"""
BagNet-9
https://github.com/wielandbrendel/bag-of-local-features-models
"""
from torch import nn

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, resnet_utils, bagnet_utils

def get_model(**kwargs):
    model = BagNet9(**kwargs)
    return model

class BagNet9(BaseModel):
    """
    class for the model BagNet9
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Assuming resize & crop size based on https://github.com/wielandbrendel/bag-of-local-features-models?tab=readme-ov-file#image-preprocessing
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
            'conv[0-9]$', 'bn1$', 'relu$', 'maxpool$', \
            'layer[0-9]$', r'layer[0-9]\.[0-9]$', '.*downsample', \
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
        Load BagNet9
        """
        model = bagnet_utils.bagnet9(pretrained=True)
        resnet_utils.fix_resnet_blocks_name_relus(
            model, custom_bottleneck= bagnet_utils.Bottleneck
        )
        model_name = 'bagnet9'
        return model, model_name
