"""
ResNet-50 VOnetNet
https://github.com/dicarlolab/vonenet
"""
import warnings
from torch import nn

try:
    import vonenet
except ImportError:
    # These libraries are optional, silently handle import error
    pass

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, resnet_utils

def get_model(**kwargs):
    model = ResNet50VOneNet(**kwargs)
    return model

class ResNet50VOneNet(BaseModel):
    """
    class for the model ResNet50 VOneNet
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
            r'vone_block.+', 'model$', \
            'model.conv1$', 'model.bn1$', 'model.relu$', 'model.maxpool$', \
            'model.layer[0-9]$', r'model.layer[0-9]\.[0-9]$', 'model.*downsample', \
            'model.fc$', r'model.*conv[0-9]\.conv$'
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
        Load ResNet50 VOneNet
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = vonenet.get_model(
                model_arch='resnet50',
                pretrained=True
            )
            model = model.module
        resnet_utils.fix_resnet_blocks_name_relus(
            model, custom_bottleneck=vonenet.back_ends.Bottleneck
        )
        model_name = 'resnet50vonenet'
        return model, model_name
