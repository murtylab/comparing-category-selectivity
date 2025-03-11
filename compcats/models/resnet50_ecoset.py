"""
ResNet-50 trained on EcoSet
https://www.kietzmannlab.org/ecoset/
https://github.com/ViCCo-Group/thingsvision/blob/master/thingsvision/custom_models/resnet50_ecoset.py
"""
import torch
from torch import nn
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils, resnet_utils

def get_model(**kwargs):
    model = ResNet50EcoSet(**kwargs)
    return model

class ResNet50EcoSet(BaseModel):
    """
    class for the model ResNet50 trained on EcoSet
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
        irrelevant_layer_regexes = [
            'conv1$', 'bn1$', 'relu$', 'maxpool$', \
            'layer[0-9]$', r'layer[0-9]\.[0-9]$', '.*downsample', \
            'fc$', 'avgpool$'
        ]
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
        Load ResNet50 trained on EcoSet
        """
        model = torchvision.models.resnet50(weights=None, num_classes=565)
        path_to_weights = "https://osf.io/gd9kn/download"
        state_dict = torch.hub.load_state_dict_from_url(
                path_to_weights, file_name="Resnet50_ecoset"
            )
        model.load_state_dict(state_dict)
        resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = 'resnet50ecoset'
        return model, model_name
