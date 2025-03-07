"""
VGG-19
https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html#torchvision.models.vgg19
"""
from torch import nn
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = VGG19ImageNet1kv1(**kwargs)
    return model

class VGG19ImageNet1kv1(BaseModel):
    """
    class for the model VGG-19 trained on ImageNet
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
        irrelevant_layer_regexes = ['features$', 'classifier$']
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
        Load VGG-19
        """
        model = torchvision.models.vgg19(weights='IMAGENET1K_V1')
        model_name = 'vgg19imagenet1kv1'
        return model, model_name
