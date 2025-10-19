"""
AlexNet: Brain-optimized model trained on the MOSAIC dataset
https://github.com/murtylab/mosaic-dataset
https://github.com/murtylab/mosaic-dataset/blob/master/mosaic/models/pretrained.py
"""
from torch import nn
import torchvision

try:
    import mosaic
except ImportError:
    # These libraries are optional, silently handle import error
    pass

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = AlexNetMosaic(**kwargs)
    return model

class AlexNetMosaic(BaseModel):
    """
    class for the model AlexNet
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # https://github.com/murtylab/mosaic-dataset/blob/master/mosaic/utils/inference.py#L11
        self.transforms = transform_utils.get_transforms()
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = ['core$', 'core.features$', 'subject_readouts.*']
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
        Load AlexNet
        """
        # Additional kwargs
        self.DEFAULT_FRAMEWORK = "multihead"
        self.framework = self.kwargs.get('framework', self.DEFAULT_FRAMEWORK)
        self.DEFAULT_SUBJECTS = "all"
        self.subjects = self.kwargs.get('subjects', self.DEFAULT_SUBJECTS)
        self.DEFAULT_VERTICES = "visual"
        self.vertices = self.kwargs.get('vertices', self.DEFAULT_VERTICES)
        model = mosaic.from_pretrained(
            backbone_name='alexnet',
            framework=self.framework,
            subjects=self.subjects,
            vertices=self.vertices
        )
        # Ignore the subject readout layers and just keep the core
        model = model.core
        model_name = f'alexnetmosaic{self.framework}{self.subjects}{self.vertices}'
        return model, model_name
