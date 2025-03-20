"""
AlexNet: IPCL
https://github.com/jacob-prince/jsputils/blob/main/jsputils/models.py
https://github.com/harvard-visionlab/open_ipcl/blob/main/models/alexnet_gn.py
https://github.com/harvard-visionlab/open_ipcl
"""
import torch
from torch import nn

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = AlexNetIPCL(**kwargs)
    return model

class AlexNetIPCL(BaseModel):
    """
    class for the model AlexNet
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # https://github.com/jacob-prince/jsputils/blob/main/jsputils/nnutils.py#L62
        # Using Custom transforms instead of model's transforms
        # Replicating the transforms from the original model
        # https://github.com/harvard-visionlab/open_ipcl?tab=readme-ov-file#load-pretrained-models-from-cloned-repo
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms()
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = ['conv_block_[0-9]$', 'fc.*', 'l2norm$']
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
        Load AlexNet IPCL ref1 (primary model) from torch hub
        """
        # model, self.transforms = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref01")
        model, _ = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref01")
        model_name = 'alexnetipclref01'
        return model, model_name
