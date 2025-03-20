"""
CORNet-Z model
https://github.com/dicarlolab/CORnet
"""
import torch
from torch import nn
from cornet.cornet_z import CORnet_Z, HASH

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, download_utils

def get_model(**kwargs):
    model = CORNetZ(**kwargs)
    return model

class CORNetZ(BaseModel):
    """
    class for the model CORNet-Z
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
        irrelevant_layer_regexes = ['decoder.*', 'V[1-4]$', 'IT$', r'.*\.output$']
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
        Load CORNet-Z
        """
        self.url = f"https://s3.amazonaws.com/cornet-models/cornet_z-{HASH}.pth"
        # Download weights
        filename = 'cornet_z.pth'
        weights_path = download_utils.download_weights(self.url, filename)
        weights = torch.load(weights_path, weights_only=False)
        weights = weights['state_dict']
        # Fix Keys
        fixed_weights = {k.replace("module.", ""): v for k, v in weights.items()}
        # Instantiate model
        model = CORnet_Z()
        # Load weights
        model.load_state_dict(fixed_weights)
        model_name = 'cornetz'
        return model, model_name
