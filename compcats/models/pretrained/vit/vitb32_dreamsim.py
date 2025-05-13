"""
ViT-B/32 DreamSim 
https://github.com/ssundaram21/dreamsim
"""
import dreamsim
from torch import nn
import torch

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ViTB32DreamSim(**kwargs)
    return model

class ViTB32DreamSim(BaseModel):
    """
    class for the model ViT-B/32 DreamSim
    """
    def __init__(self, **kwargs):
        # Default values
        self.DEFAULT_TYPE = 'open_clip_vitb32'
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # RGB(): Convert to RGB
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            'patch_embed.*', '.*blocks$', r'.*blocks\.[0-9]{1,2}$', 'head$', \
            'norm$', r'.*attn\..*', r'.*mlp\..*', 'pos_drop$'
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
        Load ViT-B/32 DreamSim
        """
        # Get DreamSim type
        self.dreamsim_type = self.kwargs.get('dreamsim_type', self.DEFAULT_TYPE)
        if self.dreamsim_type != self.DEFAULT_TYPE:
            raise ValueError(f'DreamSim type {self.dreamsim_type} not yet implemented!')
        # Get model from hugingface hub
        if not torch.cuda.is_available():
            # If no GPU, load the model on CPU
            model, _ = dreamsim.dreamsim(
                pretrained=True, dreamsim_type=self.dreamsim_type, device='cpu'
            )
        else:
            model, _ = dreamsim.dreamsim(
                pretrained=True, dreamsim_type=self.dreamsim_type
            )
        model = model.base_model.model.extractor_list[0].model # Extract model from DreamSim
        self.transforms = transform_utils.get_clip_transforms()
        model_name = 'vitb32dreamsimopenclip'
        return model, model_name
