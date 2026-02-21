"""
ViT-B/14 trained using DINOv2 with Registers
https://github.com/facebookresearch/dinov2
"""
import torch
from torch import nn

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ViTB14DINOv2Register(**kwargs)
    return model

class ViTB14DINOv2Register(BaseModel):
    """
    class for the model ViT-B/14 trained using DINOv2 with Registers
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize():
        #   mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = ['patch_embed.*', '.*blocks$', \
                                    r'.*blocks\.[0-9]{1,2}$', 'head$' , \
                                    r'.*attn\..*', r'.*mlp\..*', 'norm$', 'head$', r'.*\.ls[1-2]']
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
        Load ViT-B/14 trained using DINOv2 with Registers
        """
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', verbose=False)
        self.transforms = transform_utils.get_clip_transforms()
        model_name = 'vitb14dinov2reg'
        return model, model_name
