"""
ViT-B/16 trained on ImageNet-21K
https://github.com/Alibaba-MIIL/ImageNet21K
"""
from torch import nn

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, imagenet21k_utils

def get_model(**kwargs):
    model = ViTB16ImageNet21k(**kwargs)
    return model

class ViTB16ImageNet21k(BaseModel):
    """
    class for the model ViT-B/16 trained on ImageNet-21K
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        self.transforms = transform_utils.get_transforms(
            mean=None, std=None, resize=(224, 224)
        )
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            r'patch_embed.*', 'blocks$', r'.*blocks\.[0-9]{1,2}$', \
            'heads.head$' , r'.*attn\..*', r'.*mlp\..*', 'norm$'
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
        Load ViT-B/16
        """
        # IMAGENET21K
        model = imagenet21k_utils.create_model('vit_base_patch16_224')
        model_name = 'vitb16imagenet21k'
        return model, model_name
