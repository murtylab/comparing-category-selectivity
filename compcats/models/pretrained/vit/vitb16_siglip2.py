"""
ViT-B/16 SigLIP2
https://huggingface.co/timm/ViT-B-16-SigLIP2
"""
from torch import nn
try:
    from open_clip import create_model_from_pretrained
except ImportError:
    # These libraries are optional, silently handle import error
    pass

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ViTB16SigLIP2(**kwargs)
    return model

class ViTB16SigLIP2(BaseModel):
    """
    class for the model ViT-B/16 trained using CLIP
    """
    def __init__(self, **kwargs):
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
        irrelevant_layer_regexes = ['trunk$', r'.*patch_embed.*', '.*blocks$', \
                                    r'.*blocks\.[0-9]{1,2}$', 'head$' , \
                                    r'.*attn\..*', r'.*mlp\..*', r'.*attn_pool\..*']
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
        Load ViT-B/16 SigLIP2
        """
        # Default resolution value = 224
        self.DEFAULT_RESOLUTION = 224
        # Get resolution
        # Forcing integer value
        self.resolution = int(self.kwargs.get('resolution', self.DEFAULT_RESOLUTION))
        suffix = f'-{self.resolution}' if self.resolution != 224 else ''
        # Get model from hugingface hub
        model, _ = create_model_from_pretrained(f'hf-hub:timm/ViT-B-16-SigLIP2{suffix}')
        model = model.visual # Get visual module of the multi-modal model
        # Custom transforms for SigLIP model
        self.transforms = transform_utils.get_clip_transforms(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), resize=self.resolution
        )
        model_name = f'vitb16siglip2res{self.resolution}'
        return model, model_name
