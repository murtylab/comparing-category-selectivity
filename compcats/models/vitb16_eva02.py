"""
ViT-B/16 EVA02
https://github.com/baaivision/EVA/tree/master/EVA-02
"""
from torch import nn
import open_clip

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ViTB16EVA02(**kwargs)
    return model

class ViTB16EVA02(BaseModel):
    """
    class for the model ViT-B/16 EVA02
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # RGB(): Convert to RGB
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): 
        #   mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            'trunk$', r'.*patch_embed.*', '.*blocks$', \
            r'.*blocks\.[0-9]{1,2}$', 'trunk.norm$' , 'trunk.head$', \
            r'.*attn\..*', r'.*mlp\..*', 'trunk.rope$', 'head$'
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
        Load ViT-B/16 CLIP
        """
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-B-16", pretrained="merged2b_s8b_b131k"
        )
        model = model.visual # Get visual module of the multi-modal model
        self.transforms = transform_utils.get_clip_transforms()
        model_name = 'vitb16eva028bs8bb131k'
        return model, model_name
