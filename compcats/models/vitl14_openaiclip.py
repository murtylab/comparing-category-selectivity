"""
ViT-L/14 CLIP
https://github.com/openai/CLIP
"""
from torch import nn
import clip

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ViTL14OpenAICLIP(**kwargs)
    return model

class ViTL14OpenAICLIP(BaseModel):
    """
    class for the model ViT-L/14 trained using CLIP
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # RGB(): Convert to RGB
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            'transformer$', 'transformer.resblocks$', r'.*resblocks\.[0-9]{1,2}$', \
            'heads.head$' , r'.*attn\..*', r'.*mlp\..*', \
            'conv1$', 'ln_pre$', 'ln_post$'
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
        Load ViT-L/14 CLIP
        """
        # OpenAI CLIP
        model, _ = clip.load('ViT-L/14')
        model = model.visual # Get visual module of the multi-modal model
        model = model.float() # Convert FP16 to FP32 dtype
        # Custom transforms for CLIP model
        model_resolution = model.input_resolution # in pixels
        self.transforms = transform_utils.get_clip_transforms(resize=model_resolution)
        model_name = 'vitl14openaiclip'
        return model, model_name
