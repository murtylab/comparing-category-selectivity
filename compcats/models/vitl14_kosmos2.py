"""
ViT-L/14 KOSMOS-2
https://huggingface.co/docs/transformers/en/model_doc/kosmos-2
"""
from torch import nn
from transformers import Kosmos2ForConditionalGeneration

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ViTL14KOSMOS2(**kwargs)
    return model

class ViTL14KOSMOS2(BaseModel):
    """
    class for the model ViT-L/14 KOSMOS-2
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
        irrelevant_layer_regexes = ['embeddings.*', 'pre_layrnorm$', 'encoder$', \
                                    'encoder.layers$$', \
                                    r'.*layers\.[0-9]{1,2}$', 'head$' , \
                                    r'.*attn\..*', r'.*mlp\..*', 'post_layernorm$']
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
        Load ViT-L/14 KOSMOS-2
        """
        model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
        model = model.vision_model.model # Get visual module of the multi-modal model
        self.transforms = transform_utils.get_clip_transforms()
        model_name = 'vitl14kosmos2'
        return model, model_name
