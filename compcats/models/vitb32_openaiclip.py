"""
ViT-B/32 CLIP
https://github.com/openai/CLIP
"""
import clip

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = ViTB32OpenAICLIP(**kwargs)
    return model

class ViTB32OpenAICLIP(BaseModel):
    """
    class for the model ViT-B/32 trained using CLIP
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.all_layer_names = layer_utils.get_all_layer_names(self.model)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # RGB(): Convert to RGB
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)

    def _load_model(self):
        """
        Load ViT-B/32 CLIP
        """
        # OpenAI CLIP
        model, _ = clip.load('ViT-B/32')
        model = model.visual # Get visual module of the multi-modal model
        model = model.float() # Convert FP16 to FP32 dtype
        # Custom transforms for CLIP model
        model_resolution = model.input_resolution # in pixels
        self.transforms = transform_utils.get_clip_transforms(resize=model_resolution)
        model_name = 'vitb32openaiclip'
        return model, model_name
