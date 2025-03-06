"""
ViT-B/16 trained on ImageNet-21K
https://github.com/Alibaba-MIIL/ImageNet21K
"""

from .basemodel import BaseModel
from .utils import transform_utils, imagenet21k_utils

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

    def _load_model(self):
        """
        Load ViT-B/16
        """
        # IMAGENET21K
        model = imagenet21k_utils.create_model('vit_base_patch16_224')
        model_name = 'vitb16imagenet21k'
        return model, model_name
