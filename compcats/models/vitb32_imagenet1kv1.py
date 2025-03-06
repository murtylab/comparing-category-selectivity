"""
ViT-B/32 trained on ImageNet-1K
https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html
"""
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils

def get_model(**kwargs):
    model = ViTB32ImageNet1kv1(**kwargs)
    return model

class ViTB32ImageNet1kv1(BaseModel):
    """
    class for the model ViT-B/32 trained on ImageNet-1K
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize=(224, 224)
        )

    def _load_model(self):
        """
        Load ViT-B/32 trained on ImageNet-1K
        """
        model = torchvision.models.vit_b_32(weights='IMAGENET1K_V1')
        model_name = 'vitb32imagenet1kv1'
        return model, model_name
