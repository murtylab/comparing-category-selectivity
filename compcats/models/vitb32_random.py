"""
ViT-B/32 Untrained
https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html
"""
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils

def get_model(**kwargs):
    model = ViTB32Random(**kwargs)
    return model

class ViTB32Random(BaseModel):
    """
    class for the model ViT-B/32 Untrained
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
        Load ViT-B/32 Untrained
        """
        # Set seed to default seed if seed is not passed
        if self.seed is None:
            self.seed = self.DEFAULT_SEED
        model = torchvision.models.vit_b_32(weights=None)
        model_name = f'vitb32random{self.seed}'
        return model, model_name
