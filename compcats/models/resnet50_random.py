"""
ResNet-50 untrained
https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
"""
import torch
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils

def get_model(**kwargs):
    model = ResNet50Random(**kwargs)
    return model

class ResNet50Random(BaseModel):
    """
    class for the model ResNet50 trained on ImageNet
    """
    def __init__(self, **kwargs):
        super().__init__()
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
        Initialize ResNet50
        """
        # Seed
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        model = torchvision.models.resnet50(weights=None)
        model_name = f'resnet50random{seed}'
        return model, model_name
