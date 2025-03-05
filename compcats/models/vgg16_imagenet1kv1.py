"""
VGG-16
https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16
"""
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils

def get_model(**kwargs):
    model = VGG16ImageNet1kv1(**kwargs)
    return model

class VGG16ImageNet1kv1(BaseModel):
    """
    class for the model VGG-16 trained on ImageNet
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
        Load VGG-16
        """
        model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
        model_name = 'vgg16imagenet1kv1'
        return model, model_name
