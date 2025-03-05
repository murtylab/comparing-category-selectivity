"""
AlexNet
https://pytorch.org/vision/main/models/alexnet.html
"""
import torch
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils

def get_model(**kwargs):
    model = AlexNetRandom(**kwargs)
    return model

class AlexNetRandom(BaseModel):
    """
    class for the model AlexNet
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
        self.all_layer_names = layer_utils.get_all_layer_names(self.model)

    def _load_model(self):
        """
        Load AlexNet
        """
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        model = torchvision.models.alexnet(weights=None)
        model_name = f'alexnetrandom{seed}'
        return model, model_name
