"""
AlexNet
https://pytorch.org/vision/main/models/alexnet.html
"""
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils

def get_model(**kwargs):
    model = AlexNetImageNet1kv1(**kwargs)
    return model

class AlexNetImageNet1kv1(BaseModel):
    """
    class for the model AlexNet
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
        # Relevant layer names
        # Layer names to ignore since they are duplicates
        duplicate_layer_names = ['features', 'classifier']
        self.relevant_layer_names = [
            layer for layer in self.all_layer_names
            if layer not in duplicate_layer_names and layer not in self.irrelevant_layer_names
        ]

    def _load_model(self):
        """
        Load AlexNet
        """
        model = torchvision.models.alexnet(weights='IMAGENET1K_V1')
        model_name = 'alexnetimagenet1kv1'
        return model, model_name
