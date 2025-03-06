"""
ResNet-50 trained on SIN and IN
https://github.com/rgeirhos/texture-vs-shape/tree/master
"""
from .basemodel import BaseModel
from .utils import sin_utils, resnet_utils, transform_utils

def get_model(**kwargs):
    model = ResNet50SINandIN(**kwargs)
    return model

class ResNet50SINandIN(BaseModel):
    """
    class for the model ResNet50 trained on Stylized-ImageNet and ImageNet
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
        Load ResNet50
        """
        model = sin_utils.load_model('resnet50_trained_on_SIN_and_IN')
        model = model.module # Get the Sequential model
        model_name = 'resnet50sinandin'
        return model, model_name
