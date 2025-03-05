"""
BagNet-17
https://github.com/wielandbrendel/bag-of-local-features-models
"""
from .basemodel import BaseModel
from .utils import bagnet_utils, transform_utils

def get_model(**kwargs):
    model = BagNet17(**kwargs)
    return model

class BagNet17(BaseModel):
    """
    class for the model BagNet17
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Assuming resize & crop size based on https://github.com/wielandbrendel/bag-of-local-features-models?tab=readme-ov-file#image-preprocessing
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
        Load BagNet17
        """
        model = bagnet_utils.bagnet17(pretrained=True)
        model_name = 'bagnet17'
        return model, model_name
