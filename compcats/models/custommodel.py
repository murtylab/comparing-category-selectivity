"""
Custom model wrapper
"""
import torch
from torchvision.transforms import v2

from .basemodel import BaseModel

def get_model(model, name, transforms, **kwargs):
    """
    Get custom model
    """
    model = CustomModel(model, name, transforms, **kwargs)
    return model

class CustomModel(BaseModel):
    """
    class for custom model
    """
    def __init__(
        self,
        model,
        name,
        transforms,
        **kwargs
    ):
        super().__init__(model, name, **kwargs)
        # Transforms:
        if transforms is None:
            transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transforms = transforms
        # Relevant layer names
        self.relevant_layer_names = kwargs.get('relevant_layer_names', [])
        self.irrelevant_layer_names = \
            list(set(self.all_layer_names) - set(self.relevant_layer_names))
