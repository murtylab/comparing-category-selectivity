"""
ViT-B/32 Untrained
https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html
"""
from torch import nn
import torchvision

from .basemodel import BaseModel
from .utils import transform_utils, layer_utils

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
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = ['encoder$', 'encoder.layers$', '.*encoder_layer_[0-9]{1,2}$', \
                                    'heads.head$' , r'.*self_attention\..*', r'.*mlp\..*']
        # Layer types to ignore
        irrelevant_layer_types = [nn.Dropout]
        self.irrelevant_layer_names = layer_utils.get_layers_matching_regexes(
            self.model, irrelevant_layer_regexes
        ) + layer_utils.get_layers_matching_types(self.model, irrelevant_layer_types)
        self.relevant_layer_names = [
            layer for layer in self.all_layer_names
            if layer not in self.irrelevant_layer_names
        ]

    def _load_model(self):
        """
        Load ViT-B/32 Untrained
        """
        self._store_rng_states()
        backbone_name = 'vitb32'
        model_name = self._set_model_seed(backbone_name)
        model = torchvision.models.vit_b_32(weights=None)
        self._recover_rng_states()
        return model, model_name
