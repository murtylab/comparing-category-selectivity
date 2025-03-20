"""
ViT-B/16 SynCLR
https://github.com/google-research/syn-rep-learn/tree/main/SynCLR
"""
import torch
from torch import nn

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, synclr_utils, download_utils

def get_model(**kwargs):
    model = ViTB16SynCLR(**kwargs)
    return model

class ViTB16SynCLR(BaseModel):
    """
    class for the model ViT-B/16 trained using SynCLR
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # RGB(): Convert to RGB
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=(0.48145466, 0.4578275, 0.40821073),
        #   std=(0.26862954, 0.26130258, 0.27577711)
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = ['patch_embed.*', '.*blocks$', \
                                    r'.*blocks\.[0-9]{1,2}$', 'head$' , \
                                    r'.*attn\..*', r'.*mlp\..*', 'norm$', 'head$']
        # Layer types to ignore
        irrelevant_layer_types = [nn.Dropout, nn.Identity]
        self.irrelevant_layer_names = layer_utils.get_layers_matching_regexes(
            self.model, irrelevant_layer_regexes
        ) + layer_utils.get_layers_matching_types(self.model, irrelevant_layer_types)
        self.relevant_layer_names = [
            layer for layer in self.all_layer_names
            if layer not in self.irrelevant_layer_names
        ]

    def _load_model(self):
        """
        Load ViT-B/16 SynCLR
        """
        # https://github.com/google-research/syn-rep-learn/tree/main/SynCLR#pre-trained-models
        self.url = 'https://dl.dropboxusercontent.com/scl/fi/ogktmm26yojhlgsbwqqti/synclr_vit_b_16.pth?rlkey=2bn6j4upcrqy93d5b9fgrrj02'
        # Download weights
        filename = 'synclr_vit_b_16.pth'
        weights_path = download_utils.download_weights(self.url, filename)
        weights = torch.load(weights_path)
        weights = weights['model']
        # Fix Keys
        fixed_weights = {k.replace("module.visual.", ""): v for k, v in weights.items()}
        # Instantiate model
        model = synclr_utils.vit_base_patch16()
        # Load weights
        # Not strict as head is not present in the weights
        model.load_state_dict(fixed_weights, strict=False)
        # Using Standard CLIP transforms
        # https://github.com/google-research/syn-rep-learn/blob/main/SynCLR/eval/main_downstream_linear.py#L475
        self.transforms = transform_utils.get_clip_transforms()
        model_name = 'vitb16synclr'
        return model, model_name
