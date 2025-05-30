"""
ResNet-50 trained with blur augmentation
Jang, H., & Tong, F. (2024). Improved modeling of human vision by incorporating robustness to blur
 in convolutional neural networks. Nature Communications, 15(1), 1989.
https://www.nature.com/articles/s41467-024-45679-0
https://osf.io/upf5w/files/osfstorage
"""
import os
from urllib.request import urlretrieve
from torch import nn
import torch
import torchvision

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, resnet_utils

BLUR_MODEL_CHECKPOINT_LINKS = {
    "rn18": {
        "weak": "https://osf.io/h6x7r",
        "strong": "https://osf.io/z32va"
    },
    "rn50": {
        "weak": "https://osf.io/d29a6",
        "strong": "https://osf.io/drpfn"
    },
    "vgg16": {
        "weak": "https://osf.io/g7mqh",
        "strong": "https://osf.io/75ubj"
    },
    "vgg19": {
        "weak": "https://osf.io/km7tw",
        "strong": "https://osf.io/y7dte"
    },
    ## add the other models here
}

def get_model(**kwargs):
    model = ResNet50Blur(**kwargs)
    return model

class ResNet50Blur(BaseModel):
    """
    class for the model ResNet50 trained on blurred ImageNet
    """
    def __init__(self, **kwargs):
        self.DEFAULT_BLURSTRENGTH = "strong" # or "weak"
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms()
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            'conv1$', 'bn1$', 'relu$', 'maxpool$', \
            'layer[0-9]$', r'layer[0-9]\.[0-9]$', '.*downsample', \
            'fc$'
        ]
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
        Load ResNet50
        """
        self.blurstrength = self.kwargs.get('blurstrength', self.DEFAULT_BLURSTRENGTH)
        # Model name
        blurstrength_name = self.blurstrength
        if self.blurstrength not in BLUR_MODEL_CHECKPOINT_LINKS['rn50']:
            raise ValueError(f"Blur strength '{self.blurstrength}' is not supported.\
                             Choose from {list(BLUR_MODEL_CHECKPOINT_LINKS['rn50'].keys())}.")
        model_name = f'resnet50blur{blurstrength_name}'
        # URL for pretrained model
        osf_url = BLUR_MODEL_CHECKPOINT_LINKS['rn50'][self.blurstrength]
        # download format: https://osf.io/download/<hash>
        self.url = '/'.join(osf_url.split('/')[:-1]) + '/download/' + osf_url.split('/')[-1]
        # Download weights if not already downloaded
        if os.path.exists(f'{model_name}.pth.tar'):
            file_header = f'{model_name}.pth.tar'
        else:
            print(f'Downloading {model_name} weights from {self.url}')
            # Download weights
            file_header = urlretrieve(self.url, filename=f'{model_name}.pth.tar')
            file_header = file_header[0]
        # Load the weights
        if not torch.cuda.is_available():
            # If no GPU, load the model on CPU
            checkpoint = torch.load(
                file_header, weights_only=True, map_location='cpu'
            )['state_dict']
        else:
            checkpoint = torch.load(file_header, weights_only=True)['state_dict']
        # Load the model
        model = torchvision.models.resnet50(weights=None)
        model.load_state_dict(checkpoint, strict=False)
        # Fix the ReLU layers in the ResNet blocks
        resnet_utils.fix_resnet_blocks_name_relus(model)
        # Return the model and its name
        return model, model_name
