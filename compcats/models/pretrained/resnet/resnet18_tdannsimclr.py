"""
ResNet-18 TDANN SimCLR
https://www.sciencedirect.com/science/article/pii/S0896627324002794
"""
import os
from urllib.request import urlretrieve
from torch import nn
import torch

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, resnet_utils, tdann_utils

def get_model(**kwargs):
    model = ResNet18TDANNSimCLR(**kwargs)
    return model

class ResNet18TDANNSimCLR(BaseModel):
    """
    class for the model ResNet18 TDANN SimCLR
    """
    def __init__(self, **kwargs):
        self.DEFAULT_LOSSWEIGHT = 1
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
            'layer[0-9]$', r'layer[0-9]\.[0-9]$', '.*downsample', 'fc.*'
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
        Load ResNet18 TDANN SimCLR
        """
        self.lossweight = self.kwargs.get('lossweight', self.DEFAULT_LOSSWEIGHT)
        # Model name
        lossweight_name = str(self.lossweight).replace('.', 'pt')
        model_name = f'resnet18tdannsimclrlw{lossweight_name}'
        # URL for pretrained model
        # format: https://osf.io/<hash>
        osf_url = tdann_utils.PRETRAINED_SIMCLR_MODEL_LINKS[self.lossweight]
        # download format: https://osf.io/download/<hash>
        self.url = '/'.join(osf_url.split('/')[:-1]) + '/download/' + osf_url.split('/')[-1]
        # Download weights if not already downloaded
        if os.path.exists(f'{model_name}.torch'):
            file_header = f'{model_name}.torch'
        else:
            print(f'Downloading {model_name} weights from {self.url}')
            # Download weights
            file_header = urlretrieve(self.url, filename=f'{model_name}.torch')
            file_header = file_header[0]
        # Load the weights
        if not torch.cuda.is_available():
            # If no GPU, load the model on CPU
            checkpoint = torch.load(file_header, weights_only=False, map_location='cpu')
        else:
            checkpoint = torch.load(file_header, weights_only=False)
        state_dict = checkpoint['classy_state_dict']['base_model']['model']['trunk']
        model = tdann_utils.EshedNet()
        model.load_state_dict(state_dict, strict=False)
        model = model.base_model
        resnet_utils.fix_resnet_blocks_name_relus(model)
        return model, model_name
