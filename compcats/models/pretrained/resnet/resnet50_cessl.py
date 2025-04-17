"""
ResNet-50 Contrastive Equivariant Self-Supervised Learning (CE-SSL)
https://openreview.net/forum?id=AiMs8GPP5q
https://github.com/ThomasYerxa/CE-SSL/tree/main
"""
import os
from collections import OrderedDict
from urllib.request import urlretrieve
import torch
from torch import nn
import torchvision

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, resnet_utils

def get_model(**kwargs):
    model = ResNet50CESSL(**kwargs)
    return model

class ResNet50CESSL(BaseModel):
    """
    class for the model ResNet50 trained using CE-SSL
    """
    def __init__(self, **kwargs):
        # Default params corresponding to the top model (highest neural predictivity)
        # https://github.com/ThomasYerxa/CE-SSL/blob/main/model_loading.py#L37
        self.DEFAULT_OBJECTIVE = 'barlowtop'
        self.DEFAULT_LAMBDA = None
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
        self.objective = self.kwargs.get('objective', self.DEFAULT_OBJECTIVE).lower()
        self.lambda_ = self.kwargs.get('lambda_', self.DEFAULT_LAMBDA)
        # Check params
        assert self.objective in ['barlowtop', 'simclr', 'mmcr', 'barlow'], \
            f'Objective {self.objective} not in (SimCLR, MMCR, Barlow)'
        assert self.lambda_ in [None, 0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5], \
            f'Lambda {self.lambda_} not in [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]'
        # Model name
        lambda_str = '' if self.lambda_ is None else str(self.lambda_).replace('.', 'pt')
        model_name = f'resnet50cessl{self.objective}{lambda_str}'
        model, self.url = self._get_model(self.objective, self.lambda_, model_name)
        # Fix ReLUs
        resnet_utils.fix_resnet_blocks_name_relus(model)
        return model, model_name

    def _generate_url(self, objective, lambda_):
        """
        Generate the URL for the model weights
        """
        if objective == 'barlowtop' and lambda_ is None:
            # https://github.com/ThomasYerxa/CE-SSL/blob/main/model_loading.py#L37
            # Barlow top model
            url = "https://users.flatironinstitute.org/~tyerxa/equi_proj/training_checkpoints/final/resnet_50/Barlow/1000_epoch/ep1000-ba625000-rank0"
        else:
            raise ValueError("There is some error in downloading using urlretrieve, instead please download the model yourself and rename appropriately")
            objective_name_map = {
                'simclr': 'SimCLR',
                'mmcr': 'MMCR',
                'barlow': 'Barlow'
            }
            objective = objective_name_map[objective]
            ckpt_location = f'~tyerxa/equi_proj/training_checkpoints/final/resnet_50/{objective}/lmda_{lambda_}/ep100-ba62500-rank0'
            url = f'https://users.flatironinstitute.org/{ckpt_location}'
        return url

    def _get_model(self, objective, lambda_, model_name):
        """
        Load a model saved via Composer into a vanilla torchvision model.
        """
        # Get URL
        url = self._generate_url(objective, lambda_)
        # Download weights if not already downloaded
        if os.path.exists(f'{model_name}.torch'):
            file_header = f'{model_name}.torch'
        else:
            print(f'Downloading {model_name} weights from {url}')
            # Download weights
            file_header = urlretrieve(url, filename=f'{model_name}.torch')
            file_header = file_header[0]
        # Load state dict
        state_dict = torch.load(
            file_header, map_location=torch.device("cpu"), weights_only=False
        )["state"]["model"]
        # Load model
        model = torchvision.models.resnet.resnet50()
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if 'lin_cls' in k:
                new_sd['fc.' + k.split('.')[-1]] = v
            if ".f." not in k:
                continue
            parts = k.split(".")
            idx = parts.index("f")
            new_k = ".".join(parts[idx + 1 :])
            new_sd[new_k] = v
        model.load_state_dict(new_sd, strict=True)
        return model.eval(), url
