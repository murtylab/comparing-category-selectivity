"""
Adapted from ImageNet-21K Github repo
Link: https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/models/utils/factory.py
"""

import torch
import timm
import os
import urllib.request

# Links source: https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/MODEL_ZOO.md
MODEL_LINKS = {
    'resnet50': 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth',
    'vit_base_patch16_224': 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth',
}

def get_model_weights(model_name, download_dir=None):
    # Set download dir
    hf_home_dir = os.getenv('HF_HOME')
    if download_dir is None:
        if hf_home_dir:
            download_dir = f'{hf_home_dir}/cache/imagenet21k_weights'
        else:
            download_dir = 'imagenet21k_weights'
    os.makedirs(download_dir, exist_ok=True)
    
    # Check if model link is available
    if model_name not in MODEL_LINKS:
        raise ValueError(f"No download link available for model: {model_name}")
    model_url = MODEL_LINKS[model_name]
    
    # Download model weights
    model_path = os.path.join(download_dir, f"{model_name}.pth")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloaded {model_name} weights to {model_path}")
    
    return model_path

def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model

def create_model(model_name):
    assert model_name in MODEL_LINKS, f"Model {model_name} not implemented"

    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=0)
    elif model_name == 'vit_base_patch16_224': # notice - qkv_bias==False currently
        model_kwargs = dict(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
        model = timm.models.vision_transformer._create_vision_transformer('vit_base_patch16_224_in21k',
                                                                          pretrained=False,
                                                                          num_classes=0, **model_kwargs)

    model_path = get_model_weights(model_name)
    model = load_model_weights(model, model_path)

    return model