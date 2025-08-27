from time import time
from compcats.utils import model_utils

import main_analysis as ma
import paths

MODELS_DIR = 'models'
frois = ['FFA', 'PPA', 'EBA']
model_names = [
    # 'resnet18imagenet1kv1',
    # 'alexnetimagenet1kv1',
    # 'resnet50openaiclip',
    'vitb16siglip2res224',
    'alexnetipclref01',
    'vitb1ysynclr'
]

for model_name in model_names:
    layers = [
        ma.get_froi_models_layers(froi, froi)[model_name] for froi in frois
    ]
    layers = list(set(layers))  # Unique layers
    print(f'{model_name}, layers to extract: {layers}')
    model = ma.get_model(model_name)
    batch_size = 512
    start_time = time()
    model_allnsdstim_activations = model_utils.get_dataset_activations(
            MODELS_DIR, paths.NSD_DIR, model, layers, dataset='nsd', \
            batch_size=batch_size, device='cuda'
        )
    end_time = time()
    total_time = (end_time - start_time) / 60  # in minutes
    print(f'Time taken to extract activations for {model_name}: {total_time:.2f} minutes')