from time import time
from compcats.utils import model_utils

import main_analysis as ma
import paths

MODELS_DIR = 'models'
frois = ['FFA', 'PPA', 'EBA']
# model_names = [
#     # 'resnet50imagenet1kv2',
#     # 'resnet18imagenet1kv1',
#     # 'alexnetimagenet1kv1',
#     # 'resnet50openaiclip',
#     # 'vitb16siglip2res224',
#     # 'alexnetipclref01',
#     # 'vitb16synclr',
#     # 'resnet50faceobfuscation',
#     # 'resnet50ecoset',
#     # 'resnet50sin',
#     # 'resnet50blurstrong',
#     # 'resnet50cesslbarlowtop',
#     'bagnet33',
# ]
model_names = [
    # 'alexnetimagenet1kv1',
    # 'alexnetipclref01',
    # 'alexnetbarlowtwins',
    # 'vgg16imagenet1kv1',
    # 'resnet18imagenet1kv1',
    # 'resnet18tdannsupervisedlw10',
    # 'resnet18tdannsimclrlw1',
    # 'resnet50imagenet1kv2',
    # 'resnet50openaiclip',  # Rank 2
    # 'resnet50simclr', # Rank 6
    # 'resnet50robustl2eps3',
    # 'resnet50sin',
    # 'resnet50ecoset',
    # 'resnet50faceobfuscation',
    # 'resnet50toponettau30', # Rank 7
    # 'resnet50cesslbarlowtop',
    # 'resnet50blurstrong',
    # 'resnet101imagenet1kv1', # Rank 9
    # 'resnet101openaiclip', # Rank 4
    'bagnet33', # TODO
    # 'cornets', # Rank 8
    # 'cornetz',
    # 'convnexttinyimagenet1kv1', # Rank 3
    # 'vitb32imagenet1kv1',
    # 'vitb32openaiclip',
    # 'vitb32dreamsimopenclip',
    # 'vitb16imagenet21k',
    # 'vitb16openaiclip',
    # 'vitb16siglip2res224', # Rank 1
    # 'vitb16eva028bs8bb131k',
    # 'vitb16synclr',
    # 'vitb14dinov2', # Rank 10
    # 'vitb14dinov2reg', # Rank 5
    # 'vitl14openaiclip',
    # 'vitl14kosmos2' # TODO
]

batch_size = 128

for model_name in model_names:
    layers = [
        ma.get_froi_models_layers(froi, froi)[model_name] for froi in frois
    ]
    layers = list(set(layers))  # Unique layers
    print(f'{model_name}, layers to extract: {layers}')
    model = ma.get_model(model_name)
    for fidx, layer in enumerate(layers):
        print(f'Extracting activations for {frois[fidx]}, layer: {layer}')
        start_time = time()
        model_allnsdstim_activations = model_utils.get_dataset_activations(
                MODELS_DIR, paths.NSD_DIR, model, layers, dataset='nsd', \
                batch_size=batch_size, device='cuda'
            )
        model_allnsdstim_activations = {
            k: v.cpu() for k, v in model_allnsdstim_activations.items()
        }
        del model_allnsdstim_activations  # Free up memory
        end_time = time()
        total_time = (end_time - start_time) / 60  # in minutes
        print(f'Time taken to extract activations for {model_name} {frois[fidx]}: {total_time:.2f} minutes')
    del model  # Free up memory
