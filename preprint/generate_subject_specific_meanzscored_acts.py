# Libraries

import os
import numpy as np
import pandas as pd
import torch

# --- Custom packages ---
from compcats.localizer import localizer
from compcats.utils import model_utils
import main_analysis as ma
import paths

# Paths
RESULTS_DIR = 'results/exploration/explore_interpret'
MODELS_DIR = 'models'
PLOTS_DIR = f'{RESULTS_DIR}/plots/'
os.makedirs(PLOTS_DIR, exist_ok=True)
NSD_DIR = paths.NSD_DIR
GRILLSPECTOR_LOCALIZER_DIR = paths.GRILLSPECTOR_LOCALIZER_DIR
COCO_ANNOTATIONS_PATH = paths.COCO_ANNOTATIONS_PATH

TTHRESHOLD = 7
FROIS = ['FFA', 'PPA', 'EBA']
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]  # NSD subjects
MODEL_LOCALIZER = 'grillspector'
MODEL_LOCALIZER_DIR = GRILLSPECTOR_LOCALIZER_DIR
mean_zscored_responses_df_path = f'{RESULTS_DIR}/subject_specific_mean_zscored_responses.csv'
resp_df = pd.read_csv(mean_zscored_responses_df_path)

TOP_MODEL_NAMES = [
    # 'vitb16siglip2res224',
    # 'resnet50openaiclip', # Done
    # 'convnexttinyimagenet1kv1',
    # 'resnet101openaiclip',
    # 'vitb14dinov2reg',
    # 'resnet50simclr',
    # 'resnet50toponettau30',
    # 'cornets',
    # 'resnet101imagenet1kv1',
    # 'vitb14dinov2'
    # 'resnet18imaget1kv1',
    # 'resnet50imagenet1kv2',
    # 'resnet50faceobfuscation',
    # 'vitb16eva028bs8bb131k',
    # 'resnet50ecoset',
    'vitl14kosmos2',
    # 'vitb16synclr',
    # 'resnet18tdannsimclrlw1',
    # 'vgg16imagenet1kv1', # TODO: KeyError: (Ellipsis, tensor([[[False, False, False,  ..., False, False, False],
    # 'vitb32openaiclip',
    # 'vitb16openaiclip',
    # 'vitl14openaiclip',
    # 'resnet50robustl2eps3',
    # 'vitb32dreamsimopenclip',
    # 'resnet50blurstrong',
    # 'resnet50sin',
    # 'bagnet33', # TODO
    # 'vitb32imagenet1kv1',
    # 'alexnetimagenet1kv1',
    # 'alexnetbarlowtwins',
    # 'vitb16imagenet21k',
    # 'resnet18tdannsupervisedlw10',
    # 'cornetz', # TODO: Error: (Ellipsis, tensor([[[False, False, False,  ..., False, False, False],
    # 'alexnetipclref01'
]

for model_name in TOP_MODEL_NAMES:
    if model_name in resp_df.columns:
        # Check the length of the column, if it's the same as resp_df length, skip
        if len(resp_df[model_name]) == len(resp_df):
            print(f"Model {model_name} already in dataframe, skipping...")
            continue
    model = ma.get_model(model_name)
    layers = []
    for froi in FROIS:
        layer = ma.get_froi_models_layers(froi, froi)[model_name]
        layers.append(layer)
    layers = list(set(layers))
    model_data = model_utils.get_dataset_activations(
            MODELS_DIR, NSD_DIR, model, layers, 'nsd', subset=None, device='cpu'
        )
    if len(layers) == 1:
        print("Only one layer, wrapping in dict")
        model_data = {layers[0]: model_data}
    for froi in FROIS:
        froi_layer = ma.get_froi_models_layers(froi, froi)[model_name]
        model_froi_mask = localizer.get_froi_mask(
            MODEL_LOCALIZER_DIR, MODEL_LOCALIZER, MODELS_DIR, ma.get_model(model_name),
            froi_layer, froi, TTHRESHOLD, device='cpu'
        )
        model_froi_data = model_data[froi_layer][..., model_froi_mask].cpu().numpy()
        for subject in SUBJECTS:
            print(f"Processing model {model_name}, {froi}, subject {subject}...")
            mask = (resp_df["subject"] == subject) & (resp_df["froi"] == froi)
            nsdids = resp_df[mask]['nsd_id'].values
            model_data_sbj = model_froi_data[nsdids]
            model_response_mean = model_data_sbj.mean(axis=-1)
            model_response_mean_zscored = (model_response_mean - model_response_mean.mean()) / model_response_mean.std()
            resp_df.loc[mask, f'{model_name}_response'] = model_response_mean_zscored
            resp_df.to_csv(mean_zscored_responses_df_path, index=False)
            print(f"Done {model_name}, {froi}, subject {subject}")
        del model_froi_data, model_froi_mask
    del model_data, model
    print(f"Done {model_name}")
