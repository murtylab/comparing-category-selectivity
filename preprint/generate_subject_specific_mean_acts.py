# Libraries

import os
import numpy as np
import pandas as pd
from typing import List, Optional
import torch

# --- Custom packages ---
from compcats.datasets.fmri import nsd
from compcats.datasets.fmri.utils import nsd_ds_utils
from compcats.localizer import localizer
from compcats.utils import model_utils, coco_utils
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
SHARED1000_NSDIDS = nsd_ds_utils.get_subset_nsdids(paths.NSD_DIR, 1000)

SUBJECTS_DATA = {}
MODELS_DATA = {}

# Subjects data
for froi in FROIS:
    for subject in SUBJECTS:
        subject_data, images_df = nsd.get_betas(
            NSD_DIR, subject, froi, TTHRESHOLD, subset=None, voxels=True
        )
        # Add info to images_df
        images_df = coco_utils.add_image_info_to_df(images_df, COCO_ANNOTATIONS_PATH)
        subject_data = subject_data[~images_df['nsdId'].isin(SHARED1000_NSDIDS)]
        images_df = images_df[~images_df['nsdId'].isin(SHARED1000_NSDIDS)].reset_index(drop=True)
        SUBJECTS_DATA[(subject, froi)] = {
            'subject_data': subject_data,
            'images_df': images_df
        }
        print(f"Loaded subject {subject}, {froi}, data shape: {subject_data.shape}, images_df shape: {images_df.shape}")

def get_model_data(model_name, froi):
    if (model_name, froi) in MODELS_DATA:
        return MODELS_DATA[(model_name, froi)]
    else:
        print(f"Loading model data for {model_name}, {froi}...")
        # Get model and layer
        model = ma.get_model(model_name)
        layer = ma.get_froi_models_layers(froi, froi)[model_name]
        model_data = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer, 'nsd', froi, TTHRESHOLD, subset=None,
                froi_localizer=MODEL_LOCALIZER, froi_localizer_dir=MODEL_LOCALIZER_DIR, device='cpu'
            )
        MODELS_DATA[(model_name, froi)] = model_data
        return model_data

# Make a big csv of all mean z-scored responses
mean_responses_df_path = f'{RESULTS_DIR}/subject_specific_mean_responses.csv'
dfs = []

for froi in FROIS:
    for subject in SUBJECTS:
        subject_response_mean = SUBJECTS_DATA[(subject, froi)]['subject_data'].mean(axis=-1).cpu().numpy()
        froi_subject_df = pd.DataFrame({
            'idx_local': np.arange(len(subject_response_mean)),
            'nsd_id': SUBJECTS_DATA[(subject, froi)]['images_df']['nsdId'],
            'froi': froi,
            'subject': subject,
            'subject_response': subject_response_mean,
        })
        dfs.append(froi_subject_df)
        print(f"Done subject {subject}, {froi}")
dfs = pd.concat(dfs, axis=0, ignore_index=True)

MODEL_NAMES = ma.get_model_names()
for froi in FROIS:
    for model_name in ['resnet50imagenet1kv2']:
        model_data = get_model_data(model_name, froi)
        for subject in SUBJECTS:
            mask = (dfs["subject"] == subject) & (dfs["froi"] == froi)
            nsdids = dfs[mask]['nsd_id'].values
            model_data_sbj = model_data[nsdids]
            model_response_mean = model_data_sbj.mean(axis=-1).cpu().numpy()
            dfs.loc[mask, f'{model_name}_response'] = model_response_mean
            dfs.to_csv(mean_responses_df_path, index=False)
        print(f"Done {model_name}, {froi}")

resp_df = pd.read_csv(mean_responses_df_path)

TOP_MODEL_NAMES = [
    'vitb16siglip2res224',
    'resnet50openaiclip',
    'convnexttinyimagenet1kv1',
    'resnet101openaiclip',
    'vitb14dinov2reg',
    'resnet50simclr',
    'resnet50toponettau30',
    'cornets',
    'resnet101imagenet1kv1',
    'vitb14dinov2'
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
            resp_df.loc[mask, f'{model_name}_response'] = model_response_mean
            resp_df.to_csv(mean_responses_df_path, index=False)
            print(f"Done {model_name}, {froi}, subject {subject}")
    print(f"Done {model_name}")