# Importing the necessary libraries
import os
import time
import argparse
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm

from compcats.models import *
from compcats.datasets.fmri import nsd
from compcats.localizer import localizer
from compcats.utils import model_utils, seed_utils
from compcats.utils import responsecomparison_utils as rcutils

# Variables
# Directories
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
NSD_DIR = '/weka/scratch/weka/sinha/alishd/datasets/nsd'
# Directories
GRILLSPECTOR_LOCALIZER_DIR = '/net/vast-storage/scratch/vast/sinha/alishd/datasets/\
localizer_grillspector/fLoc_stimuli'
KANWISHER_LOCALIZER_DIR = '/net/vast-storage/scratch/vast/sinha/alishd/datasets/\
localizer_kanwisher/images'
KONKLE_LOCALIZER_DIR = '/net/vast-storage/scratch/vast/sinha/alishd/datasets/\
localizer_konkle/CompSearch'
def get_localizer_dir(localizer_name):
    if localizer_name == 'grillspector':
        return GRILLSPECTOR_LOCALIZER_DIR
    elif localizer_name == 'kanwisher':
        return KANWISHER_LOCALIZER_DIR
    elif localizer_name == 'konkle':
        return KONKLE_LOCALIZER_DIR
    else:
        raise ValueError('Localizer not implemented')
# Seed
SEED = 42
seed_utils.seed_everything(SEED)

# Models
MODELS = {
    'alexnetimagenet1kv1': alexnet_imagenet1kv1,
    'alexnetipclref01': alexnet_ipcl,
    'alexnetbarlowtwins': alexnet_barlowtwins,
    'alexnetrandom42': alexnet_random,
    'vgg16imagenet1kv1': vgg16_imagenet1kv1,
    'resnet18imagenet1kv1': resnet18_imagenet1kv1,
    'resnet50imagenet1kv2': resnet50_imagenet1kv2,
    'resnet50imagenet21k': resnet50_imagenet21k,
    'resnet50openaiclip': resnet50_openaiclip,
    'resnet50simclr': resnet50_simclr,
    'resnet50robustl2eps3': resnet50_robust,
    'resnet50sin': resnet50_sin,
    'resnet50faceobfuscation': resnet50_faceobfuscation,
    'resnet50toponettau30': resnet50_toponet,
    'resnet50random42': resnet50_random,
    'vitb32imagenet1kv1': vitb32_imagenet1kv1,
    'vitb32openaiclip': vitb32_openaiclip,
    'vitb32random42': vitb32_random,
    'vitb16imagenet21k': vitb16_imagenet21k,
    'vitb16openaiclip': vitb16_openaiclip,
    'vitb16siglip2res224': vitb16_siglip2,
    'vitl14openaiclip': vitl14_openaiclip,
}

def get_num_models():
    """
    Get number of models used for the analysis
    """
    return len(MODELS)

def get_model_names():
    """
    Get names of models used for the analysis
    """
    return list(MODELS.keys())

def get_models():
    """
    Get list of models used for the analysis
    """
    models = {}
    for model_name, model_module in MODELS.items():
        model = model_module.get_model()
        assert model.name == model_name, 'Model name mismatch'
        models[model_name] = model
    return models

# Utils
def check_row_present(row_params, results_list):
    row_present = False
    # Skip if row is already present
    if any(all(row_params[key] == d.get(key, None) for key in row_params) for d in results_list):
        row_present = True
    return row_present

def get_allsubjects_nsd515_meanrep_voxel_data():
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    subjects_data = {}
    for subject in subjects:
        subject_data, _ = nsd.get_betas(
            NSD_DIR, subject, subset=subset, voxels=True, repetition_mean=True
        )
        subjects_data[subject] = subject_data
    return subjects_data

def get_humandata_layersweep(subjects, frois, tthreshold, subset):
    """
    Get NSD mean of voxels, mean of repetitions data for specified subjects
    Args:
        subjects (list): List of subject numbers
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
        subset (int): Subset of NSD dataset
    """
    subjects_data = {}
    for froi in frois:
        subjects_data[froi] = {}
        for subject in subjects:
            subject_data, _ = nsd.get_betas(
                NSD_DIR, subject, froi=froi, tthreshold=tthreshold, \
                subset=subset, voxels=False, repetition_mean=True
            )
            subjects_data[froi][subject] = subject_data.cuda()
    return subjects_data

def model_layer_sweep(
        subjects_data, model, subset, frois, tthreshold, model_localizer
):
    """
    Generate model-human univariate comparison results to select best layer
    NSD 485 images, 4 subjects, FFA, PPA, EBA, t>7
    Args:
        subjects_data (dict): Dictionary of subject data
        model (Model): Model instance
        subset (int): Subset of NSD dataset
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
        model_localizer (str): Model localizer
    """
    # Load already computed results
    csv_pth = f'{MODELS_DIR}/{model.name}/nsd485_pearsonr_layersweep.csv'
    if os.path.exists(csv_pth):
        results_csv = pd.read_csv(csv_pth)
        results_list = results_csv.to_dict('records')
    else:
        results_list = []

    # Variables
    model_data = None
    model_froi_masks = None
    # Layers to test
    relevant_layers = model.relevant_layer_names

    # Internal functions
    # Generate NSD data for models
    def _get_model_data():
        return model_utils.get_dataset_activations(
            MODELS_DIR, NSD_DIR, model, relevant_layers, dataset='nsd', subset=485
        )
    def _get_model_froi_masks(froi):
        return localizer.get_froi_mask(
            get_localizer_dir(model_localizer), model_localizer, MODELS_DIR, \
            model, relevant_layers, froi, tthreshold
        )
    # Loop over frois
    for froi in frois:
        print(f'Running layer sweep: {model.name}, {froi}')
        for layer in tqdm(relevant_layers):
            # Check if row is present
            row_params = {
                'model_name': model.name,
                'model_layer': layer,
                'model_froi': froi,
                'model_tthreshold': tthreshold,
                'model_localizer': model_localizer,
                'fmri_dataset': f'nsd{subset}',
                'fmri_froi': froi,
                'fmri_tthreshold': tthreshold
            }
            if check_row_present(row_params, results_list):
                continue
            # Correlations
            subject_pearsonrs = []
            for subject, subject_froi_data in subjects_data[froi].items():
                # Get model data
                model_data = _get_model_data() if model_data is None else model_data
                # Get model froi masks
                model_froi_masks = _get_model_froi_masks(froi) if model_froi_masks is None \
                    else model_froi_masks
                # Get model data for the layer
                layer_froi_data = model_data[layer][
                    :, model_froi_masks[layer]
                ].mean(axis=-1)
                # Get model-human comparison results
                pearsonr_ = rcutils.pearsonr(
                    layer_froi_data, subject_froi_data
                )
                subject_pearsonrs.append(pearsonr_)
            # Save results
            results_list.append({
                **row_params,
                'pearsonr': subject_pearsonrs,
                'pearsonr_mean': np.nanmean(subject_pearsonrs),
                'pearsonr_median': np.nanmedian(subject_pearsonrs),
                'pearsonr_sem': scipy.stats.sem(subject_pearsonrs, nan_policy='omit')
            })
            # Save to CSV
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(csv_pth, index=False)

def run_allmodels_layer_sweeps():
    """
    Generate model-human univariate comparison results to select best layer
    NSD 485 images, 4 subjects, FFA, PPA, EBA, t>7
    Args:
        model (Model): Model instance
        subjects_data (dict): Dictionary of subject data
    """
    subjects = [1, 2, 5, 7]
    subset = 485
    frois = ['FFA', 'PPA', 'EBA']
    tthreshold = 7
    model_localizer = 'grillspector'
    subjects_data = get_humandata_layersweep(
        subjects, frois, tthreshold, subset
    )
    models = get_models()
    for model_name, model in models.items():
        start_time = time.time()
        print(f'Starting layer sweep for model: {model_name}')
        model_layer_sweep(
            subjects_data, model, subset, frois, tthreshold, model_localizer
        )
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # Convert to minutes
        print(f'Layer sweep for model {model_name} completed in {elapsed_time:.2f} minutes')


# Get best layer

# Analysis: Held-out z-scoring for category selective units

# Analysis: subject-subject correlations

# Analysis: model-subject correlations

# MDS


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Run analysis')
    parser.add_argument('--layersweep', action='store_true', help='Run layer sweep')
    args = parser.parse_args()

    if args.layersweep:
        run_allmodels_layer_sweeps()

if __name__ == '__main__':
    main()
