# Importing the necessary libraries
import os
import time
import argparse
import pandas as pd
import torch

from compcats.models import *
from compcats.datasets.fmri import nsd
from compcats.utils import seed_utils, misc_utils

# Analysis files
import analysis_layersweep

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

# Seed
SEED = 42
seed_utils.seed_everything(SEED)

# Models, Status of Layer sweeps
MODELS = {
    'alexnetimagenet1kv1': alexnet_imagenet1kv1, # Done
    'alexnetipclref01': alexnet_ipcl, # Done
    'alexnetbarlowtwins': alexnet_barlowtwins, # Done
    'alexnetrandom42': alexnet_random, # Done
    'resnet18imagenet1kv1': resnet18_imagenet1kv1, # Done
    'resnet50imagenet1kv2': resnet50_imagenet1kv2, # Done
    'resnet50imagenet21k': resnet50_imagenet21k, # Done
    'resnet50openaiclip': resnet50_openaiclip, # Done
    'resnet50simclr': resnet50_simclr, # Done
    'resnet50robustl2eps3': resnet50_robust, # Done
    'resnet50sin': resnet50_sin, # Done
    'resnet50faceobfuscation': resnet50_faceobfuscation, # Done
    'resnet50toponettau30': resnet50_toponet, # Done
    'resnet50random42': resnet50_random, # Done
    'vitb32imagenet1kv1': vitb32_imagenet1kv1, # Done
    'vitb32openaiclip': vitb32_openaiclip, # Done
    'vitb32random42': vitb32_random, # Done
    'vitb16imagenet21k': vitb16_imagenet21k, # Done
    'vitb16openaiclip': vitb16_openaiclip, # Done
    'vitb16siglip2res224': vitb16_siglip2, # Done
    'vitl14openaiclip': vitl14_openaiclip, # Done
    'vgg16imagenet1kv1': vgg16_imagenet1kv1, # Done
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

def get_model(model_name):
    """
    Import model module
    Args:
        model_module (str): Model name
    Returns:
        model_module (module): Model module
    """
    model_module = MODELS[model_name]
    model = model_module.get_model()
    assert model.name == model_name, 'Model name mismatch'
    return model

def get_all_models():
    """
    Get list of models used for the analysis
    """
    models = {}
    model_names = get_model_names()
    for model_name in model_names:
        model = get_model(model_name)
        models[model_name] = model
    return models

# Utils
def get_localizer_dir(localizer_name):
    """
    Get localizer directory from global directories
    Args:
        localizer_name (str): Localizer name
    Returns:
        localizer_dir (str): Localizer directory
    """
    if localizer_name == 'grillspector':
        return GRILLSPECTOR_LOCALIZER_DIR
    elif localizer_name == 'kanwisher':
        return KANWISHER_LOCALIZER_DIR
    elif localizer_name == 'konkle':
        return KONKLE_LOCALIZER_DIR
    else:
        raise ValueError('Localizer not implemented')

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

def run_allmodels_layer_sweeps():
    """
    Generate model-human univariate comparison results to select best layer
    NSD 485 images, 4 subjects, FFA, PPA, EBA, t>7
    Args:
        model (Model): Model instance
        subjects_data (dict): Dictionary of subject data
    """
    # Variables
    best_layers_csv_path = f'{MODELS_DIR}/best_layers_nsd485_median_pearsonr.csv'
    if os.path.exists(best_layers_csv_path):
        best_layers_csv = pd.read_csv(best_layers_csv_path)
        best_layers_csv_list = best_layers_csv.to_dict('records')
    else:
        best_layers_csv_list = []

    subjects = [1, 2, 5, 7]
    subset = 485
    frois = ['FFA', 'PPA', 'EBA']
    tthreshold = 7
    model_localizer = 'grillspector'
    subjects_data = get_humandata_layersweep(
        subjects, frois, tthreshold, subset
    )
    for model_name in get_model_names():
        model = get_model(model_name)
        relevant_layers = model.get_relevant_layers()
        start_time = time.time()
        print(f'Starting layer sweep for model: {model_name}')
        model_layersweep_csv = analysis_layersweep.model_layer_sweep(
            MODELS_DIR, NSD_DIR, get_localizer_dir(model_localizer), \
            subjects_data, model, relevant_layers, subset, frois, tthreshold, model_localizer
        )
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # Convert to minutes
        print(f'Layer sweep for model {model_name} completed in {elapsed_time:.2f} minutes')

        # assert layersweep is complete
        assert model_layersweep_csv is not None, 'Layer sweep not complete'
        assert len(model_layersweep_csv) == (len(relevant_layers) * len(frois)), \
            'Layer sweep not complete'

        # Empty cache
        del model
        torch.cuda.empty_cache()

        # Find best layer
        best_layer_info = analysis_layersweep.get_model_best_layer(
            model_layersweep_csv, model_name, frois
        )
        # Get best layers
        best_layers = list(set([info['model_layer'] for froi, info in best_layer_info.items()]))
        # Save best layers
        # If already present, check if best layer is same
        for froi, froi_best_layer_info in best_layer_info.items():
            froi_best_layer_present = misc_utils.check_row_present(
                {
                    'model_name': model_name,
                    'model_froi': froi,
                    'model_tthreshold': tthreshold,
                    'model_localizer': model_localizer,
                    'fmri_dataset': f'nsd{subset}',
                    'fmri_froi': froi,
                    'fmri_tthreshold': tthreshold
                }, best_layers_csv_list
            )
            if froi_best_layer_present:
                previous_best_layer = froi_best_layer_present['model_layer']
                current_best_layer = froi_best_layer_info['model_layer']
                assert previous_best_layer == current_best_layer, 'Best layer mismatch'
            else:
                best_layers_csv_list.append(froi_best_layer_info)
                best_layers_csv = pd.DataFrame(best_layers_csv_list)
                best_layers_csv.to_csv(best_layers_csv_path, index=False)

        # Delete activations of all layers except best layers to save space
        delete_layers = [layer for layer in relevant_layers if layer not in best_layers]
        # Delete localizer activations
        analysis_layersweep.delete_activations(
            f'{MODELS_DIR}/{model_name}/{model_localizer}/activations', delete_layers
        )
        # Delete NSD activations
        analysis_layersweep.delete_activations(
            f'{MODELS_DIR}/{model_name}/nsd', delete_layers
        )

    # Empty cache
    del subjects_data
    torch.cuda.empty_cache()

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
