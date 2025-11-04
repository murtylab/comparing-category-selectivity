# Importing the necessary libraries
import os
import time
import argparse
import scipy
import numpy as np
import pandas as pd
import torch

# Analysis files
import analysis_layersweep
import analysis_localizationvalidation
import paths

from compcats.models import *
from compcats.datasets.fmri import nsd, murty185, nsdsynthetic
from compcats.datasets.macaque import vinken2025
from compcats.utils import seed_utils, misc_utils, model_utils, reliability_utils
import compcats.utils.responsecomparison_utils as rcutils

# Variables
# Directories
# Get the directory of the current file
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_FILE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
MODELS_DIR = os.path.join(CURRENT_FILE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
NSD_DIR = paths.NSD_DIR
MURTY185_DIR = paths.MURTY185_DIR
VINKEN2025_DIR = paths.VINKEN2025_DIR
# Directories
GRILLSPECTOR_LOCALIZER_DIR = paths.GRILLSPECTOR_LOCALIZER_DIR
KANWISHER_LOCALIZER_DIR = paths.KANWISHER_LOCALIZER_DIR
KONKLE_LOCALIZER_DIR = paths.KONKLE_LOCALIZER_DIR
PRINCE2024_LOCALIZER_DIR = paths.PRINCE2024_LOCALIZER_DIR

# Seed
SEED = 42
seed_utils.seed_everything(SEED)

# Batch size
BATCH_SIZE = 64  # Adjust based on GPU memory

# Device
DEVICE = misc_utils.get_device()
# DEVICE = torch.device('cpu')  # Forcing CPU for reproducibility in preprint results

# Tuning-matched localizer parameters
# Reliability threshold for tuning-matched localizer
RTHRESHOLD = 0.0
# Whether to enforce unique unit per voxel in tuning-matched localizer
UNIQUE_UNIT_PER_VOXEL = False

# Results dirs
BEST_LAYERS_CSV_PATH = f'{RESULTS_DIR}/best_layers_nsd485_median_pearsonr.csv'
LOCALIZATION_VALIDATION_CSV_PATH = f'{RESULTS_DIR}/localization_validation.csv'
SUBJECTSUBJECT_CORRELATIONS_NSD515_CSV_PATH = f'{RESULTS_DIR}/subjectsubject_correlations_nsd515.csv'
MODELSUBJECT_CORRELATIONS_NSD515_CSV_PATH = f'{RESULTS_DIR}/modelsubject_correlations_nsd515.csv'
LOCALIZEREFFECT_CSV_PATH = f'{RESULTS_DIR}/localizer_effect_correlations_nsd515.csv'
TTHRESHOLDEFFECT_CSV_PATH = f'{RESULTS_DIR}/tthreshold_effect_correlations_nsd515.csv'
MODESUBJECT_CORRELATIONS_MURTY185_CSV_PATH = f'{RESULTS_DIR}/modelsubject_correlations_murty185.csv'
SUBJECTSUBJECT_CORRELATIONS_MURTY185_CSV_PATH = f'{RESULTS_DIR}/subjectsubject_correlations_murty185.csv'
CORRELATIONS_VINKEN2025_CSV_PATH = f'{RESULTS_DIR}/monkeysubjectmodel_correlations_vinken2025.csv'
CORRELATIONS_NSDSYN_CSV_PATH = f'{RESULTS_DIR}/modelsubject_correlations_nsdsynthetic.csv'

# Models, Info Log
MODELS = {
# # AlexNets
'alexnetimagenet1kv1': alexnet_imagenet1kv1, # Done
'alexnetipclref01': alexnet_ipcl, # Done
'alexnetbarlowtwins': alexnet_barlowtwins, # Done
# # VGGs
'vgg16imagenet1kv1': vgg16_imagenet1kv1, # Done
# # ResNet-18s
'resnet18imagenet1kv1': resnet18_imagenet1kv1, # Done
'resnet18tdannsupervisedlw10': resnet18_tdannsupervised, # Done
'resnet18tdannsimclrlw1': resnet18_tdannsimclr, # Done
# # ResNet-50s
'resnet50imagenet1kv2': resnet50_imagenet1kv2, # Done
'resnet50openaiclip': resnet50_openaiclip, # Done
'resnet50simclr': resnet50_simclr, # Done
'resnet50robustl2eps3': resnet50_robust, # Done
'resnet50sin': resnet50_sin, # Done
'resnet50ecoset': resnet50_ecoset, # Done
'resnet50faceobfuscation': resnet50_faceobfuscation, # Done
'resnet50toponettau30': resnet50_toponet, # Done
'resnet50cesslbarlowtop': resnet50_cessl, # Done
'resnet50blurstrong': resnet50_blur, # Done
# # ResNet-101s
'resnet101imagenet1kv1': resnet101_imagenet1kv1, # Done
'resnet101openaiclip': resnet101_openaiclip, # Done
# # # BagNets
'bagnet33': bagnet33, # Done
# # # CORNets
'cornets': cornet_s, # Done
'cornetz': cornet_z, # Done
# # # ConvNeXt
'convnexttinyimagenet1kv1': convnext_imagenet1kv1, # Done
# # # ViTs
# # # ViT-B/32
'vitb32imagenet1kv1': vitb32_imagenet1kv1, # Done
'vitb32openaiclip': vitb32_openaiclip, # Done
'vitb32dreamsimopenclip': vitb32_dreamsim, # Done
# # # ViT-B/16
'vitb16imagenet21k': vitb16_imagenet21k, # Done
'vitb16openaiclip': vitb16_openaiclip, # Done
'vitb16siglip2res224': vitb16_siglip2, # Done
'vitb16eva028bs8bb131k': vitb16_eva02, # Done
'vitb16synclr': vitb16_synclr, # Done
# # # ViT-B/14
'vitb14dinov2': vitb14_dinov2, # Done
'vitb14dinov2reg': vitb14_dinov2register, # Done
# # # ViT-L/14
'vitl14openaiclip': vitl14_openaiclip, # Done
'vitl14kosmos2': vitl14_kosmos2, # Done
# # # # Brain optimized models
'alexnetmosaicmultiheadallvisual': alexnet_mosaic,
'resnet18mosaicmultiheadallvisual': resnet18_mosaic,
'squeezenetmosaicmultiheadallvisual': squeezenet_mosaic,
'swintmosaicmultiheadallvisual': swint_mosaic,
# # Randomly initialized models
'alexnetrandom1': alexnet_random,
'alexnetrandom2': alexnet_random,
'alexnetrandom3': alexnet_random,
'alexnetrandom4': alexnet_random,
'alexnetrandom5': alexnet_random,
'resnet18random1': resnet18_random,
'resnet18random2': resnet18_random,
'resnet18random3': resnet18_random,
'resnet18random4': resnet18_random,
'resnet18random5': resnet18_random,
'resnet50random1': resnet50_random,
'resnet50random2': resnet50_random,
'resnet50random3': resnet50_random,
'resnet50random4': resnet50_random,
'resnet50random5': resnet50_random,
'vitb32random1': vitb32_random,
'vitb32random2': vitb32_random,
'vitb32random3': vitb32_random,
'vitb32random4': vitb32_random,
'vitb32random5': vitb32_random,
}

def get_num_models():
    """
    Get number of models used for the analysis
    """
    return len(MODELS)

def get_model_names(model_type='task_optimized'):
    """
    Get names of models used for the analysis
    """
    all_names = list(MODELS.keys())
    names = []
    if model_type == 'all':
        names = all_names
    elif model_type in ['random', 'untrained']:
        names = [name for name in all_names if 'random' in name]
    elif model_type == 'brain_optimized':
        names = [name for name in all_names if 'mosaic' in name]
    elif model_type == 'task_optimized':
        names = [name for name in all_names if 'random' not in name and 'mosaic' not in name]
    elif model_type == 'optimized':
        names = [name for name in all_names if 'random' not in name]
    else:
        raise ValueError(f'Invalid model model_type {model_type}, options: all, random, brain_optimized, task_optimized')
    return names

def get_model(model_name, **kwargs):
    """
    Import model module
    Args:
        model_module (str): Model name
    Returns:
        model_module (module): Model module
    """
    model_module = MODELS[model_name]
    # If random is present in the model name, detect seed from the name and add to kwargs
    if 'random' in model_name:
        seed_str = model_name.split('random')[-1]
        try:
            seed = int(seed_str)
        except ValueError as exc:
            raise ValueError(f'Invalid seed in model name: {model_name}') from exc
        kwargs['seed'] = seed
    model = model_module.get_model(**kwargs)
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

def get_froi_models_layers(model_froi, fmri_froi):
    """
    Get layers of all models used for the analysis
    Args:
        model_froi (str): Model froi
        fmri_froi (str): FMRI froi
    Returns:
        models_layers (dict): Dictionary of model layers
    """
    models_layers = {}
    # Get all model names
    model_names = get_model_names('all')
    # Load best layers csv
    best_layers_csv = pd.read_csv(BEST_LAYERS_CSV_PATH)
    for model_name in model_names:
        # Get best layer
        model_best_layer = best_layers_csv[
            (best_layers_csv['model_name'] == model_name) &
            (best_layers_csv['model_froi'] == model_froi) &
            (best_layers_csv['fmri_froi'] == fmri_froi)
        ]['model_layer']
        # Check if best layer is found
        if len(model_best_layer) == 0:
            raise ValueError(f'Best layer not found for model: {model_name}, model_froi: {model_froi}, fmri_froi: {fmri_froi}')
        elif len(model_best_layer) > 1:
            raise ValueError(f'Multiple best layers found for model: {model_name}, model_froi: {model_froi}, fmri_froi: {fmri_froi}')
        model_best_layer = model_best_layer.values[0]
        # Save best layer
        models_layers[model_name] = model_best_layer
    # Old layers
    # models_layers = {
    #     'alexnetimagenet1kv1': 'features.10',
    #     'alexnetipclref01': 'conv_block_5.0',
    #     'alexnetbarlowtwins': 'backbone.4.1',
    #     'resnet18imagenet1kv1': 'layer4.1.conv2',
    #     'resnet50imagenet1kv2': 'layer4.1.conv2',
    #     # 'resnet50imagenet21k': resnet50_imagenet21k, # Check best layer
    #     'resnet50openaiclip': 'layer4.1.conv2',
    #     # 'resnet50simclr': resnet50_simclr, # Check best layer
    #     # 'resnet50robustl2eps3': resnet50_robust,
    #     'resnet50sin': 'layer4.1.conv2',
    #     'resnet50faceobfuscation': 'layer4.1.conv2',
    #     'resnet50toponettau30': 'layer4.1.conv2',
    #     'vitb32imagenet1kv1': 'encoder.layers.encoder_layer_11.mlp',
    #     'vitb32openaiclip': 'transformer.resblocks.11.mlp',
    #     'vitb16imagenet21k': 'blocks.11.mlp',
    #     'vitb16openaiclip': 'transformer.resblocks.11.mlp',
    #     'vitb16siglip2res224': 'trunk.blocks.11.mlp',
    #     'vitl14openaiclip': 'transformer.resblocks.11.mlp',
    #     'vgg16imagenet1kv1': 'features.28',
    #     # # 'alexnetrandom42': alexnet_random,
    #     # # 'resnet50random42': resnet50_random,
    #     # # 'vitb32random42': vitb32_random,
    # }
    return models_layers

def get_best_median_worst_model_names():
    """
    Get best, median and worst model names for a given froi
    Args:
        froi (str): FMRI froi
    Returns:
        best_model_name (str): Best model name
        median_model_name (str): Median model name
        worst_model_name (str): Worst model name
    """
    raise NotImplementedError('These are not up to date, use best layers csv instead')
    return {
        'best': 'vitb16siglip2res224',
        'median': 'vitb16synclr',
        'worst': 'alexnetipclref01'
    }

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
    elif localizer_name == 'kanwishergrayscale':
        return KANWISHER_LOCALIZER_DIR
    elif localizer_name == 'konkle':
        return KONKLE_LOCALIZER_DIR
    elif localizer_name == 'prince2024':
        return PRINCE2024_LOCALIZER_DIR
    elif localizer_name == 'nsd':
        return NSD_DIR
    else:
        raise ValueError('Localizer not implemented')

def get_allsubjects_nsd515_data(frois, tthreshold, tensors=True):
    """
    Get NSD individual mean of repetitions data for all subjects
    Args:
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
    Returns:
        subjects_data (dict): Dictionary of subject data
    """
    # Variables
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    # Storing results
    subjects_data = {}
    splithalf_pearsonrs = {}
    splithalf_rdmrs = {}
    # Get subject froi mean-repetition mean-voxels data
    for froi in frois:
        subjects_data[froi] = {}
        for subject in subjects:
            subject_froi_data, _ = nsd.get_betas(
                NSD_DIR, subject, subset=subset, voxels=True, repetition_mean=True, \
                froi=froi, tthreshold=tthreshold
            )
            subjects_data[froi][subject] = subject_froi_data
    # Get subject froi split-half correlations
    for froi in frois:
        splithalf_pearsonrs[froi] = {}
        splithalf_rdmrs[froi] = {}
        for subject in subjects:
            subject_data, _ = nsd.get_betas(
                NSD_DIR, subject, froi=froi, tthreshold=tthreshold, \
                subset=subset, voxels=True, repetition_mean=False
            )
            if subject_data is None:
                splithalf_pearsonr_ = np.nan
                splithalf_rdmr_ = np.nan
            else:
                splithalf_pearsonr_ = reliability_utils.get_voxelmean_splithalf_correlation(
                    subject_data.mean(axis=-1)
                )
                splithalf_rdmr_ = reliability_utils.get_rdm_splithalf_correlation(
                    subject_data
                )
            splithalf_pearsonrs[froi][subject] = splithalf_pearsonr_
            splithalf_rdmrs[froi][subject] = splithalf_rdmr_
    if not tensors:
        for froi in frois:
            for subject in subjects:
                subjects_data[froi][subject] = subjects_data[froi][subject].cpu().numpy()
    return subjects_data, splithalf_pearsonrs, splithalf_rdmrs

def get_allmodels_nsd515_data(
    model_fmri_frois, tthreshold, localizer, \
    unitmean, tensors=True, model_type='task_optimized'
):
    """
    Get NSD515 mean of froi units data for all models
    Args:
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
        localizer (str): Localizer name
        unitmean (bool): Whether to take mean of units
        tensors (bool): Whether to return tensors
    """
    subset = 515
    models_data = {}
    for (model_froi, fmri_froi) in model_fmri_frois:
        models_data[(model_froi, fmri_froi)] = {}
        for model_name in get_model_names(model_type):
            model = get_model(model_name)
            layer = get_froi_models_layers(model_froi, fmri_froi)[model_name]
            print(f'Getting model data: {model_name}, layer: {layer}, model_froi: {model_froi}, fmri_froi: {fmri_froi}')
            model_froi_activations = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer, dataset='nsd', \
                subset=subset, \
                froi=model_froi, tthreshold=tthreshold, froi_localizer=localizer, \
                froi_localizer_dir=get_localizer_dir(localizer), batch_size=BATCH_SIZE,
                device=DEVICE, rthreshold=RTHRESHOLD, unique_unit_per_voxel=UNIQUE_UNIT_PER_VOXEL
            )
            if unitmean:
                model_froi_activations = model_froi_activations.mean(axis=-1)
            if not tensors:
                model_froi_activations = model_froi_activations.cpu().numpy()
            models_data[(model_froi, fmri_froi)][model_name] = model_froi_activations
    return models_data

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
    splithalf_pearsonrs = {}
    # Get subject froi mean-repetition mean-voxels data
    for froi in frois:
        subjects_data[froi] = {}
        for subject in subjects:
            subject_data, _ = nsd.get_betas(
                NSD_DIR, subject, froi=froi, tthreshold=tthreshold, \
                subset=subset, voxels=False, repetition_mean=True
            )
            subjects_data[froi][subject] = subject_data.to(DEVICE)
    # Get subject froi split-half correlations
    for froi in frois:
        splithalf_pearsonrs[froi] = {}
        for subject in subjects:
            subject_data, _ = nsd.get_betas(
                NSD_DIR, subject, froi=froi, tthreshold=tthreshold, \
                subset=subset, voxels=False, repetition_mean=False
            )
            splithalf_pearsonr_ = reliability_utils.get_voxelmean_splithalf_correlation(
                subject_data
            )
            splithalf_pearsonrs[froi][subject] = splithalf_pearsonr_
    return subjects_data, splithalf_pearsonrs

def get_allsubjects_murty185_data(frois):
    """
    Get Murty185 individual mean of repetitions data for all subjects
    Args:
        frois (list): List of frois
    Returns:
        subjects_data (dict): Dictionary of subject data
    """
    # Variables
    subjects = [1, 2, 3, 4]
    # Storing results
    subjects_data = {}
    # Get subject froi mean-repetition mean-voxels data
    for froi in frois:
        subjects_data[froi] = {}
        for subject in subjects:
            subject_froi_data, _ = murty185.get_betas(
                MURTY185_DIR, subject, froi, voxels=True
            )
            subjects_data[froi][subject] = subject_froi_data
    return subjects_data

def get_allmonkeys_vinken2025_nsd515data(frois, tensors=True):
    """
    Get Vinken2025 individual mean of repetitions data for all monkeys
    Splithalf metrics implemented: pearsonr, rdmr
    Args:
        frois (list): List of frois
        subset (int): Subset of Vinken2025 dataset
    Returns:
        monkeys_data (dict): Dictionary of monkey data with keys as (froi, monkey)
        monkeys_splithalfs (dict): Dictionary of monkey split-half correlations with \
            keys as (froi, monkey) and values as dict of metric keys, correlation values
    """
    # Variables
    keys = []
    # Storing results
    monkeys_data = {}
    monkeys_splithalfs = {}
    for froi in frois:
        froi_monkeys = vinken2025.get_froi_monkeys_present(froi)
        for monkey in froi_monkeys:
            key = (froi, monkey)
            assert key not in keys, 'Duplicate froi-monkey key'
            keys.append(key)
        monkeys_data[froi] = {}
        monkeys_splithalfs[froi] = {}
    keys = list(set(keys))
    for key in keys:
        froi, monkey = key
        monkeys_data[froi][monkey], _ = vinken2025.get_responses(
            VINKEN2025_DIR, monkey, froi, repetition_mean=True
        )
        # Mean split-half correlations
        monkeys_splithalfs[froi][monkey] = {
            'pearsonr': vinken2025.get_splithalf_reliability(
                paths.VINKEN2025_DIR, 'pearsonr', monkey, froi
            )[0],
            'rdmr': vinken2025.get_splithalf_reliability(
                paths.VINKEN2025_DIR, 'rdmr', monkey, froi
            )[0]
        }
        # Convert to tensors if required
        if tensors:
            monkeys_data[froi][monkey] = torch.Tensor(monkeys_data[froi][monkey]).to(DEVICE)
    return monkeys_data, monkeys_splithalfs

def get_allmodels_murty185_data(
    model_fmri_frois, tthreshold, localizer, unitmean, tensors=True, model_type='task_optimized'
):
    """
    Get Murty185 mean of froi units data for all models
    Args:
        frois (list): List of frois
        unitmean (bool): Whether to take mean of units
        tensors (bool): Whether to return tensors
    """
    models_data = {}
    for (model_froi, fmri_froi) in model_fmri_frois:
        models_data[(model_froi, fmri_froi)] = {}
        for model_name in get_model_names(model_type):
            model = get_model(model_name)
            layer = get_froi_models_layers(model_froi, fmri_froi)[model_name]
            model_froi_activations = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer, dataset='murty185', \
                froi=model_froi, tthreshold=tthreshold, froi_localizer=localizer, \
                froi_localizer_dir=get_localizer_dir(localizer), batch_size=BATCH_SIZE,
                device=DEVICE
            )
            if unitmean:
                model_froi_activations = model_froi_activations.mean(axis=-1)
            if not tensors:
                model_froi_activations = model_froi_activations.cpu().numpy()
            models_data[(model_froi, fmri_froi)][model_name] = model_froi_activations
    return models_data

# NSD Sythetic
def get_allsubjects_nsdsyn_data(frois, tthreshold, tensors=True):
    """
    Get NSD individual mean of repetitions data for all subjects
    Args:
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
    Returns:
        subjects_data (dict): Dictionary of subject data
    """
    # Variables
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 'shared_ood_nowords'
    # Storing results
    subjects_data = {}
    splithalf_pearsonrs = {}
    splithalf_rdmrs = {}
    # Get subject froi mean-repetition mean-voxels data
    for froi in frois:
        subjects_data[froi] = {}
        for subject in subjects:
            subject_froi_data, _ = nsdsynthetic.get_betas(
                NSD_DIR, subject, subset=subset, voxels=True, repetition_mean=True, \
                froi=froi, tthreshold=tthreshold
            )
            subjects_data[froi][subject] = subject_froi_data
    # Get subject froi split-half correlations
    for froi in frois:
        splithalf_pearsonrs[froi] = {}
        splithalf_rdmrs[froi] = {}
        for subject in subjects:
            splithalf_pearsonrs[froi][subject], _ = nsdsynthetic.get_splithalf_reliability(
                paths.NSD_DIR, 'pearsonr', subject, froi, tthreshold, subset=subset
            )
            splithalf_rdmrs[froi][subject], _ = nsdsynthetic.get_splithalf_reliability(
                paths.NSD_DIR, 'rdmr', subject, froi, tthreshold, subset=subset
            )
    if not tensors:
        for froi in frois:
            for subject in subjects:
                subjects_data[froi][subject] = subjects_data[froi][subject].cpu().numpy()
    return subjects_data, splithalf_pearsonrs, splithalf_rdmrs

def get_allmodels_nsdsyn_data(
    model_fmri_frois, tthreshold, localizer, \
    unitmean, tensors=True, model_type='task_optimized'
):
    """
    Get NSD515 mean of froi units data for all models
    Args:
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
        localizer (str): Localizer name
        unitmean (bool): Whether to take mean of units
        tensors (bool): Whether to return tensors
    """
    subset = 'shared_ood_nowords'
    models_data = {}
    for (model_froi, fmri_froi) in model_fmri_frois:
        models_data[(model_froi, fmri_froi)] = {}
        for model_name in get_model_names(model_type):
            model = get_model(model_name)
            layer = get_froi_models_layers(model_froi, fmri_froi)[model_name]
            model_froi_activations = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer, dataset='nsdsynthetic', \
                subset=subset, \
                froi=model_froi, tthreshold=tthreshold, froi_localizer=localizer, \
                froi_localizer_dir=get_localizer_dir(localizer), batch_size=BATCH_SIZE,
                device=DEVICE
            )
            if unitmean:
                model_froi_activations = model_froi_activations.mean(axis=-1)
            if not tensors:
                model_froi_activations = model_froi_activations.cpu().numpy()
            models_data[(model_froi, fmri_froi)][model_name] = model_froi_activations
    return models_data

# Layer sweeps
def run_analysis_nsd485layersweep():
    """
    Generate model-human univariate comparison results to select best layer
    NSD 485 images, 4 subjects, FFA, PPA, EBA, t>7
    Args:
        model (Model): Model instance
        subjects_data (dict): Dictionary of subject data
    """
    # Variables
    subjects = [1, 2, 5, 7]
    subset = 485
    model_fmri_frois_core = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]
    model_fmri_frois_extended = [
        ('FFA', 'FFA'),
        ('FFA', 'OFA'),
        ('EBA', 'EBA'),
        ('EBA', 'FBA'),
        ('PPA', 'PPA'),
        ('PPA', 'OPA'),
        ('PPA', 'RSC'),
        ('VWFA', 'VWFA'),
        ('VWFA', 'OVWFA')
    ]
    model_fmri_frois_all = model_fmri_frois_core + model_fmri_frois_extended
    # Choose model_fmri_frois, core or extended
    model_fmri_frois = model_fmri_frois_all
    tthreshold = 7
    model_localizer = 'grillspector'

    # Storing results
    best_layers_csv_path = BEST_LAYERS_CSV_PATH
    if os.path.exists(best_layers_csv_path):
        best_layers_csv = pd.read_csv(best_layers_csv_path)
        best_layers_csv_list = best_layers_csv.to_dict('records')
    else:
        best_layers_csv_list = []

    # Get subjects data
    fmri_frois = list({i[1] for i in model_fmri_frois})
    subjects_data, subjects_splithalf_pearsonrs = get_humandata_layersweep(
        subjects, fmri_frois, tthreshold, subset
    )

    # To save GPU memory, can chunk layers and run layer sweep for each chunk
    # chunk_size = 0 for no chunking
    chunk_size = 0
    if chunk_size > 0:
        print(f'Layer sweep chunk size: {chunk_size} layers')

    # Run layer sweep
    for model_name in get_model_names():
        # Check if some layersweep is already done
        model_layersweep_csv = None
        layers_swept = []
        model_layersweep_csv_path = f'{MODELS_DIR}/{model_name}/nsd485_pearsonr_layersweep.csv'
        if os.path.exists(model_layersweep_csv_path):
            model_layersweep_csv = pd.read_csv(model_layersweep_csv_path)
            layers_swept = analysis_layersweep.get_layers_already_swept(
                model_layersweep_csv, model_name, model_fmri_frois
            )
        model = get_model(model_name)
        relevant_layers = model.get_relevant_layers()
        # Remove layers already swept
        relevant_layers = [layer for layer in relevant_layers if layer not in layers_swept]
        start_time = time.time()
        print(f'\nStarting layer sweep for model: {model_name}')
        # Run layer sweep for each chunk
        if chunk_size > 0:
            layer_chunks = misc_utils.chunk_numbers(len(relevant_layers), chunk_size)
        else:
            layer_chunks = [(0, len(relevant_layers))]
        for start, end in layer_chunks:
            subset_relevant_layers = relevant_layers[start:end]
            model_layersweep_csv = analysis_layersweep.model_layer_sweep(
                MODELS_DIR, NSD_DIR, get_localizer_dir(model_localizer), \
                subjects_data, subjects_splithalf_pearsonrs, \
                model, subset_relevant_layers, subset, model_fmri_frois, tthreshold, model_localizer,
                batch_size=BATCH_SIZE
            )
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # Convert to minutes
        print(f'Layer sweep for model {model_name} completed in {elapsed_time:.2f} minutes\n')

        # assert layersweep is complete
        assert model_layersweep_csv is not None, 'Layer sweep csv not found'

        # Load the final layersweep csv
        model_layersweep_csv = pd.read_csv(model_layersweep_csv_path)

        # Empty cache
        del model
        misc_utils.clear_memory(DEVICE)

        # Find best layer
        best_layer_info = analysis_layersweep.get_model_best_layer(
            model_layersweep_csv, model_name
        )
        # Get best layers
        best_layers = {info['model_layer'] for _, info in best_layer_info.items()}
        # Save best layers
        # If already present, check if best layer is same
        for (model_froi, fmri_froi), froi_best_layer_info in best_layer_info.items():
            froi_best_layer_present = misc_utils.check_row_present(
                {
                    'model_name': model_name,
                    'model_froi': model_froi,
                    'model_tthreshold': tthreshold,
                    'model_localizer': model_localizer,
                    'fmri_dataset': f'nsd{subset}',
                    'fmri_froi': fmri_froi,
                    'fmri_tthreshold': tthreshold
                }, best_layers_csv_list
            )
            if froi_best_layer_present:
                previous_best_layer = froi_best_layer_present['model_layer']
                current_best_layer = froi_best_layer_info['model_layer']
                try:
                    assert previous_best_layer == current_best_layer, 'Best layer mismatch'
                except AssertionError :
                    print(f'Warning: overwriting best layer info for model: {model_name}, froi: {model_froi}, fmri_froi: {fmri_froi}')
                    print(f'Previous best layer: {previous_best_layer}, Current best layer: {current_best_layer}')
                    print('Breakpoint here if you want to debug')
                    breakpoint()
                    # Remove previous entry
                    best_layers_csv_list = [
                        entry for entry in best_layers_csv_list if not (
                            (entry['model_name'] == model_name) and
                            (entry['model_froi'] == model_froi) and
                            (entry['fmri_froi'] == fmri_froi)
                        )
                    ]
                    best_layers_csv_list.append(froi_best_layer_info)
            else:
                best_layers_csv_list.append(froi_best_layer_info)
            # Write csv
            best_layers_csv = pd.DataFrame(best_layers_csv_list)
            # First check if there are duplicates and remove them
            best_layers_csv = best_layers_csv.drop_duplicates(keep='first')
            best_layers_csv.to_csv(best_layers_csv_path, index=False)

        # Delete activations of all layers except best layers to save space
        delete_layers = [layer for layer in relevant_layers if layer not in best_layers]
        # Delete localizer activations
        misc_utils.delete_activations(
            f'{MODELS_DIR}/{model_name}/{model_localizer}/activations', delete_layers
        )
        # Delete NSD activations
        misc_utils.delete_activations(
            f'{MODELS_DIR}/{model_name}/nsd', delete_layers
        )

    # Empty cache
    del subjects_data
    misc_utils.clear_memory(DEVICE)

# Analysis: Held-out z-scoring for category selective units
def run_analysis_localizationvalidation():
    """
    Analysis: Held-out z-scoring for category selective units to validate localization
    """
    # Variables
    model_type = 'task_optimized'  # 'all', 'random', 'brain_optimized', 'task_optimized', 'optimized'
    model_fmri_frois = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]
    model_frois = list({i[0] for i in model_fmri_frois})
    tthreshold = 3 if model_type == 'random' else 7
    froi_localizer = 'grillspector'
    # froi_localizer = 'kanwisher'
    # froi_localizer = 'kanwishergrayscale'
    print('Validating localization for localizer: ', froi_localizer)
    # validation_localizer = 'konkle'
    # validation_localizer = 'kanwisher'
    validation_localizer = 'prince2024'
    frois_models_layers = {
        model_froi: get_froi_models_layers(model_froi, fmri_froi) \
            for (model_froi, fmri_froi) in model_fmri_frois
    }

    # Storing results
    localization_validation_csv_path = '.'.join(
        LOCALIZATION_VALIDATION_CSV_PATH.split('.')[:-1]
    ) + f'_{froi_localizer}' + f'_val_{validation_localizer}.csv'
    if os.path.exists(localization_validation_csv_path):
        localization_validation_csv = pd.read_csv(localization_validation_csv_path)
        localization_validation_csv_list = localization_validation_csv.to_dict('records')
    else:
        localization_validation_csv_list = []

    for model_name in get_model_names(model_type):
        model = get_model(model_name)
        model_all_layers = list({frois_models_layers[froi][model_name] for froi in model_frois})
        for froi in model_frois:
            model_froi_layer = frois_models_layers[froi][model_name]
            print(f'Running localization validation for model: {model_name}, froi: {froi}')
            row_params = {
                'model_name': model_name,
                'model_layer': model_froi_layer,
                'model_froi': froi,
                'model_froi_tthreshold': tthreshold,
                'model_localization_localizer': froi_localizer,
                'model_validation_localizer': validation_localizer,
            }
            # Check if row is present
            if misc_utils.check_row_present(row_params, localization_validation_csv_list):
                continue
            # Run localization validation
            froi_localizer_dir = get_localizer_dir(froi_localizer)
            validation_localizer_dir = get_localizer_dir(validation_localizer)
            tvalue, pvalue = \
                analysis_localizationvalidation.model_layer_froi_localization_validation(
                    model, model_froi_layer, froi, tthreshold, froi_localizer, froi_localizer_dir, \
                    MODELS_DIR, validation_localizer, validation_localizer_dir, \
                    batch_size=BATCH_SIZE, generate_layers=model_all_layers
                )
            # Get t-values for original localizer
            localization_tvalue, localization_pvalue = \
                analysis_localizationvalidation.model_layer_froi_localization_validation(
                    model, model_froi_layer, froi, tthreshold, froi_localizer, froi_localizer_dir, \
                    MODELS_DIR, froi_localizer, froi_localizer_dir, \
                    batch_size=BATCH_SIZE, generate_layers=model_all_layers
                )
            # Save results
            localization_validation_csv_list.append({
                **row_params,
                'validation_tvalue': tvalue,
                'validation_pvalue': pvalue,
                'localization_tvalue': localization_tvalue,
                'localization_pvalue': localization_pvalue
            })
            localization_validation_csv = pd.DataFrame(localization_validation_csv_list)
            localization_validation_csv.to_csv(localization_validation_csv_path, index=False)

        # Delete validation localizer activations to save space
        misc_utils.delete_activations(
            f'{MODELS_DIR}/{model_name}/{validation_localizer}/activations', model_all_layers
        )

# Analysis: subject-subject correlations
def run_analysis_nsd515subjectsubjectcorrelations():
    """
    Analysis: Subject-subject correlations, univariate pearsonr and multivariate RDMr
    Also normalized by split-half correlations
    """
    # Define frois
    frois_core = ['FFA', 'PPA', 'EBA']
    frois_all = frois_core + ['OFA', 'OPA', 'RSC', 'FBA', 'VWFA', 'OVWFA']

    # Variables
    frois = frois_all
    tthresholds = [3, 5, 7, 9, 11]
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515

    # Storing results
    subjectsubject_correlations_csv_path = SUBJECTSUBJECT_CORRELATIONS_NSD515_CSV_PATH
    if os.path.exists(subjectsubject_correlations_csv_path):
        subjectsubject_correlations_csv = pd.read_csv(subjectsubject_correlations_csv_path)
        subjectsubject_correlations_csv_list = subjectsubject_correlations_csv.to_dict('records')
    else:
        subjectsubject_correlations_csv_list = []

    for tthreshold in tthresholds:
        subjects_voxel_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
            get_allsubjects_nsd515_data(frois, tthreshold, tensors=True)
        for subject1_idx, subject1 in enumerate(subjects):
            for subject2_idx, subject2 in enumerate(subjects[subject1_idx+1:]):
                for froi in frois:
                    row_params = {
                        'fmri_froi': froi,
                        'fmri_tthreshold': tthreshold,
                        'fmri_dataset': f'nsd{subset}',
                        'subject1': subject1,
                        'subject2': subject2
                    }
                    # Check if row is present
                    if misc_utils.check_row_present(row_params, subjectsubject_correlations_csv_list):
                        continue
                    # Run subject-subject correlation
                    subject1_voxel_data = subjects_voxel_data[froi][subject1]
                    subject2_voxel_data = subjects_voxel_data[froi][subject2]
                    # split-half pearsonrs
                    subject1_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi][subject1]
                    subject2_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi][subject2]
                    # split-half rdmrs
                    subject1_splithalf_rdmr = subjects_splithalf_rdmrs[froi][subject1]
                    subject2_splithalf_rdmr = subjects_splithalf_rdmrs[froi][subject2]
                    if subject1_voxel_data is None or \
                        subject2_voxel_data is None or \
                        np.isnan(subject1_splithalf_pearsonr) or \
                        np.isnan(subject2_splithalf_pearsonr) or \
                        np.isnan(subject1_splithalf_rdmr) or \
                        np.isnan(subject2_splithalf_rdmr):
                        # Set to nan
                        pearsonr = np.nan
                        rdmr = np.nan
                        pvalue_pearsonr = np.nan
                        pvalue_rdmr = np.nan
                        normed_pearsonr = np.nan
                        normed_rdmr = np.nan
                    else:
                        # Pearsonr
                        pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                            subject1_voxel_data.mean(axis=-1), subject2_voxel_data.mean(axis=-1)
                        )
                        normed_pearsonr = \
                            pearsonr / np.sqrt(subject1_splithalf_pearsonr * subject2_splithalf_pearsonr)
                        # RDMr
                        rdmr, pvalue_rdmr = rcutils.rdmr(
                            subject1_voxel_data, subject2_voxel_data, return_pvalue=True
                        )
                        normed_rdmr = \
                            rdmr / np.sqrt(subject1_splithalf_rdmr * subject2_splithalf_rdmr)
                    # Save results
                    subjectsubject_correlations_csv_list.append({
                        **row_params,
                        'normed_pearsonr': normed_pearsonr,
                        'pearsonr': pearsonr,
                        'normed_rdmr': normed_rdmr,
                        'rdmr': rdmr,
                        'pvalue_rdmr': pvalue_rdmr,
                        'pvalue_pearsonr': pvalue_pearsonr,
                        'subject1_splithalf_pearsonr': subject1_splithalf_pearsonr,
                        'subject2_splithalf_pearsonr': subject2_splithalf_pearsonr,
                        'subject1_splithalf_rdmr': subject1_splithalf_rdmr,
                        'subject2_splithalf_rdmr': subject2_splithalf_rdmr
                    })
                    subjectsubject_correlations_csv = pd.DataFrame(subjectsubject_correlations_csv_list)
                    subjectsubject_correlations_csv.to_csv(
                        subjectsubject_correlations_csv_path, index=False
                    )

# Analysis: model-subject correlations
def run_analysis_nsd515modelsubjectcorrelations():
    """
    TODO
    """
    model_type = 'task_optimized'  # 'all', 'random', 'brain_optimized', 'task_optimized', 'optimized'
    # Define frois
    model_fmri_frois_core = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]
    model_fmri_frois_extended = [
        ('FFA', 'FFA'),
        ('FFA', 'OFA'),
        ('EBA', 'EBA'),
        ('EBA', 'FBA'),
        ('PPA', 'PPA'),
        ('PPA', 'OPA'),
        ('PPA', 'RSC'),
        ('VWFA', 'VWFA'),
        ('VWFA', 'OVWFA')
    ]
    model_fmri_frois_all = model_fmri_frois_core + model_fmri_frois_extended

    # Variables
    model_fmri_frois = model_fmri_frois_core
    fmri_frois = list({i[1] for i in model_fmri_frois})
    tthreshold = 3 if model_type == 'random' else 7
    # Subjects
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    # Models
    model_localizer = 'nsd'

    # Storing results
    modelsubject_correlations_csv_path = MODELSUBJECT_CORRELATIONS_NSD515_CSV_PATH
    if os.path.exists(modelsubject_correlations_csv_path):
        modelsubject_correlations_csv = pd.read_csv(modelsubject_correlations_csv_path)
        modelsubject_correlations_csv_list = modelsubject_correlations_csv.to_dict('records')
    else:
        modelsubject_correlations_csv_list = []

    subjects_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
        get_allsubjects_nsd515_data(fmri_frois, tthreshold)
    models_data = get_allmodels_nsd515_data(
        model_fmri_frois, tthreshold, model_localizer, unitmean=False, tensors=True,
        model_type=model_type
    )

    for model_name in get_model_names(model_type):
        for subject in subjects:
            for (model_froi, fmri_froi) in model_fmri_frois:
                row_params = {
                    'model_name': model_name,
                    'model_layer': get_froi_models_layers(model_froi, fmri_froi)[model_name],
                    'model_froi': model_froi,
                    'model_tthreshold': tthreshold,
                    'model_localizer': model_localizer,
                    'fmri_dataset': f'nsd{subset}',
                    'fmri_froi': fmri_froi,
                    'fmri_tthreshold': tthreshold,
                    'subject': subject
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, modelsubject_correlations_csv_list):
                    continue
                # Run model-subject correlation
                # Get data
                model_data = models_data[(model_froi, fmri_froi)][model_name]
                subject_data = subjects_data[fmri_froi][subject]
                subject_splithalf_pearsonr = subjects_splithalf_pearsonrs[fmri_froi][subject]
                subject_splithalf_rdmr = subjects_splithalf_rdmrs[fmri_froi][subject]
                # RDMr
                rdmr, pvalue_rdmr = rcutils.rdmr(
                    model_data, subject_data, return_pvalue=True
                )
                # Normalize by split-half correlation
                normed_rdmr = rdmr / np.sqrt(subject_splithalf_rdmr)
                # Pearsonr
                pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                    model_data.mean(axis=-1).cpu().numpy(), 
                    subject_data.mean(axis=-1).cpu().numpy()
                )
                # Normalize by split-half correlation
                normed_pearsonr = pearsonr / np.sqrt(subject_splithalf_pearsonr)
                # Save results
                modelsubject_correlations_csv_list.append({
                    **row_params,
                    'normed_pearsonr': normed_pearsonr,
                    'pearsonr': pearsonr,
                    'normed_rdmr': normed_rdmr,
                    'rdmr': rdmr,
                    'pvalue_rdmr': pvalue_rdmr,
                    'pvalue_pearsonr': pvalue_pearsonr,
                    'fmri_splithalf_pearsonr': subject_splithalf_pearsonr,
                    'fmri_splithalf_rdmr': subject_splithalf_rdmr
                })
                modelsubject_correlations_csv = pd.DataFrame(modelsubject_correlations_csv_list)
                modelsubject_correlations_csv.to_csv(
                    modelsubject_correlations_csv_path, index=False
                )

# Analysis: model-localizer effect
def run_analysis_nsd515localizereffect():
    """
    TODO
    """
    model_type = 'task_optimized'  # 'all', 'random', 'brain_optimized', 'task_optimized', 'optimized'
    # Define frois
    model_fmri_frois = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]

    # Variables
    fmri_frois = list({i[1] for i in model_fmri_frois})
    tthreshold = 3 if model_type == 'random' else 7
    # Subjects
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    # Models
    model_localizers = ['grillspector', 'kanwisher', 'kanwishergrayscale']

    # Storing results
    localizereffect_correlations_csv_path = LOCALIZEREFFECT_CSV_PATH
    if os.path.exists(localizereffect_correlations_csv_path):
        localizereffect_correlations_csv = pd.read_csv(localizereffect_correlations_csv_path)
        localizereffect_correlations_csv_list = localizereffect_correlations_csv.to_dict('records')
    else:
        localizereffect_correlations_csv_list = []

    subjects_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
        get_allsubjects_nsd515_data(fmri_frois, tthreshold)

    def _get_model_data(model_localizer):
        return get_allmodels_nsd515_data(
            model_fmri_frois, tthreshold, model_localizer, unitmean=False, tensors=True,
            model_type=model_type
        )

    for model_localizer in model_localizers:
        print(f'Running model-localizer effect for localizer: {model_localizer}')
        models_data = None
        for model_name in get_model_names():
            for subject in subjects:
                for (model_froi, fmri_froi) in model_fmri_frois:
                    row_params = {
                        'model_name': model_name,
                        'model_layer': get_froi_models_layers(model_froi, fmri_froi)[model_name],
                        'model_froi': model_froi,
                        'model_tthreshold': tthreshold,
                        'model_localizer': model_localizer,
                        'fmri_dataset': f'nsd{subset}',
                        'fmri_froi': fmri_froi,
                        'fmri_tthreshold': tthreshold,
                        'subject': subject
                    }
                    # Check if row is present
                    if misc_utils.check_row_present(row_params, localizereffect_correlations_csv_list):
                        continue
                    # Get model data
                    if models_data is None:
                        models_data = _get_model_data(model_localizer)
                    # Run model-subject correlation
                    # Get data
                    model_data = models_data[(model_froi, fmri_froi)][model_name]
                    subject_data = subjects_data[fmri_froi][subject]
                    subject_splithalf_pearsonr = subjects_splithalf_pearsonrs[fmri_froi][subject]
                    subject_splithalf_rdmr = subjects_splithalf_rdmrs[fmri_froi][subject]
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        model_data, subject_data, return_pvalue=True
                    )
                    # Normalize by split-half correlation
                    normed_rdmr = rdmr / np.sqrt(subject_splithalf_rdmr)
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        model_data.mean(axis=-1).cpu().numpy(),
                        subject_data.mean(axis=-1).cpu().numpy()
                    )
                    # Normalize by split-half correlation
                    normed_pearsonr = pearsonr / np.sqrt(subject_splithalf_pearsonr)
                    # Save results
                    localizereffect_correlations_csv_list.append({
                        **row_params,
                        'normed_pearsonr': normed_pearsonr,
                        'pearsonr': pearsonr,
                        'normed_rdmr': normed_rdmr,
                        'rdmr': rdmr,
                        'pvalue_rdmr': pvalue_rdmr,
                        'pvalue_pearsonr': pvalue_pearsonr,
                        'fmri_splithalf_pearsonr': subject_splithalf_pearsonr,
                        'fmri_splithalf_rdmr': subject_splithalf_rdmr
                    })
                    localizereffect_correlations_csv = pd.DataFrame(localizereffect_correlations_csv_list)
                    localizereffect_correlations_csv.to_csv(
                        localizereffect_correlations_csv_path, index=False
                    )

def run_analysis_nsd515tthresholdeffect():
    """
    TODO
    """
    model_type = 'task_optimized'  # 'all', 'random', 'brain_optimized', 'task_optimized', 'optimized'
    # Define frois
    model_fmri_frois_core = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]
    model_fmri_frois_extended = [
        ('FFA', 'FFA'),
        ('FFA', 'OFA'),
        ('EBA', 'EBA'),
        ('EBA', 'FBA'),
        ('PPA', 'PPA'),
        ('PPA', 'OPA'),
        ('PPA', 'RSC'),
        ('VWFA', 'VWFA'),
        ('VWFA', 'OVWFA')
    ]
    model_fmri_frois_all = model_fmri_frois_core + model_fmri_frois_extended

    # Variables
    model_fmri_frois = model_fmri_frois_core
    fmri_frois = list({i[1] for i in model_fmri_frois})
    tthresholds = [3, 5, 7, 9, 11]
    # Subjects
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    # Models
    model_localizer = 'grillspector'

    # Storing results
    tthresholdeffect_correlations_csv_path = TTHRESHOLDEFFECT_CSV_PATH
    if os.path.exists(tthresholdeffect_correlations_csv_path):
        tthresholdeffect_correlations_csv = pd.read_csv(tthresholdeffect_correlations_csv_path)
        tthresholdeffect_correlations_csv_list = tthresholdeffect_correlations_csv.to_dict('records')
    else:
        tthresholdeffect_correlations_csv_list = []

    for model_tthreshold in tthresholds:
        models_data = get_allmodels_nsd515_data(
            model_fmri_frois, model_tthreshold, model_localizer, unitmean=False, tensors=True,
            model_type=model_type
        )
        for fmri_tthreshold in tthresholds:
            subjects_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
                get_allsubjects_nsd515_data(fmri_frois, fmri_tthreshold)
            for model_name in get_model_names(model_type):
                for subject in subjects:
                    for (model_froi, fmri_froi) in model_fmri_frois:
                        row_params = {
                            'model_name': model_name,
                            'model_layer': get_froi_models_layers(model_froi, fmri_froi)[model_name],
                            'model_froi': model_froi,
                            'model_tthreshold': model_tthreshold,
                            'model_localizer': model_localizer,
                            'fmri_dataset': f'nsd{subset}',
                            'fmri_froi': fmri_froi,
                            'fmri_tthreshold': fmri_tthreshold,
                            'subject': subject
                        }
                        # Check if row is present
                        if misc_utils.check_row_present(row_params, tthresholdeffect_correlations_csv_list):
                            continue
                        # Run model-subject correlation
                        # Get data
                        model_data = models_data[(model_froi, fmri_froi)][model_name]
                        subject_data = subjects_data[fmri_froi][subject]
                        subject_splithalf_pearsonr = subjects_splithalf_pearsonrs[fmri_froi][subject]
                        subject_splithalf_rdmr = subjects_splithalf_rdmrs[fmri_froi][subject]
                        # RDMr
                        rdmr, pvalue_rdmr = rcutils.rdmr(
                            model_data, subject_data, return_pvalue=True
                        )
                        # Normalize by split-half correlation
                        normed_rdmr = rdmr / np.sqrt(subject_splithalf_rdmr)
                        # Pearsonr
                        pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                            model_data.mean(axis=-1).cpu().numpy(),
                            subject_data.mean(axis=-1).cpu().numpy()
                        )
                        # Normalize by split-half correlation
                        normed_pearsonr = pearsonr / np.sqrt(subject_splithalf_pearsonr)
                        # Save results
                        tthresholdeffect_correlations_csv_list.append({
                            **row_params,
                            'normed_pearsonr': normed_pearsonr,
                            'pearsonr': pearsonr,
                            'normed_rdmr': normed_rdmr,
                            'rdmr': rdmr,
                            'pvalue_rdmr': pvalue_rdmr,
                            'pvalue_pearsonr': pvalue_pearsonr,
                            'fmri_splithalf_pearsonr': subject_splithalf_pearsonr,
                            'fmri_splithalf_rdmr': subject_splithalf_rdmr
                        })
                        tthresholdeffect_correlations_csv = pd.DataFrame(tthresholdeffect_correlations_csv_list)
                        tthresholdeffect_correlations_csv.to_csv(
                            tthresholdeffect_correlations_csv_path, index=False
                        )

# Other dataset: Murty185
def run_analysis_murty185evaluation():
    """
    Analysis: Murty185 dataset evaluation for subject-subject and model-subject correlations
    """
    model_type = 'all'  # 'all', 'random', 'brain_optimized', 'task_optimized', 'optimized'
    if model_type == 'all':
        raise ValueError('t-threshold has not been decoupled for all models yet. Please specify a model type.')
    # Define frois
    model_fmri_frois_core = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]
    model_fmri_frois_all = model_fmri_frois_core

    # Variables
    model_fmri_frois = model_fmri_frois_all
    fmri_frois = list({i[1] for i in model_fmri_frois})
    tthreshold = 3 if model_type == 'random' else 7
    # Subjects
    subjects = [1, 2, 3, 4]
    # Models
    model_localizer = 'grillspector'

    # SUBJECT-SUBJECT correlations
    # Storing results
    subjectsubject_correlations_csv_path = SUBJECTSUBJECT_CORRELATIONS_MURTY185_CSV_PATH
    if os.path.exists(subjectsubject_correlations_csv_path):
        subjectsubject_correlations_csv = pd.read_csv(subjectsubject_correlations_csv_path)
        subjectsubject_correlations_csv_list = subjectsubject_correlations_csv.to_dict('records')
    else:
        subjectsubject_correlations_csv_list = []

    subjects_data = get_allsubjects_murty185_data(fmri_frois)
    for subject1_idx, subject1 in enumerate(subjects):
        for subject2_idx, subject2 in enumerate(subjects[subject1_idx+1:]):
            for froi in fmri_frois:
                row_params = {
                    'fmri_froi': froi,
                    'fmri_dataset': 'murty185',
                    'subject1': subject1,
                    'subject2': subject2
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, subjectsubject_correlations_csv_list):
                    continue
                # Run subject-subject correlation
                subject1_voxel_data = subjects_data[froi][subject1]
                subject2_voxel_data = subjects_data[froi][subject2]
                if subject1_voxel_data is None or \
                    subject2_voxel_data is None:
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        subject1_voxel_data.mean(axis=-1), subject2_voxel_data.mean(axis=-1)
                    )
                    # RDMr
                    # Convert to tensors for rdmr calculation
                    subject1_voxel_data = torch.tensor(subject1_voxel_data)
                    subject2_voxel_data = torch.tensor(subject2_voxel_data)
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        subject1_voxel_data, subject2_voxel_data, return_pvalue=True
                    )
                # Save results
                subjectsubject_correlations_csv_list.append({
                    **row_params,
                    'pearsonr': pearsonr,
                    'rdmr': rdmr,
                    'pvalue_rdmr': pvalue_rdmr,
                    'pvalue_pearsonr': pvalue_pearsonr,
                })
                subjectsubject_correlations_csv = pd.DataFrame(subjectsubject_correlations_csv_list)
                subjectsubject_correlations_csv.to_csv(
                    subjectsubject_correlations_csv_path, index=False
                )


    # MODEL-SUBJECT correlations
    # Storing results
    modelsubject_correlations_csv_path = MODESUBJECT_CORRELATIONS_MURTY185_CSV_PATH
    if os.path.exists(modelsubject_correlations_csv_path):
        modelsubject_correlations_csv = pd.read_csv(modelsubject_correlations_csv_path)
        modelsubject_correlations_csv_list = modelsubject_correlations_csv.to_dict('records')
    else:
        modelsubject_correlations_csv_list = []

    models_data = get_allmodels_murty185_data(
        model_fmri_frois, tthreshold, model_localizer, unitmean=False, tensors=True,
        model_type=model_type
    )

    for model_name in get_model_names(model_type):
        for subject in subjects:
            for (model_froi, fmri_froi) in model_fmri_frois:
                row_params = {
                    'model_name': model_name,
                    'model_layer': get_froi_models_layers(model_froi, fmri_froi)[model_name],
                    'model_froi': model_froi,
                    'model_tthreshold': tthreshold,
                    'model_localizer': model_localizer,
                    'fmri_dataset': 'murty185',
                    'fmri_froi': fmri_froi,
                    'fmri_tthreshold': tthreshold,
                    'subject': subject
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, modelsubject_correlations_csv_list):
                    continue
                # Run model-subject correlation
                # Get data
                model_data = models_data[(model_froi, fmri_froi)][model_name]
                subject_data = subjects_data[fmri_froi][subject]
                # RDMr
                # Convert to tensors for rdmr calculation
                model_data = torch.tensor(model_data) if not isinstance(model_data, torch.Tensor) else model_data
                subject_data = torch.tensor(subject_data) if not isinstance(subject_data, torch.Tensor) else subject_data
                rdmr, pvalue_rdmr = rcutils.rdmr(
                    model_data, subject_data, return_pvalue=True
                )
                # Pearsonr
                pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                    model_data.mean(axis=-1).cpu().numpy(), 
                    subject_data.mean(axis=-1).cpu().numpy()
                )
                # Save results
                modelsubject_correlations_csv_list.append({
                    **row_params,
                    'pearsonr': pearsonr,
                    'rdmr': rdmr,
                    'pvalue_rdmr': pvalue_rdmr,
                    'pvalue_pearsonr': pvalue_pearsonr,
                })
                modelsubject_correlations_csv = pd.DataFrame(modelsubject_correlations_csv_list)
                modelsubject_correlations_csv.to_csv(
                    modelsubject_correlations_csv_path, index=False
                )

# Other dataset: NSD-Synthetic
def run_analysis_nsdsyntheticevaluation():
    """
    Analysis: NSD-Synthetic dataset evaluation for subject-subject and model-subject correlations
    """
    model_type = 'task_optimized'  # 'all', 'random', 'brain_optimized', 'task_optimized', 'optimized'
    if model_type == 'all':
        raise ValueError('t-threshold has not been decoupled for all models yet. Please specify a model type.')
    # Define frois
    model_fmri_frois_core = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]
    model_fmri_frois_extended = [
        ('FFA', 'FFA'),
        ('FFA', 'OFA'),
        ('EBA', 'EBA'),
        ('EBA', 'FBA'),
        ('PPA', 'PPA'),
        ('PPA', 'OPA'),
        ('PPA', 'RSC'),
        ('VWFA', 'VWFA'),
        ('VWFA', 'OVWFA')
    ]
    model_fmri_frois_all = model_fmri_frois_core + model_fmri_frois_extended

    # Variables
    model_fmri_frois = model_fmri_frois_core
    frois_unique = list({i[1] for i in model_fmri_frois})
    fmri_tthreshold = 7
    subset = 'shared_ood_nowords'
    # Subjects
    nsd_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    # Models
    model_localizer = 'grillspector'
    model_tthreshold = 3 if model_type == 'random' else 7

    # csv for all correlations
    all_correlations_csv_path = CORRELATIONS_NSDSYN_CSV_PATH
    if os.path.exists(all_correlations_csv_path):
        all_correlations_csv = pd.read_csv(all_correlations_csv_path)
        all_correlations_csv_list = all_correlations_csv.to_dict('records')
    else:
        all_correlations_csv_list = []

    # Template for rows
    row_template = {
        'evaluation_subset': None, 
        'system1': None, 'system2': None, 'system1_type': None, 'system2_type': None,
        'system1_source': None, 'system2_source': None,
        'system1_froi': None, 'system2_froi': None, 'metric_type': None, 'metric_normed': None,
        'metric_raw': None, 'metric_pvalue': None,
        'system1_splithalf_metric': None, 'system2_splithalf_metric': None,
        'system1_info': None, 'system2_info': None
    }

    # Human-Human correlations
    subjects_voxel_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
        get_allsubjects_nsdsyn_data(frois_unique, fmri_tthreshold, tensors=True)
    for subject1_idx, subject1 in enumerate(nsd_subjects):
        for subject2_idx, subject2 in enumerate(nsd_subjects[subject1_idx+1:]):
            for (froi1, froi2) in model_fmri_frois:
                row_params = {
                    'evaluation_subset': subset,
                    'system1': f'subject_{subject1}',
                    'system2': f'subject_{subject2}',
                    'system1_type': 'human',
                    'system2_type': 'human',
                    'system1_source': 'nsdsynthetic',
                    'system2_source': 'nsdsynthetic',
                    'system1_froi': froi1,
                    'system2_froi': froi2,
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, all_correlations_csv_list):
                    continue
                # Run subject-subject correlation
                subject1_voxel_data = subjects_voxel_data[froi1][subject1]
                subject2_voxel_data = subjects_voxel_data[froi2][subject2]
                # split-half pearsonrs
                subject1_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi1][subject1]
                subject2_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi2][subject2]
                # split-half rdmrs
                subject1_splithalf_rdmr = subjects_splithalf_rdmrs[froi1][subject1]
                subject2_splithalf_rdmr = subjects_splithalf_rdmrs[froi2][subject2]
                if subject1_voxel_data is None or \
                    subject2_voxel_data is None or \
                    np.isnan(subject1_splithalf_pearsonr) or \
                    np.isnan(subject2_splithalf_pearsonr) or \
                    np.isnan(subject1_splithalf_rdmr) or \
                    np.isnan(subject2_splithalf_rdmr):
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                    normed_pearsonr = np.nan
                    normed_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        subject1_voxel_data.mean(axis=-1), subject2_voxel_data.mean(axis=-1)
                    )
                    normed_pearsonr = \
                        pearsonr / np.sqrt(subject1_splithalf_pearsonr * subject2_splithalf_pearsonr)
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        subject1_voxel_data, subject2_voxel_data, return_pvalue=True
                    )
                    normed_rdmr = \
                        rdmr / np.sqrt(subject1_splithalf_rdmr * subject2_splithalf_rdmr)
                # Save results
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'pearsonr',
                    'metric_normed': normed_pearsonr,
                    'metric_raw': pearsonr,
                    'metric_pvalue': pvalue_pearsonr,
                    'system1_splithalf_metric': subject1_splithalf_pearsonr,
                    'system2_splithalf_metric': subject2_splithalf_pearsonr,
                    'system1_info': {'tthreshold': fmri_tthreshold},
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'rdmr',
                    'metric_normed': normed_rdmr,
                    'metric_raw': rdmr,
                    'metric_pvalue': pvalue_rdmr,
                    'system1_splithalf_metric': subject1_splithalf_rdmr,
                    'system2_splithalf_metric': subject2_splithalf_rdmr,
                    'system1_info': {'tthreshold': fmri_tthreshold},
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv = pd.DataFrame(all_correlations_csv_list)
                all_correlations_csv.to_csv(
                    all_correlations_csv_path, index=False
                )

    # Model-Human correlations
    models_data = get_allmodels_nsdsyn_data(
        model_fmri_frois, model_tthreshold, model_localizer, unitmean=False, tensors=True,
        model_type=model_type
    )
    for model_name in get_model_names(model_type):
        for subject in nsd_subjects:
            for (model_froi, fmri_froi) in model_fmri_frois:
                row_params = {
                    'evaluation_subset': subset,
                    'system1': f'model_{model_name}',
                    'system2': f'subject_{subject}',
                    'system1_type': 'model',
                    'system2_type': 'human',
                    'system1_source': 'nsdsynthetic',
                    'system2_source': f'nsdsynthetic',
                    'system1_froi': model_froi,
                    'system2_froi': fmri_froi,
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, all_correlations_csv_list):
                    continue
                model_info = {
                    'model_name': model_name,
                    'tthreshold': model_tthreshold,
                    'model_layer': get_froi_models_layers(model_froi, fmri_froi)[model_name],
                    'model_localizer': model_localizer
                }
                # Get subject-model correlation
                subject_voxel_data = subjects_voxel_data[fmri_froi][subject]
                model_data = models_data[(model_froi, fmri_froi)][model_name]
                # split-half pearsonrs
                subject_splithalf_pearsonr = subjects_splithalf_pearsonrs[fmri_froi][subject]
                # split-half rdmrs
                subject_splithalf_rdmr = subjects_splithalf_rdmrs[fmri_froi][subject]
                if subject_voxel_data is None or \
                    model_data is None or \
                    np.isnan(subject_splithalf_pearsonr) or \
                    np.isnan(subject_splithalf_rdmr):
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                    normed_pearsonr = np.nan
                    normed_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        subject_voxel_data.mean(axis=-1), model_data.mean(axis=-1).cpu().numpy()
                    )
                    normed_pearsonr = \
                        pearsonr / np.sqrt(subject_splithalf_pearsonr)
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        subject_voxel_data, model_data, return_pvalue=True
                    )
                    normed_rdmr = \
                        rdmr / np.sqrt(subject_splithalf_rdmr)
                # Save results
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'pearsonr',
                    'metric_normed': normed_pearsonr,
                    'metric_raw': pearsonr,
                    'metric_pvalue': pvalue_pearsonr,
                    'system1_splithalf_metric': pd.NA,
                    'system2_splithalf_metric': subject_splithalf_pearsonr,
                    'system1_info': model_info,
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'rdmr',
                    'metric_normed': normed_rdmr,
                    'metric_raw': rdmr,
                    'metric_pvalue': pvalue_rdmr,
                    'system1_splithalf_metric': pd.NA,
                    'system2_splithalf_metric': subject_splithalf_rdmr,
                    'system1_info': model_info,
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv = pd.DataFrame(all_correlations_csv_list)
                all_correlations_csv.to_csv(
                    all_correlations_csv_path, index=False
                )

    # Assert that there are no None values left in the csv
    for row in all_correlations_csv_list:
        for key, value in row.items():
            if value is None:
                raise ValueError(f'Found None value in correlations csv for key: {key}, row: {row}')

# Other dataset: Vinken et al. 2025
def run_analysis_vinken2025evaluation():
    """
    TODO
    """
    # Define frois
    frois = [('FFA', 'FFA')]
    frois_unique = []
    for froi1, froi2 in frois:
        if froi1 not in frois_unique:
            frois_unique.append(froi1)
        if froi2 not in frois_unique:
            frois_unique.append(froi2)
    # Variables
    fmri_tthreshold = 7
    subset = 515
    # Subjects
    nsd_subjects = list(range(1, 9))
    vinken25_monkeys = [2, 3]
    # Models
    model_type = 'task_optimized'  # 'all', 'random', 'brain_optimized', 'task_optimized', 'optimized'
    if model_type == 'all':
        raise ValueError('t-threshold has not been decoupled for all models yet. Please specify a model type.')
    model_localizer = 'grillspector'
    model_tthreshold = 3 if model_type == 'random' else 7
    # Metrics
    metrics = ['pearsonr', 'rdmr']

    # csv for all correlations
    all_correlations_csv_path = CORRELATIONS_VINKEN2025_CSV_PATH
    if os.path.exists(all_correlations_csv_path):
        all_correlations_csv = pd.read_csv(all_correlations_csv_path)
        all_correlations_csv_list = all_correlations_csv.to_dict('records')
    else:
        all_correlations_csv_list = []

    # Template for rows
    # All None should be filled in later, pd.NA are for irrelevant values
    # TODO: before adding to csv, check if the row has all columns as the template 
    # and none of them is None
    row_template = {
        'evaluation_subset': None, 
        'system1': None, 'system2': None, 'system1_type': None, 'system2_type': None,
        'system1_source': None, 'system2_source': None,
        'system1_froi': None, 'system2_froi': None, 'metric_type': None, 'metric_normed': None,
        'metric_raw': None, 'metric_pvalue': None,
        'system1_splithalf_metric': None, 'system2_splithalf_metric': None,
        'system1_info': None, 'system2_info': None
    }

    # Monkey-Monkey correlations
    monkeys_data, monkeys_splithalfs = \
        get_allmonkeys_vinken2025_nsd515data(frois_unique, tensors=True)
    for monkey1_idx, monkey1 in enumerate(vinken25_monkeys):
        for monkey2_idx, monkey2 in enumerate(vinken25_monkeys[monkey1_idx+1:]):
            for (froi1, froi2) in frois:
                row_params = {
                    'evaluation_subset': subset,
                    'system1': f'monkey_{monkey1}',
                    'system2': f'monkey_{monkey2}',
                    'system1_type': 'monkey',
                    'system2_type': 'monkey',
                    'system1_source': 'vinken2025',
                    'system2_source': 'vinken2025',
                    'system1_froi': froi1,
                    'system2_froi': froi2,
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, all_correlations_csv_list):
                    continue
                # Get monkey-monkey correlation
                monkey1_neuron_data = monkeys_data[froi1][monkey1]
                monkey2_neuron_data = monkeys_data[froi2][monkey2]
                # split-half pearsonrs, TODO, keep 1 for now
                monkey1_splithalf_pearsonr = monkeys_splithalfs[froi1][monkey1]['pearsonr']
                monkey2_splithalf_pearsonr = monkeys_splithalfs[froi2][monkey2]['pearsonr']
                # split-half rdmrs
                monkey1_splithalf_rdmr = monkeys_splithalfs[froi1][monkey1]['rdmr']
                monkey2_splithalf_rdmr = monkeys_splithalfs[froi2][monkey2]['rdmr']
                if monkey1_neuron_data is None or \
                    monkey2_neuron_data is None or \
                    np.isnan(monkey1_splithalf_pearsonr) or \
                    np.isnan(monkey2_splithalf_pearsonr) or \
                    np.isnan(monkey1_splithalf_rdmr) or \
                    np.isnan(monkey2_splithalf_rdmr):
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                    normed_pearsonr = np.nan
                    normed_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        monkey1_neuron_data.mean(axis=-1), monkey2_neuron_data.mean(axis=-1)
                    )
                    normed_pearsonr = \
                        pearsonr / np.sqrt(monkey1_splithalf_pearsonr * monkey2_splithalf_pearsonr)
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        monkey1_neuron_data, monkey2_neuron_data, return_pvalue=True
                    )
                    normed_rdmr = \
                        rdmr / np.sqrt(monkey1_splithalf_rdmr * monkey2_splithalf_rdmr)
                # Save results
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'pearsonr',
                    'metric_normed': normed_pearsonr,
                    'metric_raw': pearsonr,
                    'metric_pvalue': pvalue_pearsonr,
                    'system1_splithalf_metric': monkey1_splithalf_pearsonr,
                    'system2_splithalf_metric': monkey2_splithalf_pearsonr,
                    'system1_info': pd.NA,
                    'system2_info': pd.NA
                })
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'rdmr',
                    'metric_normed': normed_rdmr,
                    'metric_raw': rdmr,
                    'metric_pvalue': pvalue_rdmr,
                    'system1_splithalf_metric': monkey1_splithalf_rdmr,
                    'system2_splithalf_metric': monkey2_splithalf_rdmr,
                    'system1_info': pd.NA,
                    'system2_info': pd.NA
                })
                all_correlations_csv = pd.DataFrame(all_correlations_csv_list)
                all_correlations_csv.to_csv(
                    all_correlations_csv_path, index=False
                )

    # Human-Human correlations
    subjects_voxel_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
        get_allsubjects_nsd515_data(frois_unique, fmri_tthreshold, tensors=True)
    for subject1_idx, subject1 in enumerate(nsd_subjects):
        for subject2_idx, subject2 in enumerate(nsd_subjects[subject1_idx+1:]):
            for (froi1, froi2) in frois:
                row_params = {
                    'evaluation_subset': subset,
                    'system1': f'subject_{subject1}',
                    'system2': f'subject_{subject2}',
                    'system1_type': 'human',
                    'system2_type': 'human',
                    'system1_source': f'nsd{subset}',
                    'system2_source': f'nsd{subset}',
                    'system1_froi': froi1,
                    'system2_froi': froi2,
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, all_correlations_csv_list):
                    continue
                # Run subject-subject correlation
                subject1_voxel_data = subjects_voxel_data[froi1][subject1]
                subject2_voxel_data = subjects_voxel_data[froi2][subject2]
                # split-half pearsonrs
                subject1_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi1][subject1]
                subject2_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi2][subject2]
                # split-half rdmrs
                subject1_splithalf_rdmr = subjects_splithalf_rdmrs[froi1][subject1]
                subject2_splithalf_rdmr = subjects_splithalf_rdmrs[froi2][subject2]
                if subject1_voxel_data is None or \
                    subject2_voxel_data is None or \
                    np.isnan(subject1_splithalf_pearsonr) or \
                    np.isnan(subject2_splithalf_pearsonr) or \
                    np.isnan(subject1_splithalf_rdmr) or \
                    np.isnan(subject2_splithalf_rdmr):
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                    normed_pearsonr = np.nan
                    normed_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        subject1_voxel_data.mean(axis=-1), subject2_voxel_data.mean(axis=-1)
                    )
                    normed_pearsonr = \
                        pearsonr / np.sqrt(subject1_splithalf_pearsonr * subject2_splithalf_pearsonr)
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        subject1_voxel_data, subject2_voxel_data, return_pvalue=True
                    )
                    normed_rdmr = \
                        rdmr / np.sqrt(subject1_splithalf_rdmr * subject2_splithalf_rdmr)
                # Save results
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'pearsonr',
                    'metric_normed': normed_pearsonr,
                    'metric_raw': pearsonr,
                    'metric_pvalue': pvalue_pearsonr,
                    'system1_splithalf_metric': subject1_splithalf_pearsonr,
                    'system2_splithalf_metric': subject2_splithalf_pearsonr,
                    'system1_info': {'tthreshold': fmri_tthreshold},
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'rdmr',
                    'metric_normed': normed_rdmr,
                    'metric_raw': rdmr,
                    'metric_pvalue': pvalue_rdmr,
                    'system1_splithalf_metric': subject1_splithalf_rdmr,
                    'system2_splithalf_metric': subject2_splithalf_rdmr,
                    'system1_info': {'tthreshold': fmri_tthreshold},
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv = pd.DataFrame(all_correlations_csv_list)
                all_correlations_csv.to_csv(
                    all_correlations_csv_path, index=False
                )

    # Monkey-Human correlations
    for monkey in vinken25_monkeys:
        for subject in nsd_subjects:
            for (froi1, froi2) in frois:
                row_params = {
                    'evaluation_subset': subset,
                    'system1': f'monkey_{monkey}',
                    'system2': f'subject_{subject}',
                    'system1_type': 'monkey',
                    'system2_type': 'human',
                    'system1_source': 'vinken2025',
                    'system2_source': f'nsd{subset}',
                    'system1_froi': froi1,
                    'system2_froi': froi2,
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, all_correlations_csv_list):
                    continue
                # Get monkey-human correlation
                monkey_neuron_data = monkeys_data[froi1][monkey]
                subject_voxel_data = subjects_voxel_data[froi2][subject]
                # split-half pearsonrs
                monkey_splithalf_pearsonr = monkeys_splithalfs[froi1][monkey]['pearsonr']
                subject_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi2][subject]
                # split-half rdmrs
                monkey_splithalf_rdmr = monkeys_splithalfs[froi1][monkey]['rdmr']
                subject_splithalf_rdmr = subjects_splithalf_rdmrs[froi2][subject]
                if monkey_neuron_data is None or \
                    subject_voxel_data is None or \
                    np.isnan(monkey_splithalf_pearsonr) or \
                    np.isnan(subject_splithalf_pearsonr) or \
                    np.isnan(monkey_splithalf_rdmr) or \
                    np.isnan(subject_splithalf_rdmr):
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                    normed_pearsonr = np.nan
                    normed_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        monkey_neuron_data.mean(axis=-1), subject_voxel_data.mean(axis=-1)
                    )
                    normed_pearsonr = \
                        pearsonr / np.sqrt(monkey_splithalf_pearsonr * subject_splithalf_pearsonr)
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        monkey_neuron_data, subject_voxel_data, return_pvalue=True
                    )
                    normed_rdmr = \
                        rdmr / np.sqrt(monkey_splithalf_rdmr * subject_splithalf_rdmr)
                # Save results
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'pearsonr',
                    'metric_normed': normed_pearsonr,
                    'metric_raw': pearsonr,
                    'metric_pvalue': pvalue_pearsonr,
                    'system1_splithalf_metric': monkey_splithalf_pearsonr,
                    'system2_splithalf_metric': subject_splithalf_pearsonr,
                    'system1_info': pd.NA,
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'rdmr',
                    'metric_normed': normed_rdmr,
                    'metric_raw': rdmr,
                    'metric_pvalue': pvalue_rdmr,
                    'system1_splithalf_metric': monkey_splithalf_rdmr,
                    'system2_splithalf_metric': subject_splithalf_rdmr,
                    'system1_info': pd.NA,
                    'system2_info': {'tthreshold': fmri_tthreshold}
                })
                all_correlations_csv = pd.DataFrame(all_correlations_csv_list)
                all_correlations_csv.to_csv(
                    all_correlations_csv_path, index=False
                )

    # Monkey-Model and Monkey-Model correlations (avoid recomputation of model data)
    models_data = get_allmodels_nsd515_data(
        frois, model_tthreshold, model_localizer, unitmean=False, tensors=True,
        model_type=model_type
    )
    for model_name in get_model_names(model_type):
        # Monkey-Model correlations
        for monkey in vinken25_monkeys:
            for (model_froi, froi2) in frois:
                row_params = {
                    'evaluation_subset': subset,
                    'system1': f'model_{model_name}',
                    'system2': f'monkey_{monkey}',
                    'system1_type': 'model',
                    'system2_type': 'monkey',
                    'system1_source': 'nsd',
                    'system2_source': 'vinken2025',
                    'system1_froi': model_froi,
                    'system2_froi': froi2,
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, all_correlations_csv_list):
                    continue
                model_info = {
                    'model_name': model_name,
                    'tthreshold': model_tthreshold,
                    'model_layer': get_froi_models_layers(model_froi, froi2)[model_name],
                    'model_localizer': model_localizer
                }
                # Get monkey-model correlation
                monkey_neuron_data = monkeys_data[froi2][monkey]
                model_data = models_data[(model_froi, froi2)][model_name]
                # split-half pearsonrs
                monkey_splithalf_pearsonr = monkeys_splithalfs[froi2][monkey]['pearsonr']
                # split-half rdmrs
                monkey_splithalf_rdmr = monkeys_splithalfs[froi2][monkey]['rdmr']
                if monkey_neuron_data is None or \
                    model_data is None or \
                    np.isnan(monkey_splithalf_pearsonr) or \
                    np.isnan(monkey_splithalf_rdmr):
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                    normed_pearsonr = np.nan
                    normed_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        monkey_neuron_data.mean(axis=-1), model_data.mean(axis=-1).cpu().numpy()
                    )
                    normed_pearsonr = \
                        pearsonr / np.sqrt(monkey_splithalf_pearsonr)
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        monkey_neuron_data, model_data, return_pvalue=True
                    )
                    normed_rdmr = \
                        rdmr / np.sqrt(monkey_splithalf_rdmr)
                # Save results
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'pearsonr',
                    'metric_normed': normed_pearsonr,
                    'metric_raw': pearsonr,
                    'metric_pvalue': pvalue_pearsonr,
                    'system1_splithalf_metric': monkey_splithalf_pearsonr,
                    'system2_splithalf_metric': pd.NA,
                    'system1_info': pd.NA,
                    'system2_info': model_info
                })
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'rdmr',
                    'metric_normed': normed_rdmr,
                    'metric_raw': rdmr,
                    'metric_pvalue': pvalue_rdmr,
                    'system1_splithalf_metric': monkey_splithalf_rdmr,
                    'system2_splithalf_metric': pd.NA,
                    'system1_info': pd.NA,
                    'system2_info': model_info
                })
                all_correlations_csv = pd.DataFrame(all_correlations_csv_list)
                all_correlations_csv.to_csv(
                    all_correlations_csv_path, index=False
                )
        # Human-Model correlations
        for subject in nsd_subjects:
            for (model_froi, froi2) in frois:
                row_params = {
                    'evaluation_subset': subset,
                    'system1': f'model_{model_name}',
                    'system2': f'subject_{subject}',
                    'system1_type': 'model',
                    'system2_type': 'human',
                    'system1_source': 'nsd',
                    'system2_source': 'nsd',
                    'system1_froi': model_froi,
                    'system2_froi': froi2,
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, all_correlations_csv_list):
                    continue
                model_info = {
                    'model_name': model_name,
                    'tthreshold': model_tthreshold,
                    'model_layer': get_froi_models_layers(model_froi, froi2)[model_name],
                    'model_localizer': model_localizer
                }
                # Get subject-model correlation
                subject_voxel_data = subjects_voxel_data[froi2][subject]
                model_data = models_data[(model_froi, froi2)][model_name]
                # split-half pearsonrs
                subject_splithalf_pearsonr = subjects_splithalf_pearsonrs[froi2][subject]
                # split-half rdmrs
                subject_splithalf_rdmr = subjects_splithalf_rdmrs[froi2][subject]
                if subject_voxel_data is None or \
                    model_data is None or \
                    np.isnan(subject_splithalf_pearsonr) or \
                    np.isnan(subject_splithalf_rdmr):
                    # Set to nan
                    pearsonr = np.nan
                    rdmr = np.nan
                    pvalue_pearsonr = np.nan
                    pvalue_rdmr = np.nan
                    normed_pearsonr = np.nan
                    normed_rdmr = np.nan
                else:
                    # Pearsonr
                    pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                        subject_voxel_data.mean(axis=-1), model_data.mean(axis=-1).cpu().numpy()
                    )
                    normed_pearsonr = \
                        pearsonr / np.sqrt(subject_splithalf_pearsonr)
                    # RDMr
                    rdmr, pvalue_rdmr = rcutils.rdmr(
                        subject_voxel_data, model_data, return_pvalue=True
                    )
                    normed_rdmr = \
                        rdmr / np.sqrt(subject_splithalf_rdmr)
                # Save results
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'pearsonr',
                    'metric_normed': normed_pearsonr,
                    'metric_raw': pearsonr,
                    'metric_pvalue': pvalue_pearsonr,
                    'system1_splithalf_metric': subject_splithalf_pearsonr,
                    'system2_splithalf_metric': pd.NA,
                    'system1_info': {'tthreshold': fmri_tthreshold},
                    'system2_info': model_info
                })
                all_correlations_csv_list.append({
                    **row_params,
                    'metric_type': 'rdmr',
                    'metric_normed': normed_rdmr,
                    'metric_raw': rdmr,
                    'metric_pvalue': pvalue_rdmr,
                    'system1_splithalf_metric': subject_splithalf_rdmr,
                    'system2_splithalf_metric': pd.NA,
                    'system1_info': {'tthreshold': fmri_tthreshold},
                    'system2_info': model_info
                })
                all_correlations_csv = pd.DataFrame(all_correlations_csv_list)
                all_correlations_csv.to_csv(
                    all_correlations_csv_path, index=False
                )

    # Assert that threre are no None values left in the csv
    for row in all_correlations_csv_list:
        for key, value in row.items():
            if value is None:
                raise ValueError(f'Found None value in correlations csv for key: {key}, row: {row}')

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Run analysis')

    parser.add_argument(
        '--nsd485layersweep', action='store_true', help='Run NSD515 layer sweep'
    )
    parser.add_argument(
        '--localizationvalidation', action='store_true', help='Run NSD515 localization validation'
    )
    parser.add_argument(
        '--nsd515subjectsubjectcorrelations', action='store_true', \
            help='Run NSD515 subject-subject correlations'
    )
    parser.add_argument(
        '--nsd515modelsubjectcorrelations', action='store_true', \
            help='Run NSD515 model-subject univariate correlations'
    )
    parser.add_argument(
        '--nsd515localizereffect', action='store_true', help='Run NSD515 localizer effect'
    )
    parser.add_argument(
        '--nsd515tthresholdeffect', action='store_true', help='Run NSD515 t-threshold effect'
    )
    parser.add_argument(
        '--murty185evaluation', action='store_true', help='Run Murty185 evaluation'
    )
    parser.add_argument(
        '--nsdsyntheticevaluation', action='store_true', help='Run NSD-Synthetic evaluation'
    )
    parser.add_argument(
        '--vinken2025evaluation', action='store_true', help='Run Vinken et al. 2025 evaluation'
    )

    args = parser.parse_args()

    if args.nsd485layersweep:
        run_analysis_nsd485layersweep()
    if args.localizationvalidation:
        run_analysis_localizationvalidation()
    if args.nsd515subjectsubjectcorrelations:
        run_analysis_nsd515subjectsubjectcorrelations()
    if args.nsd515modelsubjectcorrelations:
        run_analysis_nsd515modelsubjectcorrelations()
    if args.nsd515localizereffect:
        run_analysis_nsd515localizereffect()
    if args.nsd515tthresholdeffect:
        run_analysis_nsd515tthresholdeffect()
    if args.murty185evaluation:
        run_analysis_murty185evaluation()
    if args.nsdsyntheticevaluation:
        run_analysis_nsdsyntheticevaluation()
    if args.vinken2025evaluation:
        run_analysis_vinken2025evaluation()
    if not any(vars(args).values()):
        print('No analysis specified. Use --help for more information.')

if __name__ == '__main__':
    main()
