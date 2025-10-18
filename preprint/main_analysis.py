# Importing the necessary libraries
import os
import time
import argparse
import scipy
import numpy as np
import pandas as pd
import torch

from compcats.models import *
from compcats.datasets.fmri import nsd
from compcats.utils import seed_utils, misc_utils, model_utils, reliability_utils
import compcats.utils.responsecomparison_utils as rcutils

# Analysis files
import analysis_layersweep
import analysis_localizationvalidation

# Variables
# Directories
import paths
# Get the directory of the current file
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_FILE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
MODELS_DIR = os.path.join(CURRENT_FILE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
NSD_DIR = paths.NSD_DIR
# Directories
GRILLSPECTOR_LOCALIZER_DIR = paths.GRILLSPECTOR_LOCALIZER_DIR
KANWISHER_LOCALIZER_DIR = paths.KANWISHER_LOCALIZER_DIR
KONKLE_LOCALIZER_DIR = paths.KONKLE_LOCALIZER_DIR

# Seed
SEED = 42
seed_utils.seed_everything(SEED)

# Batch size
BATCH_SIZE = 64  # Adjust based on GPU memory

# Device
DEVICE = misc_utils.get_device()

# Results dirs
BEST_LAYERS_CSV_PATH = f'{RESULTS_DIR}/best_layers_nsd485_median_pearsonr.csv'
LOCALIZATION_VALIDATION_CSV_PATH = f'{RESULTS_DIR}/localization_validation.csv'
SUBJECTSUBJECT_CORRELATIONS_CSV_PATH = f'{RESULTS_DIR}/subjectsubject_correlations_nsd515.csv'
MODELSUBJECT_CORRELATIONS_CSV_PATH = f'{RESULTS_DIR}/modelsubject_correlations_nsd515.csv'
LOCALIZEREFFECT_CSV_PATH = f'{RESULTS_DIR}/localizer_effect_correlations_nsd515.csv'
TTHRESHOLDEFFECT_CSV_PATH = f'{RESULTS_DIR}/tthreshold_effect_correlations_nsd515.csv'

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
'vitl14kosmos2': vitl14_kosmos2, #Done
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
    model_names = get_model_names()
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

def get_allmodels_nsd515_data(model_fmri_frois, tthreshold, localizer, \
                                       unitmean, tensors=True):
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
        for model_name in get_model_names():
            model = get_model(model_name)
            layer = get_froi_models_layers(model_froi, fmri_froi)[model_name]
            model_froi_activations = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer, dataset='nsd', \
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

# Layer sweeps
def run_analysis_layersweep():
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
    model_fmri_frois = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]
    model_frois = list({i[0] for i in model_fmri_frois})
    tthreshold = 7
    froi_localizer = 'grillspector'
    # froi_localizer = 'kanwisher'
    # froi_localizer = 'kanwishergrayscale'
    print('Validating localization for localizer: ', froi_localizer)
    validation_localizer = 'konkle'
    # validation_localizer = 'kanwisher'
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

    for model_name in get_model_names():
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
def run_analysis_subjectsubjectcorrelations():
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
    subjectsubject_correlations_csv_path = SUBJECTSUBJECT_CORRELATIONS_CSV_PATH
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
def run_analysis_modelsubjectcorrelations():
    """
    TODO
    """
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
    model_fmri_frois = model_fmri_frois_all
    fmri_frois = list({i[1] for i in model_fmri_frois})
    tthreshold = 7
    # Subjects
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    # Models
    model_localizer = 'grillspector'

    # Storing results
    modelsubject_correlations_csv_path = MODELSUBJECT_CORRELATIONS_CSV_PATH
    if os.path.exists(modelsubject_correlations_csv_path):
        modelsubject_correlations_csv = pd.read_csv(modelsubject_correlations_csv_path)
        modelsubject_correlations_csv_list = modelsubject_correlations_csv.to_dict('records')
    else:
        modelsubject_correlations_csv_list = []

    subjects_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
        get_allsubjects_nsd515_data(fmri_frois, tthreshold)
    models_data = get_allmodels_nsd515_data(
        model_fmri_frois, tthreshold, model_localizer, unitmean=False, tensors=True
    )

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
def run_analysis_localizereffect():
    """
    TODO
    """
    # Define frois
    model_fmri_frois = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]

    # Variables
    fmri_frois = list({i[1] for i in model_fmri_frois})
    tthreshold = 7
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
            model_fmri_frois, tthreshold, model_localizer, unitmean=False, tensors=True
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

def run_analysis_tthresholdeffect():
    """
    TODO
    """
    # Define frois
    model_fmri_frois = [
        ('FFA', 'FFA'),
        ('PPA', 'PPA'),
        ('EBA', 'EBA')
    ]

    # Variables
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
            model_fmri_frois, model_tthreshold, model_localizer, unitmean=False, tensors=True
        )
        for fmri_tthreshold in tthresholds:
            subjects_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
                get_allsubjects_nsd515_data(fmri_frois, fmri_tthreshold)
            for model_name in get_model_names():
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


# MDS
def run_analysis_mds():
    """
    Run MDS analysis
    """
    pass

# NSD comparison
def run_analysis_nsdcomparison():
    """
    Run Full NSD comparison
    """
    pass

# Other dataset: Murty185
def run_analysis_murty185validation():
    """
    TODO
    """
    pass


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Run analysis')

    parser.add_argument(
        '--layersweep', action='store_true', help='Run layer sweep'
    )
    parser.add_argument(
        '--localizationvalidation', action='store_true', help='Run localization validation'
    )
    parser.add_argument(
        '--subjectsubjectcorrelations', action='store_true', \
            help='Run subject-subject correlations'
    )
    parser.add_argument(
        '--modelsubjectcorrelations', action='store_true', \
            help='Run model-subject univariate correlations'
    )
    parser.add_argument(
        '--mds', action='store_true', help='Run MDS'
    )
    parser.add_argument(
        '--nsdcomparison', action='store_true', help='Run NSD comparison'
    )
    parser.add_argument(
        '--murty185validation', action='store_true', help='Run Murty185 validation'
    )
    parser.add_argument(
        '--localizereffect', action='store_true', help='Run localizer effect'
    )
    parser.add_argument(
        '--tthresholdeffect', action='store_true', help='Run t-threshold effect'
    )

    args = parser.parse_args()

    if args.layersweep:
        run_analysis_layersweep()
    if args.localizationvalidation:
        run_analysis_localizationvalidation()
    if args.subjectsubjectcorrelations:
        run_analysis_subjectsubjectcorrelations()
    if args.modelsubjectcorrelations:
        run_analysis_modelsubjectcorrelations()
    if args.mds:
        run_analysis_mds()
    if args.nsdcomparison:
        run_analysis_nsdcomparison()
    if args.murty185validation:
        run_analysis_murty185validation()
    if args.localizereffect:
        run_analysis_localizereffect()
    if args.tthresholdeffect:
        run_analysis_tthresholdeffect()
    if not any(vars(args).values()):
        print('No analysis specified. Use --help for more information.')

if __name__ == '__main__':
    main()
