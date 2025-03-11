# Importing the necessary libraries
import os
import time
import argparse
import scipy
import pandas as pd
import torch

from compcats.models import *
from compcats.datasets.fmri import nsd
from compcats.utils import seed_utils, misc_utils, model_utils

# Analysis files
import analysis_layersweep
import analysis_localizationvalidation

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

# Batch size
BATCH_SIZE = 2048

# Results dirs
BEST_LAYERS_CSV_PATH = f'{MODELS_DIR}/best_layers_nsd485_median_pearsonr.csv'
LOCALIZATION_VALIDATION_CSV_PATH = f'{RESULTS_DIR}/localization_validation.csv'
SUBJECTSUBJECT_PEARSONR_CSV_PATH = f'{RESULTS_DIR}/subjectsubject_pearsonr_nsd515.csv'
MODELSUBJECT_PEARSONR_CSV_PATH = f'{RESULTS_DIR}/modelsubject_pearsonr_nsd515.csv'

# Models, Info Log
MODELS = {
    # # # AlexNets
    'alexnetimagenet1kv1': alexnet_imagenet1kv1,
    'alexnetipclref01': alexnet_ipcl,
    'alexnetbarlowtwins': alexnet_barlowtwins,
    # # # VGGs
    'vgg16imagenet1kv1': vgg16_imagenet1kv1,
    # # # ResNet-18s
    'resnet18imagenet1kv1': resnet18_imagenet1kv1,
    # # # ResNet-50s
    'resnet50imagenet1kv2': resnet50_imagenet1kv2,
    # 'resnet50imagenet21k': resnet50_imagenet21k, # Check best layer
    'resnet50openaiclip': resnet50_openaiclip,
    # 'resnet50simclr': resnet50_simclr, # Check best layer
    # 'resnet50robustl2eps3': resnet50_robust,
    'resnet50sin': resnet50_sin,
    'resnet50ecoset': resnet50_ecoset,
    'resnet50faceobfuscation': resnet50_faceobfuscation,
    'resnet50toponettau30': resnet50_toponet,
    # # # BagNets
    'bagnet33': bagnet33,
    # # # CORNets
    'cornets': cornet_s,
    'cornetz': cornet_z,
    # # # ViTs
    # # # ViT-B/32
    'vitb32imagenet1kv1': vitb32_imagenet1kv1,
    'vitb32openaiclip': vitb32_openaiclip,
    'vitb32dreamsimopenclip': vitb32_dreamsim,
    # # # ViT-B/16
    'vitb16imagenet21k': vitb16_imagenet21k,
    'vitb16openaiclip': vitb16_openaiclip,
    'vitb16siglip2res224': vitb16_siglip2,
    'vitb16eva028bs8bb131k': vitb16_eva02,
    'vitb16synclr': vitb16_synclr,
    # # # ViT-B/14
    'vitb14dinov2': vitb14_dinov2,
    'vitb14dinov2reg': vitb14_dinov2register,
    # # # ViT-L/14
    'vitl14openaiclip': vitl14_openaiclip,
    'vitl14kosmos2': vitl14_kosmos2,
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

def get_froi_models_layers(froi):
    """
    Get layers of all models used for the analysis
    Args:
        froi (str): Froi
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
            (best_layers_csv['model_froi'] == froi) &
            (best_layers_csv['fmri_froi'] == froi)
        ]['model_layer']
        # Check if best layer is found
        assert len(model_best_layer) == 1, 'Best layer not found'
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

def get_allsubjects_nsd515_meanrep_meanvoxel_data(frois, tthreshold, tensors=False):
    """
    Get NSD mean of voxels, mean of repetitions data for all subjects
    Args:
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
    Returns:
        subjects_data (dict): Dictionary of subject data
    """
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    subjects_data = {}
    for froi in frois:
        subjects_data[froi] = {}
        for subject in subjects:
            subject_froi_data, _ = nsd.get_betas(
                NSD_DIR, subject, subset=subset, voxels=False, repetition_mean=True, \
                froi=froi, tthreshold=tthreshold
            )
            if not tensors:
                subject_froi_data = subject_froi_data.cpu().numpy()
            subjects_data[froi][subject] = subject_froi_data
    return subjects_data

def get_allmodels_nsd515_meanunit_data(frois, tthreshold, localizer, tensors=False):
    """
    Get NSD515 mean of froi units data for all models
    Args:
        frois (list): List of frois
        tthreshold (float): Threshold for t-values
        localizer (str): Localizer name
    """
    subset = 515
    models_data = {}
    for froi in frois:
        models_data[froi] = {}
        for model_name in get_model_names():
            model = get_model(model_name)
            layer = get_froi_models_layers(froi)[model_name]
            model_froi_activations = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer, dataset='nsd', \
                subset=subset, \
                froi=froi, tthreshold=tthreshold, froi_localizer=localizer, \
                froi_localizer_dir=get_localizer_dir(localizer)
            ).mean(axis=-1)
            if not tensors:
                model_froi_activations = model_froi_activations.cpu().numpy()
            models_data[froi][model_name] = model_froi_activations
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
    for froi in frois:
        subjects_data[froi] = {}
        for subject in subjects:
            subject_data, _ = nsd.get_betas(
                NSD_DIR, subject, froi=froi, tthreshold=tthreshold, \
                subset=subset, voxels=False, repetition_mean=True
            )
            subjects_data[froi][subject] = subject_data.cuda()
    return subjects_data

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
    frois = ['FFA', 'PPA', 'EBA']
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
    subjects_data = get_humandata_layersweep(
        subjects, frois, tthreshold, subset
    )

    # Run layer sweep
    for model_name in get_model_names():
        model = get_model(model_name)
        relevant_layers = model.get_relevant_layers()
        start_time = time.time()
        print(f'\nStarting layer sweep for model: {model_name}')
        model_layersweep_csv = analysis_layersweep.model_layer_sweep(
            MODELS_DIR, NSD_DIR, get_localizer_dir(model_localizer), \
            subjects_data, model, relevant_layers, subset, frois, tthreshold, model_localizer
        )
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # Convert to minutes
        print(f'Layer sweep for model {model_name} completed in {elapsed_time:.2f} minutes\n')

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
        best_layers = {info['model_layer'] for froi, info in best_layer_info.items()}
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
        misc_utils.delete_activations(
            f'{MODELS_DIR}/{model_name}/{model_localizer}/activations', delete_layers
        )
        # Delete NSD activations
        misc_utils.delete_activations(
            f'{MODELS_DIR}/{model_name}/nsd', delete_layers
        )

    # Empty cache
    del subjects_data
    torch.cuda.empty_cache()

# Analysis: Held-out z-scoring for category selective units
def run_analysis_localizationvalidation():
    """
    Analysis: Held-out z-scoring for category selective units to validate localization
    """
    # Variables
    frois = ['FFA', 'PPA', 'EBA']
    tthreshold = 7
    froi_localizer = 'grillspector'
    validation_localizer = 'konkle'
    frois_models_layers = {
        froi: get_froi_models_layers(froi) for froi in frois
    }

    # Storing results
    localization_validation_csv_path = '.'.join(
        LOCALIZATION_VALIDATION_CSV_PATH.split('.')[:-1]
    ) + f'_{validation_localizer}.csv'
    if os.path.exists(localization_validation_csv_path):
        localization_validation_csv = pd.read_csv(localization_validation_csv_path)
        localization_validation_csv_list = localization_validation_csv.to_dict('records')
    else:
        localization_validation_csv_list = []

    for model_name in get_model_names():
        model = get_model(model_name)
        model_all_layers = list({frois_models_layers[froi][model_name] for froi in frois})
        for froi in frois:
            model_froi_layer = frois_models_layers[froi][model_name]
            print(f'Running localization validation for model: {model_name}, froi: {froi}')
            row_params = {
                'model_name': model_name,
                'model_layer': model_froi_layer,
                'model_froi': froi,
                'model_froi_tthreshold': tthreshold,
                'model_froi_localizer': froi_localizer,
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
            # Save results
            localization_validation_csv_list.append({
                **row_params,
                'tvalue': tvalue,
                'pvalue': pvalue
            })
            localization_validation_csv = pd.DataFrame(localization_validation_csv_list)
            localization_validation_csv.to_csv(localization_validation_csv_path, index=False)

        # Delete validation localizer activations to save space
        misc_utils.delete_activations(
            f'{MODELS_DIR}/{model_name}/{validation_localizer}/activations', model_all_layers
        )

# Analysis: subject-subject correlations (univariate)
def run_analysis_subjectsubjectunivariate():
    """
    TODO
    """
    # Variables
    frois = ['FFA', 'PPA', 'EBA']
    tthreshold = 7
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    subjects_data = get_allsubjects_nsd515_meanrep_meanvoxel_data(frois, tthreshold)

    # Storing results
    subjectsubject_pearsonr_csv_path = SUBJECTSUBJECT_PEARSONR_CSV_PATH
    if os.path.exists(subjectsubject_pearsonr_csv_path):
        subjectsubject_pearsonr_csv = pd.read_csv(subjectsubject_pearsonr_csv_path)
        subjectsubject_pearsonr_csv_list = subjectsubject_pearsonr_csv.to_dict('records')
    else:
        subjectsubject_pearsonr_csv_list = []

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
                if misc_utils.check_row_present(row_params, subjectsubject_pearsonr_csv_list):
                    continue
                # Run subject-subject correlation
                subject1_data = subjects_data[froi][subject1]
                subject2_data = subjects_data[froi][subject2]
                pearsonr, pvalue = scipy.stats.pearsonr(subject1_data, subject2_data)
                # Save results
                subjectsubject_pearsonr_csv_list.append({
                    **row_params,
                    'pearsonr': pearsonr,
                    'pvalue': pvalue
                })
                subjectsubject_pearsonr_csv = pd.DataFrame(subjectsubject_pearsonr_csv_list)
                subjectsubject_pearsonr_csv.to_csv(subjectsubject_pearsonr_csv_path, index=False)

# Analysis: model-subject correlations (univariate)
def run_analysis_modelsubjectunivariate():
    """
    TODO
    """
    # Variables
    # Common
    frois = ['FFA', 'PPA', 'EBA']
    tthreshold = 7
    # Subjects
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515
    # Models
    model_localizer = 'grillspector'

    # Storing results
    modelsubject_pearsonr_csv_path = MODELSUBJECT_PEARSONR_CSV_PATH
    if os.path.exists(modelsubject_pearsonr_csv_path):
        modelsubject_pearsonr_csv = pd.read_csv(modelsubject_pearsonr_csv_path)
        modelsubject_pearsonr_csv_list = modelsubject_pearsonr_csv.to_dict('records')
    else:
        modelsubject_pearsonr_csv_list = []

    subjects_data = get_allsubjects_nsd515_meanrep_meanvoxel_data(frois, tthreshold)
    models_data = get_allmodels_nsd515_meanunit_data(frois, tthreshold, model_localizer)

    for model_name in get_model_names():
        for subject in subjects:
            for froi in frois:
                row_params = {
                    'model_name': model_name,
                    'model_layer': get_froi_models_layers(froi)[model_name],
                    'model_froi': froi,
                    'modeltthreshold': tthreshold,
                    'model_localizer': model_localizer,
                    'fmri_dataset': f'nsd{subset}',
                    'fmri_froi': froi,
                    'fmri_tthreshold': tthreshold,
                    'subject': subject
                }
                # Check if row is present
                if misc_utils.check_row_present(row_params, modelsubject_pearsonr_csv_list):
                    continue
                # Run model-subject correlation
                model_data = models_data[froi][model_name]
                subject_data = subjects_data[froi][subject]
                pearsonr, pvalue = scipy.stats.pearsonr(model_data, subject_data)
                # Save results
                modelsubject_pearsonr_csv_list.append({
                    **row_params,
                    'pearsonr': pearsonr,
                    'pvalue': pvalue
                })
                modelsubject_pearsonr_csv = pd.DataFrame(modelsubject_pearsonr_csv_list)
                modelsubject_pearsonr_csv.to_csv(modelsubject_pearsonr_csv_path, index=False)

# MDS (univariate)
def run_analysis_mdsunivariate():
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
        '--subjectsubjectunivariate', action='store_true', \
            help='Run subject-subject univariate correlations'
    )
    parser.add_argument(
        '--modelsubjectunivariate', action='store_true', \
            help='Run model-subject univariate correlations'
    )
    parser.add_argument(
        '--mds', action='store_true', help='Run MDS'
    )
    args = parser.parse_args()

    if args.layersweep:
        run_analysis_layersweep()
    if args.localizationvalidation:
        run_analysis_localizationvalidation()
    if args.subjectsubjectunivariate:
        run_analysis_subjectsubjectunivariate()
    if args.modelsubjectunivariate:
        run_analysis_modelsubjectunivariate()
    if args.mds:
        run_analysis_mdsunivariate()

if __name__ == '__main__':
    main()
