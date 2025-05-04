"""
Script to evaluate froi similarity on a trained model
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy
import pandas as pd

from fastargs import get_current_config
from fastargs import Param, Section
# from fastargs.validation import And, OneOf

import torch

from compcats.datasets.fmri import nsd
from compcats.localizer import localizer
from compcats.utils import logging_utils, model_utils, reliability_utils, \
    misc_utils, responsecomparison_utils as rcutils

Section('data', 'data related stuff').params(
    model_dir=Param(str, 'dir for the model', required=True),
    nsd_dir=Param(str, 'NSD Directory', required=True),
    localizer_dir=Param(str, 'Localizer directory', required=True),
    output_dir=Param(str, 'directory for logging and storing results', required=True),
)

Section('model', 'model related stuff').params(
    epoch=Param(int, 'epoch number, only used if .pth is not provided', default=-1),
)

# Harcoding the localization params
# Section('localization', 'localization related stuff').params(
#     fmri_dataset=Param(str, 'fmri dataset to use', default='nsd', required=True),
#     fmri_froi=Param(str, 'fmri froi to use', default='lhq'),
#     fmri_tthreshold=Param(float, 'fmri tthreshold to use', default=0.0),
#     model_localizer=Param(str, 'model localizer to use', default='lhq'),
#     model_froi=Param(str, 'model froi to use', default='lhq'),
#     model_tthreshold=Param(float, 'model tthreshold to use', default=0.0),
# )

def get_model_nsd515_correlations(
    nsd_dir, output_dir, log_dir, localizer_dir, model, layers, model_fmri_frois,
    model_localizer, tthreshold, uid, experiment_name
):
    """
    Correlations of model's frois with NSD515 frois
    """
    fmri_frois = list({i[1] for i in model_fmri_frois})
    # Subjects
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    subset = 515

    # Storing results
    modelsubject_correlations_csv_path = os.path.join(
        log_dir, 'nsd515_correlations.csv'
    )
    # Check if modelsubject correlations csv exists
    if os.path.exists(modelsubject_correlations_csv_path):
        modelsubject_correlations_csv = pd.read_csv(modelsubject_correlations_csv_path)
        modelsubject_correlations_csv_list = modelsubject_correlations_csv.to_dict('records')
    else:
        modelsubject_correlations_csv_list = []

    subjects_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
        get_allsubjects_nsd515_data(nsd_dir, fmri_frois, tthreshold)
    model_data = {}
    # Get model data
    # Get model data for each froi
    for (model_froi, fmri_froi) in model_fmri_frois:
        if layers[(model_froi, fmri_froi)] is None:
            model_froi_data = None
        else:
            model_froi_data = model_utils.get_dataset_activations(
                output_dir, nsd_dir, model, layers[(model_froi, fmri_froi)], dataset='nsd', \
                froi=model_froi, tthreshold=tthreshold, froi_localizer=model_localizer, \
                froi_localizer_dir=localizer_dir, subset=subset
            )
        model_data[(model_froi, fmri_froi)] = model_froi_data

    for subject in subjects:
        for (model_froi, fmri_froi) in model_fmri_frois:
            row_params = {
                'experiment': experiment_name,
                'model_uid': uid,
                'model_name': model.name,
                'model_layer': layers[(model_froi, fmri_froi)],
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
            model_froi_data = model_data[(model_froi, fmri_froi)]
            subject_data = subjects_data[fmri_froi][subject]
            subject_splithalf_pearsonr = subjects_splithalf_pearsonrs[fmri_froi][subject]
            subject_splithalf_rdmr = subjects_splithalf_rdmrs[fmri_froi][subject]
            if model_froi_data is not None:
                # RDMr
                rdmr, pvalue_rdmr = rcutils.rdmr(
                    model_froi_data, subject_data, return_pvalue=True
                )
                # Normalize by split-half correlation
                normed_rdmr = rdmr / np.sqrt(subject_splithalf_rdmr)
                # Pearsonr
                pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(
                    model_froi_data.mean(axis=-1).cpu().numpy(), 
                    subject_data.mean(axis=-1).cpu().numpy()
                )
                # Normalize by split-half correlation
                normed_pearsonr = pearsonr / np.sqrt(subject_splithalf_pearsonr)
            else:
                rdmr = np.nan
                pvalue_rdmr = np.nan
                normed_rdmr = np.nan
                pearsonr = np.nan
                pvalue_pearsonr = np.nan
                normed_pearsonr = np.nan
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

def get_allsubjects_nsd515_data(nsd_dir, frois, tthreshold, tensors=True):
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
                nsd_dir, subject, subset=subset, voxels=True, repetition_mean=True, \
                froi=froi, tthreshold=tthreshold
            )
            subjects_data[froi][subject] = subject_froi_data
    # Get subject froi split-half correlations
    for froi in frois:
        splithalf_pearsonrs[froi] = {}
        splithalf_rdmrs[froi] = {}
        for subject in subjects:
            subject_data, _ = nsd.get_betas(
                nsd_dir, subject, froi=froi, tthreshold=tthreshold, \
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

def get_nsddata_layersweep(nsd_dir, subjects, frois, tthreshold, subset):
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
                nsd_dir, subject, froi=froi, tthreshold=tthreshold, \
                subset=subset, voxels=False, repetition_mean=True
            )
            subjects_data[froi][subject] = subject_data.cuda()
    # Get subject froi split-half correlations
    for froi in frois:
        splithalf_pearsonrs[froi] = {}
        for subject in subjects:
            subject_data, _ = nsd.get_betas(
                nsd_dir, subject, froi=froi, tthreshold=tthreshold, \
                subset=subset, voxels=False, repetition_mean=False
            )
            splithalf_pearsonr_ = reliability_utils.get_voxelmean_splithalf_correlation(
                subject_data
            )
            splithalf_pearsonrs[froi][subject] = splithalf_pearsonr_
    return subjects_data, splithalf_pearsonrs

def layer_sweep(
    model_uid, epoch_identifier, output_dir, nsd_dir, localizer_dir, model, layers, model_fmri_frois,
    model_localizer, tthreshold, subjects
):
    """
    Perform layer sweep on the model to choose the best layer for each froi based on
    voxel/unit averaged pearson correlation with nsd485 (t=7)
    """
    # Replace model name with uid for compatibility with library
    model.original_name = model.name
    model.name = f'{model_uid}_{epoch_identifier}'
    # Load already computed results
    csv_pth = f'{output_dir}/{model.name}/nsd485_pearsonr_layersweep.csv'
    if os.path.exists(csv_pth):
        results_csv = pd.read_csv(csv_pth)
        results_list = results_csv.to_dict('records')
        results_df = results_csv
    else:
        results_list = []

    # Variables
    model_data = None
    model_froi_masks = None
    subjects_data = None
    subjects_splithalf_pearsonrs = None
    fmri_frois = [x[1] for x in model_fmri_frois]

    # Internal functions
    # Generate NSD data for models
    def _get_model_data():
        return model_utils.get_dataset_activations(
            output_dir, nsd_dir, model, layers, dataset='nsd', subset=485
        )
    def _get_model_froi_masks(froi):
        return localizer.get_froi_mask(
            localizer_dir, model_localizer, output_dir, \
            model, layers, froi, tthreshold
        )
    # Generate subjects data
    def _get_subjects_data():
        return get_nsddata_layersweep(
            nsd_dir, subjects, fmri_frois, tthreshold, subset=485
        )

    previous_model_froi = ''
    # Loop over frois
    for (model_froi, fmri_froi) in model_fmri_frois:
        # Check if model froi masks are present
        if model_froi != previous_model_froi:
            # Empty CUDA cache
            del model_froi_masks
            torch.cuda.empty_cache()
            # Reset model froi masks
            model_froi_masks = None
        # Update previous model froi
        previous_model_froi = model_froi
        print(f'Running layer sweep: {model.name}, {model_froi}, {fmri_froi}')
        for layer in tqdm(layers):
            # Check if row is present
            row_params = {
                'model_name': model.name,
                'model_layer': layer,
                'model_froi': model_froi,
                'model_tthreshold': tthreshold,
                'model_localizer': model_localizer,
                'fmri_dataset': 'nsd485',
                'fmri_froi': fmri_froi,
                'fmri_tthreshold': tthreshold
            }
            if misc_utils.check_row_present(row_params, results_list):
                continue
            # Correlations
            subject_pearsonrs = []
            subject_normed_pearsonrs = []
            # Generate subjects data
            if subjects_data is None and subjects_splithalf_pearsonrs is None:
                # Get subjects data
                subjects_data, subjects_splithalf_pearsonrs = _get_subjects_data()
            # Loop over subjects
            for subject, subject_froi_data in subjects_data[fmri_froi].items():
                # Get model froi masks
                model_froi_masks = _get_model_froi_masks(model_froi) if model_froi_masks is None \
                    else model_froi_masks
                # Get model data
                model_data = _get_model_data() if model_data is None else model_data
                # Get model data for the layer
                layer_froi_data = model_data[layer][
                    :, model_froi_masks[layer]
                ].mean(axis=-1)
                # Get model-human comparison results
                pearsonr_ = rcutils.pearsonr(
                    layer_froi_data, subject_froi_data
                )
                subject_pearsonrs.append(pearsonr_)
                # Normalize by split-half pearsonr
                normed_pearsonr = pearsonr_ / subjects_splithalf_pearsonrs[fmri_froi][subject]
                subject_normed_pearsonrs.append(normed_pearsonr)
            # Save results
            results_list.append({
                **row_params,
                'normed_pearsonr_median': np.nanmedian(subject_normed_pearsonrs),
                'normed_pearsonr_mean': np.nanmean(subject_normed_pearsonrs),
                'normed_pearsonr_sem': scipy.stats.sem(subject_normed_pearsonrs, nan_policy='omit'),
                'pearsonr_median': np.nanmedian(subject_pearsonrs),
                'pearsonr_mean': np.nanmean(subject_pearsonrs),
                'pearsonr_sem': scipy.stats.sem(subject_pearsonrs, nan_policy='omit'),
                'pearsonr': subject_pearsonrs,
                'normed_pearsonr': subject_normed_pearsonrs,
            })
            # Save to CSV
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(csv_pth, index=False)

    # Empty CUDA cache
    del model_data
    torch.cuda.empty_cache()

    # Get best layer for each model-froi pair
    best_layers = {}
    best_layers_info = None
    for (model_froi, fmri_froi) in model_fmri_frois:
        rows = results_df[
            (results_df['model_name'] == model.name) &
            (results_df['model_froi'] == model_froi) &
            (results_df['fmri_froi'] == fmri_froi)
        ]
        if rows.empty:
            raise ValueError(f'No rows found for {model.name}, {model_froi}, {fmri_froi}')
        key = (model_froi, fmri_froi)
        if rows['normed_pearsonr_median'].isnull().all():
            tmp_row = rows.iloc[0].copy()
            tmp_row['model_layer'] = None
            if best_layers_info is None:
                best_layers_info = tmp_row
            else:
                best_layers_info = pd.concat([best_layers_info, tmp_row], axis=1)
            best_layers[key] = None
        else:
            best_row = rows.loc[rows['normed_pearsonr_median'].idxmax(skipna=True)]
            best_layers[key] = best_row['model_layer']
            if best_layers_info is None:
                best_layers_info = best_row
            else:
                best_layers_info = pd.concat([best_layers_info, best_row], axis=1)
    best_layers_info = best_layers_info.T
    return best_layers, best_layers_info

def load_model(model_dir, epoch):
    """
    Find and load the checkpoint file
    """
    assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"
    assert os.path.isdir(model_dir), f"Model directory {model_dir} is not a directory"
    # epoch substring
    if epoch > 0:
        epoch_substring = f'_epoch{epoch}'
    elif epoch == -1:
        epoch_substring = '_final'
    # Find checkpoints available
    checkpoints_available = os.listdir(model_dir)
    checkpoints_available = [x for x in checkpoints_available if x.endswith(('.pt', '.pth'))]
    # Find checkpoint
    checkpoint_file_name = [x for x in checkpoints_available if epoch_substring in x]
    # Check if checkpoint file is found
    if len(checkpoint_file_name) == 0:
        raise ValueError(f"Checkpoint file not found in {model_dir}")
    if len(checkpoint_file_name) > 1:
        raise ValueError(f"Multiple checkpoint files found in {model_dir}")
    checkpoint_file_name = checkpoint_file_name[0]
    # Checkpoint dir
    checkpoint_dir = os.path.join(model_dir, checkpoint_file_name)
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_dir, weights_only=False)
    # Load training params
    assert os.path.exists(os.path.join(model_dir, 'params.json')), \
        f"Params file not found in {model_dir}"
    with open(os.path.join(model_dir, 'params.json'), 'r', encoding='utf-8') as f:
        training_params = json.load(f)
    return checkpoint, checkpoint_dir, training_params

def copy_info(model_dir, checkpoint_dir, log_dir):
    """
    Copy the checkpoint file and params file in output directory
    """
    # Copy params file
    os.system(f'cp {model_dir}/params.json {log_dir}/training_params.json')
    # Copy checkpoint file
    os.system(f'cp {checkpoint_dir} {log_dir}')

def main(fastargs_config):
    """
    Main function
    """
    args = fastargs_config.get()
    # Load the checkpoint
    model, checkpoint_dir, training_params = \
        load_model(args.data.model_dir, args.model.epoch)
    # Update otput dir
    output_dir = str((
        Path(args.data.output_dir) / training_params['project']['name'] / 'evaluation'
    ).absolute())
    args.data.output_dir = output_dir
    # Store log
    log = {
        'args': logging_utils.nestednamespace_to_dict(args),
        'checkpoint_dir': checkpoint_dir,
        'training_params': training_params,
    }
    # Convert training params to Namespace to easy dot-access
    training_params = logging_utils.dict_to_namespace(training_params)
    # Epoch substring
    epoch_identifier = f'epoch{args.model.epoch}' if args.model.epoch > 0 else 'epochfinal'
    log['epoch_identifier'] = epoch_identifier
    # Create log dir
    model_identifier = f'{training_params.identifier.uid}_{epoch_identifier}'
    log['model_identifier'] = model_identifier
    log_dir = str((
        Path(args.data.output_dir) / model_identifier
    ).absolute())
    os.makedirs(log_dir, exist_ok=True)
    log['log_dir'] = log_dir
    # Copy the checkpoint and params file
    copy_info(args.data.model_dir, checkpoint_dir, log_dir)
    # Layer sweep
    layersweep_info = {
        'layers': model.relevant_layer_names,
        'model_fmri_frois': [('FFA', 'FFA'), ('PPA', 'PPA'), ('EBA', 'EBA')],
        'model_localizer': 'grillspector',
        'tthreshold': 7,
        'dataset': 'nsd485',
        'subjects': [1, 2, 5, 7],
    }
    best_layers, best_layers_info = layer_sweep(
        training_params.identifier.uid, epoch_identifier,
        args.data.output_dir, args.data.nsd_dir, args.data.localizer_dir,
        model, layersweep_info['layers'], layersweep_info['model_fmri_frois'],
        layersweep_info['model_localizer'], layersweep_info['tthreshold'],
        layersweep_info['subjects']
    )
    # Save best layers info
    pd.DataFrame(best_layers_info).to_csv(
        os.path.join(log_dir, 'best_layers_info.csv'), index=False
    )
    log['layersweep_info'] = layersweep_info
    log['best_layers'] = {'_'.join(k): v for k, v in best_layers.items()}
    # Evaluate similarity and store results
    get_model_nsd515_correlations(
        args.data.nsd_dir, output_dir, log_dir, args.data.localizer_dir,
        model, best_layers, layersweep_info['model_fmri_frois'],
        layersweep_info['model_localizer'], layersweep_info['tthreshold'],
        training_params.identifier.uid, training_params.project.experiment
    )
    # Store logs
    with open(os.path.join(log_dir, 'log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


if __name__ == "__main__":
    config = get_current_config()
    parser = argparse.ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main(config)
