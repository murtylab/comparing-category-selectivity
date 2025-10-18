"""
File to run layersweep analysis
"""
import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm

from compcats.localizer import localizer
from compcats.utils import model_utils, misc_utils, responsecomparison_utils as rcutils

# Device
DEVICE = misc_utils.get_device()

def model_layer_sweep(
    models_dir, nsd_dir, localizer_dir, \
    subjects_data, subjects_splithalf_pearsonrs, \
    model, layers, subset, model_fmri_frois, tthreshold, model_localizer,
    batch_size=None
):
    """
    Generate model-human univariate comparison results to select best layer
    NSD 485 images, 4 subjects, FFA, PPA, EBA, t>7
    Args:
        models_dir (str): Models directory
        nsd_dir (str): NSD directory
        localizer_dir (str): Localizer directory
        subjects_data (dict): Dictionary of subject data
        subjects_splithalf_pearsonrs (dict): Dictionary of subject split-half pearsonrs
        model (Model): Model instance
        layers (list): List of layers
        subset (int): Subset of NSD dataset
        model_fmri_frois (list): List of tuples of model and fmri frois
        tthreshold (float): Threshold for t-values
        model_localizer (str): Model localizer
    """
    # Load already computed results
    csv_pth = f'{models_dir}/{model.name}/nsd485_pearsonr_layersweep.csv'
    if os.path.exists(csv_pth):
        results_csv = pd.read_csv(csv_pth)
        results_list = results_csv.to_dict('records')
    else:
        results_list = []

    # Variables
    model_data = None
    model_froi_masks = None

    # Internal functions
    # Generate NSD data for models
    def _get_model_data():
        return model_utils.get_dataset_activations(
            models_dir, nsd_dir, model, layers, dataset='nsd', subset=485,
            batch_size=batch_size, device=DEVICE
        )
    def _get_model_froi_masks(froi):
        return localizer.get_froi_mask(
            localizer_dir, model_localizer, models_dir, \
            model, layers, froi, tthreshold, batch_size=batch_size
        )

    previous_model_froi = ''
    # Loop over frois
    for (model_froi, fmri_froi) in model_fmri_frois:
        # Check if model froi masks are present
        if model_froi != previous_model_froi:
            # Empty CUDA cache
            del model_froi_masks
            misc_utils.clear_memory(DEVICE)
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
                'fmri_dataset': f'nsd{subset}',
                'fmri_froi': fmri_froi,
                'fmri_tthreshold': tthreshold
            }
            if misc_utils.check_row_present(row_params, results_list):
                continue
            # Correlations
            subject_pearsonrs = []
            subject_normed_pearsonrs = []
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
    misc_utils.clear_memory(DEVICE)

    return pd.DataFrame(results_list)

def get_layers_already_swept(results_df, model_name, model_fmri_frois=None):
    """
    Get layers already swept for a model
    Args:
        results_df (pd.DataFrame): Results DataFrame
        model_name (str): Model name
    Returns:
        layers_swept (list): List of layers already swept
    """
    if model_fmri_frois is not None:
        pairs = set(model_fmri_frois)
        tuples = list(zip(results_df['model_froi'], results_df['fmri_froi']))
        mask_pairs = pd.Series([t in pairs for t in tuples], index=results_df.index)
        model_rows = results_df[
            (results_df['model_name'] == model_name) & mask_pairs
        ]
    else:
        model_rows = results_df[results_df['model_name'] == model_name]
    if model_rows.empty:
        raise ValueError(f'No rows found for model {model_name}')
    layers_swept = model_rows['model_layer'].unique().tolist()
    return layers_swept

def get_model_best_layer(results_df, model_name):
    """
    Get best layer for a model-froi pair
    Args:
        results_df (pd.DataFrame): Results DataFrame
        model_name (str): Model name
    Returns:
        best_layers (dict): Dictionary of best layers
    """
    best_layers = {}
    # Get all unique fMRI ROIs for this model
    model_rows = results_df[results_df['model_name'] == model_name]
    if model_rows.empty:
        raise ValueError(f'No rows found for model {model_name}')

    # Get all unique combinations of model_froi and fmri_froi
    froi_pairs = model_rows[['model_froi', 'fmri_froi']].drop_duplicates().values

    for model_froi, fmri_froi in froi_pairs:
        rows = model_rows[
            (model_rows['model_froi'] == model_froi) &
            (model_rows['fmri_froi'] == fmri_froi)
        ]

        key = (model_froi, fmri_froi)
        if rows['normed_pearsonr_median'].isnull().all():
            tmp_row = rows.iloc[0].copy()
            tmp_row['model_layer'] = None
            best_layers[key] = tmp_row
        else:
            best_row = rows.loc[rows['normed_pearsonr_median'].idxmax(skipna=True)]
            best_layers[key] = best_row.to_dict()

    return best_layers
