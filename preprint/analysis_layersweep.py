"""
File to run layersweep analysis
"""
import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
import torch

from compcats.localizer import localizer
from compcats.utils import model_utils, misc_utils, responsecomparison_utils as rcutils

def model_layer_sweep(
        models_dir, nsd_dir, localizer_dir, \
        subjects_data, model, layers, subset, frois, tthreshold, model_localizer
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
            models_dir, nsd_dir, model, layers, dataset='nsd', subset=485
        )
    def _get_model_froi_masks(froi):
        return localizer.get_froi_mask(
            localizer_dir, model_localizer, models_dir, \
            model, layers, froi, tthreshold
        )
    # Loop over frois
    for froi in frois:
        # Reset model froi masks
        model_froi_masks = None
        print(f'Running layer sweep: {model.name}, {froi}')
        for layer in tqdm(layers):
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
            if misc_utils.check_row_present(row_params, results_list):
                continue
            # Correlations
            subject_pearsonrs = []
            for subject, subject_froi_data in subjects_data[froi].items():
                # Get model froi masks
                model_froi_masks = _get_model_froi_masks(froi) if model_froi_masks is None \
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
        # Empty CUDA cache
        del model_froi_masks

    # Empty CUDA cache
    del model_data
    torch.cuda.empty_cache()
