"""Localize fROIs"""
from typing import Tuple
import os
import json
import warnings
import torch

from ..datasets.fmri import nsd
from ..utils import model_utils
from . import utils

CONTRAST_LOCALIZERS = utils.CONTRAST_LOCALIZERS
TUNINGMATCHED_LOCALIZERS = ['nsd']
RTHRESHOLD = 0.4  # Reliability threshold for voxel selection
NSD_SUBSET_TUNINGMATCHED = 485  # NSD subset for tuning-matched localizer
DEFAULT_SUBJECTS_TUNINGMATCHED = [1, 2, 5, 7]  # Default NSD subjects for tuning-matched localizer

def t_statistic(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes the t-statistic for the difference between two tensors.
    Assumes the first dimension represents samples.
    
    Args:
        tensor1 (torch.Tensor): A tensor of shape (n_samples1, ...).
        tensor2 (torch.Tensor): A tensor of shape (n_samples2, ...).
    
    Returns:
        torch.Tensor: The t-statistic tensor of shape (...).
    """
    n1 = tensor1.shape[0]
    n2 = tensor2.shape[0]
    mean_diff = tensor1.mean(dim=0) - tensor2.mean(dim=0)
    var1 = tensor1.var(dim=0, unbiased=True)
    var2 = tensor2.var(dim=0, unbiased=True)

    pooled_se = torch.sqrt(var1 / n1 + var2 / n2)
    t_stat = mean_diff / pooled_se

    # Error handling
    # Replace inf with -inf so that it can be used for masking
    t_stat[t_stat == torch.inf] = -torch.inf
    # Replace NaN with -inf so that it can be used for masking
    t_stat[torch.isnan(t_stat)] = -torch.inf

    # p-values not being calculated for computational efficiency
    calculate_pvalues = False
    if calculate_pvalues:
        raise NotImplementedError("P-value calculations have not been verified")
        # Degrees of freedom using Welch-Satterthwaite equation
        df = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
        # Calculate p-value (two-tailed)
        p_values = 2 * (1 - torch.distributions.StudentT(df).cdf(torch.abs(t_stat)))

    return t_stat

def compare_activations(tensor1, tensor2, tthreshold):
    """
    Function to calculate t-values for all elements using samples in dimension 0 of both tensors
    Args:
        tensor1(torch.Tensor): Tensor 1, shape: (n_samples1, ...)
        tensor2(torch.Tensor): Tensor 2, shape: (n_samples2, ...)
        tthreshold(float): t-threshold
    """
    assert tensor1.shape[1:] == tensor2.shape[1:], "Tensor dimensions do not match"
    t_values = t_statistic(tensor1, tensor2)
    mask = t_values >= tthreshold
    return mask, t_values

def localize(model, layer, pref_datasets, nonpref_datasets, froi, dataset_name, \
             activations_dir, froi_dir, tthreshold, return_tvals=False, **kwargs):
    """
    Localize units in a layer that respond to 
    a preferred category more than non-preferred categories
    Args:
        model():
        layer_num(int): Layer number
        pref_datasets(list): List of preferred datasets
        nonpref_datasets(list): List of non-preferred datasets
        froi(str): fROI name
        dataset_name(str): Localizer dataset name
        activations_dir(str): Directory to store activations
        froi_dir(str): Directory to store fROI masks
        tthreshold(float): t-threshold
        return_tvals(bool): Return t-values
    """
    batch_size = kwargs.get('batch_size', None)
    # Generate all activations
    # preferred categories
    pref_activations = {}
    for pref_dataset in pref_datasets:
        pref_img_class = pref_dataset.name
        pref_activation_ = model.generate_dataset_activations(
            layer, pref_dataset, activations_dir, batch_size=batch_size
        )
        pref_activations[pref_img_class] = pref_activation_
    # non-preferred categories
    nonpref_activations = {}
    for nonpref_dataset in nonpref_datasets:
        nonpref_img_class = nonpref_dataset.name
        nonpref_activation_ = model.generate_dataset_activations(
            layer, nonpref_dataset, activations_dir, batch_size=batch_size
        )
        nonpref_activations[nonpref_img_class] = nonpref_activation_
    # Compare activations
    froi_mask = None
    all_contrast_tvals = {}
    for pref_dataset in pref_datasets:
        for nonpref_dataset in nonpref_datasets:
            pref_img_class = pref_dataset.name
            nonpref_img_class = nonpref_dataset.name
            key = (pref_img_class, nonpref_img_class)
            # Get contrast mask and t-values
            contrast_mask, contrast_tvals = compare_activations(
                pref_activations[pref_img_class], \
                nonpref_activations[nonpref_img_class], \
                tthreshold
            )
            if froi_mask is None:
                froi_mask = contrast_mask
            else:
                froi_mask = froi_mask & contrast_mask
            all_contrast_tvals[key] = contrast_tvals
    # Store fROI mask
    froi_file_name = f'{model.name}_{layer}_{dataset_name}_{froi}_{tthreshold:.2f}.pt'
    torch.save(froi_mask.cpu(), f'{froi_dir}/{froi_file_name}')
    print(f'Saving fROI {froi_dir}/{froi_file_name}')
    # Store results
    mean_tvals = []
    for key, tvals in all_contrast_tvals.items():
        froi_tvals = tvals[froi_mask]
        mean_tvals.append([key, froi_tvals.mean().item()])
    # TODO: Fix results json name, froi is twice
    json_file = f'results_{model.name}_{layer}_{froi}_{dataset_name}_{froi}_{tthreshold:.2f}.json'
    json_file_path = f'{froi_dir}/{json_file}'
    results = {
        'model_name': model.name,
        'layer': layer,
        'preferred_categories': [pds.name for pds in pref_datasets],
        'nonpreferred_categories': [npds.name for npds in nonpref_datasets],
        'froi': froi,
        'localizer': dataset_name,
        'tthreshold': tthreshold,
        'total_units_len': froi_mask.numel(),
        'selected_units_len': int(froi_mask.sum().item()),
        'mean_t_values': mean_tvals,
        'unit_indices': froi_mask.flatten().nonzero(as_tuple=False).flatten().tolist()
    }
    # Save JSON
    with open(json_file_path, "w", encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4)
    if return_tvals:
        return froi_mask, all_contrast_tvals
    return froi_mask

def get_froi_mask(
    dataset_dir, dataset_name, models_dir,
    model, layers, froi, tthreshold, return_info=False,
    **kwargs
):
    """
    Get indices for neurons corresponding to froi, localized
    Args:
        localizer_type(str): grillspector, kanwisher or konkle
    """
    # Error handling
    froi = froi.upper()
    dataset_name = dataset_name.lower()
    if isinstance(layers, str):
        layers_list = [layers]
    elif isinstance(layers, list):
        layers_list = layers
    else:
        raise ValueError("layers should be a string or a list of strings")

    # kwargs
    # Batch size
    batch_size = kwargs.get('batch_size', None)
    # Subjects for tuning-matched localizer
    subjects = kwargs.get('subjects', DEFAULT_SUBJECTS_TUNINGMATCHED)
    if isinstance(subjects, int):
        subjects = [subjects]
    assert set(subjects).issubset(set(DEFAULT_SUBJECTS_TUNINGMATCHED)), \
        f"Subjects should be subset of {DEFAULT_SUBJECTS_TUNINGMATCHED}"
    # Sort subjects
    subjects = sorted(subjects)
    # Reliability threshold for tuning-matched localizer
    rthreshold = kwargs.get('rthreshold', RTHRESHOLD)
    # Unique unit per voxel matching
    unique_unit_per_voxel = kwargs.get('unique_unit_per_voxel', False)

    # Localizer type
    localizer_type = None
    if dataset_name in CONTRAST_LOCALIZERS:
        localizer_type = 'contrast'
    elif dataset_name in TUNINGMATCHED_LOCALIZERS:
        localizer_type = 'tuningmatched'
    else:
        raise ValueError(f"{dataset_name} not in {CONTRAST_LOCALIZERS + TUNINGMATCHED_LOCALIZERS}")

    # Results
    results = {}

    # Dirs
    root_dir = f'{models_dir}/{model.name}/{dataset_name}'
    activations_dir = f'{root_dir}/activations'
    os.makedirs(activations_dir, exist_ok=True)

    activations_generated = False
    # Only generate activations when needed
    pref_datasets = None
    nonpref_datasets = None
    def _generate_activations():
        # Get localizer dataset module
        dataset_module = utils.get_localizerdataset_module(dataset_name)
        # Get preferred & non-preferred datasets
        pref_datasets, nonpref_datasets = utils.get_pref_nonpref_datasets(
            dataset_dir, dataset_module, froi, model.transforms
        )
        # Generate all layer activations together in one foward pass
        for pref_dataset in pref_datasets:
            model.generate_dataset_activations(
                layers_list, pref_dataset, activations_dir, batch_size=batch_size
            )
        for nonpref_dataset in nonpref_datasets:
            model.generate_dataset_activations(
                layers_list, nonpref_dataset, activations_dir, batch_size=batch_size
            )
        return True, pref_datasets, nonpref_datasets
    subjects_voxel_nsd485_repind = {}
    model_units_nsd485 = {}
    def _generate_tuningmatched_activations() -> Tuple[dict, dict]:
        subjects_voxel_nsd485_repind = {}
        for subj in DEFAULT_SUBJECTS_TUNINGMATCHED:
            voxel_responses_repind, _ = nsd.get_betas(
                dataset_dir, subj, froi, tthreshold, subset=NSD_SUBSET_TUNINGMATCHED,
                voxels=True, repetition_mean=False
            )
            subjects_voxel_nsd485_repind[subj] = voxel_responses_repind
        model_units_nsd485 = model_utils.get_dataset_activations(
            models_dir, dataset_dir, model, layers_list, 'nsd', subset=NSD_SUBSET_TUNINGMATCHED
        )
        # Tuning-matched localization is faster on cpu
        model_units_nsd485 = {
            layer: activations.cpu() for layer, activations in model_units_nsd485.items()
        }
        return subjects_voxel_nsd485_repind, model_units_nsd485

    for layer in layers_list:
        # Mask file name
        mask_file = f'{model.name}_{layer}_{dataset_name}_{froi}_{tthreshold:.2f}'
        if localizer_type == 'tuningmatched':
            suffix = '_uniqueunitpervoxel' if unique_unit_per_voxel else ''
            mask_file += f'_{rthreshold:.2f}{suffix}'
        mask_file += '.pt'
        froi_dir = f'{root_dir}/{froi}/{layer}'
        os.makedirs(froi_dir, exist_ok=True)
        # Check if mask file is present
        if mask_file in os.listdir(froi_dir):
            mask = torch.load(f'{froi_dir}/{mask_file}')
            if localizer_type == 'tuningmatched':
                # Mask will be a dict of subject masks
                if subjects == DEFAULT_SUBJECTS_TUNINGMATCHED:
                    key = 'all'
                    mask = mask[key]
                    if not return_info:
                        mask = mask['mask']
                elif len(subjects) == 1:
                    key = str(subjects[0])
                    mask = mask[key]
                    if not return_info:
                        mask = mask['mask']
                else:
                    if return_info:
                        warnings.warn(
                            'Extra information not returned when combining multiple subjects, ' \
                            '   only mask will be returned'
                        )
                    # Combine masks of multiple subjects
                    subject_masks = [mask[str(subj)]['mask'] for subj in subjects]
                    # Stack across a new dimension and then take union across that dimension
                    stacked_masks = torch.stack(subject_masks, dim=0)
                    mask = torch.any(stacked_masks, dim=0)
                # Store result
                results[layer] = mask
            elif localizer_type == 'contrast':
                if return_info:
                    warnings.warn("t-values not present in the saved mask file")
                    results[layer] = {
                        'mask': mask,
                        'all_contrast_tvals': None
                    }
                else:
                    results[layer] = mask
            continue

        # Localization
        if localizer_type == 'contrast':
            # Generate fROI mask if not present
            # Generate activations
            if not activations_generated:
                activations_generated, pref_datasets, nonpref_datasets = _generate_activations()

            # Localize the units in the layer
            result = localize(
                model, layer, pref_datasets, nonpref_datasets, froi, dataset_name,
                activations_dir, froi_dir, tthreshold, return_tvals=return_info, **kwargs
            )
            # result is froi_mask if return_info=False
            # result is froi_mask, all_contrast_tvals if return_info=True
            if return_info:
                mask, all_contrast_tvals = result
                results[layer] = {
                    'mask': mask,
                    'all_contrast_tvals': all_contrast_tvals
                }
            else:
                results[layer] = result
        elif localizer_type == 'tuningmatched':
            # When generating the masks, do not pass reliability-subsetted voxel indices
            # for future use cases
            # Generate activations if not already done
            if not activations_generated:
                subjects_voxel_nsd485_repind, model_units_nsd485 = \
                    _generate_tuningmatched_activations()
                activations_generated = True
            # Generate all subjects tuning-matched localizer masks
            subjects_results = utils.generate_tuningmatched_localizer_masks(
                save_dir=froi_dir,
                subjects_voxel_responses_repind=subjects_voxel_nsd485_repind,
                model_unit_responses=model_units_nsd485[layer],
                model_name=model.name,
                layer=layer,
                subjects=DEFAULT_SUBJECTS_TUNINGMATCHED,
                froi=froi,
                dataset_name=dataset_name,
                subset=NSD_SUBSET_TUNINGMATCHED,
                tthreshold=tthreshold,
                rthreshold=rthreshold,
                unique_unit_per_voxel=unique_unit_per_voxel
            )
            # Get the mask for specified subjects
            key = str(subjects[0]) if len(subjects) == 1 else 'all'
            subject_results = subjects_results[key]
            # Choose only the mask if return_info is False
            if return_info:
                results[layer] = subject_results
            else:
                results[layer] = subject_results['mask']
        else:
            raise ValueError(f"{localizer_type} not recognized")

    # Error handling
    if isinstance(layers, str):
        results = results[layers]

    return results
