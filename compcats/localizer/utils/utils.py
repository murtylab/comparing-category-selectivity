"""
Utility functions for localizer datasets
"""
import json
import os
from typing import List, Union, Dict
from tqdm import tqdm
import torch

from ...datasets.localization import grillspector_localizer, \
    kanwisher_localizer, konkle_localizer, vinken2023_localizer, kanwishergrayscale_localizer, \
    prince2024_localizer
from ...utils import reliability_utils as relutils, responsecomparison_utils as rcutils

# Constants
DEFAULT_SH_METHOD = 3  # method for reliability calculation

CONTRAST_LOCALIZERS = [
    'grillspector',
    'kanwisher',
    'konkle',
    'kanwishergrayscale',
    'vinken2023',
    'prince2024'
]

def get_localizerdataset_module(dataset_name):
    """
    Get localizer dataset based on type
    """
    if dataset_name == 'kanwisher':
        dataset_module = kanwisher_localizer
    elif dataset_name == 'kanwishergrayscale':
        dataset_module = kanwishergrayscale_localizer
    elif dataset_name == 'grillspector':
        dataset_module = grillspector_localizer
    elif dataset_name == 'konkle':
        dataset_module = konkle_localizer
    elif dataset_name == 'vinken2023':
        dataset_module = vinken2023_localizer
    elif dataset_name == 'prince2024':
        dataset_module = prince2024_localizer
    else:
        raise ValueError(f'{dataset_name} unknown localizer type')
    return dataset_module

def get_pref_nonpref_datasets(dataset_dir, dataset_module, froi, transforms):
    """
    Function to get preferred & non-preferred datasets
    Args:
        dataset_dir(str): Directory where dataset is stored
        dataset_module(compcats module): Localizer dataset Module
    """
    # Get preferred & non-preferred stimuli names
    dataset_obj = dataset_module.get_localizer_dataset(dataset_dir, transforms, image_class='all')
    pref_categories, nonpref_categories = dataset_obj.get_pref_nonpref_categories(froi)
    # Get preferred & non-preferred datasets
    pref_datasets = [
        dataset_module.get_localizer_dataset(
            dataset_dir, transforms, image_class=pref_category
        ) for pref_category in pref_categories
    ]
    nonpref_datasets = [
        dataset_module.get_localizer_dataset(
            dataset_dir, transforms, image_class=nonpref_category
        ) for nonpref_category in nonpref_categories
    ]
    return pref_datasets, nonpref_datasets

def get_all_imageclass_datasets(dataset_dir, dataset_module, transforms):
    """
    Function to get all image class datasets
    Args:
        dataset_dir(str): Directory where dataset is stored
        dataset_module(compcats module): Localizer dataset Module
        transforms(torchvision.transforms): Image transformations
    """
    # Get all stimuli names
    dataset_obj = dataset_module.get_localizer_dataset(dataset_dir, transforms, image_class='all')
    imageclasses = dataset_obj.imageclasses
    imageclass_datasets = {}
    for image_class in imageclasses:
        imageclass_datasets[image_class] = dataset_module.get_localizer_dataset(
            dataset_dir, transforms, image_class=image_class
        )
    return imageclass_datasets

def get_localizer_pref_nonpref_categories(dataset_name, froi):
    """
    Get preferred & non-preferred image classes
    """
    localizer_module = get_localizerdataset_module(dataset_name)
    localizer_obj = localizer_module.get_localizer_dataset('', None)
    pref_categories, nonpref_categories = localizer_obj.get_pref_nonpref_categories(froi)
    return pref_categories, nonpref_categories


# Tuning-matched localizer utilities
def get_tuning_correlation_matrix(
    responses1: torch.Tensor,
    responses2: torch.Tensor,
    reliabilities1: torch.Tensor,
    reliabilities2: torch.Tensor,
):
    """
    Compare tuning between two sets of responses using pearson correlation.
    responses1: (num_images, num_units1)
    responses2: (num_images, num_units2)
    reliabilities1: (num_units1,)
    reliabilities2: (num_units2,)
    Returns:
        correlations: (num_units1, num_units2)
    """
    assert responses1.ndim == 2, "responses1 must be 2-dimensional"
    assert responses2.ndim == 2, "responses2 must be 2-dimensional"
    assert responses1.shape[0] == responses2.shape[0], \
        "Number of images in responses1 and responses2 must match"
    assert reliabilities1.ndim == 1, "reliabilities1 must be 1-dimensional"
    assert reliabilities2.ndim == 1, "reliabilities2 must be 1-dimensional"
    assert responses1.shape[1] == reliabilities1.shape[0], \
        "Number of units in responses1 and reliabilities1 must match"
    assert responses2.shape[1] == reliabilities2.shape[0], \
        "Number of units in responses2 and reliabilities2 must match"
    correlations = torch.zeros((responses1.shape[1], responses2.shape[1]))
    for j in tqdm(range(responses2.shape[1])):
        for i in range(responses1.shape[1]):
            corr = rcutils.pearsonr(
                responses1[:, i], responses2[:, j]
            )
            # Reliability correction
            corr /= torch.sqrt(reliabilities1[i] * reliabilities2[j])
            correlations[i, j] = corr
    return correlations

def get_best_matching_units(
    correlations_matrix: torch.Tensor
):
    """
    For each unit in set 1, get the best matching unit in set 2 based on correlations.
    correlations_matrix: (num_units1, num_units2)
    Returns:
        matched_voxel_indices: (num_voxels,)
        best_matching_unit_indices: (num_voxels,)
    """
    assert correlations_matrix.ndim == 2, \
        "correlations_matrix must be 2-dimensional"
    assert correlations_matrix.shape[1] >= correlations_matrix.shape[0], \
        "Number of units in set 2 must be greater than or equal to number of units in set 1"
    # Control for NaNs using torch nan_to_num
    correlations_matrix = torch.nan_to_num(
        correlations_matrix, nan=float('-inf')
    )
    # Variables to store results
    matched_voxel_indices = torch.full((correlations_matrix.shape[0],), -1, dtype=torch.long)
    best_matching_unit_indices = torch.full((correlations_matrix.shape[0],), -1, dtype=torch.long)
    best_matching_correlations = torch.full((correlations_matrix.shape[0],), float('-inf'), dtype=torch.float32)
    # For each unit in set 1, find the best matching unit in set 2
    for _ in range(correlations_matrix.shape[0]):
        max_corr_coords = torch.unravel_index(
            torch.argmax(correlations_matrix, dim=None),
            correlations_matrix.shape
        )
        v1, u2 = max_corr_coords
        max_corr = correlations_matrix[v1, u2]
        matched_voxel_indices[v1] = v1
        best_matching_unit_indices[v1] = u2
        best_matching_correlations[v1] = max_corr
        # Set the corresponding row and column to -inf to avoid re-selection
        correlations_matrix[v1, :] = float('-inf')
        correlations_matrix[:, u2] = float('-inf')
    # Assert all voxels have been matched
    assert (matched_voxel_indices != -1).all(), \
        "Not all voxels have been matched"
    assert (best_matching_unit_indices != -1).all(), \
        "Not all best matching units have been assigned"
    assert (best_matching_correlations != float('-inf')).all(), \
        "Not all best matching correlations have been assigned"
    return best_matching_unit_indices, best_matching_correlations

def get_tuningmatched_localizer_mask(
    voxel_responses: torch.Tensor,
    unit_responses: torch.Tensor,
    reliability_threshold: float,
    voxel_reliabilities: Union[torch.Tensor, None] = None,
):
    """
    Generate tuning-matched localizer masks for one subjects and froi.
    Args:
        voxel_individualrepetition_responses(torch.Tensor): Voxel responses with shape
            (num_images, num_repetitions, num_voxels)
        unit_responses(torch.Tensor): Model unit responses with shape (num_images, ...)
        reliability_threshold(float): Reliability threshold for voxels
        voxel_reliabilities(torch.Tensor or None): Voxel reliabilities with shape (num_voxels,).
            If None, will be computed from voxel_individualrepetition_responses.
    Returns:
        mask (torch.Tensor): Boolean mask of shape (...) indicating tuning-matched units
        voxel_indices (torch.Tensor): Indices of voxels used for tuning-matching
        unit_indices (torch.Tensor): Indices of units selected as tuning-matched, will have
            the same shape as the number of voxels used for tuning-matching
        correlations (torch.Tensor): Correlations of shape (num_voxels) indicating
            best matching correlations
    """
    # Error handling
    if voxel_reliabilities is None:
        assert voxel_responses.ndim == 3, \
            "voxel_responses must have shape (num_images, num_repetitions, num_voxels)"
    else:
        assert voxel_responses.ndim in [2, 3], \
            "voxel_responses must have shape (num_images, num_voxels) or (num_images, num_repetitions, num_voxels)"
        assert voxel_responses.shape[-1] == voxel_reliabilities.shape[0], \
            "Number of voxels in voxel_responses and voxel_reliabilities must match"
        assert voxel_reliabilities.ndim == 1, "voxel_reliabilities must have shape (num_voxels,)"
    assert unit_responses.ndim >= 2, "unit_responses must have shape (num_images, ...)"
    assert voxel_responses.shape[0] == unit_responses.shape[0], \
        "Number of images in voxel_responses and unit_responses must match"

    # Estimate voxel split-half reliabilities if not provided
    if voxel_reliabilities is None:
        voxel_reliabilities = torch.tensor(
            relutils.get_voxelwise_splithalf_correlation(
                voxel_responses, method=DEFAULT_SH_METHOD
            )
        )

    # Mean across repetitions if needed
    if voxel_responses.ndim == 3:
        voxel_responses = voxel_responses.mean(dim=1)

    # Get reliable voxel mask
    voxel_reliability_mask = voxel_reliabilities >= reliability_threshold
    voxel_indices = torch.where(voxel_reliability_mask)[0]
    reliable_voxel_responses = voxel_responses[:, voxel_reliability_mask]
    reliable_voxel_reliabilities = voxel_reliabilities[voxel_reliability_mask]

    # Get tuning correlation matrix
    correlations_matrix = get_tuning_correlation_matrix(
        reliable_voxel_responses,
        unit_responses.flatten(start_dim=1),
        reliable_voxel_reliabilities,
        torch.ones(unit_responses.shape[1:].numel()) # Assume all units have reliability of 1
    )

    # Get best matching units
    best_matching_unit_indices, correlations = get_best_matching_units(
        correlations_matrix
    )

    # Create mask for tuning-matched units with the same shape as unit_responses starting from dim 1
    mask_shape = unit_responses.shape[1:]
    tuningmatched_unit_mask = torch.zeros(mask_shape, dtype=torch.bool)
    for unit_idx in best_matching_unit_indices:
        # Convert flat index to multi-dimensional index
        multi_dim_idx = torch.unravel_index(unit_idx, mask_shape)
        tuningmatched_unit_mask[multi_dim_idx] = True

    return tuningmatched_unit_mask, voxel_indices, best_matching_unit_indices, correlations

def generate_tuningmatched_localizer_masks(
    save_dir: str,
    subjects_voxel_responses_repind: Dict[int, torch.Tensor],
    model_unit_responses: torch.Tensor,
    # Extra info
    model_name: str,
    layer: str,
    subjects: List[int],
    froi: str,
    dataset_name: str,
    subset: int,
    tthreshold: float,
    rthreshold: float,
):
    """
    Generate tuning-matched localizer masks for a model, layer and froi.
    Take in all subjects' individual repetition voxel responses and model unit
    responses for the layer.
    Generate tuning-matched localizer masks for each subject, save the dict as a torch file.
    Save extra info like voxel indices, unit indices, correlations as a json file.
    Args:
        save_dir(str): Directory to save the tuning-matched localizer masks
        subjects_voxel_responses_repind(dict): Dict of subject id to voxel responses with shape
            (num_images, num_repetitions, num_voxels)
        model_unit_responses(torch.Tensor): Model unit responses with shape (num_images, ...)
        # Extra info
        model_name(str): Name of the model
        layer(str): Layer name
        subjects(list): List of subject ids
        froi(str): fROI name
        dataset_name(str): Localizer dataset name
        subset(int): Subset number
        tthreshold(float): Threshold for tuning-matching based on correlation
        rthreshold(float): Reliability threshold for voxels
    """
    mask_file_name = \
        f'{model_name}_{layer}_{dataset_name}_{froi}_{tthreshold:.2f}_{rthreshold:.2f}.pt'
    results_file_name = \
        f'results_{model_name}_{layer}_{froi}_{dataset_name}_{tthreshold:.2f}_{rthreshold:.2f}.json'

    # Get tuning-matched localizer masks for each subject
    across_subjects_mask: Union[torch.Tensor, None] = None
    subject_all_results = {}
    for subj in subjects:
        print(f'Generating tuning-matched localizer mask for subject {subj}...')
        voxel_responses_repind = subjects_voxel_responses_repind[subj]
        tuningmatched_mask, voxel_indices, unit_indices, correlations = \
            get_tuningmatched_localizer_mask(
                voxel_responses_repind,
                model_unit_responses,
                reliability_threshold=rthreshold
            )
        subject_all_results[str(subj)] = {
            'mask': tuningmatched_mask,
            'voxel_indices': voxel_indices,
            'unit_indices': unit_indices,
            'correlations': correlations
        }
        if across_subjects_mask is None:
            across_subjects_mask = tuningmatched_mask.clone()
        else:
            across_subjects_mask |= tuningmatched_mask
    # Add the across-subjects mask
    subject_all_results['all'] = {
        'mask': across_subjects_mask
    }

    # Save masks as torch file
    assert os.path.exists(save_dir), f'Save dir {save_dir} does not exist'
    torch.save(
        subject_all_results,
        os.path.join(save_dir, mask_file_name)
    )

    # Save extra info as json file
    results = {
        'model_name': model_name,
        'layer': layer,
        'froi': froi,
        'dataset_name': dataset_name,
        'subset': subset,
        'tthreshold': tthreshold,
        'rthreshold': rthreshold,
        'subjects': subjects,
        'subjects_results': {}
    }
    for subj in subjects:
        subj_results = subject_all_results[str(subj)]
        results['subjects_results'][str(subj)] = {
            'voxel_indices': subj_results['voxel_indices'].tolist(),
            'unit_indices': subj_results['unit_indices'].tolist(),
            'correlations': subj_results['correlations'].tolist()
        }
    # Save as json
    with open(os.path.join(save_dir, results_file_name), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    return subject_all_results
