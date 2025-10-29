"""
Do ANN units match voxel tuning better than other voxels?
"""
import os
import torch

import paths
import main_analysis as ma

from compcats.datasets.fmri import nsd
from compcats.utils import model_utils, responsecomparison_utils as rcutils, \
    reliability_utils as relutils

# Paths
MODELS_DIR = 'models'
NSD_DIR = paths.NSD_DIR
RESULTS_DIR = 'results/voxel_unit_tuning_comparison'
os.makedirs(RESULTS_DIR, exist_ok=True)
BEST_MATCHING_CSV = os.path.join(RESULTS_DIR, 'voxel_best_matches_nsd485.csv')

# Variables
TO_MODEL_NAMES = ma.get_model_names(model_type='task_optimized')
UNTRAINED_MODEL_NAMES = ma.get_model_names(model_type='untrained')
TTHRESHOLD = 7 # t-value threshold
RTHRESHOLD = 0.4 # Reliability threshold
SH_METHOD = 5 # Split-half correlation method, first spearman-brown correcttion then \
# mean across the two halves across splits
DEVICE = torch.device('cpu')
FROIS = ['FFA', 'PPA', 'EBA']
SUBEJCTS = [1, 2, 5, 7]
SUBSETS = [485, 515]

def get_subject_froi_voxel_data(
    subject: int,
    froi: str,
    subset: int,
):
    """
    Get voxel data for a subject and froi for a given subset of images, and split-half
    reliabilities.
    
    """
    # Get all voxel individual repetition betas
    voxel_responses_repind, _ = nsd.get_betas(
        NSD_DIR, subject, froi, TTHRESHOLD, subset=subset, voxels=True,
        repetition_mean=False, device=DEVICE
    )
    # Get voxel split-half reliabilities
    voxel_reliabilities = relutils.get_voxelwise_splithalf_correlation(
        voxel_responses_repind, method=SH_METHOD
    )
    # Mean across repetitions
    voxel_responses_repmean = voxel_responses_repind.mean(dim=1)
    return voxel_responses_repmean, voxel_reliabilities

def get_model_unit_data(
    model_name: str,
    froi: str,
    layer: str,
    subset: int,
):
    """
    Get model unit for the full layer for a given subset of nsd.
    """
    model = ma.get_model(model_name)
    layer = ma.get_froi_models_layers(froi, froi)[model_name]
    unit_responses = model_utils.get_dataset_activations(
        MODELS_DIR, NSD_DIR, model, layer, 'nsd', subset=subset, device=DEVICE
    )
    # Flatten unit responses across dimensions except for image dimension (dim 0)
    unit_responses = unit_responses.flatten(start_dim=1)
    return unit_responses

def compare_tuning(
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
    correlations = torch.zeros((responses1.shape[1], responses2.shape[1]))
    for i in range(responses1.shape[1]):
        for j in range(responses2.shape[1]):
            corr = rcutils.pearsonr(
                responses1[:, i], responses2[:, j]
            )
            # Reliability correction
            corr /= torch.sqrt(reliabilities1[i] * reliabilities2[j])
            correlations[i, j] = corr
    return correlations

def get_best_matching_unit(
    correlations: torch.Tensor,
    reliabilities1: torch.Tensor,
    reliabilities2: torch.Tensor,
):
    """
    For each unit in set 1, get the best matching unit in set 2 based on correlations.
    correlations: (num_units1, num_units2)
    reliabilities1: (num_units1,)
    reliabilities2: (num_units2,)
    Returns:
        best_matching_indices: (num_units1,)
    """
    # Apply reliability thresholding
    reliable_units1 = reliabilities1 >= RTHRESHOLD
    reliable_units2 = reliabilities2 >= RTHRESHOLD
    correlations = correlations.clone()
    correlations[~reliable_units1, :] = -torch.inf
    correlations[:, ~reliable_units2] = -torch.inf
    # Control for NaNs
    correlations[torch.isnan(correlations)] = -torch.inf
    # Get best matching indices
    best_matching_indices = torch.argmax(correlations, dim=1)
    return best_matching_indices

def generate_voxel_voxel_comparisons():
    """
    Generate voxel-voxel tuning comparisons across subjects and frois.
    """
    for subset in SUBSETS:
            for froi in FROIS:
                for subject1 in SUBEJCTS:
                    for subject2 in SUBEJCTS:
                        if subject1 >= subject2:
                            continue
                        print(f'Processing Subject {subject1} vs Subject {subject2}, FROI {froi}, Subset {subset}')
                        # Get voxel data for both subjects
                        voxel_responses1, voxel_reliabilities1 = get_subject_froi_voxel_data(
                            subject1, froi, subset
                        )
                        voxel_responses2, voxel_reliabilities2 = get_subject_froi_voxel_data(
                            subject2, froi, subset
                        )
                        # Compare voxels to voxels
                        correlations = compare_tuning(
                            voxel_responses1, voxel_responses2,
                            voxel_reliabilities1, voxel_reliabilities2
                        )
                        # Get best matching voxels
                        best_matching_indices = get_best_matching_unit(
                            correlations, voxel_reliabilities1, voxel_reliabilities2
                        )
                        # Save results
                        result_path = os.path.join(
                            RESULTS_DIR,
                            f'subject{subject1}_vs_subject{subject2}_froi{froi}_subset{subset}_voxel_voxel_comparison.pt'
                        )
                        torch.save({
                            'correlations': correlations,
                            'best_matching_indices': best_matching_indices,
                            'voxel_reliabilities1': voxel_reliabilities1,
                            'voxel_reliabilities2': voxel_reliabilities2,
                        }, result_path)