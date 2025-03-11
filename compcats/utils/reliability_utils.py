import numpy as np
import torch

from compcats.utils import responsecomparison_utils as rcutils

def fix_type(data):
    # Convert to appropriate type
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data, dtype=torch.float32)
    elif isinstance(data, torch.Tensor):
        if data.dtype != torch.float32:
            data = data.to(torch.float32)
    else:
        raise ValueError('Data type not recognized while calculating split-half correlations')
    return data

####################################################################################################
# VOXEL-WISE SPLITHALF CORRELATION
####################################################################################################
def get_voxelwise_splithalf_correlation(subject_data, image_subset=..., method=3):
    assert method in [1, 2, 3], 'Method must be 1, 2, or 3'
    # Convert to appropriate type
    subject_data = fix_type(subject_data)
    # Check dimensions
    assert subject_data.ndim == 3 and subject_data.shape[1]==3, \
        'Data must be 3D in the shape of n_stimuli, n_repetitions=3, n_voxels'
    # Voxel-wise pearson correlations
    voxel_corrs = []
    for vox_id in range(subject_data.shape[-1]):
        voxel_data = subject_data[image_subset, :, vox_id]
        if method==1:
            corr = _get_pearsonr_splithalf_correlarion_method1(voxel_data)
        elif method==2:
            corr = _get_pearsonr_splithalf_correlarion_method2(voxel_data)
        elif method==3:
            corr = _get_pearsonr_splithalf_correlarion_method3(voxel_data)
        if isinstance(corr, np.floating):
            corr = corr.item()
        voxel_corrs.append(corr)
    return np.array(voxel_corrs)

def _get_pearsonr_splithalf_correlarion_method1(voxel_data):
    """
    Mean across remaining repetitions and then calculate correlation
    Split each repetition into individual and take correlation across each pair
    """
    # Error handle
    assert voxel_data.ndim == 2, 'Data must be 2D in the shape of n_stimuli, n_repetitions'
    corrs_ij = []
    idx_combinations = [
        (0, 1),
        (1, 2),
        (0, 2)
    ]
    for (i, j) in idx_combinations:
        di = voxel_data[:, i]
        dj = voxel_data[:, j]
        corr_ij = rcutils.pearsonr(di, dj)
        corrs_ij.append(corr_ij)
    # Spearman-Brown correction
    return 2*np.mean(corrs_ij)/(1+np.mean(corrs_ij))

def _get_pearsonr_splithalf_correlarion_method2(voxel_data):
    """
    Mean across remaining repetitions and then calculate correlation
    Split each repetition into individual and take correlation across each individual and
    mean of remaining repetitions
    """
    # Error handle
    assert voxel_data.ndim == 2, 'Data must be 2D in the shape of n_stimuli, n_repetitions'
    corrs_ijk = []
    idx_combinations = [
        (0, (1, 2)),
        (1, (0, 2)),
        (0, (1, 2))
    ]
    for (i, (j, k)) in idx_combinations:
        di = voxel_data[:, i]
        dj = voxel_data[:, j]
        dk = voxel_data[:, k]
        djk = (dj + dk)/2
        corr_ijk = rcutils.pearsonr(di, djk)
        corrs_ijk.append(corr_ijk)
    return 2*np.mean(corrs_ijk)/(1+np.mean(corrs_ijk))

def _get_pearsonr_splithalf_correlarion_method3(voxel_data):
    """
    Split each repetition into individual and take correlation across each individual and
    mean of remaining repetitions, perform the spearman-brown correction and return the
    maximum of the three correlations
    """
    # Error handle
    assert voxel_data.ndim == 2, 'Data must be 2D in the shape of n_stimuli, n_repetitions'
    corrs_ijk = []
    idx_combinations = [
        (0, (1, 2)),
        (1, (0, 2)),
        (0, (1, 2))
    ]
    for (i, (j, k)) in idx_combinations:
        di = voxel_data[:, i]
        dj = voxel_data[:, j]
        dk = voxel_data[:, k]
        djk = (dj + dk)/2
        corr_ijk = rcutils.pearsonr(di, djk)
        corrs_ijk.append(corr_ijk)
    max_corr = np.max(corrs_ijk)
    corrected_corr = 2*max_corr/(1+max_corr)
    return corrected_corr

####################################################################################################
# RDM SPLITHALF CORRELATION
####################################################################################################
def get_rdm_splithalf_correlation(subject_data, image_subset=..., method=3):
    assert method in [1, 2, 3], 'Method must be 1, 2, or 3'
    # Convert to appropriate type
    subject_data = fix_type(subject_data)
    # Check dimensions
    assert subject_data.ndim == 3 and subject_data.shape[1]==3, \
        'Data must be 3D in the shape of n_stimuli, n_repetitions=3, n_voxels'
    # Subset data
    subset_data = subject_data[image_subset, :, :]
    # RDM Correlations
    if method==1:
        corr = _get_rdm_splithalf_correlation_method1(subset_data)
    elif method==2:
        corr = _get_rdm_splithalf_correlation_method2(subset_data)
    elif method==3:
        corr = _get_rdm_splithalf_correlation_method3(subset_data)
    if isinstance(corr, np.floating):
        corr = corr.item()
    return corr

def _get_rdm_splithalf_correlation_method1(subset_data):
    # Error handle
    assert subset_data.ndim == 3, \
        'Data must be 3D in the shape of n_stimuli, n_repetitions, n_voxels'
    # RDM Correlations
    corrs_ij = []
    idx_combinations = [
        (0, 1),
        (1, 2),
        (0, 2)
    ]
    for (i, j) in idx_combinations:
        di = subset_data[:, i, :]
        dj = subset_data[:, j, :]
        corr_ij = rcutils.rdmr(di, dj)
        corrs_ij.append(corr_ij)
    # Spearman-Brown correction
    return 2*np.mean(corrs_ij)/(1+np.mean(corrs_ij))

def _get_rdm_splithalf_correlation_method2(subset_data):
    # Error handle
    assert subset_data.ndim == 3, \
        'Data must be 3D in the shape of n_stimuli, n_repetitions, n_voxels'
    # RDM Correlations
    corrs_ijk = []
    idx_combinations = [
        (0, (1, 2)),
        (1, (0, 2)),
        (0, (1, 2))
    ]
    for (i, (j, k)) in idx_combinations:
        di = subset_data[:, i, :]
        dj = subset_data[:, j, :]
        dk = subset_data[:, k, :]
        djk = (dj + dk)/2
        corr_ijk = rcutils.rdmr(di, djk)
        corrs_ijk.append(corr_ijk)
    # Spearman-Brown correction
    return 2*np.mean(corrs_ijk)/(1+np.mean(corrs_ijk))

def _get_rdm_splithalf_correlation_method3(subset_data):
    """
    Split each repetition into individual and take correlation across each individual and
    mean of remaining repetitions, perform the spearman-brown correction and return the
    maximum of the three correlations
    """
    # Error handle
    assert subset_data.ndim == 3, \
        'Data must be 3D in the shape of n_stimuli, n_repetitions, n_voxels'
    # RDM Correlations
    corrs_ijk = []
    idx_combinations = [
        (0, (1, 2)),
        (1, (0, 2)),
        (0, (1, 2))
    ]
    for (i, (j, k)) in idx_combinations:
        di = subset_data[:, i, :]
        dj = subset_data[:, j, :]
        dk = subset_data[:, k, :]
        djk = (dj + dk)/2
        corr_ijk = rcutils.rdmr(di, djk)
        corrs_ijk.append(corr_ijk)
    max_corr = np.max(corrs_ijk)
    corrected_corr = 2*max_corr/(1+max_corr)
    return corrected_corr

####################################################################################################
# VOXEL-AVERAGED PEARSON SPLITHALF CORRELATION
####################################################################################################
def get_voxelmean_splithalf_correlation(subject_data, image_subset=..., method=3):
    assert method in [1, 2, 3], 'Method must be 1, 2, or 3'
    # Convert to appropriate type
    subject_data = fix_type(subject_data)
    # Check dimensions
    assert subject_data.ndim in [2, 3] and subject_data.shape[1]==3, \
        'Data must be 3D in the shape of n_stimuli, n_repetitions=3, n_voxels'
    # Subset data
    subset_data = None
    if subject_data.ndim == 2:
        subset_data = subject_data[image_subset, :]
    elif subject_data.ndim == 3:
        subset_data = subject_data[image_subset, :, :].mean(axis=-1)
    # Pearson Correlations
    if method==1:
        corr = _get_pearsonr_splithalf_correlarion_method1(subset_data)
    elif method==2:
        corr = _get_pearsonr_splithalf_correlarion_method2(subset_data)
    elif method==3:
        corr = _get_pearsonr_splithalf_correlarion_method3(subset_data)
    if isinstance(corr, np.floating):
        corr = corr.item()
    return corr
