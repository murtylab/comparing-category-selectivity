from typing import Union, Any
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
def get_voxelwise_splithalf_correlation(
    subject_data: Union[np.ndarray, torch.Tensor],
    image_subset: Union[np.ndarray, torch.Tensor, Any] = ...,
    method: int = 3,
) -> np.ndarray:
    assert method in list(range(1, 6)), 'Method must be 1, 2, 3, 4, or 5'
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
        elif method==4:
            corr = _get_pearsonr_splithalf_correlarion_method4(voxel_data)
        elif method==5:
            corr = _get_pearsonr_splithalf_correlarion_method5(voxel_data)
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

def _get_pearsonr_splithalf_correlarion_method4(voxel_data):
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
        corrected_corr_ijk = 2*corr_ijk/(1+corr_ijk)
        corrs_ijk.append(corrected_corr_ijk)
    return np.mean(corrs_ijk)

def _get_pearsonr_splithalf_correlarion_method5(voxel_data):
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
        corr_ijk = 2*corr_ijk/(1+corr_ijk)
        corrs_ijk.append(corr_ijk)
    max_corr = np.max(corrs_ijk)
    return max_corr

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
    assert method in [1, 2, 3, 4, 5], 'Method must be 1, 2, 3, 4, or 5'
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
    elif method==4:
        corr = _get_pearsonr_splithalf_correlarion_method4(subset_data)
    elif method==5:
        corr = _get_pearsonr_splithalf_correlarion_method5(subset_data)
    if isinstance(corr, np.floating):
        corr = corr.item()
    return corr


####################################################################################################
# NEURON-AVERAGED PEARSON SPLITHALF CORRELATION
####################################################################################################
def get_neuron_population_splithalf_correlation(
    responses: np.ndarray,
    metric: str = "pearsonr",
    n_boot: int = 100,
    spearman_brown: bool = True,
    random_state: int | None = 42,
):
    """
    Estimate population-level split-half reliability across stimuli for macaque data.

    Parameters
    ----------
    responses : np.ndarray
        Array of shape (n_stimuli, n_repeats, n_electrodes)
        Each stimulus may have NaNs or zeros for missing repeats.
    n_boot : int
        Number of random splits for bootstrapping.
    metric : {"pearsonr", "rdmr"}
        "pearsonr": reliability of mean response across electrodes.
        "rdmr": reliability of population pattern (RDM correlation).
    spearman_brown : bool
        Whether to apply Spearman-Brown correction to each split.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    reliability : float
        Mean (Spearman-Brown-corrected) reliability across bootstraps.
    reliability_splits : np.ndarray
        Reliability values (uncorrected and corrected) for each bootstrap split.
    """
    # Error handle
    metric = metric.lower()
    assert metric in ["pearsonr", "rdmr"], "metric must be 'pearsonr' or 'rdmr'"
    assert responses.ndim == 3, \
        "responses must be 3D in the shape of n_stimuli, n_repeats, n_electrodes"
    # Set random state
    rng = np.random.default_rng(random_state)
    n_stimuli, n_repeats, n_electrodes = responses.shape
    if isinstance(responses, torch.Tensor):
        responses = responses.cpu().numpy()

    split_rs = []
    for _ in range(n_boot):
        # --- Create random splits per stimulus ---
        half1, half2 = [], []
        for stim in range(n_stimuli):
            valid_reps = np.where(~np.isnan(responses[stim, :].mean(axis=-1)))[0]
            n_valid = len(valid_reps)
            if n_valid < 2:
                continue
            idx = rng.permutation(valid_reps)
            split = n_valid // 2
            reps1, reps2 = idx[:split], idx[split:]
            if len(reps1) == 0 or len(reps2) == 0:
                continue
            half1.append(np.nanmean(responses[stim, reps1, :], axis=0))
            half2.append(np.nanmean(responses[stim, reps2, :], axis=0))

        half1, half2 = np.array(half1), np.array(half2)
        if half1.size == 0 or half2.size == 0:
            continue

        # --- Compute population-level correlation ---
        if metric == "pearsonr":
            # Mean across electrodes in each half → 1 vector per half
            v1 = torch.Tensor(half1.mean(axis=1))
            v2 = torch.Tensor(half2.mean(axis=1))
            r = rcutils.pearsonr(v1, v2)
        elif metric == "rdmr":
            half1, half2 = torch.Tensor(half1), torch.Tensor(half2)
            r = rcutils.rdmr(half1, half2)
        else:
            raise ValueError("metric must be 'pearsonr' or 'rdmr'")

        # --- Apply Spearman–Brown correction ---
        if spearman_brown:
            r = (2 * r) / (1 + r) if np.isfinite(r) else np.nan

        split_rs.append(r)

    reliability = np.nanmean(split_rs)
    return reliability, np.array(split_rs)
