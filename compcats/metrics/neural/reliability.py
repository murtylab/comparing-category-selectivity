"""
Reliability metrics
"""
import warnings
from typing import Tuple
import numpy as np
from scipy import stats
import torch

from compcats.utils import responsecomparison_utils as rcutils

def responsiveness(
    data: np.ndarray,
    baseline_window: Tuple[int, int],
    response_window: Tuple[int, int],
    p_threshold: float = 0.01,
):
    """
    Args:
        data: A 3D or 4D array containing the neural data (n_sites, n_trials, n_timepoints)
            or (n_sites, n_categories, n_trials, n_timepoints). Trials can have all NaNs to
            accomodate for variable number of trials
        baseline_window: A tuple specifying the start and end indices of the baseline window
        response_window: A tuple specifying the start and end indices of the response window
        p_threshold: The p-value threshold for determining significant responses
    """
    # Error handling
    assert data.ndim in [3, 4], "Data must be a 3D or 4D array"
    assert len(baseline_window) == 2, "Baseline window must be a tuple of two integers"
    assert baseline_window[0] < baseline_window[1], "Baseline window must be a valid range"
    assert len(response_window) == 2, "Response window must be a tuple of two integers"
    assert response_window[0] < response_window[1], "Response window must be a valid range"
    assert p_threshold > 0, "Threshold must be a positive value"
    # Fix data shape if needed
    if data.ndim == 3:
        warnings.warn("Responsiveness metric: no categories provided, assuming 1 category")
        # Add a dimension at 1
        data = np.expand_dims(data, axis=1)
    # Estimate mean of the windows
    baseline_rates = np.nanmean(data[:, :, :, baseline_window[0]:baseline_window[1]], axis=-1)
    response_rates = np.nanmean(data[:, :, :, response_window[0]:response_window[1]], axis=-1)
    # Conduct a statistical test for each site and category
    n_sites, n_categories = data.shape[:2]
    sites_responsiveness = np.full((n_sites,), False) # Bool array
    for sidx in range(n_sites):
        site_pvalues = np.full((n_categories,), np.Inf)
        for cidx in range(n_categories):
            sc_baseline = baseline_rates[sidx, cidx, :] # Shape: (n_trials,)
            sc_response = response_rates[sidx, cidx, :] # Shape: (n_trials,)

            # Remove NaN values (for variable trial counts)
            sc_baseline_clean = sc_baseline[~np.isnan(sc_baseline)]
            sc_response_clean = sc_response[~np.isnan(sc_response)]

            assert len(sc_baseline_clean) > 1, "No valid baseline data"
            assert len(sc_response_clean) > 1, "No valid response data"

            # Conduct an independent t-test
            _, p_value = stats.ttest_ind(sc_response_clean, sc_baseline_clean)
            site_pvalues[cidx] = p_value

        # Check if any category is significantly responsive
        if np.any(site_pvalues < p_threshold):
            sites_responsiveness[sidx] = True
    return sites_responsiveness

def persource_bootstrapped_splithalf_correlation(
    responses: np.ndarray,
    metric: str = "pearsonr",
    n_boot: int = 100,
    spearman_brown: bool = True,
    random_state: int | None = 42,
):
    """
    Estimate bootstrapped split-half reliability per source

    Parameters
    ----------
    responses : np.ndarray
        Array of shape (n_electrodes, n_stimuli, n_repeats)
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
        "responses must be 3D in the shape of n_electrodes, n_stimuli, n_repeats"
    # Set random state
    rng = np.random.default_rng(random_state)
    n_electrodes, n_stimuli, n_repeats = responses.shape
    if isinstance(responses, torch.Tensor):
        responses = responses.cpu().numpy()

    # Assert that all valid reps are before invalid reps (i.e., no NaNs in the middle)
    for eidx in range(n_electrodes):
        for sidx in range(n_stimuli):
            first_invalid_rep_idx = np.min(np.where(np.isnan(responses[eidx, sidx, :]))[0]) if np.any(np.isnan(responses[eidx, sidx, :])) else n_repeats
            last_valid_rep_idx = np.max(np.where(~np.isnan(responses[eidx, sidx, :]))[0]) if len(np.where(~np.isnan(responses[eidx, sidx, :]))[0]) > 0 else -1
            assert first_invalid_rep_idx == n_repeats or last_valid_rep_idx < first_invalid_rep_idx, "Invalid NaN placement in responses"
    # Reduce n_repeats if needed to the max number of valid repetitions
    max_n_reps = np.max([np.sum(~np.isnan(responses[electrode_idx, stim, :])) for electrode_idx in range(n_electrodes) for stim in range(n_stimuli)])
    if max_n_reps < n_repeats:
        responses = responses[:, :, :max_n_reps]
        print(f"Reducing number of repeats from {n_repeats} to {max_n_reps} for split-half reliability estimation")

    electrode_reliabilities = []
    for electrode_idx in range(n_electrodes):
        electrode_responses = responses[electrode_idx]
        split_rs = []
        for bidx in range(n_boot):
            # --- Create random splits per stimulus ---
            half1, half2 = [], []
            for stim in range(n_stimuli):
                valid_reps = np.where(~np.isnan(electrode_responses[stim, :]))[0]
                n_valid = len(valid_reps)
                if n_valid < 2:
                    continue
                idx = rng.permutation(valid_reps)
                split = n_valid // 2
                reps1, reps2 = idx[:split], idx[split:]
                if len(reps1) == 0 or len(reps2) == 0:
                    continue
                half1.append(np.nanmean(electrode_responses[stim, reps1], axis=0))
                half2.append(np.nanmean(electrode_responses[stim, reps2], axis=0))

            half1, half2 = np.array(half1), np.array(half2)
            if half1.size == 0 or half2.size == 0:
                continue

            # --- Compute population-level correlation ---
            if metric == "pearsonr":
                r = rcutils.pearsonr(
                    torch.Tensor(half1),
                    torch.Tensor(half2)
                )
            elif metric == "rdmr":
                half1, half2 = torch.Tensor(half1), torch.Tensor(half2)
                r = rcutils.rdmr(half1, half2)
            else:
                raise ValueError("metric must be 'pearsonr' or 'rdmr'")

            # --- Apply Spearman–Brown correction ---
            if spearman_brown:
                r = (2 * r) / (1 + r) if np.isfinite(r) else np.nan

            split_rs.append(r)
        electrode_reliabilities.append(split_rs)
    electrode_reliabilities = np.stack(electrode_reliabilities)
    mean_reliabilities = np.nanmean(electrode_reliabilities, axis=-1)
    return mean_reliabilities, electrode_reliabilities
