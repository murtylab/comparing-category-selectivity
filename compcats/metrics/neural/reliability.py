"""
Reliability metrics
"""
import warnings
from typing import Tuple
import numpy as np
from scipy import stats

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
