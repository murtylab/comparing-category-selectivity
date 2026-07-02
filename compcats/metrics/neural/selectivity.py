"""
Functions for selectivity-related metrics
"""
from typing import Tuple, Union
import numpy as np

def dprime(
    site_data_baseline_sub: np.ndarray,
    stimuli_categories: np.ndarray,
    preferred_category: str,
    nonpreferred_categories: Union[list, None] = None,
    perm_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute d-prime (category selectivity index) for each site comparing
    a preferred category against all other categories, with permutation test p-values.
    
    Args:
        site_data_baseline_sub: Shape (n_sites, n_images, n_max_repetitions)
            Baseline-subtracted firing rates for each trial.
            NaN values are ignored (for variable number of repetitions).
        
        stimuli_categories: Shape (n_images,)
            Category label for each image (e.g., array of strings like 'body', 'face', etc.)
            Must have same length as axis=1 of site_data_baseline_sub.
        
        preferred_category: str
            The category to compare against all others (e.g., 'body').
        
        nonpreferred_categories: list or None
            Optional list of specific non-preferred categories to compare against.
            If None, all categories except the preferred one are used.
        
        perm_iterations: int
            Number of permutations for significance testing (default: 1000).
    
    Returns:
        d_prime: Shape (n_sites,)
            d-prime value for each site (preferred category vs others).
            Formula: d' = (μ_pref - μ_other) / sqrt((σ²_pref + σ²_other) / 2)
        
        p_values: Shape (n_sites,)
            Permutation test p-value for each site (H0: category labels are random).
    
    Example:
        >>> site_data = np.random.randn(50, 100, 6)  # 50 sites, 100 images, 6 reps each
        >>> categories = np.array(['body']*50 + ['face']*30 + ['object']*20)
        >>> d_prime, p_vals = compute_dprime_per_site(site_data, categories, 'body')
        >>> print(d_prime.shape, p_vals.shape)
        (50,) (50,)
        >>> selective = p_vals < 0.05
        >>> print(f"Selective sites: {np.sum(selective)}/{len(selective)}")
    """
    n_sites, n_images, n_max_reps = site_data_baseline_sub.shape

    # Validate inputs
    assert len(stimuli_categories) == n_images, \
        f"stimuli_categories length {len(stimuli_categories)} != n_images {n_images}"
    assert preferred_category in stimuli_categories, \
        f"preferred_category '{preferred_category}' not found in stimuli_categories"

    # Get indices for preferred vs other categories
    pref_mask = stimuli_categories == preferred_category
    if nonpreferred_categories is not None:
        other_mask = np.isin(stimuli_categories, nonpreferred_categories)
    else:
        other_mask = ~pref_mask
    pref_indices = np.where(pref_mask)[0]
    other_indices = np.where(other_mask)[0]

    d_prime = np.zeros(n_sites)
    p_values = np.ones(n_sites)  # Default to p=1 (not significant)

    for site_idx in range(n_sites):
        # Collect all trials for preferred category
        pref_trials = []
        for img_idx in pref_indices:
            trials = site_data_baseline_sub[site_idx, img_idx, :]
            valid_trials = trials[~np.isnan(trials)]
            pref_trials.extend(valid_trials)

        # Collect all trials for other categories
        other_trials = []
        for img_idx in other_indices:
            trials = site_data_baseline_sub[site_idx, img_idx, :]
            valid_trials = trials[~np.isnan(trials)]
            other_trials.extend(valid_trials)

        pref_trials = np.array(pref_trials)
        other_trials = np.array(other_trials)

        # Compute d-prime only if both distributions have samples
        if len(pref_trials) > 0 and len(other_trials) > 0:
            mu_pref = np.mean(pref_trials)
            mu_other = np.mean(other_trials)

            sigma2_pref = np.var(pref_trials, ddof=1)  # Sample variance
            sigma2_other = np.var(other_trials, ddof=1)

            # d-prime formula
            pooled_var = (sigma2_pref + sigma2_other) / 2
            d_prime[site_idx] = (mu_pref - mu_other) / np.sqrt(pooled_var)

            # Permutation test
            # Combine all trials and their labels
            all_trials = np.concatenate([pref_trials, other_trials])
            true_labels = np.concatenate([
                np.ones(len(pref_trials)),  # 1 = preferred category
                np.zeros(len(other_trials))  # 0 = other categories
            ])

            # Count how many permutations yield d-prime >= observed
            perm_count = 0
            for _ in range(perm_iterations):
                # Shuffle labels
                shuffled_labels = np.random.permutation(true_labels)
                perm_pref_trials = all_trials[shuffled_labels == 1]
                perm_other_trials = all_trials[shuffled_labels == 0]

                # Compute permuted d-prime
                mu_pref_perm = np.mean(perm_pref_trials)
                mu_other_perm = np.mean(perm_other_trials)
                sigma2_pref_perm = np.var(perm_pref_trials, ddof=1)
                sigma2_other_perm = np.var(perm_other_trials, ddof=1)
                pooled_var_perm = (sigma2_pref_perm + sigma2_other_perm) / 2
                d_prime_perm = (mu_pref_perm - mu_other_perm) / np.sqrt(pooled_var_perm)

                # Count if permuted d-prime is >= observed (absolute value for two-tailed)
                if np.abs(d_prime_perm) >= np.abs(d_prime[site_idx]):
                    perm_count += 1

            # Compute p-value (two-tailed)
            p_values[site_idx] = (perm_count + 1) / (perm_iterations + 1)
        else:
            d_prime[site_idx] = np.nan
            p_values[site_idx] = np.nan

    return d_prime, p_values
