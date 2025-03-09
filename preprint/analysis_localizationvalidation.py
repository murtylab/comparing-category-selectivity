import numpy as np
import scipy

from compcats.utils import model_utils
import compcats.localizer.utils as localizer_utils

def model_layer_froi_localization_validation(
    model, layer, froi, tthreshold, froi_localizer, froi_localizer_dir, models_dir, \
    validation_localizer, validation_localizer_dir
):
    """
    Run localization validation for a model layer and froi
    Args:
        model(str): Model name
        layer(str): Layer name
        froi(str): Froi name
        tthreshold(float): T-threshold for froi mask
        froi_localizer(str): Froi localizer name
        froi_localizer_dir(str): Froi localizer directory
        models_dir(str): Models directory
        validation_localizer(str): Validation localizer name
        validation_localizer_dir(str): Validation localizer directory
    Returns:
        tvalue(float): T-value
        pvalue(float): P-value
    """
    # Get froi activations for validation localizer
    validation_localizer_activations = model_utils.get_dataset_activations(
        models_dir, validation_localizer_dir, model, layer, dataset='localizer', \
        localizer_type=validation_localizer, \
        froi=froi, tthreshold=tthreshold, froi_localizer=froi_localizer,\
        froi_localizer_dir=froi_localizer_dir
    )
    # Take mean of froi units
    # Also move to cpu and convert to numpy
    validation_localizer_activations_mean = {}
    for category, category_activations in validation_localizer_activations.items():
        validation_localizer_activations_mean[category] = \
            category_activations.mean(axis=-1).cpu().numpy()
    # Get pref, nonpref categories
    pref_categories, nonpref_categories = localizer_utils.get_localizer_pref_nonpref_categories(
        validation_localizer, froi
    )
    assert len(pref_categories) == 1, "Only one preferred category implemented"
    pref_categories = pref_categories[0]
    # Get pref, nonpref activations
    pref_activations = validation_localizer_activations_mean[pref_categories]
    nonpref_activations = np.concatenate(
        [validation_localizer_activations_mean[category] for category in nonpref_categories]
    )
    # Compare pref vs. all non-pref activations
    tvalue, pvalue = scipy.stats.ttest_ind(
        pref_activations, nonpref_activations, equal_var=False, nan_policy='raise', \
        alternative='greater'
    )
    return tvalue, pvalue
