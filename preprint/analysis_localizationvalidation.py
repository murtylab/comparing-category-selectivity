import numpy as np
import scipy

from compcats.utils import model_utils
import compcats.localizer.utils as localizer_utils

CATEGORIES_CONSIDERED = [
    'faces', 'bodies', 'scenes', 'places', 'objects'
]

def model_layer_froi_localization_validation(
    model, layer, froi, tthreshold, froi_localizer, froi_localizer_dir, models_dir, \
    validation_localizer, validation_localizer_dir, batch_size=768, generate_layers=None
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
        batch_size(int): Batch size
        generate_layers(list): List of layers to generate activations to improve efficiency
    Returns:
        validation_localizer_category_normality_pvalues(list): List of p-values from normality tests
        tvalues(list): List of t-values from t-tests
        t_pvalues(list): List of p-values from t-tests
        uvalues(list): List of u-values from mannwhitneyu tests
        u_pvalues(list): List of p-values from mannwhitneyu tests
    """
    # Error handling
    if not isinstance(layer, str) and np.isnan(layer):
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    # Determine layers to generate
    layers_list = None
    if generate_layers:
        layers_list = generate_layers
    else:
        layers_list = [layer]
    for layer_ in layers_list:
        if not isinstance(layer_, str) and np.isnan(layer_):
            layers_list.remove(layer_)
    if len(layers_list) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    # Get froi activations for validation localizer
    validation_localizer_activations = model_utils.get_dataset_activations(
        models_dir, validation_localizer_dir, model, layers_list, dataset='localizer', \
        localizer_type=validation_localizer, \
        froi=froi, tthreshold=tthreshold, froi_localizer=froi_localizer,\
        froi_localizer_dir=froi_localizer_dir, batch_size=batch_size
    )
    # Get required layer activations
    validation_localizer_activations = validation_localizer_activations[layer]
    # Take mean of froi units
    # Also move to cpu and convert to numpy
    validation_localizer_category_normality_pvalues = []
    validation_localizer_activations_mean = {}
    for category, category_activations in validation_localizer_activations.items():
        unitaveraged_category_activations = category_activations.mean(axis=1).cpu().numpy()
        validation_localizer_activations_mean[category] = unitaveraged_category_activations
        # Test normality
        validation_localizer_category_normality_pvalues.append(
            scipy.stats.shapiro(unitaveraged_category_activations).pvalue
        )
    # Get pref, nonpref categories
    pref_categories, nonpref_categories = localizer_utils.get_localizer_pref_nonpref_categories(
        validation_localizer, froi
    )
    assert len(pref_categories) == 1, "Only one preferred category implemented"
    pref_categories = pref_categories[0]
    assert pref_categories in CATEGORIES_CONSIDERED, \
        "Preferred category not in considered categories"
    # Consider only non-pref categories that are in considered categories
    nonpref_categories = [
        category for category in nonpref_categories if category in CATEGORIES_CONSIDERED
    ]
    # Get pref, nonpref activations
    pref_activations = validation_localizer_activations_mean[pref_categories]
    nonpref_activations = [
        validation_localizer_activations_mean[category] for category in nonpref_categories
    ]
    # Compare all pairs of pref vs. non-pref activations and report the t-value, p-values
    # Also report the u-values and p-values from mannwhitneyu test
    tvalues, t_pvalues = [], []
    uvalues, u_pvalues = [], []
    for nonpref_activation in nonpref_activations:
        if np.isnan(nonpref_activation).all():
            tvalues.append(float('nan'))
            t_pvalues.append(float('nan'))
            uvalues.append(float('nan'))
            u_pvalues.append(float('nan'))
            continue
        tvalue, pvalue = scipy.stats.ttest_ind(
            pref_activations, nonpref_activation, equal_var=False, nan_policy='raise', \
            alternative='greater'
        )
        tvalues.append(tvalue)
        t_pvalues.append(pvalue)
        uvalue, pvalue = scipy.stats.mannwhitneyu(
            pref_activations, nonpref_activation, alternative='greater'
        )
        uvalues.append(uvalue)
        u_pvalues.append(pvalue)
    # nonpref_activations = np.concatenate(
    #     [validation_localizer_activations_mean[category] for category in nonpref_categories]
    # )
    # Compare pref vs. all non-pref activations
    # tvalue, pvalue = scipy.stats.ttest_ind(
    #     pref_activations, nonpref_activations, equal_var=False, nan_policy='raise', \
    #     alternative='greater'
    # )
    return validation_localizer_category_normality_pvalues, tvalues, t_pvalues, uvalues, u_pvalues
