import pandas as pd
import torch

from compcats.localizer import localizer
from compcats.datasets.fmri import nsd
from compcats.utils import model_utils, seed_utils

import main_analysis as ma
import paths

RESULTS_DIR = 'results'
MODELS_DIR = 'models'

GRILLSPECTOR_LOCALIZER_DIR = paths.GRILLSPECTOR_LOCALIZER_DIR
KANWISHER_LOCALIZER_DIR = paths.KANWISHER_LOCALIZER_DIR

iou_df_path = f'{RESULTS_DIR}/localizer_ious.csv'

seed_utils.seed_everything(42)
ious_df = []
# Variables
tthreshold = 7
frois = ['FFA', 'PPA', 'EBA']
n_bootstrap = 50
localizers_name_dir = {
    'grillspector': GRILLSPECTOR_LOCALIZER_DIR,
    'kanwisher': KANWISHER_LOCALIZER_DIR
}
all_models = ma.get_all_models()


# Get IoU between localizers for best layers of each model in each fROI
for fidx, froi in enumerate(frois):
    best_layers = ma.get_froi_models_layers(froi, froi)
    set_num_units = [] # (only_gs, only_kw, both, total, iou)
    for model_name, model in all_models.items():
        layer = best_layers[model_name]
        gs_loc_froi_mask = localizer.get_froi_mask(
            GRILLSPECTOR_LOCALIZER_DIR, 'grillspector', MODELS_DIR,
            model, layer, froi, tthreshold
        )
        kw_loc_froi_mask = localizer.get_froi_mask(
            KANWISHER_LOCALIZER_DIR, 'kanwisher', MODELS_DIR,
            model, layer, froi, tthreshold
        )
        a = gs_loc_froi_mask
        b = kw_loc_froi_mask
        total = a.numel()
        only_a = torch.logical_and(a, ~b).sum().item()
        only_b = torch.logical_and(~a, b).sum().item()
        both   = torch.logical_and(a, b).sum().item()
        neither = torch.logical_and(~a, ~b).sum().item()
        intersection = torch.logical_and(a, b).sum().float()
        union = torch.logical_or(a, b).sum().float()
        iou = (intersection / union).item()
        ious_df.append({
            'froi': froi,
            'model_localizer': 'union',
            'model_name': model_name,
            'tthreshold': tthreshold,
            'layer': layer,
            'bootstrap_iter': 0,
            'iou': iou
        })


# Loop over fROIs
for froi in frois:
    # Loop over localizers
    for model_localizer, model_localizer_dir in localizers_name_dir.items():
        # Loop over models
        for model_name, model in all_models.items():
            # Get model layer
            layer = ma.get_froi_models_layers(froi, froi)[model_name]
            # Get n_samples x n_units matrix for all categories
            # Get the activations for this category
            loc_categories_activations = model_utils.get_dataset_activations(
                MODELS_DIR, model_localizer_dir, model, layer, 
                'localizer', froi, tthreshold, localizer_type=model_localizer
            )
            categories = list(loc_categories_activations.keys())
            for bootstrap_iter in range(n_bootstrap):
                # Loop over categories
                # dict_keys(['bodies', 'faces', 'places', 'objects', 'characters', 'scrambledobjects'])
                category_activations_splits = {1: {}, 2: {}}
                for category, category_activations in loc_categories_activations.items():
                    category_activations = loc_categories_activations[category]
                    # Divide them into two (almost) equal subsets
                    n_samples_split = category_activations.shape[0] // 2
                    # Choose n_samples_split random samples for split 1
                    # The rest go to split 2
                    # Get random indices for split 1
                    random_indices = torch.randperm(category_activations.shape[0])[:n_samples_split]
                    remaining_indices = torch.arange(category_activations.shape[0])
                    remaining_indices = remaining_indices[~torch.isin(remaining_indices, random_indices)]
                    assert len(set(random_indices.tolist()).intersection(set(remaining_indices.tolist()))) == 0, \
                        "Random indices and remaining indices overlap!"
                    category_activations_splits[1][category] = category_activations[random_indices]
                    category_activations_splits[2][category] = category_activations[remaining_indices]
                # get preffered and non-preffered categories
                pref_category, nonpref_categories = localizer.utils.get_localizer_pref_nonpref_categories(model_localizer, froi)
                pref_category = pref_category[0]  # get the string from the list
                froi_mask_subsets = {1: None, 2: None}
                for nonpref_category in nonpref_categories:
                    for sidx, split_activations in category_activations_splits.items():
                        contrast_mask, contrast_tvals = localizer.compare_activations(
                            category_activations_splits[sidx][pref_category], \
                            category_activations_splits[sidx][nonpref_category], \
                            tthreshold
                        )
                        if froi_mask_subsets[sidx] is None:
                            froi_mask_subsets[sidx] = contrast_mask
                        else:
                            froi_mask_subsets[sidx] = froi_mask_subsets[sidx] & contrast_mask
                # Now froi_mask_subsets[1] and froi_mask_subsets[2] contain the fROI masks for the two splits
                # estimate the Intersection over Union (IoU) between the two splits
                intersection = torch.logical_and(froi_mask_subsets[1], froi_mask_subsets[2]).sum().float()
                union = torch.logical_or(froi_mask_subsets[1], froi_mask_subsets[2]).sum().float()
                iou = (intersection / union).item()
                ious_df.append({
                    'froi': froi,
                    'model_localizer': model_localizer,
                    'tthreshold': tthreshold,
                    'model_name': model_name,
                    'layer': layer,
                    'bootstrap_iter': bootstrap_iter,
                    'iou': iou
                })
        print(f'{froi}, {model_localizer} done.')
ious_df = pd.DataFrame(ious_df)
# Save the dataframe
ious_df.to_csv(iou_df_path, index=False)
# Display the dataframe
# display(ious_df)