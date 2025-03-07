"""Localize fROIs"""
import os
import json
import torch

from . import utils

def t_statistic(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes the t-statistic for the difference between two tensors.
    Assumes the first dimension represents samples.
    
    Args:
        tensor1 (torch.Tensor): A tensor of shape (n_samples1, ...).
        tensor2 (torch.Tensor): A tensor of shape (n_samples2, ...).
    
    Returns:
        torch.Tensor: The t-statistic tensor of shape (...).
    """
    n1 = tensor1.shape[0]
    n2 = tensor2.shape[0]
    mean_diff = tensor1.mean(dim=0) - tensor2.mean(dim=0)
    var1 = tensor1.var(dim=0, unbiased=True)
    var2 = tensor2.var(dim=0, unbiased=True)

    pooled_se = torch.sqrt(var1 / n1 + var2 / n2)
    t_stat = mean_diff / pooled_se

    # p-values not being calculated for computational efficiency
    calculate_pvalues = False
    if calculate_pvalues:
        raise NotImplementedError("P-value calculations have not been verified")
        # Degrees of freedom using Welch-Satterthwaite equation
        df = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
        # Calculate p-value (two-tailed)
        p_values = 2 * (1 - torch.distributions.StudentT(df).cdf(torch.abs(t_stat)))

    return t_stat

def compare_activations(tensor1, tensor2, tthreshold):
    """
    Function to calculate t-values for all elements using samples in dimension 0 of both tensors
    Args:
        tensor1(torch.Tensor): Tensor 1, shape: (n_samples1, ...)
        tensor2(torch.Tensor): Tensor 2, shape: (n_samples2, ...)
        tthreshold(float): t-threshold
    """
    assert tensor1.shape[1:] == tensor2.shape[1:], "Tensor dimensions do not match"
    t_values = t_statistic(tensor1, tensor2)
    mask = t_values >= tthreshold
    return mask, t_values

def localize(model, layer, pref_datasets, nonpref_datasets, froi, dataset_name, \
             activations_dir, froi_dir, tthreshold, return_tvals=False):
    """
    Localize units in a layer that respond to 
    a preferred category more than non-preferred categories
    Args:
        model():
        layer_num(int): Layer number
        pref_datasets(list): List of preferred datasets
        nonpref_datasets(list): List of non-preferred datasets
        froi(str): fROI name
        dataset_name(str): Localizer dataset name
        activations_dir(str): Directory to store activations
        froi_dir(str): Directory to store fROI masks
        tthreshold(float): t-threshold
        return_tvals(bool): Return t-values
    """
    # Generate all activations
    # preferred categories
    pref_activations = {}
    for pref_dataset in pref_datasets:
        pref_img_class = pref_dataset.name
        pref_activation_ = model.generate_dataset_activations(
            layer, pref_dataset, activations_dir
        )
        pref_activations[pref_img_class] = pref_activation_
    # non-preferred categories
    nonpref_activations = {}
    for nonpref_dataset in nonpref_datasets:
        nonpref_img_class = nonpref_dataset.name
        nonpref_activation_ = model.generate_dataset_activations(
            layer, nonpref_dataset, activations_dir
        )
        nonpref_activations[nonpref_img_class] = nonpref_activation_
    # Compare activations
    froi_mask = None
    all_contrast_tvals = {}
    for pref_dataset in pref_datasets:
        for nonpref_dataset in nonpref_datasets:
            pref_img_class = pref_dataset.name
            nonpref_img_class = nonpref_dataset.name
            key = (pref_img_class, nonpref_img_class)
            # Get contrast mask and t-values
            contrast_mask, contrast_tvals = compare_activations(
                pref_activations[pref_img_class], \
                nonpref_activations[nonpref_img_class], \
                tthreshold
            )
            if froi_mask is None:
                froi_mask = contrast_mask
            else:
                froi_mask = froi_mask & contrast_mask
            all_contrast_tvals[key] = contrast_tvals
    # Store fROI mask
    froi_file_name = f'{model.name}_{layer}_{dataset_name}_{froi}_{tthreshold:.2f}.pt'
    torch.save(froi_mask.cpu(), f'{froi_dir}/{froi_file_name}')
    print(f'Saving fROI {froi_dir}/{froi_file_name}')
    # Store results
    mean_tvals = []
    for key, tvals in all_contrast_tvals.items():
        froi_tvals = tvals[froi_mask]
        mean_tvals.append([key, froi_tvals.mean().item()])
    json_file = f'results_{model.name}_{layer}_{froi}_{dataset_name}_{froi}_{tthreshold:.2f}.json'
    json_file_path = f'{froi_dir}/{json_file}'
    results = {
        'model_name': model.name,
        'layer': layer,
        'preferred_categories': [pds.name for pds in pref_datasets],
        'nonpreferred_categories': [npds.name for npds in nonpref_datasets],
        'froi': froi,
        'localizer': dataset_name,
        'tthreshold': tthreshold,
        'total_units_len': froi_mask.numel(),
        'selected_units_len': int(froi_mask.sum().item()),
        'mean_t_values': mean_tvals,
        'unit_indices': froi_mask.flatten().nonzero(as_tuple=False).flatten().tolist()
    }
    # Save JSON
    with open(json_file_path, "w", encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4)
    if return_tvals:
        return froi_mask, all_contrast_tvals
    return froi_mask

def get_froi_mask(
        dataset_dir, dataset_name, models_dir, model, layers, froi, tthreshold, return_tvals=False
):
    """
    Get indices for neurons corresponding to froi, localized
    Args:
        localizer_type(str): grillspector, kanwisher or konkle
    """
    # Error handling
    froi = froi.upper()
    dataset_name = dataset_name.lower()
    if isinstance(layers, str):
        layers_list = [layers]
    elif isinstance(layers, list):
        layers_list = layers
    else:
        raise ValueError("layers should be a string or a list of strings")

    # Results
    results = {}

    # Dirs
    root_dir = f'{models_dir}/{model.name}/{dataset_name}'
    activations_dir = f'{root_dir}/activations'
    os.makedirs(activations_dir, exist_ok=True)

    # Get localizer dataset module
    dataset_module = utils.get_localizerdataset_module(dataset_name)
    # Get preferred & non-preferred datasets
    pref_datasets, nonpref_datasets = utils.get_pref_nonpref_datasets(
        dataset_dir, dataset_module, froi, model.transforms
    )

    # Generate all layer activations together in one foward pass
    for pref_dataset in pref_datasets:
        model.generate_dataset_activations(layers_list, pref_dataset, activations_dir)
    for nonpref_dataset in nonpref_datasets:
        model.generate_dataset_activations(layers_list, nonpref_dataset, activations_dir)

    for layer in layers_list:
        # Mask file name
        mask_file = f'{model.name}_{layer}_{dataset_name}_{froi}_{tthreshold:.2f}.pt'
        froi_dir = f'{root_dir}/{froi}/{layer}'
        os.makedirs(froi_dir, exist_ok=True)
        # Check if mask file is present
        if mask_file in os.listdir(froi_dir):
            mask = torch.load(f'{froi_dir}/{mask_file}')
            if return_tvals:
                raise Warning("fROI mask already present, not returning t-values")
            return mask

        # Generate fROI mask if not present
        # Localize the units in the layer
        result = localize(model, layer, pref_datasets, nonpref_datasets, froi, \
                                        dataset_name, activations_dir, froi_dir, tthreshold, \
                                        return_tvals=return_tvals)
        # result is froi_mask if return_tvals=False
        # result is froi_mask, all_contrast_tvals if return_tvals=True
        if return_tvals:
            mask, all_contrast_tvals = result
            results[layer] = {
                'mask': mask,
                'all_contrast_tvals': all_contrast_tvals
            }
        else:
            results[layer] = result

    # Error handling
    if isinstance(layers, str):
        results = results[layers]

    return results
