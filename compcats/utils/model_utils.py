import torch

from compcats.datasets.fmri import nsd, murty185
from compcats.localizer import localizer
from compcats.localizer import utils as localizer_utils


def get_dataset_activations(
    models_dir, dataset_dir, \
    model, layers, dataset='nsd', \
    froi = None, tthreshold = None, \
    froi_localizer = None, froi_localizer_dir = None, \
    **kwargs
):
    """
    Function to generate model activations of a particular layer for different datasets
    Args:
        data_dir(str): Directory where dataset is stored
        model(): Model
        layer_name(str): Layer name
        dataset(str): nsd or murty185
    kwargs:
        nsd_subset_len(int): Number of images to consider from NSD dataset
    """
    # Store activations
    dataset_activations = {}

    # kwargs
    batch_size = kwargs.get('batch_size', 768)
    device = kwargs.get('device', 'cuda')
    if not torch.cuda.is_available():
        device = 'cpu'


    # Error handling
    dataset = dataset.lower()
    if isinstance(layers, str):
        layers_list = [layers]
    elif isinstance(layers, list):
        layers_list = layers
    else:
        raise ValueError('layers should be a string or a list')

    # Get activations for different datasets
    if dataset == 'nsd':
        subset = kwargs.get('subset', None)
        save_dir = f'{models_dir}/{model.name}/nsd'
        dataset_name = nsd.get_stumli_dataset_name(subset=subset)
        # Check if dataset activations are already present
        dataset_activations_present, _ = model.check_dataset_activations_present(
            layers_list, dataset_name, save_dir
        )
        if dataset_activations_present:
            dataset_activations = model.load_dataset_activations(
                layers_list, dataset_name, save_dir
            )
            # Move activations to device
            dataset_activations = {k: v.to(device) for k, v in dataset_activations.items()}
        else:
            nsd_dataset = nsd.get_stimuli_dataset(dataset_dir, model.transforms, subset=subset)
            dataset_activations = model.generate_dataset_activations(
                layers_list, nsd_dataset, save_dir, batch_size, device
            )
    elif dataset == 'murty185':
        save_dir = f'{models_dir}/{model.name}/murty185'
        murty185_dataset = murty185.get_stimuli_dataset(dataset_dir, model.transforms)
        dataset_activations = model.generate_dataset_activations(
            layers_list, murty185_dataset, save_dir, batch_size, device
        )
    elif dataset == 'localizer':
        imgclass_activations = {} # Store activations for different classes
        localizer_type = kwargs.get('localizer_type', None) # get localizer type
        assert localizer_type is not None, "localizer_type not provided" # Error handling
        # Get localizer module
        localizer_module = localizer_utils.get_localizerdataset_module(localizer_type)
        # Get localizer image class datasets
        imageclass_datasets = localizer_utils.get_all_imageclass_datasets(
            dataset_dir, localizer_module, model.transforms
        )
        # Save dir
        save_dir = f'{models_dir}/{model.name}/{localizer_type}/activations'
        # Generate activations for all image classes
        for imageclass, imageclass_dataset in imageclass_datasets.items():
            imgclass_activations[imageclass] = model.generate_dataset_activations(
                layers_list, imageclass_dataset, save_dir, batch_size, device
            )
        for layer in layers_list:
            dataset_activations[layer] = {}
            for imgclass, imgclass_activation in imgclass_activations.items():
                dataset_activations[layer][imgclass] = imgclass_activation[layer]
    else:
        raise ValueError(f'{dataset} unknown dataset')

    # Get fROI activations if froi is provided
    if froi is not None and tthreshold is not None and froi_localizer is not None:
        froi_masks = localizer.get_froi_mask(
            froi_localizer_dir, froi_localizer, models_dir, \
            model, layers_list, froi, tthreshold, batch_size=batch_size
        )
        for layer in layers_list:
            froi_mask_layer = froi_masks[layer].to(device)
            if isinstance(dataset_activations[layer], dict):
                for imgclass in dataset_activations[layer].keys():
                    dataset_activations[layer][imgclass] = \
                        dataset_activations[layer][imgclass][:, froi_mask_layer]
            else:
                dataset_activations[layer] = dataset_activations[layer][:, froi_mask_layer]

    # Error handling
    if isinstance(layers, str):
        dataset_activations = dataset_activations[layers]
    return dataset_activations
