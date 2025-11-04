import warnings

from compcats.datasets.fmri import nsd, murty185, nsdsynthetic
from compcats.datasets.image import afhqv2, celebahq, ava, things
from compcats.localizer import localizer
from compcats.localizer import utils as localizer_utils
from compcats.utils import misc_utils

# Device
DEVICE = misc_utils.get_device()
DEFAULT_NUM_WORKERS = 1

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
    if device == 'cpu' and DEVICE.type != 'cpu':
        warnings.warn("Device is set to CPU but GPU is available. Consider using GPU for faster computation.")
    else:
        device = DEVICE
    num_workers = kwargs.get('num_workers', DEFAULT_NUM_WORKERS)

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
        save_dir = f'{models_dir}/{model.name}/nsd/activations'
        dataset_name = nsd.get_stimuli_dataset_name(subset=subset)
        # Check if dataset activations are already present
        dataset_activations_present, _ = model.check_dataset_activations_present(
            layers_list, dataset_name, save_dir
        )
        if dataset_activations_present:
            dataset_activations = model.load_dataset_activations(
                layers_list, dataset_name, save_dir
            )
            # Move activations to device
            for k, v in dataset_activations.items():
                try:
                    dataset_activations[k] = v.to(device)
                except Exception as e:
                    msg = str(e).lower()
                    # catch CUDA OOM, generic OOM, and MPS/resource_tracker semaphore issues
                    if ('cuda out of memory' in msg) or ('out of memory' in msg) or \
                       ('resource_tracker' in msg) or ('leaked semaphore' in msg) or ('mps' in msg):
                        print(f"Memory error when loading activations for layer {k}. Keeping activations on CPU and returning.")
                        device = 'cpu'
                        # Ensure all activations are on CPU
                        for kk, vv in dataset_activations.items():
                            try:
                                dataset_activations[kk] = vv.cpu()
                            except Exception:
                                # if .cpu() fails, leave as-is
                                dataset_activations[kk] = vv
                        return dataset_activations
                    else:
                        raise e
        else:
            nsd_dataset = nsd.get_stimuli_dataset(dataset_dir, model.transforms, subset=subset)
            dataset_activations = model.generate_dataset_activations(
                layers_list, nsd_dataset, save_dir, batch_size, device
            )
    elif dataset == 'nsdsynthetic':
        subset = kwargs.get('subset', 'all')
        save_dir = f'{models_dir}/{model.name}/nsdsynthetic/activations'
        dataset_name = nsdsynthetic.get_stimuli_dataset_name(
            nsdsynthetic.extract_subset_type(subset)
        )
        # Check if dataset activations are already present
        dataset_activations_present, _ = model.check_dataset_activations_present(
            layers_list, dataset_name, save_dir
        )
        if dataset_activations_present:
            dataset_activations = model.load_dataset_activations(
                layers_list, dataset_name, save_dir
            )
            # Move activations to device
            for k, v in dataset_activations.items():
                dataset_activations[k] = v.to(device)
        else:
            nsdsynthetic_dataset = nsdsynthetic.get_stimuli_dataset(
                dataset_dir, model.transforms, subset=subset
            )
            dataset_activations = model.generate_dataset_activations(
                layers_list, nsdsynthetic_dataset, save_dir, batch_size, device
            )
    elif dataset == 'murty185':
        save_dir = f'{models_dir}/{model.name}/murty185/activations'
        murty185_dataset = murty185.get_stimuli_dataset(dataset_dir, model.transforms)
        dataset_activations = model.generate_dataset_activations(
            layers_list, murty185_dataset, save_dir, batch_size, device, num_workers=num_workers
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
                layers_list, imageclass_dataset, save_dir, batch_size, device,
                num_workers=num_workers
            )
        for layer in layers_list:
            dataset_activations[layer] = {}
            for imgclass, imgclass_activation in imgclass_activations.items():
                dataset_activations[layer][imgclass] = imgclass_activation[layer]
    elif dataset == 'celebahq':
        celebahq_dataset = celebahq.get_val_dataset(
            dataset_dir, model.transforms
        )
        celebahq_dataset.name = 'celebahq'
        save_dir = f'{models_dir}/{model.name}/{celebahq_dataset.name}/activations'
        dataset_activations = model.generate_dataset_activations(
            layers_list, celebahq_dataset, save_dir, batch_size, device, num_workers=num_workers
        )
    elif dataset == 'afhqv2':
        afhqv2_dataset = afhqv2.get_val_dataset(
            dataset_dir, model.transforms
        )
        afhqv2_dataset.name = 'afhqv2'
        save_dir = f'{models_dir}/{model.name}/{afhqv2_dataset.name}/activations'
        dataset_activations = model.generate_dataset_activations(
            layers_list, afhqv2_dataset, save_dir, batch_size, device, num_workers=num_workers
        )
    elif dataset == 'ava':
        landscape_only = kwargs.get('landscape_only', True)
        vanishing_point_images = kwargs.get('vanishing_point_images', None)
        ava_dataset = ava.get_stimuli_dataset(
            dataset_dir, model.transforms,
            landscape_only=landscape_only,
            vanishing_point_images=vanishing_point_images
        )
        save_dir = f'{models_dir}/{model.name}/{ava_dataset.name}/activations'
        dataset_activations = model.generate_dataset_activations(
            layers_list, ava_dataset, save_dir, batch_size, device, num_workers=num_workers
        )
    elif dataset == 'things':
        save_dir = f'{models_dir}/{model.name}/things/activations'
        things_dataset = things.get_stimuli_dataset(
            dataset_dir, model.transforms
        )
        dataset_activations = model.generate_dataset_activations(
            layers_list, things_dataset, save_dir, batch_size, device, num_workers=num_workers
        )
    else:
        raise ValueError(f'{dataset} unknown dataset')

    # Get fROI activations if froi is provided
    if froi is not None and tthreshold is not None and froi_localizer is not None:
        froi_masks = localizer.get_froi_mask(
            froi_localizer_dir, froi_localizer, models_dir, \
            model, layers_list, froi, tthreshold, **kwargs
        )
        # If froi_masks is a dict with values as dicts, then only extract masks
        # and leave out extra info like t-values or voxel indices
        if isinstance(froi_masks, dict):
            if isinstance(list(froi_masks.values())[0], dict):
                for layer in layers_list:
                    froi_masks[layer] = froi_masks[layer]['mask']
        # Apply froi masks to dataset activations
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
