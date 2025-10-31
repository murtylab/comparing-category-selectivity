from compcats.datasets.localization import grillspector_localizer, \
    kanwisher_localizer, konkle_localizer, vinken2023_localizer, kanwishergrayscale_localizer, \
    prince2024_localizer
# Constants
DEFAULT_SH_METHOD = 3  # method for reliability calculation

CONTRAST_LOCALIZERS = [
    'grillspector',
    'kanwisher',
    'konkle',
    'kanwishergrayscale',
    'vinken2023',
    'prince2024'
]

def get_localizerdataset_module(dataset_name):
    """
    Get localizer dataset based on type
    """
    if dataset_name == 'kanwisher':
        dataset_module = kanwisher_localizer
    elif dataset_name == 'kanwishergrayscale':
        dataset_module = kanwishergrayscale_localizer
    elif dataset_name == 'grillspector':
        dataset_module = grillspector_localizer
    elif dataset_name == 'konkle':
        dataset_module = konkle_localizer
    elif dataset_name == 'vinken2023':
        dataset_module = vinken2023_localizer
    elif dataset_name == 'prince2024':
        dataset_module = prince2024_localizer
    else:
        raise ValueError(f'{dataset_name} unknown localizer type')
    return dataset_module

def get_pref_nonpref_datasets(dataset_dir, dataset_module, froi, transforms):
    """
    Function to get preferred & non-preferred datasets
    Args:
        dataset_dir(str): Directory where dataset is stored
        dataset_module(compcats module): Localizer dataset Module
    """
    # Get preferred & non-preferred stimuli names
    dataset_obj = dataset_module.get_localizer_dataset(dataset_dir, transforms, image_class='all')
    pref_categories, nonpref_categories = dataset_obj.get_pref_nonpref_categories(froi)
    # Get preferred & non-preferred datasets
    pref_datasets = [
        dataset_module.get_localizer_dataset(
            dataset_dir, transforms, image_class=pref_category
        ) for pref_category in pref_categories
    ]
    nonpref_datasets = [
        dataset_module.get_localizer_dataset(
            dataset_dir, transforms, image_class=nonpref_category
        ) for nonpref_category in nonpref_categories
    ]
    return pref_datasets, nonpref_datasets

def get_all_imageclass_datasets(dataset_dir, dataset_module, transforms):
    """
    Function to get all image class datasets
    Args:
        dataset_dir(str): Directory where dataset is stored
        dataset_module(compcats module): Localizer dataset Module
        transforms(torchvision.transforms): Image transformations
    """
    # Get all stimuli names
    dataset_obj = dataset_module.get_localizer_dataset(dataset_dir, transforms, image_class='all')
    imageclasses = dataset_obj.imageclasses
    imageclass_datasets = {}
    for image_class in imageclasses:
        imageclass_datasets[image_class] = dataset_module.get_localizer_dataset(
            dataset_dir, transforms, image_class=image_class
        )
    return imageclass_datasets

def get_localizer_pref_nonpref_categories(dataset_name, froi):
    """
    Get preferred & non-preferred image classes
    """
    localizer_module = get_localizerdataset_module(dataset_name)
    localizer_obj = localizer_module.get_localizer_dataset('', None)
    pref_categories, nonpref_categories = localizer_obj.get_pref_nonpref_categories(froi)
    return pref_categories, nonpref_categories
