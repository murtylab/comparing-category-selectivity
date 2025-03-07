from compcats.datasets.stimuli import grillspector_localizer, \
    kanwisher_localizer, konkle_localizer, vinken23_localizer

def get_localizerdataset_module(dataset_name):
    """
    Get localizer dataset based on type
    """
    if dataset_name == 'kanwisher':
        dataset_module = kanwisher_localizer
    elif dataset_name == 'grillspector':
        dataset_module = grillspector_localizer
    elif dataset_name == 'konkle':
        dataset_module = konkle_localizer
    elif dataset_name == 'vinken23':
        dataset_module = vinken23_localizer
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
    pref_category, nonpref_categories = dataset_obj.get_pref_nonpref_categories(froi)
    # Get preferred & non-preferred datasets
    pref_dataset = dataset_module.get_localizer_dataset(dataset_dir, transforms, image_class=pref_category)
    nonpref_datasets = [
        dataset_module.get_localizer_dataset(dataset_dir, transforms, image_class=nonpref_category) \
        for nonpref_category in nonpref_categories
    ]
    return pref_dataset, nonpref_datasets
