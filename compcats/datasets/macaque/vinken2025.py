"""
Code to load the Vinken2025 macaque dataset
Vinken, K., Sharma, S., & Livingstone, M. S. (2025). Mapping macaque to human cortex with natural scene responses. Proceedings of the National Academy of Sciences, 122(40), e2512619122.
Data Link: https://osf.io/vh2pd/files/237u5
"""
from typing import Union, Tuple, Any
import os
import numpy as np
import pandas as pd

from .utils import vinken2025_utils

HUMAN_FROIS = ['FFA', 'ATL']
MONKEY_FROIS = ['V1', 'CIT', 'AIT']
MONKEY_NAME_MAP = {
    1: 'Anthony'.upper(),
    2: 'OtherGuy'.upper(),
    3: 'Paul'.upper(),
    4: 'Red2'.upper(),
    5: 'Baby1'.upper(),
}
MONKEY_FROI_COLLECTED_MAP = {
    1: 'V1',
    2: 'CIT',
    3: 'CIT',
    4: 'AIT',
    5: 'AIT',
}
FROI_MONKEY_PRESENT_MAP = {
    'V1': [1],
    'CIT': [2, 3],
    'AIT': [4, 5],
}
HUMAN_MONKEY_FROI_MAP = {
    'V1': 'V1',
    'FFA': 'CIT',
    'ATL': 'AIT',
}

def get_froi_monkeys_present(froi: str) -> list:
    """
    Get monkeys present for a froi
    Args:
        froi(str): froi name
    Returns:
        list: List of monkeys present
    """
    if froi in MONKEY_FROIS:
        pass
    elif froi in HUMAN_FROIS:
        # Map to monkey froi
        froi = HUMAN_MONKEY_FROI_MAP[froi]
    else:   
        raise ValueError(f'FROI {froi} not recognized in Vinken2025 dataset')
    return FROI_MONKEY_PRESENT_MAP[froi]

def get_responses(
    root_dir: str,
    monkey: int,
    froi: str,
    subset: int = 515,
    rthreshold: Union[int, float] = 0.4,
    **kwargs: Any
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Get responses for Vinken2025 dataset
    Args:
        root_dir(str): Root directory
        monkey(int): Subject number
        froi(str): froi name
        rthreshold(float): Reliability threshold
        subset(str): Subset of dataset
    Returns:
        np.ndarray: Responses
    """
    # Kwargs
    reptition_mean = kwargs.get('reptition_mean', True)
    nsd_dir = kwargs.get('nsd_dir', None)
    # Make sure data is preprocessed
    preprocessed_dir = vinken2025_utils.check_data_preprocessed(root_dir)
    if preprocessed_dir is None:
        assert nsd_dir is not None, 'NSD directory must be provided to preprocess Vinken2025 dataset'
        preprocessed_dir = vinken2025_utils.preprocess_vinken2025_data(root_dir, nsd_dir)
    # Error handle
    froi = froi.upper()
    assert subset in [485, 515, 1000], f'Subset {subset} not recognized'
    if froi in HUMAN_FROIS:
        # Map to monkey froi
        froi = HUMAN_MONKEY_FROI_MAP[froi]
    elif froi not in MONKEY_FROIS:
        raise ValueError(f'FROI {froi} not recognized in Vinken2025 dataset')
    froi_monkeys_present = get_froi_monkeys_present(froi)
    assert monkey in froi_monkeys_present, (
        f'Monkey {monkey} does not have data for froi {froi}'
    )
    # Get monkey name
    monkey_name = MONKEY_NAME_MAP[monkey]
    # Load preprocessed data
    all_responses = np.load(
        os.path.join(
            preprocessed_dir,
            'responses.npy'
        ),
        allow_pickle=True
    ).item()
    responses = all_responses[(froi, monkey_name)]
    # Select units based on reliability threshold
    if rthreshold > 0:
        # Load reliabilities
        all_reliabilities = np.load(
            os.path.join(
                preprocessed_dir,
                'reliabilities.npy'
            ),
            allow_pickle=True
        ).item()
        reliabilities = all_reliabilities[(froi, monkey_name)]
        # Get indices of units above threshold
        valid_unit_indices = np.where(reliabilities >= rthreshold)[0]
        responses = responses[:, :, valid_unit_indices]
    # Stimuli info
    stim_info_df = pd.read_csv(
        os.path.join(
            preprocessed_dir,
            'stim_info.csv'
        )
    )
    # Get indices to subset responses
    subset_info = stim_info_df[
        (stim_info_df[f'shared{subset}'] == True) & \
        (stim_info_df['allmonkeys'] == True)
    ]
    subset_indices = subset_info.index.values
    # Subset responses
    responses = responses[subset_indices, :]
    # Average across repetitions if specified
    if reptition_mean:
        responses = np.nanmean(responses, axis=1)  # shape: (num_stimuli, num_units)
    return responses, subset_info
