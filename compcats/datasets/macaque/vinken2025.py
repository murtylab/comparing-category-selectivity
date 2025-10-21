"""
Code to load the Vinken2025 macaque dataset
Vinken, K., Sharma, S., & Livingstone, M. S. (2025). Mapping macaque to human cortex with natural scene responses. Proceedings of the National Academy of Sciences, 122(40), e2512619122.
Data Link: https://osf.io/vh2pd/files/237u5
"""

import os
import numpy as np

from ..fmri import nsd
from ..fmri.nsd import nsd_ds_utils
from .utils import vinken2025_utils

MONKEY_NAME_MAP = {
    '1': 'Anthony',
    '2': 'OtherGuy',
    '3': 'Paul',
    '4': 'Red2',
    '5': 'Baby1',
}
MONKEY_FROI_COLLECTED_MAP = {
    '1': 'V1',
    '2': 'CIT',
    '3': 'CIT',
    '4': 'AIT',
    '5': 'AIT',
}
FROI_MONKEY_PRESENT_MAP = {
    'V1': ['1'],
    'CIT': ['2', '3'],
    'AIT': ['4', '5'],
}
MONKEY_HUMAN_FROI_MAP = {
    'V1': 'V1',
    'CIT': 'FFA',
    'AIT': 'ATL',
}

# Useful functions from nsd
def get_subset_stimuli(nsd_dir, subset=1000):
    """
    Get subset of stimuli
    Args:
        subset(str): Subset of NSD dataset
    """
    assert subset in [515, 485, 1000], "Subset must be one of [515, 485, 1000]"
    stimuli, nsdids = nsd.get_subset_stimuli(nsd_dir, subset)
    return stimuli, nsdids
