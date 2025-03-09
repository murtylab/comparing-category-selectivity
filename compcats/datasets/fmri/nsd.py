"""Code to load the NSD dataset"""
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from compcats.datasets.fmri.utils import nsd_ds_utils

# Useful functions from utils
def get_subset_stimuli(root_dir, subset=None):
    """
    Get subset of stimuli
    Args:
        root_dir(str): Root directory
        subset(str): Subset of NSD dataset
    """
    stimuli, nsdids = nsd_ds_utils.get_subset_stimuli(root_dir, subset)
    return stimuli, nsdids

def get_subset_data(root_dir, subject, subset, **kwargs):
    """
    Get subset of data
    Args:
        root_dir(str): Root directory
        subject(int): Subject number
        subset(str): Subset of NSD dataset
    """
    return nsd_ds_utils.get_subset_data(root_dir, subject, subset, **kwargs)

def get_froi_mask(root_dir, froi, subject, tthreshold):
    """
    Get froi mask
    Args:
        root_dir(str): Root directory
        froi(str): Path to froi mask
        subject(int): Subject number
        tthreshold(float): Threshold for t-values
    """
    return nsd_ds_utils.get_froi_mask(root_dir, froi, subject, tthreshold)

# Stimuli
class NSDStimuli(Dataset):
    """
    Dataset for loading the stimuli in the NSD dataset
    """
    def __init__(self, root_dir, transform, subset=None):
        self.root_dir = root_dir # NSD base directory
        # Set default transform
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform # Transformations to be applied
        # Subset
        assert subset in [None, 515, 485, 1000], 'NSD subset length not implemented yet, only 485, 515 or 1000 available'
        self.subset = subset
        self.all_stimuli, self.nsdids = get_subset_stimuli(root_dir, self.subset)
        # Name of the dataset to store activations
        dataset_name = 'nsdcoco'
        self.name = f'{dataset_name}{self.subset}' if self.subset else dataset_name

    def __len__(self):
        return len(self.all_stimuli)

    def __getitem__(self, idx):
        image_array = self.all_stimuli[idx, :, :, :] # Numpy array (H, W, C)
        image = Image.fromarray(image_array) # PIL Image
        image = self.transform(image)
        return image

def get_stimuli_dataset(root_dir, transform=None, subset=None):
    """
    Get NSD stimuli dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = NSDStimuli(root_dir, transform=transform, subset=subset)
    return dataset

def get_betas(root_dir, subject, froi=None, tthreshold=0, subset=None, voxels=False, **kwargs):
    """
    Get betas for a subject
    Args:
        root_dir(str): Root directory
        subject(int): Subject number
        froi(str): Path to froi mask
        tthreshold(float): Threshold for t-values
        subset(str): Subset of NSD dataset
        voxels(bool): Return voxels or mean of voxels
    """
    betas, ptid_nsdid_map = get_subset_data(root_dir, subject, subset, **kwargs)
    # Check for repetition mean
    repetition_mean = kwargs.get('repetition_mean', True)
    if froi:
        froi_mask, froi_tvals = get_froi_mask(root_dir, froi, subject, tthreshold)
        # If no voxels in the area
        if np.sum(froi_mask) == 0:
            return None, None
        # Transpose to match beta shape
        froi_mask = froi_mask.transpose(2, 1, 0)
        # Convert to torch.Tensor
        froi_mask = torch.BoolTensor(froi_mask)
        # Apply mask
        if repetition_mean:
            betas = betas[:, froi_mask]
        else:
            betas = betas[:, :, froi_mask]
    # Mean across voxels if needed
    if not voxels:
        if repetition_mean:
            betas = betas.reshape(betas.shape[0], -1) # Flatten the voxel dimensions
        else:
            betas = betas.reshape(betas.shape[0], betas.shape[1], -1) # Flatten the voxel dimensions
        betas = betas.mean(axis=-1)
    return betas, ptid_nsdid_map
