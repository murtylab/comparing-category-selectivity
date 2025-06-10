"""Code to load the NSD dataset"""
import os
import h5py
import ast
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import nsd_ds_utils

def get_stimuli_info(root_dir, nowords, shared_only):
    """
    Function to get the stimuli info, and generate if needed
    Trials and stimuli info were same across subjects
    Args:
        root_dir(str): Root directory
        words(bool): Whether to use words or not
    """
    processed_stimuli_dir = f'{root_dir}/nsdsynthetic_processed'
    os.makedirs(processed_stimuli_dir, exist_ok=True)
    processed_stim_info_file_name = ''
    if shared_only:
        processed_stim_info_file_name += 'shared'
    if nowords:
        processed_stim_info_file_name += 'nowords'
    if not nowords and not shared_only:
        processed_stim_info_file_name = 'all'
    processed_stim_info_file_name += 'stimuli_info.csv'
    if os.path.exists(f'{processed_stimuli_dir}/{processed_stim_info_file_name}'):
        return pd.read_csv(f'{processed_stimuli_dir}/{processed_stim_info_file_name}')
    stim_info_csv_path = f'{root_dir}/nsddata/experiments/nsdsynthetic/nsdsyntheticimageinformation.csv'
    stim_info = pd.read_csv(stim_info_csv_path)
    # Get info about trials
    trial_info_mat = f'{root_dir}/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat'
    trial_stim_ids = sio.loadmat(trial_info_mat)['masterordering'][0]
    # Convert to 0-indexed
    # List of 0-indexed stimuli ids according to the trial order in the experiment
    trial_stim_ids = trial_stim_ids - 1
    # Loop over stimuli and get the corresponding info
    stim_trial_ids = []
    stim_num_reps = []
    for stim_idx, _ in stim_info.iterrows():
        # Get the trial ids for the current stimulus
        trial_ids = np.where(trial_stim_ids == stim_idx)[0].tolist()
        num_reps = len(trial_ids)
        # Add the index to the dataframe
        stim_trial_ids.append(trial_ids)
        stim_num_reps.append(num_reps)
    # Add the trial ids and number of repetitions to the dataframe
    stim_info['trials'] = stim_trial_ids
    stim_info['num_repetitions'] = stim_num_reps
    if shared_only:
        # Remove subject specific stimuli
        subjectspecific_idxs = stim_info[stim_info['Image class'] == 'Chromatic noise'].index
        # Remove subject specific from stimuli info
        stim_info = stim_info.drop(subjectspecific_idxs)
        stim_info = stim_info.reset_index(drop=True)
    if nowords:
        # Remove words from stimuli
        words_idxs = stim_info[stim_info['Image class'] == 'Single words'].index
        # Remove words from stimuli info
        stim_info = stim_info.drop(words_idxs)
        stim_info = stim_info.reset_index(drop=True)
    # Save the stimuli info
    stim_info.to_csv(f'{processed_stimuli_dir}/{processed_stim_info_file_name}')
    return stim_info

def get_subject_stimuli(root_dir, subject, nowords, shared_only):
    """
    Get subset of stimuli
    Args:
        root_dir(str): Root directory
        subject(int): Subject
        nowords(bool): Whether to use words or not
    """
    # Error handling
    assert subject in range(1, 9), 'Invalid subject'
    # Stimuli info
    stim_info = get_stimuli_info(root_dir, nowords, shared_only)
    # Processed stimuli directory
    processed_stimuli_dir = f'{root_dir}/nsdsynthetic_processed/subj0{subject}'
    os.makedirs(processed_stimuli_dir, exist_ok=True)
    processed_stimuli_file_name = f'subj0{subject}_'
    if shared_only:
        processed_stimuli_file_name += 'shared'
    if nowords:
        processed_stimuli_file_name += 'nowords'
    if not shared_only and not nowords:
        processed_stimuli_file_name += 'all'
    processed_stimuli_file_name += 'stimuli.pt'
    if os.path.exists(f'{processed_stimuli_dir}/{processed_stimuli_file_name}'):
        return torch.load(f'{processed_stimuli_dir}/{processed_stimuli_file_name}'), stim_info
    # Path to stimuli
    common_stimuli_path = f'{root_dir}/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5'
    subject_stimuli_path = f'{root_dir}/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_colorstimuli_subj0{subject}.hdf5'
    # Load stimuli
    with h5py.File(common_stimuli_path, 'r') as chdf5:
        common_stimuli = chdf5['imgBrick'][:]
    if not shared_only:
        with h5py.File(subject_stimuli_path, 'r') as shdf5:
            subject_stimuli = shdf5['imgBrick'][:]
        # Concatenate stimuli
        stimuli = np.concatenate((common_stimuli, subject_stimuli), axis=0)
    else:
        stimuli = common_stimuli
    # Check if words are to be used
    if nowords:
        # Get indexes to delete
        all_idxs = stim_info['Image number'].values
        # Convert to 0-indexed
        all_idxs = all_idxs - 1
        # Get indexes to delete
        delete_idxs = list(set(range(stimuli.shape[0])) - set(all_idxs))
        # Delete indexes
        stimuli = np.delete(stimuli, delete_idxs, axis=0)
    # Convert to torch tensor
    stimuli = torch.from_numpy(stimuli)
    # Save stimuli
    stimuli_path = f'{processed_stimuli_dir}/{processed_stimuli_file_name}'
    # Save the stimuli
    torch.save(stimuli, stimuli_path)
    return stimuli, stim_info

def get_subject_betas(root_dir, subject, nowords, shared_only):
    """
    Get subset of data
    Args:
        root_dir(str): Root directory
        subject(int): Subject number
    """
    processed_betas_dir = f'{root_dir}/nsdsynthetic_processed/subj0{subject}'
    os.makedirs(processed_betas_dir, exist_ok=True)
    processed_betas_file_name = f'subj0{subject}_'
    if shared_only:
        processed_betas_file_name += 'shared'
    if nowords:
        processed_betas_file_name += 'nowords'
    if not shared_only and not nowords:
        processed_betas_file_name += 'all'
    processed_betas_file_name += f'stimuli_repmean_betas.pt'
    if os.path.exists(f'{processed_betas_dir}/{processed_betas_file_name}'):
        return torch.load(f'{processed_betas_dir}/{processed_betas_file_name}')
    # Path to betas
    subject_betas_dir = f'{root_dir}/nsddata_betas/ppdata/subj0{subject}/func1pt8mm/nsdsyntheticbetas_fithrf_GLMdenoise_RR'
    subject_betas_hdf5_name = 'betas_nsdsynthetic.hdf5'
    # Load the hdf5 file
    subject_betas_path = os.path.join(subject_betas_dir, subject_betas_hdf5_name)
    with h5py.File(subject_betas_path, 'r') as hdf5_file:
        # Get the betas
        subject_betas_alltrials = np.array(hdf5_file['betas'][:])
        # Shape: 744, d1, d2, d3
        # Convert type to float32 and divide by 300 as they have been stored after multiplying with 300
        subject_betas_alltrials = subject_betas_alltrials.astype(np.float32) / 300.0
        # Convert to torch tensor
        subject_betas_alltrials = torch.from_numpy(subject_betas_alltrials)
    # Take mean across repetitions
    stim_info = get_stimuli_info(root_dir, nowords, shared_only)
    # Make a new tensor to store the betas
    betas = torch.zeros(
        (len(stim_info), subject_betas_alltrials.shape[1],
         subject_betas_alltrials.shape[2], subject_betas_alltrials.shape[3])
    )
    for stim_idx, stim_row in stim_info.iterrows():
        # Get the trial ids for the current stimulus
        trial_ids = stim_row['trials']
        # Convert to list
        trial_ids = ast.literal_eval(trial_ids)
        # Get the number of repetitions
        num_reps = stim_row['num_repetitions']
        # Make a tensor to store the betas for this stimuli
        # Shape: num_reps, d1, d2, d3
        stim_betas = torch.zeros(
            (num_reps, subject_betas_alltrials.shape[1],
             subject_betas_alltrials.shape[2], subject_betas_alltrials.shape[3])
        )
        # Loop over the trial ids and get the betas
        for trial_idx, trial_id in enumerate(trial_ids):
            # Get the betas for the current trial
            stim_betas[trial_idx, :, :, :] = subject_betas_alltrials[trial_id, :, :, :]
        # Get the betas for the current stimulus
        betas[stim_idx, :, :, :] = stim_betas.mean(axis=0)
    # Save the betas
    betas_path = f'{processed_betas_dir}/{processed_betas_file_name}'
    torch.save(betas, betas_path)
    return betas

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

# Dataset name
def get_stimuli_dataset_name(subject, nowords, shared_only):
    """
    Get dataset name
    Args:
        subset(str): Subset of NSD dataset
    """
    dataset_name = 'nsdsynthetic'
    if shared_only:
        dataset_name += 'shared'
    if nowords:
        dataset_name += 'nowords'
    if not shared_only and not nowords:
        dataset_name += 'all'
    return f'{dataset_name}subj0{subject}'

# Stimuli
class NSDSyntheticStimuli(Dataset):
    """
    Dataset for loading the stimuli in the NSD Synthetic dataset
    """
    def __init__(self, root_dir, subject, transform, nowords=False, shared_only=False):
        self.root_dir = root_dir # NSD base directory
        self.subject = subject # Subject number
        self.nowords = nowords # Whether to use words or not
        self.shared_only = shared_only # Whether to use shared stimuli only
        # Set default transform
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform # Transformations to be applied
        self.all_stimuli, self.stim_info = get_subject_stimuli(
            self.root_dir, self.subject, self.nowords, self.shared_only
        ) # Get stimuli
        # Name of the dataset to store activations
        self.name = get_stimuli_dataset_name(
            self.subject, nowords=self.nowords, shared_only=self.shared_only
        )
        # Check if words are to be used
        if self.nowords:
            # # Remove words from stimuli
            # words_idxs = self.stim_info[self.stim_info['Image class'] == 'Single words'].index
            # # Remove words from stimuli info
            # self.stim_info = self.stim_info.drop(words_idxs)
            # self.stim_info = self.stim_info.reset_index(drop=True)
            # # Remove words from stimuli
            # self.all_stimuli = np.delete(self.all_stimuli, words_idxs, axis=0)
            # Make the stimuli square
            self.all_stimuli = self.all_stimuli[:, :, 323:1037, :] # (N, H, W, C)

    def __len__(self):
        return len(self.all_stimuli)

    def __getitem__(self, idx):
        image_array = self.all_stimuli[idx, :, :, :] # Numpy array (H, W, C)
        image = Image.fromarray(image_array) # PIL Image
        image = self.transform(image)
        return image

def get_subset_info(subset):
    """
    Convert subset arg to words
    """
    subset = subset.lower() if isinstance(subset, str) else None
    assert subset in [None, 'all', 'nowords', 'shared_nowords', 'shared'], \
        "Invalid subset, allowed: None/'all', 'nowords'"
    if subset is None:
        subset = 'all'
    nowords = False
    shared_only = False
    if 'shared' in subset:
        shared_only = True
    if 'nowords' in subset:
        nowords = True
    return nowords, shared_only

def get_stimuli_dataset(root_dir, subject, transform=None, subset=None):
    """
    Get NSD stimuli dataset, shape: N_images, H, W, C
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    nowords, shared_only = get_subset_info(subset)
    dataset = NSDSyntheticStimuli(
        root_dir, subject, transform=transform, nowords=nowords, shared_only=shared_only
    )
    return dataset

def get_betas(root_dir, subject, froi=None, tthreshold=0, subset=None, voxels=False, **kwargs):
    """
    Get betas for a subject
    Args:
        root_dir(str): Root directory
        subject(int): Subject number
        froi(str): Path to froi mask
        tthreshold(float): Threshold for t-values
        subset(str): Subset of NSD Synthetic dataset
            Options: None/'all', 'nowords', 'shared_nowords', 'shared'
        voxels(bool): Return voxels or mean of voxels
    """
    nowords, shared_only = get_subset_info(subset)
    # Get the betas
    betas = get_subject_betas(root_dir, subject, nowords, shared_only)
    # Get stim info
    stim_info = get_stimuli_info(root_dir, nowords, shared_only)
    # Check for repetition mean
    repetition_mean = kwargs.get('repetition_mean', True)
    assert repetition_mean is True, "Only repetition mean has been implemented for now"
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
    return betas, stim_info
