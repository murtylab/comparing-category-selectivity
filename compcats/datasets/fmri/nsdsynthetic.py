"""Code to load the NSD Synthetic dataset"""
import os
import ast
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import nsd_ds_utils

MAX_NUM_REPETITIONS = 10 # Max number of repetitions for any stimulus
SUBSET_TYPE_KEYS = ['nowords', 'shared_only', 'ood_only', 'all']

def _check_subset_type(subset_type):
    """
    Check if subset_type is valid
    Args:
        subset_type(dict): Subset info
    """
    if not isinstance(subset_type, dict):
        raise ValueError('subset_type must be a dictionary')
    for key in SUBSET_TYPE_KEYS:
        if key not in subset_type:
            raise ValueError(f'subset_type must contain key: {key}')
        if not isinstance(subset_type[key], bool):
            raise ValueError(f'subset_type[{key}] must be a boolean')
    # If all are False, raise error
    if not any(subset_type.values()):
        raise ValueError('At least one value in subset_type must be True')
    # If 'all' is True, others must be False
    if subset_type['all']:
        for key in SUBSET_TYPE_KEYS:
            if key != 'all' and subset_type[key]:
                raise ValueError('If subset_type["all"] is True, other keys must be False')
    # If any of 'shared_only' or 'ood_only' or 'nowords is True, 'all' must be False
    if subset_type['shared_only'] or subset_type['ood_only'] or subset_type['nowords']:
        if subset_type['all']:
            raise ValueError('If any of "shared_only", "ood_only", or "nowords" is True, "all" must be False')

def get_subset_stimuli_info(root_dir, subset_type):
    """
    Function to get the stimuli info, and generate if needed
    Trials and stimuli info were same across subjects
    Args:
        root_dir(str): Root directory
        words(bool): Whether to use words or not
    """
    _check_subset_type(subset_type)
    processed_stimuli_dir = f'{root_dir}/nsdsynthetic_processed'
    os.makedirs(processed_stimuli_dir, exist_ok=True)
    processed_stim_info_file_name = ''
    if subset_type['all']:
        processed_stim_info_file_name += 'all'
    if subset_type['shared_only']:
        processed_stim_info_file_name += 'shared'
    if subset_type['ood_only']:
        processed_stim_info_file_name += 'ood'
    if subset_type['nowords']:
        processed_stim_info_file_name += 'nowords'
    processed_stim_info_file_name += '_stimuli_info.csv'
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
    if subset_type['shared_only']:
        # Remove subject specific stimuli
        subjectspecific_idxs = stim_info[stim_info['Image class'] == 'Chromatic noise'].index
        # Remove subject specific from stimuli info
        stim_info = stim_info.drop(subjectspecific_idxs)
        stim_info = stim_info.reset_index(drop=True)
    if subset_type['ood_only']:
        # Keep only non-naturalistic stimuli. i.e. remove rows with 'Natural scenes' substring \
        # in 'Image class' column
        naturalistic_idxs = stim_info[stim_info['Image class'].str.contains('Natural scenes')].index
        stim_info = stim_info.drop(naturalistic_idxs)
        stim_info = stim_info.reset_index(drop=True)
    if subset_type['nowords']:
        # Remove words from stimuli
        words_idxs = stim_info[stim_info['Image class'] == 'Single words'].index
        # Remove words from stimuli info
        stim_info = stim_info.drop(words_idxs)
        stim_info = stim_info.reset_index(drop=True)
    # Save the stimuli info
    stim_info.to_csv(f'{processed_stimuli_dir}/{processed_stim_info_file_name}')
    return stim_info

def get_subset_stimuli(root_dir, subset_type, subject=None):
    """
    Get subset of stimuli
    Args:
        root_dir(str): Root directory
        subset_type(dict): Subset info
        subject(int): Subject number
    """
    # Processed stimuli directory
    processed_stimuli_dir = f'{root_dir}/nsdsynthetic_processed'
    processed_stimuli_file_name = ''
    # Error handling
    if subject is not None:
        assert subject in range(1, 9), 'Invalid subject'
        processed_stimuli_dir += f'/subj0{subject}'
        processed_stimuli_file_name += f'subj0{subject}_'
    os.makedirs(processed_stimuli_dir, exist_ok=True)
    # Stimuli info
    stim_info = get_subset_stimuli_info(root_dir, subset_type)
    if subset_type['all']:
        processed_stimuli_file_name += 'all'
    if subset_type['shared_only']:
        processed_stimuli_file_name += 'shared'
    if subset_type['ood_only']:
        processed_stimuli_file_name += 'ood'
    if subset_type['nowords']:
        processed_stimuli_file_name += 'nowords'
    processed_stimuli_file_name += 'stimuli.npy'
    if os.path.exists(f'{processed_stimuli_dir}/{processed_stimuli_file_name}'):
        return torch.load(f'{processed_stimuli_dir}/{processed_stimuli_file_name}'), stim_info
    # Path to stimuli
    common_stimuli_path = f'{root_dir}/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5'
    subject_stimuli_path = f'{root_dir}/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_colorstimuli_subj0{subject}.hdf5'
    # Load common stimuli
    with h5py.File(common_stimuli_path, 'r') as chdf5:
        common_stimuli = chdf5['imgBrick'][:]
    # Remove unwanted stimuli
    # Get indexes to delete
    all_idxs = np.array(stim_info['Image number'].values)
    # Convert to 0-indexed
    all_idxs = all_idxs - 1
    # Get indexes to delete
    delete_idxs = list(set(range(common_stimuli.shape[0])) - set(all_idxs))
    # Delete indexes
    common_stimuli = np.delete(common_stimuli, delete_idxs, axis=0)
    # Stimuli
    stimuli = common_stimuli
    # If not shared only, add subject specific stimuli
    if not subset_type['shared_only']:
        with h5py.File(subject_stimuli_path, 'r') as shdf5:
            subject_stimuli = shdf5['imgBrick'][:]
        # Concatenate stimuli
        stimuli = np.concatenate((stimuli, subject_stimuli), axis=0)
    # Check number of stimuli
    assert stimuli.shape[0] == len(stim_info), \
        f'Expected number of stimuli: {len(stim_info)}, got: {stimuli.shape[0]}'
    # Convert to torch tensor
    # stimuli = torch.from_numpy(stimuli)
    # Keep as numpy for PIL transforms compatibility
    # Convert to square images if no words
    if subset_type['nowords']:
        # Make the stimuli square
        stimuli = stimuli[:, :, 323:1037, :]
    # Save stimuli
    stimuli_path = f'{processed_stimuli_dir}/{processed_stimuli_file_name}'
    # Save the stimuli
    torch.save(stimuli, stimuli_path)
    return stimuli, stim_info

def get_subject_betas(root_dir, subject, subset_type, repetition_mean=True):
    """
    Get subset of data
    Args:
        root_dir(str): Root directory
        subject(int): Subject number
        subset_type(dict): Subset info
        repetition_mean(bool): Whether to take mean across repetitions
    """
    processed_betas_dir = f'{root_dir}/nsdsynthetic_processed/subj0{subject}'
    os.makedirs(processed_betas_dir, exist_ok=True)
    processed_betas_file_name = f'subj0{subject}_'
    if subset_type['all']:
        processed_betas_file_name += 'all'
    if subset_type['shared_only']:
        processed_betas_file_name += 'shared'
    if subset_type['ood_only']:
        processed_betas_file_name += 'ood'
    if subset_type['nowords']:
        processed_betas_file_name += 'nowords'
    repmean_flag_str = 'mean' if repetition_mean else 'individual'
    processed_betas_file_name += f'_rep{repmean_flag_str}_betas.pt'
    if os.path.exists(f'{processed_betas_dir}/{processed_betas_file_name}'):
        return torch.load(f'{processed_betas_dir}/{processed_betas_file_name}')
    # Get stimuli info
    stim_info = get_subset_stimuli_info(root_dir, subset_type)
    max_num_repetitions = stim_info['num_repetitions'].max().item()
    assert max_num_repetitions == MAX_NUM_REPETITIONS, \
        f'Expected max num repetitions to be {MAX_NUM_REPETITIONS}, got {max_num_repetitions}'
    # Generate repmean from repindividual if it exists
    if 'repmean' in processed_betas_file_name:
        # If repmean betas are requested, check if repindividual betas exist
        repindividual_betas_file_name = processed_betas_file_name.replace(
            'repmean', 'repindividual'
        )
        if os.path.exists(f'{processed_betas_dir}/{repindividual_betas_file_name}'):
            print('Loading repindividual betas to compute repmean betas')
            repindividual_betas = torch.load(
                f'{processed_betas_dir}/{repindividual_betas_file_name}'
            )
            assert repindividual_betas.ndim == 5, \
                f'Expected repindividual betas to have 5 dimensions, got {repindividual_betas.ndim}'
            assert repindividual_betas.shape[1] == max_num_repetitions, \
                f'Expected repindividual betas to have shape[1] == {max_num_repetitions}, got {repindividual_betas.shape[1]}'
            assert repindividual_betas.shape[0] == len(stim_info), \
                f'Expected repindividual betas to have shape[0] == {len(stim_info)}, got {repindividual_betas.shape[0]}'
            # Take mean across repetitions
            repmean_betas = torch.nanmean(repindividual_betas, dim=1)
            # Save the repmean betas
            torch.save(repmean_betas, f'{processed_betas_dir}/{processed_betas_file_name}')
            return repmean_betas
    # Check if responses for the full set ('all') of stimuli exist, in which case just subset
    allstim_betas_file_name = f'subj0{subject}_all_rep{repmean_flag_str}_betas.pt'
    if os.path.exists(f'{processed_betas_dir}/{allstim_betas_file_name}'):
        print('Loading full stimuli betas to subset')
        allstim_betas = torch.load(f'{processed_betas_dir}/{allstim_betas_file_name}')
        assert allstim_betas.shape[0] == 284, \
            f'Expected allstim betas to have shape[0] == 284, got {allstim_betas.shape[0]}'
        # Get indexes to keep
        keep_idxs = torch.Tensor(stim_info['Image number'].values) - 1 # Convert to 0-indexed
        # Convert to long tensor
        keep_idxs = keep_idxs.long()
        # Subset the betas
        betas = allstim_betas[keep_idxs, ...]
    else:
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
        # Make a new tensor to store the betas
        betas = torch.full(
            (
                len(stim_info),
                max_num_repetitions,
                subject_betas_alltrials.shape[1],
                subject_betas_alltrials.shape[2],
                subject_betas_alltrials.shape[3]
            ),
            float('nan')
        )
        for stim_idx, stim_row in stim_info.iterrows():
            # Get the trial ids for the current stimulus
            trial_ids = stim_row['trials']
            # Convert to list
            if isinstance(trial_ids, str):
                # If trial_ids is a string, convert it to a list
                trial_ids = ast.literal_eval(trial_ids)
            # Get the number of repetitions
            # num_reps = stim_row['num_repetitions']
            # Make a tensor to store the betas for this stimuli
            # Shape: num_reps, d1, d2, d3
            stim_betas = torch.full(
                (
                    max_num_repetitions,
                    subject_betas_alltrials.shape[1],
                    subject_betas_alltrials.shape[2],
                    subject_betas_alltrials.shape[3]
                ),
                float('nan')
            )
            # Loop over the trial ids and get the betas
            for trial_idx, trial_id in enumerate(trial_ids):
                # Get the betas for the current trial
                stim_betas[trial_idx, :, :, :] = subject_betas_alltrials[trial_id, :, :, :]
            # Append to the betas tensor
            betas[stim_idx, :, :, :, :] = stim_betas
        if repetition_mean:
            # Take mean across repetitions
            betas = torch.nanmean(betas, dim=1)
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
def get_stimuli_dataset_name(subset_type, subject=None):
    """
    Get dataset name
    Args:
        subset(str): Subset of NSD dataset
    """
    _check_subset_type(subset_type)
    nowords = subset_type['nowords']
    shared_only = subset_type['shared_only']
    ood_only = subset_type['ood_only']
    dataset_name = 'nsdsynthetic'
    if shared_only:
        dataset_name += 'shared'
    if ood_only:
        dataset_name += 'ood'
    if nowords:
        dataset_name += 'nowords'
    if not shared_only and not nowords:
        dataset_name += 'all'
    if subject is not None:
        dataset_name += f'subj0{subject}'
    return dataset_name

# Stimuli
class NSDSyntheticStimuli(Dataset):
    """
    Dataset for loading the stimuli in the NSD Synthetic dataset
    """
    def __init__(self, root_dir, transform, subset=None, **kwargs):
        # Error handling
        self.subject = kwargs.get('subject', None)
        self.subset_type = _extract_subset_type(subset)
        _check_subset_type(self.subset_type)
        if not self.subset_type['shared_only'] and self.subject is None:
            raise ValueError('Subject must be specified if subset is not shared')
        self.root_dir = root_dir # NSD base directory
        # Set default transform
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform # Transformations to be applied
        self.all_stimuli, self.stim_info = get_subset_stimuli(
            self.root_dir, self.subset_type, self.subject
        ) # Get stimuli
        # Name of the dataset to store activations
        self.name = get_stimuli_dataset_name(
            subset_type=self.subset_type, subject=self.subject
        )
        # Check if words are to be used
        # if self.nowords:
            # # Remove words from stimuli
            # words_idxs = self.stim_info[self.stim_info['Image class'] == 'Single words'].index
            # # Remove words from stimuli info
            # self.stim_info = self.stim_info.drop(words_idxs)
            # self.stim_info = self.stim_info.reset_index(drop=True)
            # # Remove words from stimuli
            # self.all_stimuli = np.delete(self.all_stimuli, words_idxs, axis=0)
            # Make the stimuli square
            # self.all_stimuli = self.all_stimuli[:, :, 323:1037, :] # (N, H, W, C)

    def __len__(self):
        return len(self.all_stimuli)

    def __getitem__(self, idx):
        image_array = self.all_stimuli[idx, :, :, :] # Torch tensor (H, W, C)
        image = Image.fromarray(image_array) # PIL Image
        image = self.transform(image)
        return image

def _extract_subset_type(subset):
    """
    Convert subset arg to words
    """
    subset = subset.lower() if isinstance(subset, str) else None
    possible_subsets = [
        None, 'all', 'nowords', 'shared_nowords', 'shared',
        'shared_ood_nowords'
    ]
    assert subset in possible_subsets, \
        f"Invalid subset, {subset}, possible options: {possible_subsets}"
    all_stimuli = False
    nowords = False
    shared_only = False
    ood_only = False
    if subset is None:
        subset = 'all'
    if subset == 'all':
        all_stimuli = True
    if 'shared' in subset:
        shared_only = True
    if 'nowords' in subset:
        nowords = True
    if 'ood' in subset:
        ood_only = True
    subset_type = {
        'nowords': nowords,
        'shared_only': shared_only,
        'ood_only': ood_only,
        'all': all_stimuli
    }
    _check_subset_type(subset_type)
    return subset_type

def get_stimuli_dataset(root_dir, transform=None, subset=None, **kwargs):
    """
    Get NSD stimuli dataset, shape: N_images, H, W, C
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = NSDSyntheticStimuli(
        root_dir, transform=transform, subset=subset, **kwargs
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
    # Check for repetition mean
    repetition_mean = kwargs.get('repetition_mean', True)
    # Get subset info
    subset_type = _extract_subset_type(subset)
    # Get the betas
    betas = get_subject_betas(root_dir, subject, subset_type, repetition_mean)
    # Get stim info
    subset_stimuli_info_csv = get_subset_stimuli_info(root_dir, subset_type)
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
            # Assertion to check for NaNs, there should be no NaNs after masking
            assert not torch.isnan(betas).any(), 'NaNs found in betas after masking'
        else:
            betas = betas.reshape(betas.shape[0], betas.shape[1], -1) # Flatten the voxel dimensions
        betas = betas.mean(dim=-1)
    # Check number of betas match number of stimuli
    assert betas.shape[0] == len(subset_stimuli_info_csv), \
        f'Expected number of betas: {len(subset_stimuli_info_csv)}, got: {betas.shape[0]}'
    return betas, subset_stimuli_info_csv
