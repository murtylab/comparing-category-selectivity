import os
import glob
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib

import torch

# Stimuli
def get_subject_cocoids(root_dir, subject):
    # Estimate number of trials for a subject
    betas_dir = f'{root_dir}/nsddata_betas/ppdata/subj0{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'
    subject_nsessions = len(glob.glob(f'{betas_dir}/betas_session*.hdf5'))
    subject_ntrials = subject_nsessions*750
    # Get related stimuli coco ids
    stim_info_csv_path = f'{root_dir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    all_info_csv = pd.read_csv(stim_info_csv_path)
    info = all_info_csv[all_info_csv['shared1000'] == True]
    info_trials = info[(info[f'subject{subject}_rep0']<subject_ntrials) & \
                       (info[f'subject{subject}_rep1']<subject_ntrials) & \
                        (info[f'subject{subject}_rep2']<subject_ntrials)]
    subject_cocoids = list(info_trials['cocoId'])
    return subject_cocoids

def get_subset_nsdids(root_dir, subset):
    """
    Get nsdids of shared1000 and non-shared1000 stimuli
    """
    subset_nsdids = None
    # Stimuli csv
    stim_info_csv_path = f'{root_dir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    all_info_csv = pd.read_csv(stim_info_csv_path)
    # shared1000 subset in csv
    shared1000_info = all_info_csv[all_info_csv['shared1000'] == True]
    shared1000_nsdids = list(shared1000_info['nsdId'])
    if subset is None:
        subset_nsdids = list(all_info_csv['nsdId'])
    elif subset == 1000:
        # Get shared1000 nsdids
        subset_nsdids = shared1000_nsdids
    elif subset in [485, 515]:
        shared515_cocoids = get_shared515_cocoids(root_dir)
        shared515_nsdids = list(shared1000_info[shared1000_info['cocoId'].isin(shared515_cocoids)]['nsdId'])
        if subset == 515:
            # Get shared515 nsdids
            subset_nsdids = shared515_nsdids
        if subset == 485:
            # Get shared485 nsdids
            subset_nsdids = list(set(shared1000_nsdids) - set(shared515_nsdids))
    else:
        raise NotImplementedError('NSD subset length not implemented yet, only 485, 515 or 1000 available')
    # Sort nsdids
    subset_nsdids.sort()
    return subset_nsdids

def get_shared515_cocoids(root_dir):
    common_cocoids = None
    for sbj_idx in range(1, 9):
        sbjids = get_subject_cocoids(root_dir, sbj_idx)
        if common_cocoids is None:
            common_cocoids = set(sbjids)
        else:
            common_cocoids = common_cocoids.intersection(set(sbjids))
    assert len(common_cocoids) == 515, 'Incorrect number of coco ids in shared515 subset'
    common_cocoids = sorted(common_cocoids)
    return common_cocoids

def get_subset_stimuli(root_dir, subset=None):
    all_stimuli_dir = f'{root_dir}/nsddata_stimuli/stimuli/nsd'
    stimuli_file_name = 'nsd_stimuli.hdf5'
    processed_subset_dir = f'{root_dir}/nsd_processed/stimuli/nsd'
    os.makedirs(processed_subset_dir, exist_ok=True)
    # Get subset nsdids
    subset_nsdids = get_subset_nsdids(root_dir, subset)
    if subset is None:
        stim_hdf5 = h5py.File(f'{all_stimuli_dir}/{stimuli_file_name}', 'r')
        all_stimuli = stim_hdf5['imgBrick'] # (73000, 425, 425, 3)
        all_stimuli = all_stimuli[:] # Convert to numpy array
    else:
        subset_stimuli_file_name = stimuli_file_name.split('.')[0] + f'{subset}.npy'
        if subset_stimuli_file_name in os.listdir(processed_subset_dir):
            all_stimuli = np.load(f'{processed_subset_dir}/{subset_stimuli_file_name}')
        else:
            stim_hdf5 = h5py.File(f'{all_stimuli_dir}/{stimuli_file_name}', 'r')
            all_stimuli = stim_hdf5['imgBrick']
            all_stimuli = all_stimuli[subset_nsdids, :, :, :]
            np.save(f'{processed_subset_dir}/{subset_stimuli_file_name}', all_stimuli)
    return all_stimuli, subset_nsdids

# Betas
def get_session_numbers_recorded(root_dir, subject):
    """
    Get number of sessions recorded for a subject
    """
    betas_dir = f'{root_dir}/nsddata_betas/ppdata/subj0{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'
    session_files = glob.glob(f'{betas_dir}/betas_session*.hdf5')
    session_numbers = [int(os.path.basename(f).split('betas_session')[1].split('.hdf5')[0]) for f in session_files]
    return session_numbers

def get_session_betas(root_dir, subject, session_num, trials_info):
    """
    Get required betas in a single session
    """
    betas_dir = f'{root_dir}/nsddata_betas/ppdata/subj0{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'
    betas_file_name = f'betas_session{session_num:02d}.hdf5'
    session_betas_hdf5 = h5py.File(f'{betas_dir}/{betas_file_name}', 'r')['betas'][:]
    # Convert type to float32 and divide by 300 as they have been stored after multiplying with 300
    # https://cvnlab.slite.page/p/6CusMRYfk0/Untitled#7dfe1d13
    # Converting to float32 to save memory, convert to float64 after z-scoring
    session_betas_hdf5 = session_betas_hdf5.astype(np.float32) / 300.0
    # z-score within session to account for inter-session variability
    session_betas_mean = np.mean(session_betas_hdf5, axis=0)
    session_betas_std = np.std(session_betas_hdf5, axis=0)
    # Handle divide by zero, turn it into NaN
    session_betas_std[session_betas_std == 0] = np.nan
    # z-score in chunks and store in original array to reduce memory load
    for stim_i in range(session_betas_hdf5.shape[0]):
        session_betas_hdf5[stim_i] = (session_betas_hdf5[stim_i] - session_betas_mean) / session_betas_std
    session_trials_info = trials_info[trials_info[:, 1] == session_num]
    session_betas = {}
    # trial_number: trial_betas
    for tr_info in session_trials_info:
        session_betas[tr_info[0]] = session_betas_hdf5[tr_info[2]].astype(float)
    # Free-up memory
    del session_betas_hdf5, session_betas_mean, session_betas_std
    return session_betas

def get_stim_betas(subject, stim_info, all_trial_betas, repetition_mean=True):
    """
    Get betas for stimuli, mean across repetitions
    """
    nsdid_idx_mapping = []
    all_betas = None
    all_trial_betas_keys = list(all_trial_betas.keys()) # All trails
    data_shape = all_trial_betas[all_trial_betas_keys[0]].shape
    for idx, (row_idx, row) in enumerate(stim_info.iterrows()):
        n_repetitions = sum([row[f'subject{subject}_rep{i}'] in all_trial_betas_keys for i in range(3)])
        # Skip this stimuli if no repetitions were collected
        if n_repetitions == 0:
            continue
        nsdid_idx_mapping.append([row['nsdId'], idx, \
                                row['cocoId'], row['cocoSplit'], \
                                row['cropBox'], row['loss'], row['flagged'], \
                                row['BOLD5000'], n_repetitions])
        stim_betas = None
        for rep_idx in range(3): # Maximum 3 repetitions for each image
            rep_trial_idx = row[f'subject{subject}_rep{rep_idx}']
            if rep_trial_idx in all_trial_betas_keys:
                rep_betas = all_trial_betas[rep_trial_idx]
                rep_betas = torch.FloatTensor(rep_betas)
                rep_betas = rep_betas.unsqueeze(0) # Add a dimension
                stim_betas = rep_betas if stim_betas is None else torch.cat((stim_betas, rep_betas), 0)
            else:
                nan_tensor = torch.full(data_shape, float('nan')) # Make a Float Tensor full of NaNs
                nan_tensor = nan_tensor.unsqueeze(0) # Add a dimension
                stim_betas = nan_tensor if stim_betas is None else torch.cat((stim_betas, nan_tensor), 0)
            print(f'Stim {idx}, Trial {rep_trial_idx}, Rep {rep_idx}')
        if repetition_mean:
            # Taking mean across small tensors then concatenating is more space and time efficient 
            # than concatenating and then taking mean across a huge tensor
            # Take average across repetitions
            stim_betas = torch.nanmean(stim_betas, 0, keepdim=True)
            # Resulting shape will be (n_images, n_voxels_dim1, n_voxels_dim2, n_voxels_dim3)
        else:
            stim_betas = stim_betas.unsqueeze(0)
        # Resulting shape will be (n_images, 3, n_voxels_dim1, n_voxels_dim2, n_voxels_dim3)
        # Some of these out of 3 may be NaNs depending on number of repetitions
        all_betas = stim_betas if all_betas is None else torch.cat((all_betas, stim_betas), 0)
    return all_betas, nsdid_idx_mapping

def get_shared1000_data(root_dir, subject, repetition_mean=True):
    """
    Function to generate a PyTorch file with shared1000 betas
    Args:
        root_dir(str): Path to model data storage
        subject(int): Subject number is 1, 2, 5 or 7
    """
    assert subject in [1, 2, 5, 7], f'shared1000 not available for subject {subject}'
    store_path = f'{root_dir}/nsd_processed/subj0{subject}'
    os.makedirs(store_path, exist_ok=True)
    if repetition_mean:
        betas_file_name = 'shared1000_repmean_betas.pt'
    else:
        betas_file_name = 'shared1000_repindividual_betas.pt'
    idx_file_name = 'nsdid_mapping_shared1000.csv'
    if (betas_file_name in os.listdir(store_path)) and (idx_file_name in os.listdir(store_path)):
        # print(f'Loading shared1000 betas mean across repetitions for subject {subject} from {store_path}/{betas_file_name}')
        stim_betas = torch.load(f'{store_path}/{betas_file_name}')
        nsdid_idx_mapping = pd.read_csv(f'{store_path}/{idx_file_name}')
        return stim_betas, nsdid_idx_mapping
    # Stimuli csv
    stim_info_csv_path = f'{root_dir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    all_info_csv = pd.read_csv(stim_info_csv_path)
    info = all_info_csv[all_info_csv['shared1000'] == True]
    # Select useful columns
    info = info[['cocoId', 'nsdId', \
                f'subject{subject}_rep0', \
                f'subject{subject}_rep1', \
                f'subject{subject}_rep2', \
                'cocoSplit', \
                'cropBox', 'loss', 'flagged', 'BOLD5000']]
    # Load required data
    trials = list(info[f'subject{subject}_rep0']) + list(info[f'subject{subject}_rep1']) + list(info[f'subject{subject}_rep2'])
    # Remove all zeros
    trials = [item for item in trials if item != 0]
    trials.sort()
    trial_sessioninfo = []
    for tr in trials:
        # Trial, session number, index in the session
        trial_sessioninfo.append([tr, ((tr-1)//750)+1, (tr%750)-1])
    trial_sessioninfo = np.array(trial_sessioninfo)
    # Storing betas for trials
    trial_betas = {}
    # Iterate over sessions
    session_numbers = np.unique(trial_sessioninfo[:, 1])
    for session_num in tqdm(session_numbers):
        # Get required betas from a session
        session_betas = get_session_betas(root_dir, subject, session_num, trial_sessioninfo)
        trial_betas = trial_betas | session_betas
    # Take average across repetitions for a stimuli
    stim_betas, nsdid_idx_mapping = get_stim_betas(subject, info, trial_betas, repetition_mean=repetition_mean)
    # Store betas
    torch.save(stim_betas, f'{store_path}/{betas_file_name}')
    # Store ID mapping
    idx_df = pd.DataFrame(nsdid_idx_mapping, columns=['nsdId', 'ptId', 'cocoId', \
                                                    'cocoSplit', 'cropBox', 'loss', \
                                                    'flagged', 'BOLD5000', 'num_repetitions'])
    idx_df.to_csv(f'{store_path}/{idx_file_name}')
    return stim_betas, idx_df

def get_shared515_data(root_dir, subject, repetition_mean=True):
    """
    Function to generate a PyTorch file with shared515 betas
    """
    assert subject in range(1, 9), f'Incorrect subject number {subject}'
    store_path = f'{root_dir}/nsd_processed/subj0{subject}'
    os.makedirs(store_path, exist_ok=True)
    if repetition_mean:
        betas_file_name = 'shared515_repmean_betas.pt'
    else:
        betas_file_name = 'shared515_repindividual_betas.pt'
    idx_file_name = 'nsdid_mapping_shared515.csv'
    if (betas_file_name in os.listdir(store_path)) and (idx_file_name in os.listdir(store_path)):
        # print(f'Loading shared515 betas mean across repetitions for subject {subject} from {store_path}/{betas_file_name}')
        stim_betas = torch.load(f'{store_path}/{betas_file_name}')
        nsdid_idx_mapping = pd.read_csv(f'{store_path}/{idx_file_name}')
        return stim_betas, nsdid_idx_mapping
    # Stimuli csv
    stim_info_csv_path = f'{root_dir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    all_info_csv = pd.read_csv(stim_info_csv_path)
    info = all_info_csv[all_info_csv['shared1000'] == True]
    # Get 515 common cocoids
    common_cocoids = get_shared515_cocoids(root_dir)
    info = info[info['cocoId'].isin(common_cocoids)]
    # Select useful columns
    info = info[['cocoId', 'nsdId', \
                f'subject{subject}_rep0', \
                f'subject{subject}_rep1', \
                f'subject{subject}_rep2', \
                'cocoSplit', \
                'cropBox', 'loss', 'flagged', 'BOLD5000']]
    # Load required data
    trials = list(info[f'subject{subject}_rep0']) + list(info[f'subject{subject}_rep1']) + list(info[f'subject{subject}_rep2'])
    # Remove all zeros
    trials = [item for item in trials if item != 0]
    trials.sort()
    trial_sessioninfo = []
    for tr in trials:
        trial_sessioninfo.append([tr, ((tr-1)//750)+1, (tr%750)-1])
    trial_sessioninfo = np.array(trial_sessioninfo)
    # Storing betas for trials
    trial_betas = {}
    # Iterate over sessions
    session_numbers = np.unique(trial_sessioninfo[:, 1])
    for session_num in tqdm(session_numbers):
        # Get required betas from a session
        session_betas = get_session_betas(root_dir, subject, session_num, trial_sessioninfo)
        trial_betas = trial_betas | session_betas
    # Take average across repetitions for a stimuli
    stim_betas, nsdid_idx_mapping = get_stim_betas(subject, info, trial_betas, repetition_mean=repetition_mean)
    # Store betas
    torch.save(stim_betas, f'{store_path}/{betas_file_name}')
    # Store ID mapping
    idx_df = pd.DataFrame(nsdid_idx_mapping, columns=['nsdId', 'ptId', 'cocoId', \
                                                    'cocoSplit', 'cropBox', 'loss', \
                                                    'flagged', 'BOLD5000', 'num_repetitions'])
    idx_df.to_csv(f'{store_path}/{idx_file_name}')
    return stim_betas, idx_df

def get_shared485_data(root_dir, subject, repetition_mean=True):
    """
    Function to generate a PyTorch file with shared485 betas
    """
    assert subject in [1, 2, 5, 7], f'shared1000 subset 485 not available for subject {subject}'
    store_path = f'{root_dir}/nsd_processed/subj0{subject}'
    os.makedirs(store_path, exist_ok=True)
    if repetition_mean:
        betas_file_name = 'shared485_repmean_betas.pt'
    else:
        betas_file_name = 'shared485_repindividual_betas.pt'
    idx_file_name = 'nsdid_mapping_shared485.csv'
    if (betas_file_name in os.listdir(store_path)) and (idx_file_name in os.listdir(store_path)):
        # print(f'Loading subset485 betas mean across repetitions for subject {subject} from {store_path}/{betas_file_name}')
        stim_betas = torch.load(f'{store_path}/{betas_file_name}')
        nsdid_idx_mapping = pd.read_csv(f'{store_path}/{idx_file_name}')
        return stim_betas, nsdid_idx_mapping
    # Stimuli csv
    stim_info_csv_path = f'{root_dir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    all_info_csv = pd.read_csv(stim_info_csv_path)
    info = all_info_csv[all_info_csv['shared1000'] == True]
    # Get 515 common cocoids
    common_cocoids = get_shared515_cocoids(root_dir)
    info = info[~info['cocoId'].isin(common_cocoids)]
    # Select useful columns
    info = info[['cocoId', 'nsdId', \
                f'subject{subject}_rep0', \
                f'subject{subject}_rep1', \
                f'subject{subject}_rep2', \
                'cocoSplit', \
                'cropBox', 'loss', 'flagged', 'BOLD5000']]
    # Load required data
    trials = list(info[f'subject{subject}_rep0']) + list(info[f'subject{subject}_rep1']) + list(info[f'subject{subject}_rep2'])
    # Remove all zeros
    trials = [item for item in trials if item != 0]
    trials.sort()
    trial_sessioninfo = []
    for tr in trials:
        trial_sessioninfo.append([tr, ((tr-1)//750)+1, (tr%750)-1])
    trial_sessioninfo = np.array(trial_sessioninfo)
    # Storing betas for trials
    trial_betas = {}
    # Iterate over sessions
    session_numbers = np.unique(trial_sessioninfo[:, 1])
    for session_num in tqdm(session_numbers):
        # Get required betas from a session
        session_betas = get_session_betas(root_dir, subject, session_num, trial_sessioninfo)
        trial_betas = trial_betas | session_betas
    # Take average across repetitions for a stimuli
    stim_betas, nsdid_idx_mapping = get_stim_betas(subject, info, trial_betas, repetition_mean=repetition_mean)
    # Store betas
    torch.save(stim_betas, f'{store_path}/{betas_file_name}')
    # Store ID mapping
    idx_df = pd.DataFrame(nsdid_idx_mapping, columns=['nsdId', 'ptId', 'cocoId', \
                                                    'cocoSplit', 'cropBox', 'loss', \
                                                    'flagged', 'BOLD5000', 'num_repetitions'])
    idx_df.to_csv(f'{store_path}/{idx_file_name}')
    return stim_betas, idx_df

def get_allstimuli_data(root_dir, subject, repetition_mean=True):
    """
    Function to generate a PyTorch file with betas for all stimuli
    Args:
        root_dir(str): Path to model data storage
        subject(int): Subject number between [1, 8]
    """
    # Store path for processed data
    store_path = f'{root_dir}/nsd_processed/subj0{subject}'
    os.makedirs(store_path, exist_ok=True)
    if repetition_mean:
        betas_file_name = 'allstimuli_repmean_betas.pt'
    else:
        betas_file_name = 'allstimuli_repindividual_betas.pt'
    idx_file_name = 'nsdid_mapping_allstimuli.csv'
    # Load if data already processed
    if (betas_file_name in os.listdir(store_path)) and (idx_file_name in os.listdir(store_path)):
        # print(f'Loading shared1000 betas mean across repetitions for subject {subject} from {store_path}/{betas_file_name}')
        stim_betas = torch.load(f'{store_path}/{betas_file_name}')
        nsdid_idx_mapping = pd.read_csv(f'{store_path}/{idx_file_name}')
        return stim_betas, nsdid_idx_mapping
    elif ('allstimuli_repindividual_betas.pt' in os.listdir(store_path)) and \
         (idx_file_name in os.listdir(store_path)):
        # If individual repetition data is present, then just load and take average across it
        print(f'Processed repetition mean betas for all stimuli not found but individual betas found, \
              so taking mean across repetitions for subject {subject}')
        repind_stim_betas = torch.load(f'{store_path}/allstimuli_repindividual_betas.pt')
        stim_betas = torch.nanmean(repind_stim_betas, 1) # Take mean across the repetition dimension
        torch.save(stim_betas, f'{store_path}/{betas_file_name}')
        nsdid_idx_mapping = pd.read_csv(f'{store_path}/{idx_file_name}')
        return stim_betas, nsdid_idx_mapping
    print(f'Processed betas for all stimuli not found, generating for subject {subject}')
    print('Beware: This requires a lot of memory (>256GB), this process might crash if not enough RAM is available')
    # Stimuli csv
    stim_info_csv_path = f'{root_dir}/nsddata/experiments/nsd/nsd_stim_info_merged.csv'
    all_info_csv = pd.read_csv(stim_info_csv_path)
    # Filter rows where at least one repetition is non-zero
    all_info_csv = all_info_csv[(all_info_csv[f'subject{subject}_rep0'] != 0) | 
                                (all_info_csv[f'subject{subject}_rep1'] != 0) | 
                                (all_info_csv[f'subject{subject}_rep2'] != 0)]
    # Select useful columns
    info = all_info_csv[['cocoId', 'nsdId', \
                f'subject{subject}_rep0', \
                f'subject{subject}_rep1', \
                f'subject{subject}_rep2', \
                'cocoSplit', \
                'cropBox', 'loss', 'flagged', 'BOLD5000']]
    # Load required data
    trials = list(info[f'subject{subject}_rep0']) + list(info[f'subject{subject}_rep1']) + list(info[f'subject{subject}_rep2'])
    # Remove all zeros
    trials = [item for item in trials if item != 0]
    trials.sort()
    trial_sessioninfo = []
    for tr in trials:
        # Trial, session number, index in the session
        trial_sessioninfo.append([tr, ((tr-1)//750)+1, (tr%750)-1])
    trial_sessioninfo = np.array(trial_sessioninfo)
    # Storing betas for trials
    trial_betas = {}
    # Iterate over sessions
    # session_numbers = np.unique(trial_sessioninfo[:, 1]) # Ideal number of sessions
    # Number of sessions planned may be more than number of sessions recorded
    session_numbers = get_session_numbers_recorded(root_dir, subject)
    session_numbers.sort()
    for session_num in tqdm(session_numbers):
        # Get required betas from a session
        session_betas = get_session_betas(root_dir, subject, session_num, trial_sessioninfo)
        trial_betas = trial_betas | session_betas
    # Take average across repetitions for a stimuli
    stim_betas, nsdid_idx_mapping = get_stim_betas(subject, info, trial_betas, repetition_mean=repetition_mean)
    # Store betas
    torch.save(stim_betas, f'{store_path}/{betas_file_name}')
    # Store ID mapping
    idx_df = pd.DataFrame(nsdid_idx_mapping, columns=['nsdId', 'ptId', 'cocoId', \
                                                    'cocoSplit', 'cropBox', 'loss', \
                                                    'flagged', 'BOLD5000', 'num_repetitions'])
    idx_df.to_csv(f'{store_path}/{idx_file_name}')
    return stim_betas, idx_df

def get_subset_data(root_dir, subject, subset, **kwargs):
    """
    Function to generate a PyTorch file with shared1000 betas
    """
    repetition_mean = kwargs.get('repetition_mean', True)
    if subset == 1000:
        return get_shared1000_data(root_dir, subject, repetition_mean)
    elif subset == 515:
        return get_shared515_data(root_dir, subject, repetition_mean)
    elif subset == 485:
        return get_shared485_data(root_dir, subject, repetition_mean)
    else:
        return get_allstimuli_data(root_dir, subject, repetition_mean)

# fROI
# Function to threshold the NSD fROI mask over a t-value threshold
def threshold_froi_mask(root_dir, subject, froi, froi_mask, tthreshold):
    """
    Threshold the froi mask >= tthreshold.
    Args:
        root_dir (str): The root directory containing the NSD dataset.
        subject (int): The subject number.
        froi (str): The functional region of interest (fROI) name.
        froi_mask (numpy.ndarray): The mask of the fROI.
        tthreshold (float): The threshold value for t-values.
    Returns:
        tuple: A tuple containing:
            - froi_thresholded_mask (numpy.ndarray): The thresholded mask.
            - froi_thresholded_tvals (numpy.ndarray): The corresponding t-values of the thresholded mask.
    """
    # Variables
    face_areas = ['OFA', 'FFA_1', 'FFA_2', 'FFA', 'MTL_FACES', 'ATL_FACES']
    body_areas = ['EBA', 'FBA_1', 'FBA_2', 'FBA', 'MTL_BODIES']
    place_areas = ['OPA', 'PPA', 'RSC']
    word_areas = ['OVWFA', 'VWFA_1', 'VWFA_2', 'VWFA', 'MFS_WORDS', 'MTL_WORDS']
    # Load t-values of the whole brain
    tvals_dir = f'{root_dir}/nsddata/ppdata/subj0{subject}/func1pt8mm'
    froi_mask = froi_mask.astype(bool)
    if froi in face_areas:
        tvals_file_name = 'floc_facestval.nii.gz'
    elif froi in body_areas:
        tvals_file_name = 'floc_bodiestval.nii.gz'
    elif froi in place_areas:
        tvals_file_name = 'floc_placestval.nii.gz'
    elif froi in word_areas:
        tvals_file_name = 'floc_wordtval.nii.gz'
    else:
        raise NotImplementedError(f'{froi} mask not found in NSD dataset')
    # Load t-values of the whole brain
    froi_all_tvals = nib.load(f'{tvals_dir}/{tvals_file_name}').get_fdata()
    # Get t-values of selected areas in the froi
    froi_selectedareas_tvals = np.full_like(froi_all_tvals, -np.inf)
    froi_selectedareas_tvals[froi_mask] = froi_all_tvals[froi_mask]
    # Threshold the voxels
    froi_thresholded_mask = froi_selectedareas_tvals >= tthreshold
    froi_thresholded_tvals = np.full_like(froi_all_tvals, -np.inf)
    froi_thresholded_tvals[froi_thresholded_mask] = froi_all_tvals[froi_thresholded_mask]
    # Return the thresholded mask and the corresponding t-values
    return froi_thresholded_mask, froi_thresholded_tvals

# Function to load an NSD fROI mask for a specific subject
def get_froi_mask(root_dir, froi, subject, tthreshold=0):
    """
    Get the functional region of interest (fROI) mask for a given subject.
    Args:
        root_dir (str): The root directory where the NSD dataset is stored.
        froi (str): The name of the functional region of interest (e.g., 'OFA', 'FFA_1', 'PPA').
        subject (int): The subject number [1, 8].
        tthreshold (float, optional): The threshold value to apply to the mask. Default is 0.
    Returns:
        tuple: A tuple containing:
            - froi_mask (numpy.ndarray): The binary mask for the selected fROI.
            - froi_tvals (numpy.ndarray): The t-values associated with the selected fROI mask.
    Raises:
        NotImplementedError: If the specified fROI is not found in the NSD dataset.
    Reference: https://cvnlab.slite.page/p/X_7BBMgghj/ROIs
    """

    froi = froi.upper()
    mask_dir = f'{root_dir}/nsddata/ppdata/subj0{subject}/func1pt8mm/roi'
    selected_areas_idxs = [] # enumerated indices for specific areas within froi
    # Dictionary to map fROI name to mask file name and selected areas
    # 1: OFA, 2: FFA-1, 3: FFA-2, 4:mTL-faces, 5: aTL-faces
    # 1: EBA, 2: FBA-1, 3: FBA-2, 4: mTL-bodies
    # 1: OPA, 2: PPA, 3: RSC
    # 1: OVWFA, 2: VWFA-1, 3: VWFA-2, 4: mfs-words, 5: mTL-words
    froi_info = {
        # NSD general
        'NSDGENERAL': ('nsdgeneral.nii.gz', [1]),
        # Early visual areas
        'V1': {'prf-visualrois.nii.gz': [1, 2]},
        'V1D': {'prf-visualrois.nii.gz': [1]},
        'V1V': {'prf-visualrois.nii.gz': [2]},
        'V2': {'prf-visualrois.nii.gz': [3, 4]},
        'V2D': {'prf-visualrois.nii.gz': [3]},
        'V2V': {'prf-visualrois.nii.gz': [4]},
        'V3': {'prf-visualrois.nii.gz': [5, 6]},
        'V3D': {'prf-visualrois.nii.gz': [5]},
        'V3V': {'prf-visualrois.nii.gz': [6]},
        'V4': {'prf-visualrois.nii.gz': [7]},
        # Eccentricity-based visual areas
        'ECC0PT5': {'prf-eccrois.nii.gz': [1]},
        'ECC0': {'prf-eccrois.nii.gz': [2]},
        'ECC1': {'prf-eccrois.nii.gz': [3]},
        'ECC2': {'prf-eccrois.nii.gz': [4]},
        'ECC4': {'prf-eccrois.nii.gz': [5]},
        'ECC4+': {'prf-eccrois.nii.gz': [6]},
        # Higher-level regions
        'OFA': ('floc-faces.nii.gz', [1]),
        'FFA_1': ('floc-faces.nii.gz', [2]),
        'FFA_2': ('floc-faces.nii.gz', [3]),
        'FFA': ('floc-faces.nii.gz', [2, 3]),
        'MTL_FACES': ('floc-faces.nii.gz', [4]),
        'ATL_FACES': ('floc-faces.nii.gz', [5]),
        'EBA': ('floc-bodies.nii.gz', [1]),
        'FBA_1': ('floc-bodies.nii.gz', [2]),
        'FBA_2': ('floc-bodies.nii.gz', [3]),
        'FBA': ('floc-bodies.nii.gz', [2, 3]),
        'MTL_BODIES': ('floc-bodies.nii.gz', [4]),
        'OPA': ('floc-places.nii.gz', [1]),
        'PPA': ('floc-places.nii.gz', [2]),
        'RSC': ('floc-places.nii.gz', [3]),
        'OVWFA': ('floc-words.nii.gz', [1]),
        'VWFA_1': ('floc-words.nii.gz', [2]),
        'VWFA_2': ('floc-words.nii.gz', [3]),
        'VWFA': ('floc-words.nii.gz', [2, 3]),
        'MFS_WORDS': ('floc-words.nii.gz', [4]),
        'MTL_WORDS': ('floc-words.nii.gz', [5])
    }
    if froi in froi_info.keys():
        mask_file_name = froi_info[froi][0]
        selected_areas_idxs = froi_info[froi][1]
    else:
        raise NotImplementedError(f'{froi} mask not found in NSD dataset')
    # Load all areas in froi
    froi_all_areas = nib.load(f'{mask_dir}/{mask_file_name}').get_fdata()
    # Select voxels in specific areas in fROI
    froi_mask = np.isin(froi_all_areas, selected_areas_idxs)
    # Threshold the mask according to tthreshold
    froi_tvals = None
    if tthreshold is not None:
        froi_mask, froi_tvals = threshold_froi_mask(root_dir, subject, froi, froi_mask, tthreshold)
    return froi_mask, froi_tvals
