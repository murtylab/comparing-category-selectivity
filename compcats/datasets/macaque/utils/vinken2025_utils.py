# Import libraries
import os
import h5py
import numpy as np
import pandas as pd

from ...fmri.utils import nsd_ds_utils

NSDID_MISSING = 16422

def process_mat_data(mat_path):
    """
    Process mat data to extract reliabilities and responses
    Args:
        mat_data(dict): Loaded mat data
    Returns:
        dict: reliabilities
        dict: responses
    """
    f = h5py.File(mat_path, 'r')
    refs = f['#refs#']
    data = f['data']
    reliabilities = data['rel'][:][0]
    stimids = data['stimid'][:][0] - 1 # Convert to zero-based indexing
    stimnames = []
    nsdids = []
    stimname_keys = data['stimname'][:]
    for key in stimname_keys:
        encoded_value = refs[key[0]]
        stimname = ''.join([chr(c[0]) for c in encoded_value])
        # Clean stimname
        stimname = stimname.strip()
        stimname = stimname.rsplit('\\', maxsplit=1)[-1]
        stimnames.append(stimname)
        nsdid = stimname.split('_')[-1].split('.')[0].replace('nsd', '')
        nsdids.append(int(nsdid) - 1) # Convert to zero-based indexing
    responses = data['trialrates'][:]
    if 'V1' in mat_path:
        froi, hemisphere, monkey_name = \
            os.path.basename(mat_path).rsplit('.', maxsplit=1)[0].split('_')
    else:
        froi, hemisphere, _, monkey_name = \
            os.path.basename(mat_path).rsplit('.', maxsplit=1)[0].split('_')
    processed_data = {
        'froi': froi.upper(),
        'hemisphere': hemisphere.upper(),
        'monkey_name': monkey_name.upper(),
        'reliabilities': reliabilities,
        'responses': responses,
        'stimids': stimids,
        'stimnames': stimnames,
        'nsdids': nsdids,
    }
    return processed_data

def calculate_num_repetitions(stimdata):
    """
    Calculate number of repetitions for each stimulus
    Args:
        stimdata(np.ndarray): Stimulus data (total_sessions_possible x num_units)
    Returns:
        int: number of repetitions
    """
    num_repetitions = 0
    for repidx in range(stimdata.shape[0]):
        repmean = stimdata[repidx, :].mean()
        if not np.isnan(repmean):
            num_repetitions += 1
    return num_repetitions

def preprocess_vinken2025_data(vinken2025_dir, nsd_dir):
    """
    Preprocess Vinken2025 dataset
    Args:
        vinken2025_dir(str): Path to Vinken2025 dataset
    """
    processed_dir = os.path.join(vinken2025_dir, 'vinken2025_processed')
    # Reuturn if preprocessed data exists
    if os.path.exists(processed_dir):
        return
    os.makedirs(processed_dir, exist_ok=True)
    # Check if raw data exists
    assert os.path.exists(vinken2025_dir), f'Download Vinken2025 dataset to {vinken2025_dir} first'
    # Stimuli info
    stim_info = []
    shared1000_nsdids = nsd_ds_utils.get_subset_nsdids(nsd_dir, subset=1000)
    shared485_nsdids = nsd_ds_utils.get_subset_nsdids(nsd_dir, subset=485)
    shared515_nsdids = nsd_ds_utils.get_subset_nsdids(nsd_dir, subset=515)
    for idx, nsdid in enumerate(shared1000_nsdids):
        stim_info.append({
            'stim_index': idx,
            'nsdid': nsdid,
            'allmonkeys': nsdid != NSDID_MISSING,
            'shared1000': True,
            'shared485': nsdid in shared485_nsdids,
            'shared515': nsdid in shared515_nsdids,
        })
    # Storing results
    reliabilities = {}
    responses = {}
    splithalfs = {}
    # load all mat files
    mat_file_names = [f for f in os.listdir(vinken2025_dir) if f.endswith('.mat')]
    processed_mats = {}
    for mat_file_name in mat_file_names:
        mat_path = os.path.join(vinken2025_dir, mat_file_name)
        processed_mat = process_mat_data(mat_path)
        key = (processed_mat['froi'], processed_mat['monkey_name'])
        processed_mats[key] = processed_mat
    # Loop over each key and collate data
    for key, processed_mat in processed_mats.items():
        reliabilities[key] = processed_mat['reliabilities']
        resps = processed_mat['responses']
        responses[key] = resps
        stim_nreps = {}
        nsdids = processed_mat['nsdids']
        for idx, nsdid in enumerate(nsdids):
            if nsdid == NSDID_MISSING:
                stim_nreps[nsdid] = 0
            else:
                stim_data = resps[:, idx, :]
                stim_nreps[nsdid] = calculate_num_repetitions(stim_data)
        # Add a column for number of repetitions in stim_info by matching nsdids
        for stim in stim_info:
            nsdid = stim['nsdid']
            stim['num_repetitions_' + processed_mat['monkey_name']] = stim_nreps.get(nsdid, 0)
        splithalfs[key] = None # TODO: implement split-half reliabilities
    # Save processed data as npz and csv, the npy should be dictionary with keys from the processed_mats
    reliabilities_path = os.path.join(processed_dir, 'reliabilities.npy')
    responses_path = os.path.join(processed_dir, 'responses.npy')
    splithalfs_path = os.path.join(processed_dir, 'splithalfs.npy')
    stiminfo_path = os.path.join(processed_dir, 'stim_info.csv')
    np.save(reliabilities_path, reliabilities, allow_pickle=True)
    np.save(responses_path, responses, allow_pickle=True)
    np.save(splithalfs_path, splithalfs, allow_pickle=True)
    stim_info_df = pd.DataFrame(stim_info)
    stim_info_df.to_csv(stiminfo_path, index=False)


# Write function to get split-half reliabilities for each subject x froi
