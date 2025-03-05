"""
This module provides functionality to download and load the Murty185 dataset, including both stimuli images and betas data.
Submodules:
- download_murty185_dataset: Downloads and extracts the Murty185 dataset.
- get_murty185_betas: Retrieves betas data for a specified subject and functional region of interest (froi).
- Murty185Betas: A PyTorch Dataset class for loading Murty185 betas data.
- Murty185Stimuli: A PyTorch Dataset class for loading stimuli images from the Murty185 dataset.
- get_murty185stimuli_dataset: Returns a Murty185Stimuli dataset instance.
- get_murty185betas_dataset: Returns a Murty185Betas dataset instance.
"""

import os
import pickle
import gdown
import zipfile
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def download_murty185_dataset(dir):
    # Skip download if already exists
    if os.path.exists(os.path.join(dir, 'all_data.pickle')) and os.path.exists(os.path.join(dir, 'images_185')):
        return None
    os.makedirs(dir, exist_ok=True)
    file_url_id = '1LyVcLxDOA_Lti8pw6KCBPQf5iLJ0sDiE'
    zip_path = f'{dir}/data_murty185.zip'
    gdown.download(id=file_url_id, output=zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dir)
    
    # Move extracted subfolders to dir
    extracted_folder = os.path.join(dir, 'data_murty185')
    for item in os.listdir(extracted_folder):
        s, d = os.path.join(extracted_folder, item), os.path.join(dir, item)
        if os.path.isdir(s):
            os.rename(s, d) if not item.startswith(('_', '.')) else os.rmdir(s)
        else:
            os.replace(s, d) if not item.startswith(('_', '.')) else os.remove(s)
    os.rmdir(extracted_folder)
    os.remove(zip_path)
    return None


class Murty185Stimuli(Dataset):
    """
    Dataset for loading the stimuli in the Murty185 dataset
    """
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.root_dir = root_dir
        download_murty185_dataset(self.root_dir)
        self.transform = transform
        self.name = 'murty185'
        images_dir = f'{root_dir}/images_185'
        self.image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


def get_stimuli_dataset(root_dir, transform):
    """
    Get Murty185 stimuli dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = Murty185Stimuli(root_dir, transform=transform)
    return dataset


def get_betas(root_dir, subject, froi, voxels=False):
    froi = froi.lower()
    assert subject in [-1, 1, 2, 3, 4], 'Invalid subject' 
    assert froi in ['ffa', 'ppa', 'eba'], 'Invalid froi'
    download_murty185_dataset(root_dir)
    pickle_data = pickle.load(open(f'{root_dir}/all_data.pickle', 'rb'))
    data = np.concatenate(
        (pickle_data[f'p{subject}'][f'r{froi}'], pickle_data[f'p{subject}'][f'l{froi}']), 
        axis=1)
    if not voxels:
        data = data.mean(axis=1)
    # Return 2 vars for compatibility with other datasets
    return data, None
