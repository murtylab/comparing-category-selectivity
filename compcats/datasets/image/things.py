"""Code to load the THINGS dataset"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class THINGSDataset(Dataset):
    """
    Dataset for THINGS database
    https://osf.io/jum2f
    """
    def __init__(self, root_dir, transform):
        self.name = 'things'
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths_csv_path = f'{self.root_dir}/Metadata/Image-specific/image_paths.csv'
        self.image_paths_csv = pd.read_csv(self.image_paths_csv_path, header=None)
        self.image_paths = list(self.image_paths_csv.iloc[:, 0])
        # Capitalize the first I in each image path
        # Path is Images/class/img_name.jpg
        self.image_paths = ['I' + s[1:] for s in self.image_paths]
        # image_class = image_class.strip().lower()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_full_path = os.path.join(self.root_dir, img_path)
        # Example - {root_dir}/Images/aardvark/aardvark_02s.jpg
        # Image class is the folder name
        # image_class = img_name.split('/')[-1].split('_')[0]
        image = Image.open(img_full_path)
        if self.transform:
            image = self.transform(image)
        return image #, img_path

def get_stimuli_dataset(root_dir, transform):
    """
    Get THINGS dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = THINGSDataset(root_dir, transform=transform)
    return dataset
