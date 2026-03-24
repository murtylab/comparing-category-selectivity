"""Code to load a custom dataset"""
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Dataset for custom database
    """
    def __init__(self, root_dir, image_names, dataset_name, transform):
        self.name = dataset_name
        self.root_dir = root_dir
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_full_path = os.path.join(self.root_dir, img_name)
        # Example - {root_dir}/Images/aardvark/aardvark_02s.jpg
        # Image class is the folder name
        # image_class = img_name.split('/')[-1].split('_')[0]
        image = Image.open(img_full_path)
        if self.transform:
            image = self.transform(image)
        return image #, img_path

def get_stimuli_dataset(root_dir, image_names, dataset_name, transform):
    """
    Get Custom dataset
    Args:
        root_dir(str): Root directory
        image_names(list): List of image names
        dataset_name(str): Name of the dataset
        transform(torchvision.transforms): Image transformations
    """
    dataset = CustomDataset(root_dir, image_names, dataset_name, transform=transform)
    return dataset
