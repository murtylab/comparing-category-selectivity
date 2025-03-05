"""
Code to load Grill-Spector localizer
Vinken, K., Prince, J. S., Konkle, T., & Livingstone, M. S. (2023). The neural code for “face cells” is not face-specific. Science Advances, 9(35), eadg1736.
"""
import os
import numpy as np
import scipy
from PIL import Image
from torch.utils.data import Dataset

class Vinkenetal2023Dataset(Dataset):
    """
    Dataset for Vinken et al 2023 localizer
    data/images.mat:
        - imarray: WxHxCxI matrix for I images with width W, hight H, color
                    channels C
        NOTE: imarray is actually HxWxCxI instead of WxHxCxI (incorrect in documentation)
        Link: https://osf.io/ymszt/
        - imsets: vector with I image condition indices for 3 conditions.
        - imsets_labels: cell array with the 3 image condition labels.
    The labels are ['human_faces', 'monkey_faces', 'nonfaces']
    The images are 224x224x3
    """
    def __init__(self, root_dir, transform, image_class='all'):
        self.root_dir = root_dir
        # Load the mat
        self.mat = scipy.io.loadmat(os.path.join(root_dir, 'images.mat'))
        # Get the labels & idxs
        self.labels_arr = self.mat['imsets_labels'][0]
        self.labels_names = [str(l[0]) for l in self.labels_arr]
        # Labels are ['human_faces', 'monkey_faces', 'nonfaces']
        self.labels_idxs = {l: i+1 for i, l in enumerate(self.labels_names)}
        self.image_labels = self.mat['imsets'][0] # This will be a 1379 length array
        self.all_images = self.mat['imarray'] # This will be a 224x224x3x1379 array, HxWxCxI
        # Transforms
        self.transform = transform
        # Image class
        image_class = image_class.strip().lower()
        self.name = image_class
        self.images = self._get_images(self.name, self.image_labels, self.all_images)

    def _get_images(self, image_class, image_labels, all_images):
        """Function to get paths to image files"""
        if image_class == 'humanfaces':
            image_idxs = np.where(image_labels == self.labels_idxs['human_faces'])[0]
        elif image_class == 'monkeyfaces':
            image_idxs = np.where(image_labels == self.labels_idxs['monkey_faces'])[0]
        elif image_class == 'nonfaces':
            image_idxs = np.where(image_labels == self.labels_idxs['nonfaces'])[0]
        elif image_class == 'faces':
            # human_faces, monkey_faces
            image_idxs = np.where((image_labels == self.labels_idxs['human_faces']) | \
                                    (image_labels == self.labels_idxs['monkey_faces']))[0]
        elif image_class == 'all':
            # human_faces, monkey_faces, nonfaces
            image_idxs = np.where((image_labels == self.labels_idxs['human_faces']) | \
                                    (image_labels == self.labels_idxs['monkey_faces']) | \
                                    (image_labels == self.labels_idxs['nonfaces']))[0]
        else:
            raise ValueError(f'{image_class} images not implemented in localizer')
        images = all_images[:, :, :, image_idxs]
        return images

    def __len__(self):
        return self.images.shape[-1]

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        image = Image.fromarray(self.images[:, :, :, idx]) # HxWxC
        if self.transform:
            image = self.transform(image)
        return image

def get_stimuli_dataset(root_dir, transform, image_class='all'):
    """
    Get Vinken et al 2023 dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = Vinkenetal2023Dataset(root_dir, transform=transform, image_class=image_class)
    return dataset
