"""
Code to load Konkle localizer
Cohen, M. A., Alvarez, G. A., Nakayama, K., & Konkle, T. (2017). Visual search for object categories is predicted by the representational architecture of high-level visual cortex. Journal of neurophysiology, 117(1), 388-402.
"""
import os
import glob

from .base_localizer import BaseLocalizerDataset

class KonkleLocalizerDataset(BaseLocalizerDataset):
    """
    Dataset for Konkle localizer
    """
    def __init__(self, root_dir, transform, image_class='all'):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        image_class = image_class.strip().lower()
        self.name = f'konkle_{image_class}'
        self.image_files = self._get_image_files(self.name, self.root_dir)
        # Contrasts
        self.froi_contrasts = {
            'FFA': {
                'preffered': 'faces',
                'nonpreffered': ['bodies', 'scenes', 'objects']
            },
            'PPA': {
                'preffered': 'scenes',
                'nonpreffered': ['bodies', 'faces', 'objects']
            },
            'EBA': {
                'preffered': 'bodies',
                'nonpreffered': ['faces', 'scenes', 'objects']
            },
        }

    def _get_image_files(self, image_class, root_dir):
        """Function to get paths to image files"""
        all_classes = []
        for stim_class in os.listdir(root_dir):
            if not stim_class.startswith('.'): # Ignore files that start with .
                all_classes.append(stim_class)
        all_classes = set(all_classes)
        if image_class == 'bodies':
            # bodies
            image_files = sorted(
                glob.glob(f'{root_dir}/bodies/bodies*.jpg')
            )
        elif image_class == 'faces':
            # faces
            image_files = sorted(
                glob.glob(f'{root_dir}/faces/faces*.jpg')
            )
        elif image_class == 'scenes':
            # buildings
            image_files = sorted(
                glob.glob(f'{root_dir}/buildings/buildings*.jpg')
            )
        elif image_class == 'objects':
            # cars, chairs, hammers, phones
            # One option is to choose only cars for consistency with 
            # Grill-Spector localizer, but this is not implemented here
            image_files = sorted(
                glob.glob(f'{root_dir}/cars/cars*.jpg') + \
                glob.glob(f'{root_dir}/chairs/chairs*.jpg') + \
                glob.glob(f'{root_dir}/hammers/hammers*.jpg') + \
                glob.glob(f'{root_dir}/phones/phones*.jpg')
            )
        elif image_class in all_classes:
            image_files = sorted(
                glob.glob(f'{root_dir}/{image_class}/{image_class}*.jpg')
            )
        elif image_class == 'all':
            image_files = sorted(
                glob.glob(f'{root_dir}/*/*.jpg')
            )
        else:
            raise ValueError(f'{image_class} images not implemented in localizer')
        return image_files


def get_localizer_dataset(root_dir, transform, image_class='all'):
    """
    Get Konkle Localizer dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = KonkleLocalizerDataset(root_dir, transform=transform, image_class=image_class)
    return dataset
