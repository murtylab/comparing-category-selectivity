"""
Code to load Grill-Spector localizer
Stigliani, A., Weiner, K. S., & Grill-Spector, K. (2015). Temporal processing capacity in high-level visual cortex is domain specific. Journal of Neuroscience, 35(36), 12412-12424.
"""
import os
import glob

from .base_localizer import BaseLocalizerDataset

class GrillSpectorLocalizerDataset(BaseLocalizerDataset):
    """
    Dataset for Grill-Spector localizer
    """
    def __init__(self, root_dir, transform, image_class='all'):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        image_class = image_class.strip().lower()
        self.name = f'grillspector_{image_class}'
        self.image_files = self._get_image_files(image_class)
        # Contrasts
        self.froi_contrasts = {
            'FFA': {
                'preffered': 'faces',
                'nonpreffered': ['bodies', 'places', 'objects', 'characters', 'scrambledobjects']
            },
            'PPA': {
                'preffered': 'places',
                'nonpreffered': ['bodies', 'faces', 'objects', 'characters', 'scrambledobjects']
            },
            'EBA': {
                'preffered': 'bodies',
                'nonpreffered': ['faces', 'places', 'objects', 'characters', 'scrambledobjects']
            },
            'VWFA': {
                'preffered': 'characters',
                'nonpreffered': ['bodies', 'places', 'faces', 'objects', 'scrambledobjects']
            }
        }
        self.imageclasses = [
            'bodies', 'faces', 'places', 'objects', 'characters', 'scrambledobjects'
        ]

    def _get_image_files(self, image_class):
        """Function to get paths to image files"""
        root_dir = self.root_dir
        all_classes = []
        if image_class not in [
            'all', 'bodies', 'faces', 'places', 'objects', 'characters', 'scrambledobjects'
        ]:
            # Get all classes if requested image class not found
            all_classes = {fl.split('-')[0] for fl in os.listdir(root_dir)}
        if image_class == 'bodies':
            # body, limb
            image_files = sorted(
                glob.glob(f'{root_dir}/body-*.jpg') + \
                glob.glob(f'{root_dir}/limb-*.jpg')
            )
        elif image_class == 'faces':
            # adult, child
            image_files = sorted(
                glob.glob(f'{root_dir}/adult-*.jpg') + \
                glob.glob(f'{root_dir}/child-*.jpg')
            )
        elif image_class == 'characters':
            # word, number
            image_files = sorted(
                glob.glob(f'{root_dir}/word-*.jpg') + \
                glob.glob(f'{root_dir}/number-*.jpg')
            )
        elif image_class == 'places':
            # house, corridor
            image_files = sorted(
                glob.glob(f'{root_dir}/corridor-*.jpg') + \
                glob.glob(f'{root_dir}/house-*.jpg')
            )
        elif image_class == 'objects':
            # car, instrument
            image_files = sorted(
                glob.glob(f'{root_dir}/instrument-*.jpg') + \
                glob.glob(f'{root_dir}/car-*.jpg')
            )
        elif image_class == 'scrambledobjects':
            # scrambled
            image_files = sorted(
                glob.glob(f'{root_dir}/scrambled-*.jpg')
            )
        elif image_class == 'all':
            image_files = sorted(
                glob.glob(f'{root_dir}/*.jpg')
            )
        elif image_class in all_classes:
            image_files = sorted(
                glob.glob(f'{root_dir}/{image_class}-*.jpg')
            )
        else:
            raise ValueError(f'{image_class} images not implemented in localizer')
        return image_files

def get_localizer_dataset(root_dir, transform, image_class='all'):
    """
    Get Grill-Spector Localizer dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = GrillSpectorLocalizerDataset(root_dir, transform=transform, image_class=image_class)
    return dataset
