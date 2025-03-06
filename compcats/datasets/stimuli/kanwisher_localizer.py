"""
Code to load Kanwisher localizer
Julian, J. B., Fedorenko, E., Webster, J., & Kanwisher, N. (2012). An algorithmic method for functionally defining regions of interest in the ventral visual pathway. Neuroimage, 60(4), 2357-2364.
"""
import glob

from .base_localizer import BaseLocalizerDataset

class KanwisherLocalizerDataset(BaseLocalizerDataset):
    """
    Dataset for Kanwisher localizer
    """
    def __init__(self, root_dir, transform, image_class='all'):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        image_class = image_class.strip().lower()
        if image_class == 'bodies':
            self.name = 'bodies'
            self.image_files = sorted(glob.glob(f'{root_dir}/Bodies/*.png'))
        elif image_class == 'faces':
            self.name = 'faces'
            self.image_files = sorted(glob.glob(f'{root_dir}/Faces/*.png'))
        elif image_class == 'objects':
            self.name = 'objects'
            self.image_files = sorted(glob.glob(f'{root_dir}/Objects/*.png'))
        elif image_class == 'scenes':
            self.name = 'scenes'
            self.image_files = sorted(glob.glob(f'{root_dir}/Scenes/*.png'))
        elif image_class == 'scrambled_objects':
            self.name = 'scrambledobjects'
            self.image_files = sorted(glob.glob(f'{root_dir}/scrambled15G/*.png'))
        elif image_class == 'all':
            self.name = 'all'
            self.image_files = sorted(glob.glob(f'{root_dir}/**/*.png'))
        else:
            raise ValueError(f'{image_class} images not implemented in localizer')
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

def get_localizer_dataset(root_dir, transform, image_class='all'):
    """
    Get Kanwisher Localizer dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = KanwisherLocalizerDataset(root_dir, transform=transform, image_class=image_class)
    return dataset
