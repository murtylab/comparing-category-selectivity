"""
Code to load Prince 2024 localizer
Prince, J. S., Alvarez, G. A., & Konkle, T. (2024). Contrastive learning explains the emergence and function of visual category-selective regions. Science Advances, 10(39), eadl1776.
"""
import glob

from .base_localizer import BaseLocalizerDataset

class Prince2024LocalizerDataset(BaseLocalizerDataset):
    """
    Dataset for Prince 2024 localizer
    """
    def __init__(self, root_dir, transform, image_class='all'):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        image_class = image_class.strip().lower()
        self.name = f'prince2024_{image_class}'
        self.image_files = self._get_image_files(image_class)
        # Contrasts
        self.froi_contrasts = {
            'FFA': {
                'preffered': 'faces',
                'nonpreffered': [
                    'bodies', 'scenes', 'objects', 'words', 'scrambledobjects', 'scrambledwords'
                ]
            },
            'PPA': {
                'preffered': 'scenes',
                'nonpreffered': [
                    'bodies', 'faces', 'objects', 'words', 'scrambledobjects', 'scrambledwords'
                ]
            },
            'EBA': {
                'preffered': 'bodies',
                'nonpreffered': [
                    'faces', 'scenes', 'objects', 'words', 'scrambledobjects', 'scrambledwords'
                ]
            },
            'VWFA': {
                'preffered': 'words',
                'nonpreffered': [
                    'bodies', 'faces', 'scenes', 'objects', 'scrambledobjects', 'scrambledwords'
                ]
            }
        }
        self.imageclasses = [
            'faces', 'bodies', 'scenes', 'objects', 'words', 'scrambledobjects', 'scrambledwords'
        ]

    def _get_image_files(self, image_class):
        """Function to get paths to image files"""
        root_dir = self.root_dir
        image_class = image_class.strip().lower()
        # Get all classes if not all images requested
        if image_class not in [
            'all', 'bodies', 'faces', 'scenes', 'objects', 'words', 'scrambled',
            'scrambledobjects', 'scrambledwords'
        ]:
            raise ValueError(f'{image_class} images not implemented in localizer')
        if image_class == 'all':
            image_files = sorted(
                glob.glob(f'{root_dir}/*/*.jpg')
            )
        elif image_class == 'faces':
            # faces
            image_files = sorted(
                glob.glob(f'{root_dir}/1-Faces/*.jpg')
            )
        elif image_class == 'bodies':
            # bodies
            image_files = sorted(
                glob.glob(f'{root_dir}/2-Bodies/*.jpg')
            )
        elif image_class == 'scenes':
            # scenes
            image_files = sorted(
                glob.glob(f'{root_dir}/3-Scenes/*.jpg')
            )
        elif image_class == 'words':
            # scenes
            image_files = sorted(
                glob.glob(f'{root_dir}/4-Words/*.jpg')
            )
        elif image_class == 'objects':
            # objects
            image_files = sorted(
                glob.glob(f'{root_dir}/5-Objects/*.jpg')
            )
        elif image_class == 'scrambledobjects':
            # scrambled objects
            image_files = sorted(
                glob.glob(f'{root_dir}/6-ScrambledObjects/*.jpg')
            )
        elif image_class == 'scrambledwords':
            # scrambled words
            image_files = sorted(
                glob.glob(f'{root_dir}/7-ScrambledWords/*.jpg')
            )
        elif image_class == 'scrambled':
            # scrambled objects and scrambled words
            image_files = sorted(
                glob.glob(f'{root_dir}/6-ScrambledObjects/*.jpg') + \
                glob.glob(f'{root_dir}/7-ScrambledWords/*.jpg')
            )
        else:
            raise ValueError(f'{image_class} images not implemented in localizer')
        return image_files


def get_localizer_dataset(root_dir, transform, image_class='all'):
    """
    Get Prince2024 Localizer dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = Prince2024LocalizerDataset(root_dir, transform=transform, image_class=image_class)
    return dataset
