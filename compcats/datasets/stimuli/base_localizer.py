import os
from PIL import Image

from torch.utils.data import Dataset

class BaseLocalizerDataset(Dataset):
    """
    Base class for localizer datasets
    """
    def __init__(self):
        super().__init__()
        self.root_dir = None
        self.image_files = None
        self.name = None
        self.transform = None
        self.froi_contrasts = None
    
    def get_pref_nonpref_categories(self, froi):
        """
        Get preferred & non-preferred
        """
        froi = froi.upper()
        pref_category = self.froi_contrasts[froi]['preffered']
        nonpref_categories = self.froi_contrasts[froi]['nonpreffered']
        return pref_category, nonpref_categories

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        # Convert grayscale image to RGB
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image