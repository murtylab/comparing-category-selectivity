"""Code to load the ImageNet-Sketch dataset"""
import datasets
from torch.utils.data import Dataset

class ImageNetSketchDataset(Dataset):
    """
    Dataset for ImageNet Sketch
    https://huggingface.co/datasets/songweig/imagenet_sketch
    """
    def __init__(self, transform):
        self.name = 'imagenetsketch'
        self.transform = transform
        hf_dataset = datasets.load_dataset('imagenet_sketch', trust_remote_code=True, split='train')
        self.hf_dataset = hf_dataset.with_format('torch')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]['image']
        label = self.hf_dataset[idx]['label'].item()
        if self.transform:
            image = self.transform(image)
        return image, label

def get_stimuli_dataset(transform):
    """
    Get ImageNet Sketch dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = ImageNetSketchDataset(transform=transform)
    return dataset
