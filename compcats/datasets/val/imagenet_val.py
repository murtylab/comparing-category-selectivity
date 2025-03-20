"""
Script to load the ImageNet validation dataset
"""
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_loader(root, transform, batch_size=32, num_workers=4):
    """
    Get ImageNet validation dataset
    """
    dataset = get_dataset(root, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=False, shuffle=False
    )
    return dataloader

def get_dataset(root, transform):
    """
    Get ImageNet validation dataset
    """
    dataset = ImageFolder(root, transform=transform)
    return dataset
