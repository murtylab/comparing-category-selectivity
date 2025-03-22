"""
Script to load the ImageNet validation dataset
"""
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from ffcv.loader import Loader, OrderOption

from compcats.models.utils.transform_utils import get_imagenet_val_transforms

def get_loader(root, transform=None, batch_size=768, num_workers=12, ffcv=False, **kwargs):
    """
    Get ImageNet validation dataset
    """
    dataloader = None
    if transform is None:
        transform = get_imagenet_val_transforms(
            ffcv=ffcv, device=kwargs.get("device", "cuda:0")
        )
    if ffcv:
        val_path = Path(root)
        assert val_path.is_file(), f"Path {val_path} is not a file"
        torch.cuda.synchronize()
        dataloader = Loader(
            val_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines=transform,
            distributed=False
        )
        # TODO: Implement distributed loading
    else:
        dataset = get_dataset(root, transform)
        torch.cuda.synchronize()
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
