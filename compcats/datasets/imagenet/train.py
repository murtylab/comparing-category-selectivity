"""
Script to load the ImageNet validation dataset
"""
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from ffcv.loader import Loader, OrderOption

from compcats.models.utils.transform_utils import get_imagenet_train_transforms

def get_loader(root, transform=None, batch_size=768, num_workers=12, ffcv=True, **kwargs):
    """
    Get ImageNet training dataset
    """
    dataloader = None
    if transform is None:
        transform = get_imagenet_train_transforms(
            ffcv=ffcv, device=kwargs.get("device", "cuda:0")
        )
    if ffcv:
        train_path = Path(root)
        assert train_path.is_file(), f"Path {train_path} is not a file"
        order = OrderOption.RANDOM # if distributed else OrderOption.QUASI_RANDOM
        torch.cuda.synchronize()
        dataloader = Loader(
            train_path,
            batch_size=batch_size,
            num_workers=num_workers,
            os_cache=kwargs.get("os_cache", True),
            order=order,
            drop_last=True,
            pipelines=transform['pipeline'],
            distributed=False
        )
        dataloader_dict = {
            'decoder': transform['decoder'],
            'dataloader': dataloader,
        }
        # TODO: Implement distributed loading
    else:
        raise NotImplementedError("FFCV is the only supported data loader")
        dataset = get_dataset(root, transform)
        torch.cuda.synchronize()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, drop_last=False, shuffle=False
        )
    return dataloader_dict

def get_dataset(root, transform):
    """
    Get ImageNet training dataset
    """
    dataset = ImageFolder(root, transform=transform)
    return dataset
