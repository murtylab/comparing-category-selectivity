"""
Script to load the AFHQv2 dataset
https://github.com/clovaai/stargan-v2/
"""
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

try:
    from ffcv.loader import Loader, OrderOption
except ImportError:
    # These libraries are optional, silently handle import error
    pass

from ...models.utils.transform_utils import \
    get_contrastive_train_transforms, get_contrastive_val_transforms
from ...models.utils.transform_utils import \
    get_imagenet_train_transforms, get_imagenet_val_transforms

def get_train_loader(root, transform=None, batch_size=768, num_workers=12, ffcv=True, **kwargs):
    """
    Get AFHQv2 training dataset
    """
    dataloader = None
    if isinstance(transform, str):
        transform = transform.lower()
    if transform is None or transform == 'contrastive':
        transform = get_contrastive_train_transforms(
            ffcv=ffcv, device=kwargs.get("device", "cuda:0")
        )
    elif transform == 'imagenet':
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
        dataset = get_train_dataset(root, transform)
        torch.cuda.synchronize()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, drop_last=False, shuffle=False
        )
    return dataloader_dict

def get_train_dataset(root, transform):
    """
    Get AFHQv2 training dataset
    """
    dataset = ImageFolder(root, transform=transform)
    return dataset

def get_val_loader(root, transform=None, batch_size=768, num_workers=12, ffcv=True, **kwargs):
    """
    Get AFHQv2 validation dataset
    """
    dataloader = None
    if isinstance(transform, str):
        transform = transform.lower()
    drop_last = False
    if transform is None or transform == 'contrastive':
        transform = get_contrastive_val_transforms(
            ffcv=ffcv, device=kwargs.get("device", "cuda:0")
        )
        drop_last = True
    elif transform == 'imagenet':
        transform = get_imagenet_val_transforms(
            ffcv=ffcv, device=kwargs.get("device", "cuda:0"), resolution=256
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
            drop_last=drop_last,
            pipelines=transform,
            distributed=False
        )
        # TODO: Implement distributed loading
    else:
        raise NotImplementedError("FFCV is the only supported data loader")
        dataset = get_val_dataset(root, transform)
        torch.cuda.synchronize()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, drop_last=False, shuffle=False
        )
    return dataloader

def get_val_dataset(root, transform):
    """
    Get AFHQv2 validation dataset
    """
    dataset = ImageFolder(root, transform=transform)
    return dataset
