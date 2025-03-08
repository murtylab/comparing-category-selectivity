"""Utils for Image transforms"""
import numpy as np
import torch
from torchvision.transforms import v2
# PyTorch recommends using v2 for transforms
# https://pytorch.org/vision/main/transforms.html#v1-or-v2-which-one-should-i-use

class SquarePad:
    """
    Updated from:
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        fill = 0 # Default value
        # Other strategies to pad can also be used
        # https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Pad.html
        return v2.functional.pad(image, padding, fill, 'constant')

def get_transforms(mean, std, resize=(224, 224)):
    """
    Get standard transforms for the model
    """
    # Init Compose
    # Square Padding: Pad shorter side to longer side
    transforms = v2.Compose([
        SquarePad(),
    ])

    # Resize: Resize to input size
    if isinstance(resize, int):
        # Handle int size, as int only resizes the smaller dimension
        transforms.transforms.append(v2.Resize((resize, resize)))
    elif isinstance(resize, tuple):
        transforms.transforms.append(v2.Resize(resize))
    elif isinstance(resize, list):
        transforms.transforms.append(v2.Resize(resize))
    elif isinstance(resize, v2._geometry.Resize):
        transforms.transforms.append(resize)
    else:
        raise ValueError(f'Incorrect resize type {type(resize)}')
    
    # ToImage: Convert to Image (subclass of Tensor)
    transforms.transforms.append(v2.ToImage())

    # ToDtype: Convert to FloatTensor and scale to [0, 1]
    # Here Dtype is mimicking (deparcated) ToTensor
    # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    transforms.transforms.append(v2.ToDtype(dtype=torch.float32, scale=True))

    # Normalize: Normalize with mean and std
    if mean is not None and std is not None:
        transforms.transforms.append(v2.Normalize(mean, std))
    
    return transforms

def get_clip_transforms(mean=None, std=None, resize=None):
    """
    Get transforms for CLIP models
    mean and std are the same across models, only resize is different
    https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
    mean, std are not used in CLIP models, but added for consistency
    """
    # Init Compose
    # Square Padding: Pad shorter side to longer side
    transforms = v2.Compose([
        SquarePad(),
    ])

    # Resize: Resize to input size
    assert isinstance(resize, int), "Only int resize is supported for CLIP models"
    transforms.transforms.append(v2.Resize((resize, resize),
                                           interpolation=v2.InterpolationMode.BICUBIC))

    # RGB: Convert to RGB
    transforms.transforms.append(v2.RGB())

    # ToImage: Convert to Image (subclass of Tensor)
    transforms.transforms.append(v2.ToImage())

    # ToDtype: Convert to FloatTensor and scale to [0, 1]
    # Here Dtype is mimicking (deparcated) ToTensor
    # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    transforms.transforms.append(v2.ToDtype(dtype=torch.float32, scale=True))

    # Normalize: Normalize with mean and std
    # https://github.com/openai/CLIP/blob/main/clip/clip.py#L85C19-L85C92
    if mean is None and std is None:
        transforms.transforms.append(v2.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ))
    elif mean is not None and std is not None:
        transforms.transforms.append(v2.Normalize(mean, std))
    else:
        raise ValueError("Both mean and std should be provided or None")
    
    return transforms
