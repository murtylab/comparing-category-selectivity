"""Utils for Image transforms"""
import numpy as np
import torch
from torchvision.transforms import v2
# PyTorch recommends using v2 for transforms
# https://pytorch.org/vision/main/transforms.html#v1-or-v2-which-one-should-i-use

# ffcv
try:
    import ffcv.transforms as ffcv_transforms
    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
    from ffcv.fields.basics import IntDecoder
except ImportError:
    # These libraries are optional, silently handle import error
    pass

from compcats import constants

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

def get_transforms(
    mean=None, std=None, resize=(224, 224)
):
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
    if mean is None and std is None:
        transforms.transforms.append(v2.Normalize(
            constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD
        ))
    else:
        transforms.transforms.append(v2.Normalize(mean, std))

    return transforms

def get_clip_transforms(
    mean=None, std=None, resize=(224, 224)
):
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
    if isinstance(resize, int):
        resize = (resize, resize)
    transforms.transforms.append(
        v2.Resize(resize, interpolation=v2.InterpolationMode.BICUBIC)
    )

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
            constants.OPENAI_CLIP_MEAN, constants.OPENAI_CLIP_STD
        ))
    elif mean is not None and std is not None:
        transforms.transforms.append(v2.Normalize(mean, std))
    else:
        raise ValueError("Both mean and std should be provided or None")

    return transforms

def get_imagenet_val_transforms(ffcv=False, resolution=256, device='cuda:0'):
    """
    Get transforms for ImageNet validation dataset
    """
    transforms = None
    if ffcv:
        cropper = CenterCropRGBImageDecoder(
            (resolution, resolution), ratio=constants.IMAGENET_DEFAULT_VAL_CROP_RATIO
        )
        image_pipeline = [
            cropper,
            ffcv_transforms.ToTensor(),
            ffcv_transforms.ToDevice(torch.device(device), non_blocking=True),
            ffcv_transforms.ToTorchImage(),
            ffcv_transforms.NormalizeImage(
                np.array(constants.IMAGENET_DEFAULT_MEAN) * 255,
                np.array(constants.IMAGENET_DEFAULT_STD) * 255,
                np.float16
            )
        ]
        label_pipeline = [
            IntDecoder(),
            ffcv_transforms.ToTensor(),
            ffcv_transforms.Squeeze(),
            ffcv_transforms.ToDevice(torch.device(device),
            non_blocking=True)
        ]
        transforms = {
            'image': image_pipeline,
            'label': label_pipeline
        }
    else:
        # Resize to 256x256
        # CenterCrop to 224x224
        # Normalize to ImageNet mean and std
        transforms = v2.Compose([
            v2.Resize((256, 256)),
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD)
        ])
    return transforms

def get_imagenet_train_transforms(ffcv=False, device='cuda:0'):
    """
    Get transforms for ImageNet training dataset
    """
    transforms = None
    default_imagenet_size = 224
    if ffcv:
        train_image_decoder = RandomResizedCropRGBImageDecoder((
            default_imagenet_size, default_imagenet_size
        ))
        image_pipeline = [
            train_image_decoder,
            ffcv_transforms.RandomHorizontalFlip(),
            ffcv_transforms.ToTensor(),
            ffcv_transforms.ToDevice(torch.device(device), non_blocking=True),
            ffcv_transforms.ToTorchImage(),
            ffcv_transforms.NormalizeImage(
                np.array(constants.IMAGENET_DEFAULT_MEAN) * 255,
                np.array(constants.IMAGENET_DEFAULT_STD) * 255,
                np.float16
            )
        ]
        label_pipeline = [
            IntDecoder(),
            ffcv_transforms.ToTensor(),
            ffcv_transforms.Squeeze(),
            ffcv_transforms.ToDevice(torch.device(device), non_blocking=True)
        ]
        transforms = {
            'decoder': train_image_decoder,
            'pipeline': {
                'image': image_pipeline,
                'label': label_pipeline
            }
        }
    else:
        # RandomResizedCrop to 224x224
        # RandomHorizontalFlip
        # Normalize to ImageNet mean and std
        transforms = v2.Compose([
            v2.RandomResizedCrop((224, 224)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float16, scale=True),
            v2.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD)
        ])
    return transforms

# Transforms for datasets for contrastive objective

def get_contrastive_train_transforms(ffcv=False, device='cuda:0'):
    """
    Get transforms for ImageNet training dataset
    """
    transforms = None
    default_size = 224
    if ffcv:
        train_image_decoder = RandomResizedCropRGBImageDecoder((
            default_size, default_size
        ), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))
        image_pipeline = [
            train_image_decoder,
            ffcv_transforms.RandomHorizontalFlip(flip_prob=0.5),
            ffcv_transforms.RandomTranslate(padding=int(224/4)),
            ffcv_transforms.RandomBrightness(magnitude=0.5, p=0.05),
            ffcv_transforms.RandomContrast(magnitude=0.5, p=0.05),
            ffcv_transforms.RandomSaturation(magnitude=0.5, p=0.05),
            ffcv_transforms.ToTensor(),
            ffcv_transforms.ToDevice(torch.device(device), non_blocking=True),
            ffcv_transforms.ToTorchImage(),
            ffcv_transforms.NormalizeImage(
                np.array(constants.IMAGENET_DEFAULT_MEAN) * 255,
                np.array(constants.IMAGENET_DEFAULT_STD) * 255,
                np.float16
            )
        ]
        label_pipeline = [
            IntDecoder(),
            ffcv_transforms.ToTensor(),
            ffcv_transforms.Squeeze(),
            ffcv_transforms.ToDevice(torch.device(device), non_blocking=True)
        ]
        transforms = {
            'decoder': train_image_decoder,
            'pipeline': {
                'image': image_pipeline,
                'label': label_pipeline
            }
        }
    else:
        # RandomResizedCrop to 224x224
        # RandomHorizontalFlip
        # Normalize to ImageNet mean and std
        transforms = v2.Compose([
            v2.RandomResizedCrop((224, 224)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float16, scale=True),
            v2.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD)
        ])
    return transforms

def get_contrastive_val_transforms(ffcv=False, device='cuda:0'):
    """
    Get transforms for ImageNet validation dataset
    """
    transforms = None
    if ffcv:
        cropper = CenterCropRGBImageDecoder((224, 224), ratio=224/256)
        image_pipeline = [
            cropper,
            ffcv_transforms.ToTensor(),
            ffcv_transforms.ToDevice(torch.device(device), non_blocking=True),
            ffcv_transforms.ToTorchImage(),
            ffcv_transforms.NormalizeImage(
                np.array(constants.IMAGENET_DEFAULT_MEAN) * 255,
                np.array(constants.IMAGENET_DEFAULT_STD) * 255,
                np.float16
            )
        ]
        label_pipeline = [
            IntDecoder(),
            ffcv_transforms.ToTensor(),
            ffcv_transforms.Squeeze(),
            ffcv_transforms.ToDevice(torch.device(device),
            non_blocking=True)
        ]
        transforms = {
            'image': image_pipeline,
            'label': label_pipeline
        }
    else:
        # Resize to 256x256
        # CenterCrop to 224x224
        # Normalize to ImageNet mean and std
        transforms = v2.Compose([
            v2.Resize((256, 256)),
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD)
        ])
    return transforms
