"""
ResNet-50
https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
A Study of Face Obfuscation in ImageNet
Yang, K., Yau, J. H., Fei-Fei, L., Deng, J., & Russakovsky, O. (2022, June). A study of face obfuscation in imagenet. In International Conference on Machine Learning (pp. 25313-25330). PMLR.
https://huggingface.co/kaiyuy/imagenet-blur-pretrained-models/tree/main
https://github.com/princetonvisualai/imagenet-face-obfuscation/tree/main
"""
from collections import OrderedDict
import torch
import torchvision
from huggingface_hub import hf_hub_download

from .basemodel import BaseModel
from .utils import transform_utils

def get_model(**kwargs):
    model = ResNet50FaceObfuscation(**kwargs)
    return model

class ResNet50FaceObfuscation(BaseModel):
    """
    class for the model ResNet50 trained on ImageNet
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize=(224, 224)
        )

    def _load_model(self):
        """
        Load ResNet50
        """
        self.checkpoints_dir = hf_hub_download(repo_id="kaiyuy/imagenet-blur-pretrained-models", filename="resnet50_blurred_ILSVRC.pth")
        checkpoint = torch.load(self.checkpoints_dir)
        state_dict = OrderedDict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        model = torchvision.models.resnet50(weights=None)
        model.load_state_dict(state_dict)
        model_name = 'resnet50faceobfuscation'
        return model, model_name
