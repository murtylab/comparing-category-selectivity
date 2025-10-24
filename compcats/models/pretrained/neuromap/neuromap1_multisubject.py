"""
NeuroMap 1.0 single-subject model.
Reference:
Kushal, Mayukh, etc
"""
import torch
from torch import nn
import torchvision
from functools import reduce

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils

def get_layer_by_name(model: nn.Module, layer_name: str):
    """Retrieve a pytorch layer from a model by it's name"""
    if layer_name != "":
        names = layer_name.split(sep=".")
    else:
        return model
    return reduce(getattr, names, model)

class ForwardHook:
    def __init__(self, model, hook_layer_name: str):
        layer = get_layer_by_name(model=model, layer_name=hook_layer_name)
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None
        self.hook_layer_name = hook_layer_name

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

    def __repr__(self):
        return f"ForwardHook(hook_layer_name={self.hook_layer_name})"

class ParallelBottleneckConfig:
    layer_name: str
    roi_names: list[str]
    num_voxels: list[int]

    def __post_init__(self):
        assert len(self.roi_names) == len(self.num_voxels), \
            f"Expected roi_names and num_voxels to be of the same length, got {len(self.roi_names)} and {len(self.num_voxels)} instead"
    def dict(self):
        return {
            "layer_name": self.layer_name,
            "roi_names": self.roi_names,
            "num_voxels": self.num_voxels
        }

class NeuromapModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        bottleneck_configs: list[ParallelBottleneckConfig]
    ):
        
        super().__init__()
        self.backbone = backbone
        self.bottleneck_configs = bottleneck_configs 

        self.forward_hooks = {}
        self.num_voxels_per_roi = {}
        self.roi_names = []
        if self.bottleneck_configs is None:
            print("\033[93mWARNING: No bottleneck configs provided, model has no brain bottlenecks\033[0m")
        else:
            for config in self.bottleneck_configs:
                for num_voxels, roi_name in zip(config.num_voxels, config.roi_names): 
                    self.forward_hooks[roi_name] = ForwardHook(
                        model=backbone,
                        hook_layer_name=f"{config.layer_name}.mappers.{roi_name}.mapper"
                    )
                    self.num_voxels_per_roi[roi_name] = num_voxels
                    self.roi_names.append(roi_name)
            

    def forward(self, x):
        logits = self.backbone(x)
        outputs = {}
        outputs["logits"] = logits

        for roi_name in self.forward_hooks:
            outputs[roi_name] = {
                "predicted_voxels": self.forward_hooks[roi_name].output,
                "roi_name": roi_name
            }
            assert outputs[roi_name]["predicted_voxels"].ndim == 2, \
                f"Expected output for ROI {roi_name} to be of shape (batch_size, num_voxels), got {outputs[roi_name].shape} instead"
            
            assert outputs[roi_name]["predicted_voxels"].shape[-1] == self.num_voxels_per_roi[roi_name], \
                f"Expected output for ROI {roi_name} to have {self.num_voxels_per_roi[roi_name]} voxels, got {outputs[roi_name].shape[-1]} instead"

        return outputs

    @classmethod
    def from_pretrained(
        cls,
        filename: str,
        device: str = "cpu"
    ):
        checkpoint = torch.load(filename, weights_only=False, map_location=device)
        return checkpoint["model_pickle"].to(device)

def get_model(**kwargs):
    model = NeuroMap1MultiSubject(**kwargs)
    return model

class NeuroMap1MultiSubject(BaseModel):
    """
    class for the model ResNet18 trained on ImageNet
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Transforms:
        # SquarePad(): Padding shorter side to make it square
        # Resize(): Resizing to 224x224
        # ToImage(): Convert to Image (subclass of Tensor)
        # ToDtype(): Convert to FloatTensor and scale to [0, 1]
        # Normalize(): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.transforms = transform_utils.get_transforms()
        # Relevant layer names
        # Layer names to ignore
        irrelevant_layer_regexes = [
            'conv1$', 'bn1$', 'relu$', 'maxpool$', \
            'layer[0-9]$', r'layer[0-9]\.[0-9]$', '.*downsample', 'fc.*'
        ]
        # Layer types to ignore
        irrelevant_layer_types = [nn.Dropout, nn.Identity]
        self.irrelevant_layer_names = layer_utils.get_layers_matching_regexes(
            self.model, irrelevant_layer_regexes
        ) + layer_utils.get_layers_matching_types(self.model, irrelevant_layer_types)
        self.relevant_layer_names = [
            layer for layer in self.all_layer_names
            if layer not in self.irrelevant_layer_names
        ]
        # Number of classes
        self.num_classes = 1000

    def _load_model(self):
        """
        Load ResNet18
        """
        # IMAGENET1K_V1
        model = NeuromapModel.from_pretrained(
            filename='/Users/adipani3/projects/catsel/comparing-category-selectivity/preprint/models/neuromap1_checkpoints/robust_final_checkpoint.pt'
        )
        # resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = 'neuromap1multisubject'
        return model, model_name