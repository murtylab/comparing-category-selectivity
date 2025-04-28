"""
ResNet-18
https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
"""
from torch import nn
import torchvision

from ...basemodel import BaseModel
from ...utils import transform_utils, layer_utils, resnet_utils
from ...architectures import resnet18

def get_model(**kwargs):
    model = ResNet18ImageNet1kv1(**kwargs)
    return model

class ResNet18ImageNet1kv1(BaseModel):
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
        model = torchvision.models.resnet18(weights='DEFAULT')
        resnet_utils.fix_resnet_blocks_name_relus(model)
        model_name = 'resnet18imagenet1kv1'
        return model, model_name

    def replace_fc(self, num_classes):
        """
        Replace the fully connected layer with a new one
        """
        # Update the number of classes
        self.num_classes = num_classes
        # Replace the last fully connected layer with a new one
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def add_projector(self, projector):
        """
        Add a projector to the model
        """
        # Initialize the architecture with the projector
        model_with_projector = resnet18.get_architecture(
            num_classes=self.num_classes, projector=projector
        )
        # Not passing kwargs to the architecture, this may create issues
        # Load the state dict of the current model into the new one
        load_info = model_with_projector.model.load_state_dict(
            self.model.state_dict(), strict=False
        )
        # Assert the missing keys are the ones we expect
        assert all(
            key.startswith('projector') for key in load_info.missing_keys
        ), f"Unexpected missing keys: {load_info.missing_keys}"
        # Update the model
        self.model = model_with_projector
