from torch import nn
import torchvision

class Bottleneck_NamedReLU(nn.Module):
    def __init__(
        self,
        conv1,
        bn1,
        conv2,
        bn2,
        conv3,
        bn3,
        downsample
    ):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv2
        self.bn2 = bn2
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3
        self.bn3 = bn3
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class BasicBlock_NamedReLU(nn.Module):
    def __init__(
        self,
        conv1,
        bn1,
        conv2,
        bn2,
        downsample
    ):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

def fix_resnet_blocks_name_relus(model, custom_basicblock=None, custom_bottleneck=None):
    """
    Replaces the ReLU activations in the ResNet blocks of a given model with NamedReLU activations.

    This function iterates over the children modules of the given model and replaces instances of
    `torchvision.models.resnet.Bottleneck` and `torchvision.models.resnet.BasicBlock` with custom
    blocks (`Bottleneck_NamedReLU` and `BasicBlock_NamedReLU`, respectively) that use NamedReLU
    activations instead of the standard ReLU.

    Args:
        model (torch.nn.Module): The model whose ResNet blocks will be modified.

    Returns:
        None
    """
    for name, module in model.named_children():
        print(name, type(module))
        if isinstance(module, torchvision.models.resnet.Bottleneck):
            new_block = Bottleneck_NamedReLU(
                conv1=module.conv1,
                bn1=module.bn1,
                conv2=module.conv2,
                bn2=module.bn2,
                conv3=module.conv3,
                bn3=module.bn3,
                downsample=module.downsample
            )
            new_block.load_state_dict(module.state_dict())
            setattr(model, name, new_block)
        elif custom_bottleneck and isinstance(module, custom_bottleneck):
            new_block = Bottleneck_NamedReLU(
                conv1=module.conv1,
                bn1=module.bn1,
                conv2=module.conv2,
                bn2=module.bn2,
                conv3=module.conv3,
                bn3=module.bn3,
                downsample=module.downsample
            )
            new_block.load_state_dict(module.state_dict())
            setattr(model, name, new_block)
        elif isinstance(module, torchvision.models.resnet.BasicBlock):
            new_block = BasicBlock_NamedReLU(
                conv1=module.conv1,
                bn1=module.bn1,
                conv2=module.conv2,
                bn2=module.bn2,
                downsample=module.downsample
            )
            new_block.load_state_dict(module.state_dict())
            setattr(model, name, new_block)
        elif custom_basicblock and isinstance(module, custom_basicblock):
            new_block = BasicBlock_NamedReLU(
                conv1=module.conv1,
                bn1=module.bn1,
                conv2=module.conv2,
                bn2=module.bn2,
                downsample=module.downsample
            )
            new_block.load_state_dict(module.state_dict())
            setattr(model, name, new_block)
        else:
            fix_resnet_blocks_name_relus(
                module, \
                custom_basicblock = custom_basicblock, \
                custom_bottleneck = custom_bottleneck
            )
