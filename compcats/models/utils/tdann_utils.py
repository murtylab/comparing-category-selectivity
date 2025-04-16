import torch
from torch import nn
import torchvision

PRETRAINED_SIMCLR_MODEL_LINKS = {
    # simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw0_checkpoints
    0: 'https://osf.io/8q32t',
    # simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lw01_checkpoints
    0.1: 'https://osf.io/5z8b3',
    # simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_checkpoints
    1: 'https://osf.io/exugm',
    # simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx2_checkpoints
    2: 'https://osf.io/zaber',
    # simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx5_checkpoints
    5: 'https://osf.io/zrdh5',
    # simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx10_checkpoints
    10: 'https://osf.io/fp9eh',
    # simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_lwx100_checkpoints
    100: 'https://osf.io/zax52',
}

PRETRAINED_SUPERVISED_MODEL_LINKS = {
    # supervised_resnet18_swappedon_SineGrating2019_isoswap_3_lw0_checkpoints
    0: 'https://osf.io/4gq5n',
    # supervised_spatial_resnet18_swappedon_SineGrating2019_lw01_checkpoints
    0.1: 'https://osf.io/ckemq',
    # supervised_spatial_resnet18_swappedon_SineGrating2019_checkpoints
    1: 'https://osf.io/xnhtd',
    # supervised_spatial_resnet18_swappedon_SineGrating2019_lwx2_checkpoints
    2: 'https://osf.io/qasyf',
    # supervised_spatial_resnet18_swappedon_SineGrating2019_lwx5_checkpoints
    5: 'https://osf.io/xz5ky',
    # supervised_spatial_resnet18_swappedon_SineGrating2019_lwx10_checkpoints
    10: 'https://osf.io/4zfj2',
    # supervised_spatial_resnet18_swappedon_SineGrating2019_lwx100_checkpoints
    100: 'https://osf.io/j5kt4',
}

class EshedNet(nn.Module):
    def __init__(self):
        """Create a new EshedNet

        Inputs:
            model_config: an AttrDict (like a dictionary, but with dot syntax support)
                that specifies the parameters for the model trunk. Specifically, we will
                expect that "model_config.TRUNK.TRUNK_PARAMS.position_dir" exists

            model_name: VISSL will pass the model name as the second arg, but we don't
                use it for anything in this case
        """
        super(EshedNet, self).__init__()

        # self.positions = self._load_positions(model_config.TRUNK.TRUNK_PARAMS.position_dir)
        self.base_model = torchvision.models.resnet18(weights=None)

        # remove the FC layer, we're not going to need it
        self.base_model.fc = nn.Identity()

    # VISSL requires this signature for forward passes
    def forward(self, x: torch.Tensor):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        maxpool = self.base_model.maxpool(x)

        x_1_0 = self.base_model.layer1[0](maxpool)
        x_1_1 = self.base_model.layer1[1](x_1_0)
        x_2_0 = self.base_model.layer2[0](x_1_1)
        x_2_1 = self.base_model.layer2[1](x_2_0)
        x_3_0 = self.base_model.layer3[0](x_2_1)
        x_3_1 = self.base_model.layer3[1](x_3_0)
        x_4_0 = self.base_model.layer4[0](x_3_1)
        x_4_1 = self.base_model.layer4[1](x_4_0)

        x = self.base_model.avgpool(x_4_1)
        flat_outputs = torch.flatten(x, 1)
        return flat_outputs
