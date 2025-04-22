"""
Script to train a model on ImageNet dataset using FFCV
Args are a mix of torchvision classification and ffcv:
https://github.com/pytorch/vision/blob/main/references/classification/train.py#L393
https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py#L36
"""
import argparse
import wandb

from fastargs import get_current_config
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from compcats.models.trainer.imagenet1k import ImageNet1kTrainer
from compcats.models import architectures
from compcats.utils import logging_utils

Section('seed', 'seed').params(
    global_seed=Param(int, 'global seed', default=42),
    model_seed=Param(int, 'model seed', default=42)
)

Section('data', 'data related stuff').params(
    train_data_path=Param(str, '.ffcv file to use for training', required=True),
    val_data_path=Param(str, '.ffcv file to use for validation', required=True),
    output_dir=Param(str, 'directory for logging and storing checkpoints', required=True),
    device=Param(str, 'device to use for training', default='cuda:0'),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('project', 'project related stuff').params(
    name=Param(str, 'wandb project name', required=True),
    experiment=Param(str, 'experiment name', required=True),
)

Section('model', 'model details').params(
    architecture=Param(And(str, OneOf(dir(architectures))), required=True),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    resolution_schedule_type=Param(OneOf(['fixed', 'ramp']), default='ramp'),
    fixed_resolution=Param(int, 'the constant resolution', default=224),
    min_resolution=Param(int, 'the minimum (starting) resolution', default=160),
    max_resolution=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('lr', 'lr scheduling').params(
    # Only implemented cyclic, not step
    # https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py#L100
    # step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    # step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

def main(fastargs_config):
    """
    Main function to parse arguments and train the model
    """
    args = fastargs_config.get()
    nested_args = logging_utils.namespace_to_dict(args)
    wandb.init(
        project = args.project.name,
        config = nested_args
    )
    trainer = ImageNet1kTrainer(args)
    trainer.run()

if __name__ == "__main__":
    config = get_current_config()
    parser = argparse.ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main(config)
