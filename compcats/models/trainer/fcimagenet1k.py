"""
Trainer class to train an architecture on ImageNet dataset
For Baseline training, ffcv only
Sources:
https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py
https://github.com/pytorch/vision/blob/main/references/classification/train.py
"""
from argparse import Namespace
import time
from uuid import uuid4
from pathlib import Path
import json
import wandb
from tqdm import tqdm
import numpy as np
import torchmetrics
import torch
from torch.amp import autocast, GradScaler

from .. import architectures
from .. import pretrained as pretrained_module
from ..losses import facecontrastive as fcloss
from ...datasets.image import imagenet1k as imagenet
from ...datasets.image import celebahq, afhqv2, lhq
from ...utils import seed_utils, logging_utils, iterator_utils

def get_trainer():
    """
    Get the trainer class
    """
    return FaceContrastiveImageNet1kTrainer

class FaceContrastiveImageNet1kTrainer:
    """
    Trainer class to train an architecture on ImageNet dataset as a baseline
    """
    def __init__(self, args):
        # Args
        self.args = args
        self.uid = str(uuid4())
        print(f'\nUID: {self.uid}\n')
        # Add identifiers
        self.args.identifier = Namespace()
        self.args.identifier.uid = self.uid
        self.args.identifier.training_strategy = "facecontrastive_imagenet1k"
        # Log uid
        wandb.config.update({"uid": self.uid})
        # Seed
        seed_utils.seed_everything(self.args.seed.global_seed)
        # Setup dataloders
        # Train dataloader
        # Mixing ratios
        self.mixing_ratios = {
            'celebahq': 0.25,
            'afhqv2': 0.25,
            'lhq': 0.5
        }
        self.args.identifier.mixing_ratios = Namespace(**self.mixing_ratios)
        wandb.config.update({"mixing_ratios": self.mixing_ratios})
        # Train dataloader
        train_data_paths = {
            'imagenet1k': self.args.data.imagenet1k_train_data_path,
            'celebahq': self.args.data.celebahq_train_data_path,
            'afhqv2': self.args.data.afhqv2_train_data_path,
            'lhq': self.args.data.lhq_train_data_path
        }
        train_loader_dict = self._create_train_loader(
            train_data_paths, self.args.training.batch_size,
            self.args.data.num_workers, self.args.data.device, self.args.data.in_memory
        )
        self.train_loaders = {k: v['dataloader'] for k, v in train_loader_dict.items()}
        # Decoder to control the resolution
        self.train_image_decoders = {k: v['decoder'] for k, v in train_loader_dict.items()}
        # Validation dataloader
        val_data_paths = {
            'imagenet1k': self.args.data.imagenet1k_val_data_path,
            'celebahq': self.args.data.celebahq_val_data_path,
            'afhqv2': self.args.data.afhqv2_val_data_path,
            'lhq': self.args.data.lhq_val_data_path
        }
        self.val_loaders = self._create_val_loader(
            val_data_paths, self.args.validation.resolution,
            self.args.validation.batch_size, self.args.data.num_workers, self.args.data.device
        )
        # Setup model
        self.num_classes = 1000
        self.args.identifier.num_classes = self.num_classes
        wandb_config_model = dict(wandb.config["model"])
        wandb_config_model["num_classes"] = self.num_classes
        wandb.config.update({"model": wandb_config_model}, allow_val_change=True)
        self.model, self.scaler = self._create_model_and_scaler(
            self.args.model.architecture, self.args.data.device, self.args.seed.model_seed,
            self.args.model.pretrained, self.args.model.pretrained_model,
            num_classes=self.num_classes, projector_dim=self.args.model.projector_dim
        )
        # Setup optimizer
        self.optimizer, self.train_ce_loss, self.train_contrastive_loss, \
            self.val_ce_loss, self.val_contrastive_loss = \
                self._create_optimizer(
                self.args.training.momentum, self.args.training.optimizer,
                self.args.training.weight_decay, self.args.training.label_smoothing,
                self.args.data.device, initial_lr=self.args.lr.lr,
                loss_version=self.args.training.loss_version,
                loss_temperature=self.args.training.loss_temperature
            )
        # Setup logger
        self.val_meters, self.log_folder, self.start_time = self._initialize_logger(
            self.args.data.output_dir, self.args.model.architecture, self.uid,
            self.args.data.device, self.num_classes, self.args
        )
        # Initialize global step
        self.global_step = 0
        # lr schedule
        self.lr_schedule = self._create_lr_schedule(
            self.args.lr.lr_schedule_type, self.args.lr.lr, self.args.training.epochs,
            lr_peak_epoch=self.args.lr.lr_peak_epoch
        )
        # Log lr schedule
        wandb.config.update({"lr_schedule": self.lr_schedule})
        # Resolution schedule
        if self.args.resolution.resolution_schedule_type == 'fixed':
            self.resolution_schedule = np.array(
                [self.args.resolution.fixed_resolution] * (self.args.training.epochs + 1)
            )
        elif self.args.resolution.resolution_schedule_type == 'ramp':
            self.resolution_schedule = self._create_resolution_ramp_schedule(
                min_res=self.args.resolution.min_resolution,
                max_res=self.args.resolution.max_resolution,
                start_ramp=self.args.resolution.start_ramp,
                end_ramp=self.args.resolution.end_ramp,
                epochs=self.args.training.epochs
            )
        else:
            raise ValueError(
                f"Unknown resolution schedule type {self.args.resolution.resolution_schedule_type}"
            )
        # Log resolution schedule
        wandb.config.update({"resolution_schedule": self.resolution_schedule})
        # Validation, training time augmentation
        self.lr_tta = self.args.validation.lr_tta
        # Device
        self.device = self.args.data.device

    def run(self):
        """
        Run the training loop
        """
        if self.args.training.eval_only:
            self._eval_and_log()
        else:
            self._train(self.args.training.epochs, self.log_folder)

    def _create_lr_schedule(self, lr_schedule_type, initial_lr, epochs, **kwargs):
        assert lr_schedule_type == 'cyclic', "Only cyclic LR schedule is implemented"

        # https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py#L110
        xs = [0, kwargs.get('lr_peak_epoch'), epochs]
        ys = [1e-4 * initial_lr, initial_lr, 0]
        epoch_lrs = np.interp(np.arange(epochs+1), xs, ys)
        # Add an extra element at the start for first validation epoch
        epoch_lrs = [1e-4 * initial_lr] + epoch_lrs.tolist()
        return epoch_lrs

    def _create_resolution_ramp_schedule(
        self, min_res, max_res, start_ramp, end_ramp,
        epochs
    ):
        """
        Create the resolution schedule if variable resolution is used
        """
        # https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py#L173
        assert min_res <= max_res, "min_res should be less than max_res"
        assert start_ramp < end_ramp, "start_ramp should be less than end_ramp"
        assert start_ramp < epochs, "start_ramp should be less than epochs"
        assert end_ramp <= epochs, "end_ramp should be less than or equal to epochs"

        # While ramping up, linearly interpolate to the nearest multiple of 32
        ramping_resolutions = []
        for epoch_ in range(start_ramp, end_ramp):
            # Interpolate the resolution to the nearest multiple of 32
            epoch_ramping_resolution_interp = np.interp(
                [epoch_], [start_ramp, end_ramp], [min_res, max_res]
            )
            epoch_ramping_resolution = int(np.round(epoch_ramping_resolution_interp[0] / 32)) * 32
            # Append the ramping resolution
            ramping_resolutions.append(epoch_ramping_resolution)

        epoch_resolutions = [min_res] * start_ramp + \
            ramping_resolutions + \
            [max_res] * (epochs + 1 - end_ramp)

        # Add an extra element at the start for first validation epoch
        epoch_resolutions = [min_res] + epoch_resolutions

        return epoch_resolutions

    def _create_train_loader(
        self,
        train_data_paths,
        batch_size,
        num_workers,
        device,
        os_cache,
    ):
        """
        Create the training dataloader
        """
        batch_sizes = {
            'imagenet1k': int(batch_size/2),
            'celebahq': int(batch_size / 2 * self.mixing_ratios['celebahq']),
            'afhqv2': int(batch_size / 2 * self.mixing_ratios['afhqv2']),
            'lhq': int(batch_size / 2 * self.mixing_ratios['lhq'])
        }
        train_loaders = {
            'imagenet1k': imagenet.get_train_loader(
                train_data_paths['imagenet1k'], batch_size=batch_sizes['imagenet1k'],
                num_workers=num_workers, ffcv=True, device=device, os_cache=os_cache
            ),
            'celebahq': celebahq.get_train_loader(
                train_data_paths['celebahq'], batch_size=batch_sizes['celebahq'],
                transform = 'contrastive',
                num_workers=num_workers, ffcv=True, device=device, os_cache=os_cache
            ),
            'afhqv2': afhqv2.get_train_loader(
                train_data_paths['afhqv2'], batch_size=batch_sizes['afhqv2'],
                transform = 'contrastive',
                num_workers=num_workers, ffcv=True, device=device, os_cache=os_cache
            ),
            'lhq': lhq.get_train_loader(
                train_data_paths['lhq'], batch_size=batch_sizes['lhq'],
                transform = 'contrastive',
                num_workers=num_workers, ffcv=True, device=device, os_cache=os_cache
            )
        }
        return train_loaders

    def _create_val_loader(
        self,
        val_data_paths,
        resolution,
        batch_size,
        num_workers,
        device,
    ):
        """
        Create the validation dataloader
        """
        batch_sizes = {
            'imagenet1k': int(batch_size/2),
            'celebahq': int(batch_size / 2 * self.mixing_ratios['celebahq']),
            'afhqv2': int(batch_size / 2 * self.mixing_ratios['afhqv2']),
            'lhq': int(batch_size / 2 * self.mixing_ratios['lhq'])
        }
        val_loaders = {
            'imagenet1k': imagenet.get_val_loader(
                val_data_paths['imagenet1k'], batch_size=batch_sizes['imagenet1k'],
                resolution=resolution,
                num_workers=num_workers, ffcv=True, device=device
            ),
            'celebahq': celebahq.get_val_loader(
                val_data_paths['celebahq'], batch_size=batch_sizes['celebahq'],
                transform='contrastive', resolution=resolution,
                num_workers=num_workers, ffcv=True, device=device
            ),
            'afhqv2': afhqv2.get_val_loader(
                val_data_paths['afhqv2'], batch_size=batch_sizes['afhqv2'],
                transform='contrastive', resolution=resolution,
                num_workers=num_workers, ffcv=True, device=device
            ),
            'lhq': lhq.get_val_loader(
                val_data_paths['lhq'], batch_size=batch_sizes['lhq'],
                transform='contrastive', resolution=resolution,
                num_workers=num_workers, ffcv=True, device=device
            )
        }
        return val_loaders

    def _create_model_and_scaler(
        self, arch, device, model_seed, pretrained, pretrained_model, num_classes, projector_dim
    ):
        scaler = GradScaler()
        if pretrained:
            raise NotImplementedError(
                "Projector for Pretrained models are not implemented yet"
            )
            model = getattr(pretrained_module, pretrained_model).get_model()
        else:
            model = getattr(architectures, arch).get_architecture(
                seed=model_seed, num_classes=num_classes, projector=projector_dim
            )

        model = model.to(memory_format=torch.channels_last)
        model = model.to(device)

        return model, scaler

    def _create_optimizer(
        self, momentum, optimizer, weight_decay,
        label_smoothing, device, initial_lr, loss_version, loss_temperature
    ):
        # Initial lr is just for logging purposes
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if 'bn' in k]
        other_params = [v for k, v in all_params if not 'bn' in k]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        optimizer = torch.optim.SGD(param_groups, lr=initial_lr, momentum=momentum)
        train_ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
        train_contrastive_loss = fcloss.get_loss_fn(
            loss_version=loss_version, loss_temperature=loss_temperature
        ).to(device)
        # Validation loss
        val_ce_loss = torch.nn.CrossEntropyLoss().to(device)
        val_contrastive_loss = fcloss.get_loss_fn(
            loss_version=loss_version, loss_temperature=loss_temperature
        ).to(device)

        return optimizer, train_ce_loss, train_contrastive_loss, val_ce_loss, val_contrastive_loss

    def _initialize_logger(self, folder, arch_name, uid, device, num_classes, args):
        val_meters = {
            'top_1': torchmetrics.Accuracy(
                task='multiclass', num_classes=num_classes, top_k=1
            ).to(device),
            'top_5': torchmetrics.Accuracy(
                task='multiclass', num_classes=num_classes, top_k=5
            ).to(device),
            'loss': MeanScalarMetric().to(device),
            'ce_loss': MeanScalarMetric().to(device),
            'contrastive_loss': MeanScalarMetric().to(device)
        }

        # Logging
        log_folder = (
            Path(folder)/ args.project.name / args.project.experiment / arch_name / str(uid)
        ).absolute()
        log_folder.mkdir(parents=True)
        # Start time
        start_time = time.time()
        # Save params
        print(f'=> Logging in {log_folder}')
        with open(log_folder / 'params.json', 'w+', encoding='utf-8') as handle:
            json.dump(logging_utils.nestednamespace_to_dict(args), handle, indent=4)

        return val_meters, log_folder, start_time

    def _train(self, epochs, log_folder):
        """
        Train the model for the given number of epochs
        """
        # Validate the model and log the results without training
        # This is to get the first validation results
        # before training starts
        self._eval_and_log({
            'epoch': 0,
            'resolution': self.resolution_schedule[0]
        })
        for epoch in range(1, epochs+1):
            # Run one epoch of training
            start_train_loop_time = time.time()
            resolution = self.resolution_schedule[epoch]
            # Update the resolution of the train image decoders
            for ds_name in self.train_image_decoders.keys():
                self.train_image_decoders[ds_name].output_size = (resolution, resolution)
            # Train loop
            self._train_loop(epoch)
            train_loop_time = time.time() - start_train_loop_time
            # Validate the model and log the results
            self._eval_and_log({
                'epoch': epoch,
                'resolution': resolution,
                'train_loop_time': train_loop_time
            })
            # Save the model checkpoint
            torch.save(
                self.model,
                log_folder / f'{self.model.name}_epoch{epoch}.pt'
            )
        # Save the final checkpoint
        torch.save(
            self.model,
            log_folder / f'{self.model.name}_final.pt'
        )

    def _eval_and_log(self, extra_metrics=None):
        """
        Evaluate the model and log the results
        """
        start_val = time.time()
        stats = self._val_loop()
        val_time = time.time() - start_val
        wandb.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'val_top_1': stats['top_1'],
                'val_top_5': stats['top_5'],
                'val_loss': stats['loss'],
                'val_ce_loss': stats['ce_loss'],
                'val_contrastive_loss': stats['contrastive_loss'],
                'val_loop_time': val_time
            }, **extra_metrics))
        return stats

    def _train_loop(self, epoch):
        model = self.model
        model.train()

        lr_start, lr_end = self.lr_schedule[epoch], self.lr_schedule[epoch + 1]
        iters = len(self.train_loaders['imagenet1k'])
        # Interpolate the learning rate
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        # Convert dataloaders for contrastive loss to cyclic iterators to compensate for
        # large number of samples in imagenet
        # 1st positive dataset
        dataloader_pos1 = iterator_utils.convert_to_cyclic(
            self.train_loaders['celebahq']
        )
        # 2nd positive dataset
        dataloader_pos2 = iterator_utils.convert_to_cyclic(
            self.train_loaders['afhqv2']
        )
        # Negative dataset
        dataloader_neg = iterator_utils.convert_to_cyclic(
            self.train_loaders['lhq']
        )

        imagenet1k_iterator = tqdm(self.train_loaders['imagenet1k'], desc='Training loop')
        for ix, (imagenet1k_images, imagenet1k_target) in enumerate(imagenet1k_iterator):
            self.global_step += 1
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]
            # Get images for contrastive loss
            images_pos1, _ = next(dataloader_pos1)
            images_pos2, _ = next(dataloader_pos2)
            images_neg, _ = next(dataloader_neg)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device):
                # Cross Entropy loss
                fc_output, _ = self.model(imagenet1k_images)
                ce_loss = self.train_ce_loss(fc_output, imagenet1k_target)
                # Contrastive loss
                _, embed_pos1 = self.model(images_pos1)
                _, embed_pos2 = self.model(images_pos2)
                _, embed_neg = self.model(images_neg)
                contrastive_loss = self.train_contrastive_loss(
                    embed_pos1, embed_pos2, embed_neg
                )
                loss_train = ce_loss + self.args.training.loss_weight * contrastive_loss
            wandb.log(
                {
                    "global_step": self.global_step,
                    "train_loss": loss_train,
                    "train_ce_loss": ce_loss,
                    "train_contrastive_loss": contrastive_loss,
                    "learning_rate": lrs[ix],
                    "epoch": epoch
                }
            )
            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

    def _val_loop(self):
        model = self.model
        model.eval()
        # Convert dataloaders for contrastive loss to cyclic iterators to compensate for
        # large number of samples in imagenet
        # 1st positive dataset
        dataloader_pos1 = iterator_utils.convert_to_cyclic(
            self.val_loaders['celebahq']
        )
        # 2nd positive dataset
        dataloader_pos2 = iterator_utils.convert_to_cyclic(
            self.val_loaders['afhqv2']
        )
        # Negative dataset
        dataloader_neg = iterator_utils.convert_to_cyclic(
            self.val_loaders['lhq']
        )
        contrastive_iterator = tqdm(
            zip([dataloader_pos1, dataloader_pos2, dataloader_neg]),
            desc='Validation loop',
            total=max(
                len(dataloader_pos1), len(dataloader_pos2), len(dataloader_neg)
            )
        )
        with torch.no_grad():
            with autocast(self.device):
                # Imagenet1k validation
                for images, target in tqdm(self.val_loaders['imagenet1k']):
                    fc_output, _ = self.model(images)
                    if self.lr_tta:
                        fc_output_tta, _ = self.model(torch.flip(images, dims=[3]))
                        # Add the outputs
                        fc_output += fc_output_tta
                    # Calculate accuracy
                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](fc_output, target)
                    # Calculate cross-entropy loss
                    ce_loss = self.val_ce_loss(fc_output, target)
                    # Update loss meters
                    self.val_meters['ce_loss'](ce_loss)
                # Contrastive validation
                for ((images_pos1, _), (images_pos2, _), (images_neg, _)) in contrastive_iterator:
                    # Get the outputs
                    _, projector_output_pos1 = self.model(images_pos1)
                    _, projector_output_pos2 = self.model(images_pos2)
                    _, projector_output_neg = self.model(images_neg)
                    # Calculate contrastive loss
                    contrastive_loss = self.val_contrastive_loss(
                        projector_output_pos1, projector_output_pos2, projector_output_neg
                    )
                    loss_val = ce_loss + self.args.training.loss_weight * contrastive_loss
                    # Update loss meters
                    self.val_meters['contrastive_loss'](contrastive_loss)
                    self.val_meters['loss'](loss_val)
        # Compute stats over the entire validation set
        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        # Reset meters
        _ = [meter.reset() for meter in self.val_meters.values()]
        return stats

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    """
    Custom torchmetric for validation loop
    https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py#L479
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state('sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: torch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count
