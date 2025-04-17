"""
Trainer class to train an architecture on ImageNet dataset
For Baseline training, ffcv only
Sources:
https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py
https://github.com/pytorch/vision/blob/main/references/classification/train.py
https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt
"""
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
from ...datasets.image import imagenet1k as imagenet
from ...datasets.image import imagenet16 as imagenet16c
from ...utils import seed_utils

class ImageNet16Trainer:
    """
    Trainer class to train an architecture on ImageNet dataset as a baseline
    """
    def __init__(self, args):
        # Experiment name
        self.experiment_name = 'Manipulation_ImageNet16c'
        wandb.config.update({"experiment_name": self.experiment_name})
        # Args
        self.args = args
        self.uid = str(uuid4())
        # Log uid
        wandb.config.update({"uid": self.uid})
        # Seed
        self.seed = self.args.seed
        seed_utils.seed_everything(self.seed)
        # Device
        self.device = self.args.device
        # Maps for conversion of imagenet1k to imagenet16c
        self.imagenet1k_16c_map, self.imagenet16c_label_weights = \
            self._imagenet1k_to_imagenet16c(self.device)
        # Setup dataloders
        # Train dataloader
        train_loader_dict = self._create_train_loader(
            self.args.train_data_path, self.args.batch_size, self.args.workers,
            self.args.device, self.args.os_cache
        )
        self.train_loader = train_loader_dict['dataloader']
        # Decoder to control the resolution
        self.train_image_decoder = train_loader_dict['decoder']
        # Validation dataloader
        self.val_loader = self._create_val_loader(
            self.args.val_data_path, self.args.batch_size, self.args.workers,
            self.args.device
        )
        # Setup model
        self.num_classes = 17
        wandb.config.update({"num_classes": self.num_classes})
        self.model, self.scaler = self._create_model_and_scaler(
            self.args.architecture, self.args.device, self.args.model_seed,
            num_classes=self.num_classes
        )
        # Setup optimizer
        self.optimizer, self.train_loss, self.val_loss, self.val_loss_unweighted = \
            self._create_optimizer(
                self.args.momentum, self.args.optimizer, self.args.weight_decay,
                self.args.label_smoothing, self.imagenet16c_label_weights
            )
        # Setup logger
        self.val_meters, self.log_folder, self.start_time = self._initialize_logger(
            self.args.output_dir, self.args.architecture, self.uid, self.args.device, self.args
        )
        # Initialize global step
        self.global_step = 0
        # lr schedule
        self.lr_schedule = self._create_lr_schedule(
            self.args.lr_scheduler_type, self.args.lr, self.args.epochs,
            lr_peak_epoch=self.args.lr_peak_epoch
        )
        # Log lr schedule
        wandb.config.update({"lr_schedule": self.lr_schedule.tolist()})
        # Resolution schedule
        if self.args.resolution_schedule_type == 'fixed':
            self.resolution_schedule = np.array(
                [self.args.fixed_resolution] * (self.args.epochs + 1)
            )
        elif self.args.resolution_schedule_type == 'ramp':
            self.resolution_schedule = self._create_resolution_ramp_schedule(
                min_res=self.args.min_resolution, max_res=self.args.max_resolution,
                start_ramp=self.args.start_ramp, end_ramp=self.args.end_ramp,
                epochs=self.args.epochs
            )
        else:
            raise ValueError(
                f"Unknown resolution schedule type {self.args.resolution_schedule_type}"
            )
        # Log resolution schedule
        wandb.config.update({"resolution_schedule": self.resolution_schedule.tolist()})
        # Validation, training time augmentation
        self.lr_tta = self.args.lr_tta

    def run(self):
        """
        Run the training loop
        """
        if self.args.eval_only:
            self._eval_and_log()
        else:
            self._train(self.args.epochs, self.log_folder)

    def _imagenet1k_to_imagenet16c(self, device):
        """
        Maps to convert imagenet1k to imagenet16c, and the label weights to tackle the
            class imbalance
        """
        imagenet1k_16c_map = imagenet16c.get_imagenet_1k_16_map()
        imagenet1k_16c_map = imagenet1k_16c_map.to(device)
        imagenet_16c_label_weights = imagenet16c.get_imagenet16_label_weights(
            normalized=True
        )
        imagenet_16c_label_weights = imagenet_16c_label_weights.to(device)
        return imagenet1k_16c_map, imagenet_16c_label_weights

    def _create_lr_schedule(self, lr_schedule_type, initial_lr, epochs, **kwargs):
        assert lr_schedule_type == 'cyclic', "Only cyclic LR schedule is implemented"

        # https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py#L110
        xs = [0, kwargs.get('lr_peak_epoch'), epochs]
        ys = [1e-4 * initial_lr, initial_lr, 0]
        epoch_lrs = np.interp(np.arange(epochs+1), xs, ys)
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
        # [ramping_resolution] * (end_ramp - start_ramp) + \

        return epoch_resolutions

    def _create_train_loader(
        self,
        train_data_path,
        batch_size,
        num_workers,
        device,
        os_cache,
    ):
        """
        Create the training dataloader
        """
        return imagenet.get_train_loader(
            train_data_path, batch_size=batch_size, num_workers=num_workers, ffcv=True,
            device=device, os_cache=os_cache
        )

    def _create_val_loader(
        self,
        val_data_path,
        batch_size,
        num_workers,
        device,
    ):
        """
        Create the validation dataloader
        """
        return imagenet.get_val_loader(
            val_data_path, batch_size=batch_size, num_workers=num_workers, ffcv=True,
            device=device
        )

    def _create_model_and_scaler(
        self, arch, device, model_seed, num_classes
    ):
        scaler = GradScaler()
        model = getattr(architectures, arch).get_architecture(
            seed=model_seed, num_classes=num_classes
        )

        model = model.to(memory_format=torch.channels_last)
        model = model.to(device)

        return model, scaler

    def _create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, label_weights):
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

        optimizer = torch.optim.SGD(param_groups, lr=1, momentum=momentum)
        train_loss = torch.nn.CrossEntropyLoss(weight=label_weights, label_smoothing=label_smoothing)
        val_loss = torch.nn.CrossEntropyLoss(weight=label_weights)
        val_loss_unweighted = torch.nn.CrossEntropyLoss()

        return optimizer, train_loss, val_loss, val_loss_unweighted

    def _initialize_logger(self, folder, arch_name, uid, device, args):
        val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=1).to(device),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(device),
            'loss': MeanScalarMetric().to(device),
            'loss_unweighted': MeanScalarMetric().to(device)
        }

        # Logging
        log_folder = (Path(folder)/ args.project_name / arch_name / str(uid)).absolute()
        log_folder.mkdir(parents=True)
        # Start time
        start_time = time.time()
        # Save params
        print(f'=> Logging in {log_folder}')
        with open(log_folder / 'params.json', 'w+', encoding='utf-8') as handle:
            json.dump(vars(args), handle, indent=4)

        return val_meters, log_folder, start_time

    def _train(self, epochs, log_folder):
        """
        Train the model for the given number of epochs
        """
        for epoch in range(epochs):
            # Run one epoch of training
            start_train_loop_time = time.time()
            resolution = self.resolution_schedule[epoch]
            # Update the resolution of the train image decoder
            self.train_image_decoder.output_size = (resolution, resolution)
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
                'val_loop_time': val_time
            }, **extra_metrics))
        return stats

    def _train_loop(self, epoch):
        model = self.model
        model.train()

        lr_start, lr_end = self.lr_schedule[epoch].item(), self.lr_schedule[epoch + 1].item()
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            self.global_step += 1
            # Transform target to Imagenet16c
            target = self.imagenet1k_16c_map[target]
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device):
                output = self.model(images)
                loss_train = self.train_loss(output, target)
            wandb.log(
                {
                    "global_step": self.global_step,
                    "train_loss": loss_train,
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
        with torch.no_grad():
            with autocast(self.device):
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if self.lr_tta:
                        output += self.model(torch.flip(images, dims=[3]))
                    # Transform target to Imagenet16c
                    target = self.imagenet1k_16c_map[target]
                    # Calculate accuracy
                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)
                    # Calculate loss
                    loss_val = self.val_loss(output, target)
                    loss_val_unweighted = self.val_loss_unweighted(output, target)
                    self.val_meters['loss'](loss_val)
                    self.val_meters['loss_unweighted'](loss_val_unweighted)
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
