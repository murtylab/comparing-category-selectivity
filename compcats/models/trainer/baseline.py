"""
Trainer class to train an architecture on ImageNet dataset
For Baseline training, ffcv only
Sources:
https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py
https://github.com/pytorch/vision/blob/main/references/classification/train.py
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
from ...datasets import imagenet

class BaselineTrainer:
    """
    Trainer class to train an architecture on ImageNet dataset as a baseline
    """
    def __init__(self, args):
        self.args = args
        self.uid = str(uuid4())
        # Setup dataloders
        self.train_loader = self._create_train_loader(
            self.args.train_data_path, self.args.batch_size, self.args.workers,
            self.args.device, self.args.os_cache
        )
        self.val_loader = self._create_val_loader(
            self.args.val_data_path, self.args.batch_size, self.args.workers,
            self.args.device
        )
        # Setup model
        self.model, self.scaler = self._create_model_and_scaler(
            self.args.architecture, self.args.device
        )
        # Setup optimizer
        self.optimizer, self.loss = self._create_optimizer(
            self.args.momentum, self.args.optimizer, self.args.weight_decay,
            self.args.label_smoothing
        )
        # Setup logger
        self.val_meters, self.log_folder, self.start_time = self._initialize_logger(
            self.args.output_dir, self.args.architecture, self.uid, self.args.device, self.args
        )
        # Initialize global step
        self.global_step = 0
        # lr schedule
        self.lr_schedule = self._create_lr_schedule(
            self.args.lr_schedule, self.args.lr, self.args.epochs,
            lr_peak_epoch=self.args.lr_peak_epoch
        )
        # Validation, channels last?
        self.lr_tta = self.args.lr_tta
        # Device
        self.device = self.args.device

    def run(self):
        """
        Run the training loop
        """
        if self.args.eval_only:
            self._eval_and_log()
        else:
            self._train(self.args.epochs, self.log_folder)

    def _create_lr_schedule(self, lr_schedule_type, initial_lr, epochs, **kwargs):
        assert lr_schedule_type == 'cyclic', "Only cyclic LR schedule is implemented"

        # https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py#L110
        xs = [0, kwargs.get('lr_peak_epoch'), epochs]
        ys = [1e-4 * initial_lr, initial_lr, 0]
        epoch_lrs = np.interp(np.arange(epochs), xs, ys)
        return epoch_lrs

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
        return imagenet.train.get_loader(
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
        return imagenet.validation.get_loader(
            val_data_path, batch_size=batch_size, num_workers=num_workers, ffcv=True,
            device=device
        )

    def _create_model_and_scaler(
        self, arch, device
    ):
        scaler = GradScaler()
        model = getattr(architectures, arch)()

        model = model.to(memory_format=torch.channels_last)
        model = model.to(device)

        return model, scaler

    def _create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd'

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        optimizer = torch.optim.SGD(param_groups, lr=1, momentum=momentum)
        loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        return optimizer, loss

    def _initialize_logger(self, folder, arch_name, uid, device, args):
        val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=1).to(device),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(device),
            'loss': MeanScalarMetric().to(device)
        }

        # Logging
        log_folder = (Path(folder) / arch_name / str(uid)).absolute()
        log_folder.mkdir(parents=True)
        # Start time
        start_time = time.time()
        # Save params
        print(f'=> Logging in {log_folder}')
        with open(log_folder / 'params.json', 'w+') as handle:
            json.dump(args, handle)
        
        return val_meters, log_folder, start_time

    def _train(self, epochs, log_folder):
        """
        Train the model for the given number of epochs
        """
        for epoch in range(epochs):
            # Run one epoch of training
            train_loss = self._train_loop(epoch)
            # Validate the model and log the results
            self._eval_and_log({
                'train_loss': train_loss,
                'epoch': epoch,
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
        wandb.log(
            {
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time
            }
        )
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
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device):
                output = self.model(images)
                loss_train = self.loss(output, target)
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
                    # Calculate accuracy
                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)
                    # Calculate loss
                    loss_val = self.loss(output, target)
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
