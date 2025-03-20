"""
Script to evaluate the model on the ImageNet validation dataset
"""
from tqdm import tqdm
import torch
import torchmetrics

from compcats.datasets.val import imagenet_val, imagenetffcv_val
from ..utils import transform_utils

def evaluate_model(
    root,
    model,
    batch_size=768,
    num_workers=12,
    device='cuda',
    ffcv=False,
    **kwargs
):
    """
    Evaluate the model on the validation dataset
    """
    # Load the model
    model.eval()
    model.to(device)

    # Get default imagenet validation transforms
    transforms = transform_utils.get_imagenet_val_transforms()

    # Load the validation dataset
    val_dataloader = None
    if ffcv:
        val_dataloader = imagenetffcv_val.get_loader(
            root, transforms, batch_size=batch_size, num_workers=num_workers
        )
    else:
        val_dataloader = imagenet_val.get_loader(
            root, transforms, batch_size=batch_size, num_workers=num_workers
        )

    # Metrics
    top1_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(device)
    top5_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(device)
    loss = torch.nn.CrossEntropyLoss().to(device)

    # Evaluate the model
    with torch.inference_mode():
        total_loss = 0.0
        total_top1_acc = 0.0
        total_top5_acc = 0.0
        total_batches = len(val_dataloader)

        for batch, target in tqdm(val_dataloader):
            batch = batch.to(device)
            target = target.to(device)
            output = model(batch)

            # Compute the loss and accuracy
            batch_loss = loss(output, target)
            batch_top1_acc = top1_accuracy(output, target)
            batch_top5_acc = top5_accuracy(output, target)

            # Accumulate the loss and accuracy
            total_loss += batch_loss.item()
            total_top1_acc += batch_top1_acc
            total_top5_acc += batch_top5_acc

        avg_loss = total_loss / total_batches
        avg_top1_acc = total_top1_acc / total_batches
        avg_top5_acc = total_top5_acc / total_batches

        print(f'Average Loss: {avg_loss}, Average Top-1 Accuracy: {avg_top1_acc}, Average Top-5 Accuracy: {avg_top5_acc}')

    return avg_loss, avg_top1_acc, avg_top5_acc
