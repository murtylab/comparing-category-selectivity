"""
Script to train a model on ImageNet dataset using FFCV
"""
import argparse
import wandb

from compcats.models.trainer.baseline import BaselineTrainer

def main(args):
    """
    Main function to parse arguments and train the model
    """
    wandb.init(
        project=args.project_name,
        config = None
    )
    trainer = BaselineTrainer(args)
    trainer.run()

def get_args_parser(add_help=True):
    """
    https://github.com/pytorch/vision/blob/main/references/classification/train.py#L393
    """

    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help
    )

    # Dirs
    parser.add_argument(
        "--train-data-path", default=None, type=str, help="training ffcv dataset path"
    )
    parser.add_argument(
        "--val-data-path", default=None, type=str, help="validation ffcv dataset path"
    )
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")

    # Model parameters
    parser.add_argument(
        "--architecture", default="resnet18", type=str,
        choices=["resnet18", "resnet50", "alexnet"],
        help="model name"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", default=88, type=int, metavar="N",
        help="number of total epochs to run"
    )
    parser.add_argument("--optimizer", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, 
        help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument(
        "--lr-scheduler-type", default="cyclic", type=str,
        help="the lr scheduler (default: cyclic)"
    )
    parser.add_argument(
        "--lr-peak-epoch", default=0, type=int,
        help="the number of epochs to warmup (default: 0)"
    )

    # Hardware
    parser.add_argument(
        "--device", default="cuda:0", type=str,
        help="device (Use cuda or cpu Default: cuda)"
    )
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=12, type=int, metavar="N",
        help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "--os-cache", default=True, type=bool,
        help="Whether the OS cache should be used for data loading"
    )

    # Misc
    # parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--lr-tta", default=False, type=bool,
        help="use test-time augmentation for validation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--project-name", default=False, type=str,
        help="project name for wandb"
    )

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
