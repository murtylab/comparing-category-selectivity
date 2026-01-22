"""
Submodule for the different architectures to be trained
"""

ARCH_ABBEREVIATIONS = {
    'alexnet': 'alxnet',
    'resnet18': 'r18',
    'resnet50': 'r50',
    'vitb32': 'vitb32'
}

from . import alexnet
from . import resnet18
from . import resnet50
from . import vitb32
