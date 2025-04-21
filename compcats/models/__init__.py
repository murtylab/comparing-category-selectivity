"""
This module initializes and imports various neural network models.
"""

##########################
# Evaluation
##########################

from .evaluate.imagenet1k import evaluate as imagenet1k_evaluataion


##########################
# Pretrained models
##########################

from .pretrained import *
