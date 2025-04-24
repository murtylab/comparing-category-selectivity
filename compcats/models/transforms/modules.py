"""
Source: Haider Al-Tahan
https://github.com/haideraltahan/ImageNet/blob/master/modules/transforms.py
"""

from typing import Callable, Optional, Tuple
from ffcv.pipeline.operation import Operation

from ffcv.pipeline.state import State
from ffcv.pipeline.allocation_query import AllocationQuery
from dataclasses import replace
import torch

import random

import torchvision.transforms.v2 as transforms


class RandomChoice(Operation):
    def __init__(self):
        super(RandomChoice, self).__init__()
        self.greyscale = transforms.Grayscale(3)
        self.solarize = transforms.RandomSolarize(0.569, 1.0)
        self.gaussian = transforms.GaussianBlur(3)
        self.transforms = [self.greyscale, self.solarize, self.gaussian]

    def generate_code(self) -> Callable:
        @torch.no_grad()
        def pick_a_corner(x, dst):
            t = random.choices(self.transforms)[0]
            dst = t(x)
            return dst

        return pick_a_corner

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        new_shape = previous_state.shape
        new_state = replace(previous_state, shape=new_shape)
        mem_allocation = AllocationQuery(new_shape, previous_state.dtype)
        return (new_state, mem_allocation)


class ColorJitter(Operation):
    def __init__(self):
        super(ColorJitter, self).__init__()
        self.colorjitter = transforms.ColorJitter(0.3, 0.3, 0.3)

    def generate_code(self) -> Callable:
        @torch.no_grad()
        def pick_a_corner(x, dst):
            dst = self.colorjitter(x)
            return dst

        return pick_a_corner

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
