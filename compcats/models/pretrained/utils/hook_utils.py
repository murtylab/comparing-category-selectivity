import torch.nn as nn
from functools import reduce

def get_layer_by_name(model: nn.Module, layer_name: str):
    """Retrieve a pytorch layer from a model by it's name"""
    if layer_name != "":
        names = layer_name.split(sep=".")
    else:
        return model
    return reduce(getattr, names, model)

class ForwardHook:
    def __init__(
        self,
        model,
        hook_layer_name: str
    ):
        layer = get_layer_by_name(
            model=model,
            layer_name=hook_layer_name
        )
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()
