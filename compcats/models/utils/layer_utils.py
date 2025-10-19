"""
Util functions for working with layers in PyTorch models
"""
import warnings
import re
import torch.nn as nn

def get_all_layer_names(model):
    """
    Get all layer names in a model
    """
    return [name for name, _ in model.named_modules() if name != ""]

def get_all_dropout_layer_names(model):
    """
    Get all dropout layer names in a model
    """
    return [name for name, module in model.named_modules() if isinstance(module, nn.Dropout)]

def get_layers_matching_regexes(model, regex_list):
    """
    Get all layer names in a model that match any of the regex patterns in the provided list.
    """
    matching_layers = []
    if len(regex_list) == 0:
        return matching_layers
    for name, _ in model.named_modules():
        if any(re.match(pattern, name) for pattern in regex_list):
            matching_layers.append(name)
    return matching_layers

def get_layers_matching_types(model, module_types):
    """
    Get all layer names in a model that are instances of any of the provided module types.
    """
    module_types = tuple(module_types)  # Convert to tuple if single type or list provided
    matching_layers = []
    if len(module_types) == 0:
        return matching_layers
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            matching_layers.append(name)
    return matching_layers

def replace_relu_with_out_of_place(module):
    """
    Recursively replaces all in-place ReLU activations with out-of-place ReLU in a given model.
    Works with models containing nested structures like ResNet.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))  # Replace ReLU
        else:
            replace_relu_with_out_of_place(child)  # Recursively apply to child modules

def check_for_inplace_operations(module, parent_name="model", raise_error=True):
    """
    Recursively checks all modules in a model and raises an error if any module
    has an in-place operation (e.g., ReLU(inplace=True)).
    """
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}"  # Track module path

        # Check for in-place ReLU or other modules with inplace=True attribute
        if hasattr(child, "inplace") and child.inplace:
            if raise_error:
                raise RuntimeError(f"In-place operation detected: {full_name} ({child})")
            # If raise_error is False, just issue a warning
            warnings.warn(
                f"Atleast one in-place operation detected: {full_name} ({child}), " + \
                "make sure this in intended."
            )
            return

        # Recursively check child modules
        check_for_inplace_operations(child, full_name)
