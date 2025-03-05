import torch.nn as nn

def get_all_layer_names(model):
    """
    Get all layer names in a model
    """
    return [name for name, _ in model.named_modules() if name != ""]


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

def check_for_inplace_operations(module, parent_name="model"):
    """
    Recursively checks all modules in a model and raises an error if any module
    has an in-place operation (e.g., ReLU(inplace=True)).
    """
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}"  # Track module path

        # Check for in-place ReLU or other modules with inplace=True attribute
        if hasattr(child, "inplace") and child.inplace:
            raise RuntimeError(f"In-place operation detected: {full_name} ({child})")

        # Recursively check child modules
        check_for_inplace_operations(child, full_name)