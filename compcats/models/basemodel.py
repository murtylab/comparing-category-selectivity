"""Base Model"""
import torch

from compcats.models.utils import hook_utils, layer_utils

class BaseModel():
    """
    class for the Base model
    """
    def __init__(self, **kwargs):
        self.transforms = None
        self.model, self.name = self._load_model()
        self.cpu() # Move to device during intialization
        # Replace all in-place ReLU with out-of-place ReLU, so that pre-ReLU outputs are preserved
        layer_utils.replace_relu_with_out_of_place(self.model)
        # Check for in-place operations
        layer_utils.check_for_inplace_operations(self.model)
        # Get all layer names
        self.all_layer_names = layer_utils.get_all_layer_names(self.model)

    def to(self, device):
        """
        Send model to GPU or CPU
        """
        self.model = self.model.to(device)

    def cpu(self):
        """
        Send model to CPU
        """
        self.to("cpu")
    
    def cuda(self):
        """
        Send model to CPU
        """
        self.to("cuda")
    
    def device(self):
        """
        Get device
        """
        return next(self.model.parameters()).device

    def eval(self):
        """
        Turn on Evaluation mode
        """
        self.model.eval()

    def _load_model(self):
        """
        Load the model
        """
        raise NotImplementedError("_load_model method should be overridden")

    def get_activations(self, batch, layers):
        """
        Args:
            batch (torch.Tensor): Image batch
            layer_name(str or list): Number of layer after which activations needed
        """
        # Storign activations
        activations = {}
        # If only one layer is requested, convert it to a list
        if isinstance(layers, str):
            layers_list = [list]
        else:
            layers_list = layers
        # Register hooks
        hooks = {}
        for layer in layers_list:
            hook = hook_utils.ForwardHook(self.model, layer)
            hooks[layer] = hook
        # Forward pass
        with torch.no_grad():
            _ = self.model(batch)
        # Get activations for all layers requested
        for layer in layers_list:
            activations[layer] = hooks[layer].output
            hooks[layer].close()
        # If only one layer is requested, return only the activations
        if isinstance(layers, str):
            activations = activations[layers]
        return activations
        if len(layers) == 1:
            activations = activations[layers[0]]
        return activations

    def get_froi_activations(self, img, layer_name, unit_indices):
        """
        Args:
            img(torch.Tensor): Image
            layer_name(int): Number of layer after which activations needed
            unit_indices(torch.Tensor): Indices of the units in the fROI
        """
        raise NotImplementedError("get_froi_activations method not yet implemented")
        activations = self.get_activations(img, layer_name)
        activations = activations.view(activations.size(0), -1)
        froi_activations = activations[unit_indices]
        return froi_activations
