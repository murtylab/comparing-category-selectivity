"""Base Model"""
import os
from tqdm import tqdm
import torch

from compcats.models.utils import hook_utils, layer_utils

class BaseModel(torch.nn.Module):
    """
    class for the Base model
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Default seed when untrained models are initialized without passing a seed
        self.DEFAULT_SEED = 42
        # Seed if passed
        self.seed = kwargs.get('seed', None)
        self.transforms = None
        self.model, self.name = self._load_model()
        self.cpu() # Move to device during intialization
        # Replace all in-place ReLU with out-of-place ReLU, so that pre-ReLU outputs are preserved
        layer_utils.replace_relu_with_out_of_place(self.model)
        # Check for in-place operations
        layer_utils.check_for_inplace_operations(self.model)
        # Initialize model if seed was set
        if self.seed:
            self._initialize_model()
        # Get all layer names
        self.all_layer_names = layer_utils.get_all_layer_names(self.model)
        # Store irrelevant layer names, to be overridden in child classes
        self.relevant_layer_names = None
        self.irrelevant_layer_names = None

    def forward(self, x):
        return self.model(x)

    def train_step(self, x, target, criterion, optimizer):
        raise NotImplementedError("train_step method should be updated: only boilerplate code is provided")
        optimizer.zero_grad()
        output = self.forward(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()

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

    def _initialize_model(self):
        """
        Initializes a model using its default initialization methods while 
            preserving global randomness.
        Raises an error if a module has weights but does not implement reset_parameters().
        
        Args:
            model (nn.Module): The PyTorch model to initialize.
            seed (int): Seed for deterministic weight initialization.
        
        Returns:
            nn.Module: The initialized model.
        """
        print(f"Initializing {self.name} with seed = {self.seed} after loading the model")
        gen = torch.Generator(device = self.device())
        gen.manual_seed(self.seed)  # Use a separate generator to avoid affecting global randomness

        def init_weights(m):
            if hasattr(m, 'reset_parameters'):
                with torch.no_grad():
                    m.reset_parameters()  # Use the default initialization
            elif any(p.requires_grad for p in m.parameters()):  # Check if module has weights
                raise ValueError(f"Module {m} has weights but does not implement 'reset_parameters', \
                                 intialization will not consistent")

        # Apply the initialization recursively to all submodules in the model
        def apply_recursive(m):
            # Initialize the current module's weights
            if isinstance(m, torch.nn.Module):  # Check if it's a container module (like nn.Sequential)
                for child in m.children():
                    apply_recursive(child)  # Apply recursively to submodules
            else:
                if not not isinstance(m, torch.nn.Sequential):
                    init_weights(m)

        apply_recursive(self.model) # Apply initialization to all layers recursively

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
        # Turn on evaluation mode
        # Very important to turn on evaluation mode to avoid any dropout
        self.eval()
        # Storing activations
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

    def generate_dataset_activations(self, layers, dataset, save_dir=None, batch_size=768, device='cuda'):
        """
        Function to generate activations for a dataset
        Args:
            layer(str): Layer name
            dataset(torch.utils.data.Dataset): Dataset
            save_dir(str): Directory to save activations without the layer name
        """
        if isinstance(layers, str):
            layers = [layers]
        # init activations
        activations = {}

        # Error handling for path for compatibility with multiple layers
        # If only one layer is requested and save_dir is provided with layer name as the last folder
        # then remove the layer name from the save_dir
        if save_dir and len(layers) == 1:
            last_folder_name = save_dir.split('/')[-1]
            if last_folder_name == layers[0]:
                save_dir = '/'.join(save_dir.split('/')[:-1])

        # Check which layers to generate, and load already generated activations
        layers_to_generate = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for layer in layers:
                layer_dir = f'{save_dir}/{layer}'
                activations_file_name = f'{self.name}_{layer}_{dataset.name}.pt'
                if os.path.isdir(layer_dir):
                    if activations_file_name not in os.listdir(layer_dir):
                        layers_to_generate.append(layer)
                    else:
                        layer_activations = torch.load(f'{layer_dir}/{activations_file_name}')
                        layer_activations = layer_activations.to(device)
                        activations[layer] = layer_activations
                else:
                    os.makedirs(layer_dir)
                    layers_to_generate.append(layer)
        else:
            layers_to_generate = layers

        # If all activations are already generated, load them and return
        if len(layers_to_generate) == 0:
            if len(layers) == 1:
                activations = activations[layers[0]]
            return activations

        # Start generating activations
        # Setup
        # Set device
        self.to(device)
        # Create a DataLoader to iterate over the dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # Turn on evaluation mode
        self.eval()
        # Generate activations
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Send images to device
                batch = batch.to(device)
                # Forward pass
                outputs_batch_layers = self.get_activations(batch, layers_to_generate)
                # Store activations
                for layer in layers_to_generate:
                    layer_output = outputs_batch_layers[layer]
                    if layer in activations:
                        activations[layer] = torch.vstack((activations[layer], layer_output))
                    else:
                        activations[layer] = layer_output

        # Store activations
        if save_dir:
            for layer in layers_to_generate:
                layer_dir = f'{save_dir}/{layer}'
                # Layer dir should already exist
                activations_file_name = f'{self.name}_{layer}_{dataset.name}.pt'
                torch.save(activations[layer].cpu(), f'{layer_dir}/{activations_file_name}')

        # Return activations
        if len(layers) == 1:
            activations = activations[layers[0]]
        return activations


    def get_froi_activations(self, img, layer_name, froi_mask):
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
