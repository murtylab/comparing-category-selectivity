"""
Miscellaneous utility functions.
"""
import os
import torch

def check_row_present(row_params, results_list):
    """
    Check if row is present in the results list.
    Args:
        row_params (dict): Row parameters.
        results_list (list): List of results.
    Returns:
        row_present (bool): True if row is present, False otherwise.
    """
    row_present = None
    # Skip if row is already present
    for row in results_list:
        if all(row_params[key] == row.get(key, None) for key in row_params):
            row_present = row
            break
    return row_present

def delete_activations(activations_dir, layer_names):
    """
    Delete saved activations except for the specified layers to save space
    Args:
        activations_dir (str): Activations directory
        layer_names (list): List of layer names
    """
    num_layers_found = 0
    for layer in layer_names:
        layer_activations_dir = f'{activations_dir}/{layer}'
        if os.path.exists(layer_activations_dir):
            os.system(f'rm -rf {layer_activations_dir}')
            num_layers_found += 1
    if num_layers_found > 0:
        print(f'Deleted activations for {num_layers_found} layers in {activations_dir}')

def chunk_numbers(N, chunk_size=10):
    """
    Splits a range of numbers from 0 to N into chunks of specified size.
    Parameters:
        N (int): The upper limit of the range (inclusive).
        chunk_size (int, optional): The size of each chunk. Default is 10.
    Returns:
        list of tuples: A list where each tuple contains the start and end indices of each chunk.
    """
    return [(i, min(i + chunk_size, N + 1)) for i in range(0, N + 1, chunk_size)]

def get_device():
    """
    Get the available device (GPU if available, else CPU).
    Returns:
        device (torch.device): The available device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def clear_memory(device):
    """
    Clear memory by emptying CUDA cache if GPU is available.
    Args:
        device (torch.device): The device to clear memory for.
    """
    # Error handling
    if isinstance(device, str):
        device = torch.device(device)
    # Clear memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
