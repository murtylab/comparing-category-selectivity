"""
Miscellaneous utility functions.
"""
import os

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