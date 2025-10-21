"""
Miscellaneous utility functions.
"""
from typing import List, Dict, Any, Union
import os
import json
import pandas as pd
import torch

# csv utils
def check_row_present(row_params, results_list):
    """
    Check if a row (dict) is already present in a list of results.
    Handles nested dicts/lists by comparing their JSON representations.

    Args:
        row_params (dict): Row parameters.
        results_list (list[dict]): List of existing results.

    Returns:
        dict | None: The matching row dict if found, otherwise None.
    """
    def normalize(v):
        """Convert values to a canonical comparable form."""
        # Convert dicts/lists recursively to sorted JSON strings
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, sort_keys=True)
            except TypeError:
                return str(v)
        return v

    for row in results_list:
        if all(normalize(row_params[key]) == normalize(row.get(key, None))
               for key in row_params):
            return row  # Return the full matching row
    return None

def to_csv(data: List[Dict[Union[str, int, float], Any]], path: str) -> None:
    """
    Save a list of dictionaries to CSV. If any columns contain dicts,
    they are JSON-encoded automatically.

    Parameters
    ----------
    data : list of dict
        Each dict represents a row (key = column name, value = value).
    path : str
        Output CSV file path.
    """
    if not data:
        raise ValueError("Input data list is empty")

    # Build DataFrame
    df = pd.DataFrame(data)

    # Detect columns that contain dicts in any row
    dict_cols = [
        col for col in df.columns
        if df[col].apply(lambda x: isinstance(x, dict)).any()
    ]

    # Convert dicts to JSON strings for saving
    for col in dict_cols:
        df[col] = df[col].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else pd.NA
        )

    # Save
    df.to_csv(path, index=False)

def read_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV created by `to_csv`, decoding any JSON-like columns
    back into Python dicts while preserving numeric dtypes.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with dict columns restored and numeric types preserved.
    """
    df = pd.read_csv(path, low_memory=False)

    def try_json_load(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (int, float)):  # already numeric
            return x
        if isinstance(x, str) and x.strip().startswith("{") and x.strip().endswith("}"):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return x
        return x

    # Detect and decode JSON-like columns
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if not sample.empty and sample.str.strip().str.startswith("{").any():
            df[col] = df[col].apply(try_json_load)

    # Re-infer best dtypes (Int64, Float64, string, etc.)
    df = df.convert_dtypes()

    return df

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
