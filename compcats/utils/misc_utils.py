"""
Miscellaneous utility functions.
"""

def check_row_present(row_params, results_list):
    """
    Check if row is present in the results list.
    Args:
        row_params (dict): Row parameters.
        results_list (list): List of results.
    Returns:
        row_present (bool): True if row is present, False otherwise.
    """
    row_present = False
    # Skip if row is already present
    if any(all(row_params[key] == d.get(key, None) for key in row_params) for d in results_list):
        row_present = True
    return row_present
