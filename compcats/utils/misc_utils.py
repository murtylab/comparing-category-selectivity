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
    row_present = None
    # Skip if row is already present
    for row in results_list:
        if all(row_params[key] == row.get(key, None) for key in row_params):
            row_present = row
            break
    return row_present
