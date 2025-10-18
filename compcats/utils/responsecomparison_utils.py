"""Utils for comparing responses across models and fMRI data"""
from typing import Dict, Union, Tuple
import numpy as np
from scipy.stats import spearmanr as scipy_spearmanr
from sklearn.model_selection import KFold
import ot # POT: Python Optimal Transport
import torch

def pearsonr(vector1, vector2, return_pvalue=False):
    """
    Calculate Pearson correlation between two tensors
    Args:
        vector1 (torch.Tensor): 1D tensor
        vector2 (torch.Tensor): 1D tensor
    Returns:
        correlation (float): Pearson correlation
    """
    if return_pvalue:
        raise NotImplementedError('P-value calculation not implemented')
    if not torch.is_tensor(vector1) or not torch.is_tensor(vector2):
        raise ValueError('Input must be tensors')
    assert vector1.ndim == 1 and vector2.ndim == 1, 'Input must be 1D tensors'
    assert vector1.size(0) == vector2.size(0), 'Vectors must have the same length'
    with torch.no_grad():
        correlation = torch.corrcoef(torch.vstack((vector1, vector2)))[0, 1].item()
    return correlation

def rdmr(matrix1, matrix2, return_pvalue=False):
    """
    Calculate correlation between two RDMs
    Args:
        matrix1 (torch.Tensor): n_samples n_dimensions
        matrix2 (torch.Tensor): n_samples n_dimensions
    Returns:
        rho (float): Spearman's rank correlation
    """
    if not torch.is_tensor(matrix1) or not torch.is_tensor(matrix2):
        raise ValueError('Input must be tensors')
    assert matrix1.ndim == 2 and matrix2.ndim == 2, 'Input must be 2D tensors'
    assert matrix1.size(0) == matrix2.size(0), 'Matrices must have the same number of samples'
    with torch.no_grad():
        rsa1 = torch.cdist(matrix1, matrix1, p=2) / matrix1.size(0)
        rsa1 = rsa1[torch.tril(torch.ones_like(rsa1), diagonal=-1).bool()].flatten()
        rsa2 = torch.cdist(matrix2, matrix2, p=2) / matrix2.size(0)
        rsa2 = rsa2[torch.tril(torch.ones_like(rsa2), diagonal=-1).bool()].flatten()
        result = spearmanr(rsa1, rsa2, return_pvalue=return_pvalue)
        del rsa1, rsa2
    # Result is a tuple of rho and p-value if return_pvalue is True
    # Otherwise, it is just rho
    return result

def spearmanr(vector1, vector2, return_pvalue=False):
    """
    Calculate Spearman's rank correlation between two tensors
    Args:
        vector1 (torch.Tensor): 1D tensor
        vector2 (torch.Tensor): 1D tensor
        return_pvalue (bool): Whether to return p-value
    Returns:
        rho (float): Spearman's rank correlation
        p_value (float): P-value (if return_pvalue is True)
    """
    if not torch.is_tensor(vector1) or not torch.is_tensor(vector2):
        raise ValueError('Input must be tensors')
    assert vector1.ndim == 1 and vector2.ndim == 1, 'Input must be 1D tensors'
    assert vector1.size(0) == vector2.size(0), 'Vectors must have the same length'

    if return_pvalue:
        rho, p_value = scipy_spearmanr(vector1.cpu().numpy(), vector2.cpu().numpy())
        # assert np.isclose(scipy_rho, rho, atol=1e-3), \
        #     f"Spearman correlation mismatch: {scipy_rho} (scipy) vs {rho} (torch)"
        if isinstance(p_value, np.floating):
            p_value = p_value.item()
        return rho, p_value

    with torch.no_grad():
        # Step 1: Rank the data using argsort
        rank_x = torch.argsort(torch.argsort(vector1, descending=False), descending=False)
        rank_y = torch.argsort(torch.argsort(vector2, descending=False), descending=False)

        # Step 2: Compute the differences in ranks (d)
        d = rank_x - rank_y

        # Step 3: Compute Spearman's rank correlation
        n = vector1.size(0)
        # rho = 1 - (6 * torch.sum(d ** 2)) / (n * (n ** 2 - 1))
        rho = 1 - (6.0 * torch.sum(d.to(torch.float64) ** 2)) / (float(n) * (float(n)**2 - 1))
        rho = rho.item()
        assert -1 <= rho <= 1, f"Invalid Spearman correlation: {rho}"

        # Empty cuda cache
        del rank_x, rank_y, d

    return rho

def util_cdist(X,Y):
    norms_X = np.sum(X**2, axis=1)
    norms_Y = np.sum(Y**2, axis=1)
    cross_term = np.dot(X, Y.T)
    dist_matrix = norms_X[:, np.newaxis] + norms_Y[np.newaxis, :] - 2 * cross_term
    return dist_matrix  

def softmatchingdistance(X, Y, itermax=400):
    # https://github.com/anshksoni/NeuroAIMetrics/blob/main/utils/metrics.py#L111
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    if torch.is_tensor(Y):
        Y = Y.cpu().numpy()
    rem=[]
    for i in range(X.shape[1]):
        if np.all(X[:,i]==0):
            rem.append(i)
    X=np.delete(X, rem, axis=1)
    rem=[]
    for i in range(Y.shape[1]):
        if np.all(Y[:,i]==0):
            rem.append(i)
    Y=np.delete(Y, rem, axis=1)

    score=[]
    kf = KFold(n_splits=5,  shuffle=True, random_state = 42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        nx = X[train_idx].shape[1]
        ny = Y[train_idx].shape[1]
        dist_matrix = util_cdist(X[train_idx].T, Y[train_idx].T) 
        soft_assignments, log = ot.emd(
            np.ones(nx) / nx,
            np.ones(ny) / ny,
            dist_matrix,
            numItermax=100000*itermax,
            log=True
        )
        if log['warning'] != None:
            print('Did not converge, increase itermax')
            return np.nan
        dist_matrix = util_cdist(X[test_idx].T, Y[test_idx].T)
        score.append( np.sqrt(np.sum(soft_assignments * dist_matrix)) / len(test_idx))

    return score

def remove_constant_units(X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Remove units (columns) that have all-zero or constant responses.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_units)
        Response matrix.
    tol : float, optional
        Tolerance for detecting zero variance (default = 1e-8).

    Returns
    -------
    X_filtered : np.ndarray, shape (n_samples, n_units_filtered)
        Response matrix with constant or all-zero units removed.
    mask : np.ndarray, shape (n_units,)
        Boolean mask of retained units (True = kept).
    """
    # Compute variance along samples
    var = np.var(X, axis=0)

    # Keep only columns with variance > tol
    mask = var > tol
    X_filtered = X[:, mask]

    return X_filtered

def sfd_compute_normalization_params(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute normalization parameters from training data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_units)
        Training data matrix.

    Returns
    -------
    params : dict
        Dictionary containing column means, column norms, and Frobenius norm.
    """
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean
    col_norms = np.linalg.norm(X_centered, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1.0  # avoid divide-by-zero
    fro_norm = np.linalg.norm(X_centered / col_norms)
    return {"mean": mean, "col_norms": col_norms, "fro_norm": fro_norm}

def sfd_apply_normalization(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Apply training-based normalization to new data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_units)
        Data to normalize (test or validation set).
    params : dict
        Output from compute_normalization_params().

    Returns
    -------
    X_normed : array-like, shape (n_samples, n_units)
        Normalized data.
    """
    X_centered = X - params["mean"]
    X_scaled = X_centered / params["col_norms"]
    X_normed = X_scaled / params["fro_norm"]
    return X_normed

def sfd_train(X, Y, itermax=1000) -> Tuple[float, Dict[str, Union[np.ndarray, Dict]]]:
    """
    Soft Matching Distance training between two response matrices X and Y
    """
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    if torch.is_tensor(Y):
        Y = Y.cpu().numpy()
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    assert X.ndim == 2 and Y.ndim == 2, "X and Y must be 2D arrays"
    # Remove constant units
    X = remove_constant_units(X)
    Y = remove_constant_units(Y)
    # Compute normalization parameters
    X_params = sfd_compute_normalization_params(X)
    Y_params = sfd_compute_normalization_params(Y)
    # Apply normalization
    X_normed = sfd_apply_normalization(X, X_params)
    Y_normed = sfd_apply_normalization(Y, Y_params)
    # Compute soft matching distance training params
    nx = X_normed.shape[1]
    ny = Y_normed.shape[1]
    dist_matrix = util_cdist(X_normed.T, Y_normed.T)
    soft_assignments, log = ot.emd(
        np.ones(nx) / nx,
        np.ones(ny) / ny,
        dist_matrix,
        numItermax=100000*itermax,
        log=True
    )
    # Check convergence
    if log['warning'] is not None:
        print('Soft matching distance training did not converge, increase itermax, breakpointing for debug')
        print(log)
        # breakpoint()
        raise ValueError('Soft matching distance training did not converge, increase itermax')
    # Train soft matching distance
    score = np.sqrt(np.sum(soft_assignments * dist_matrix)) / X.shape[0]
    return score, {
        "X_norm_params": X_params,
        "Y_norm_params": Y_params,
        "soft_assignments": soft_assignments
    }

def sfd_test(X, Y, sfd_train_params):
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    if torch.is_tensor(Y):
        Y = Y.cpu().numpy()
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    assert X.ndim == 2 and Y.ndim == 2, "X and Y must be 2D arrays"
    # Remove constant units
    X = remove_constant_units(X)
    Y = remove_constant_units(Y)
    # Apply normalization
    X_normed = sfd_apply_normalization(X, sfd_train_params["X_norm_params"])
    Y_normed = sfd_apply_normalization(Y, sfd_train_params["Y_norm_params"])
    # Compute distance matrix
    dist_matrix = util_cdist(X_normed.T, Y_normed.T)
    # Compute soft matching distance
    score = np.sqrt(np.sum(sfd_train_params["soft_assignments"] * dist_matrix)) / X.shape[0]
    return score

def sfd(X_train, Y_train, X_test, Y_test, itermax=1000) -> Tuple[float, float]:
    """
    Soft Matching Distance between two response matrices X and Y
    # https://github.com/anshksoni/NeuroAIMetrics/blob/main/utils/metrics.py#L111
    Args:
        X_train (torch.Tensor): n_samples n_dimensions
        Y_train (torch.Tensor): n_samples n_dimensions
        X_test (torch.Tensor): n_samples n_dimensions
        Y_test (torch.Tensor): n_samples n_dimensions
        itermax (int): Maximum number of iterations for OT solver
    Returns:
        score (float): Soft Matching Distance
    """
    train_score, sfd_train_params = sfd_train(X_train, Y_train, itermax=itermax)
    test_score = sfd_test(X_test, Y_test, sfd_train_params)
    return train_score, test_score
