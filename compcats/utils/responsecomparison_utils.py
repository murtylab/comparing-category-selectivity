"""Utils for comparing responses across models and fMRI data"""
import torch
from torch.distributions import StudentT

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
    Returns:
        rho (float): Spearman's rank correlation
    """
    if not torch.is_tensor(vector1) or not torch.is_tensor(vector2):
        raise ValueError('Input must be tensors')
    assert vector1.ndim == 1 and vector2.ndim == 1, 'Input must be 1D tensors'
    assert vector1.size(0) == vector2.size(0), 'Vectors must have the same length'
    with torch.no_grad():
        # Step 1: Rank the data using argsort
        rank_x = torch.argsort(torch.argsort(vector1, descending=False), descending=False)
        rank_y = torch.argsort(torch.argsort(vector2, descending=False), descending=False)

        # Step 2: Compute the differences in ranks (d)
        d = rank_x - rank_y

        # Step 3: Compute Spearman's rank correlation
        n = vector1.size(0)
        rho = 1 - (6 * torch.sum(d ** 2)) / (n * (n ** 2 - 1))

        if return_pvalue:
            raise NotImplementedError('P-value calculation not implemented')
            # Step 4: Compute the t-statistic
            t_statistic = rho * \
                torch.sqrt(torch.tensor(n - 2, dtype=torch.float32)) / \
                    torch.sqrt(1 - rho ** 2)

            # Step 5: Compute the p-value using torch.distributions.StudentT
            student_t = StudentT(df=n-2)
            p_value = 2 * (1 - student_t.cdf(torch.abs(t_statistic)))
            del student_t

        # Empty cuda cache
        del rank_x, rank_y, d

    if return_pvalue:
        return rho.item(), p_value.item()
    return rho.item()
