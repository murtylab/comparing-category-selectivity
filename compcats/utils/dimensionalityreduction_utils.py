import torch

def mds_pytorch(
    dist_matrix, n_components=2, max_iter=1000, lr=0.01, tol=1e-5, log=True, device='cuda'
):
    """
    Perform MDS using stress minimization in PyTorch.
    
    Args:
        dist_matrix (Tensor): The pairwise distance matrix of shape (n_samples, n_samples)
        n_components (int): The number of dimensions to reduce to (default is 2)
        max_iter (int): The maximum number of iterations for optimization (default is 1000)
        lr (float): The learning rate for gradient descent (default is 0.01)
        tol (float): The tolerance for convergence (default is 1e-5)
        log (bool): Whether to print the stress at each iteration (default is True)
        device (str): The device to use for computation (default is 'cuda')
    Returns:
        Tensor: A matrix of shape (n_samples, n_components) with the reduced data
        float: The final stress value
    """
    n_samples = dist_matrix.size(0)

    # Randomly initialize the lower-dimensional configuration
    Y = torch.randn((n_samples, n_components), requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam([Y], lr=lr)

    def compute_stress(Y, dist_matrix):
        # Compute pairwise distances in the lower-dimensional space
        dist_lower = torch.cdist(Y, Y)
        # Compute stress (the difference between original distances and the ones in lower dimension)
        stress = torch.sqrt(torch.sum((dist_lower - dist_matrix) ** 2))
        return stress
    
    # Move data to device
    dist_matrix = dist_matrix.to(device)
    Y = Y.to(device)

    for iteration in range(max_iter):
        optimizer.zero_grad()

        # Calculate the stress
        stress = compute_stress(Y, dist_matrix)

        # Compute gradients and update Y
        stress.backward()
        optimizer.step()

        if log:
            # Print stress every 100 iterations for monitoring
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Stress = {stress.item()}")

            # Check for convergence (if stress change is below tolerance)
            if iteration > 0 and abs(previous_stress - stress.item()) < tol:
                print(f"Converged at iteration {iteration}")
                break

        previous_stress = stress.item()

    return Y.detach().cpu().numpy(), stress.item()
