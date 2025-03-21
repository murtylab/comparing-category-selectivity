from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import spearmanr

import compcats.utils.responsecomparison_utils as rcutils

import main_analysis as ma

def calculate_stress(original_distances, projected_distances):
    """
    Calculate the Kruskal's Stress between two distance matrices.
    
    Args:
    original_distances (torch.Tensor): The original distance matrix (n x n).
    projected_distances (torch.Tensor): The distance matrix after MDS (n x n).
    
    Returns:
    torch.Tensor: The stress value.
    """
    # Compute the squared differences between the original and projected distances
    diff = original_distances - projected_distances
    squared_diff = diff ** 2
    
    # Calculate the sum of squared differences
    numerator = torch.sum(squared_diff)
    
    # Calculate the sum of squared original distances
    denominator = torch.sum(original_distances ** 2)
    
    # Calculate stress
    stress = torch.sqrt(numerator / denominator)
    
    return stress.item()

model_fmri_frois = [
    ('FFA', 'FFA'),
    ('PPA', 'PPA'),
    ('EBA', 'EBA'),
]
fmri_frois = list({froi for froi, _ in model_fmri_frois})
tthreshold = 7
model_localizer = 'grillspector'

subjects_data, subjects_splithalf_pearsonrs, subjects_splithalf_rdmrs = \
    ma.get_allsubjects_nsd515_data(fmri_frois, tthreshold)
for froi in subjects_data.keys():
    for subject in subjects_data[froi].keys():
        subjects_data[froi][subject] = subjects_data[froi][subject].mean(axis=-1).cuda()
models_data = ma.get_allmodels_nsd515_data(
    model_fmri_frois, tthreshold, model_localizer, unitmean=True
)

annotate_models = False

for (model_froi, fmri_froi) in model_fmri_frois:
    print(f'Computing MDS for {model_froi} and {fmri_froi}')
    froi_subjects_data = subjects_data[fmri_froi]
    froi_models_data = models_data[(model_froi, fmri_froi)]
    # Ignore resnet50imagenet21k model
    if 'resnet50imagenet21k' in froi_models_data:
        print(f"Ignoring resnet50imagenet21k model for {model_froi} and {fmri_froi}")
        del froi_models_data['resnet50imagenet21k']
    froi_subjects_splithalf_pearsonrs = subjects_splithalf_pearsonrs[fmri_froi]

    keys = list(froi_subjects_data.keys()) + list(froi_models_data.keys())
    froi_splithalf_pearsonr_vector = []
    for key in keys:
        if key in froi_subjects_splithalf_pearsonrs:
            froi_splithalf_pearsonr_vector.append(froi_subjects_splithalf_pearsonrs[key])
        elif key in froi_models_data:
            froi_splithalf_pearsonr_vector.append(1)

    # Compute reliability matrix
    reliability_vector = torch.tensor(froi_splithalf_pearsonr_vector)
    reliability_matrix = torch.sqrt(reliability_vector[:, None] * reliability_vector[None, :])
    # Ensure diagonal elements are 1
    diag_indices = torch.arange(reliability_matrix.shape[0])
    reliability_matrix[diag_indices, diag_indices] = 1.0

    responses = []
    for key in keys:
        if key in froi_subjects_data:
            responses.append(froi_subjects_data[key])
        elif key in froi_models_data:
            responses.append(froi_models_data[key])

    responses = torch.stack(responses, dim=0)

    # Compute pairwise distances
    n_systems = responses.shape[0]
    distances = torch.zeros((n_systems, n_systems))

    for i in range(n_systems):
        for j in range(i + 1, n_systems):  # Compute only upper triangular (symmetric)
            dist = 1 - (rcutils.pearsonr(responses[i], responses[j]) / reliability_matrix[i, j])
            distances[i, j] = dist
            distances[j, i] = dist  # Symmetric matrix

    epsilon = 1e-5
    print(f'{distances[distances<=0].numel()} our of {distances.numel()} values are negative, clamping to {epsilon}')
    distances = torch.clamp(distances, min=epsilon)

    mds = MDS(n_components=2, dissimilarity='precomputed')
    mds_coords = mds.fit_transform(distances.cpu().numpy())

    # Test if MDS distances are correct
    mds_coords_tensor = torch.tensor(mds_coords)
    n_systems = len(keys)
    lowdim_distances = torch.zeros((n_systems, n_systems))
    for i in range(n_systems):
        for j in range(i + 1, n_systems):  # Compute only upper triangular (symmetric)
            dist = torch.norm(mds_coords_tensor[i] - mds_coords_tensor[j])
            lowdim_distances[i, j] = dist
            lowdim_distances[j, i] = dist  # Symmetric matrix
    print('Stress:', calculate_stress(distances, lowdim_distances))

    # Find which indices correspond to subjects and which to models
    subject_indices = [i for i, key in enumerate(keys) if key in froi_subjects_data]
    model_indices = [i for i, key in enumerate(keys) if key in froi_models_data]

    # Calculate mean distances from each model to all subjects
    model_to_subjects_orig = torch.zeros(len(model_indices))
    model_to_subjects_mds = torch.zeros(len(model_indices))

    for i, model_idx in enumerate(model_indices):
        # Mean distance from this model to all subjects in original space
        model_to_subjects_orig[i] = distances[model_idx, subject_indices].mean()
        
        # Mean distance from this model to all subjects in MDS space
        model_to_subjects_mds[i] = lowdim_distances[model_idx, subject_indices].mean()

    # Calculate Spearman rank correlation
    sp_corr, p_value = spearmanr(model_to_subjects_orig.numpy(), model_to_subjects_mds.numpy())

    print(f"Spearman rank correlation between original and MDS distances: {sp_corr:.3f} (p={p_value:.3e})")

    human_color = [0, 0, 1]
    model_color = [1, 0, 0]
    # colors = [human_color if key in froi_subjects_data else model_color for key in keys]
    facecolors = [human_color + [0.5] if key in froi_subjects_data else model_color + [0.5] for key in keys]
    edgecolors = [human_color + [1] if key in froi_subjects_data else model_color + [1] for key in keys]
    plt.figure(figsize=(5, 5), dpi=300)
    # plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=colors, s=100)
    plt.scatter(mds_coords[:, 0], mds_coords[:, 1], fc=facecolors, ec=edgecolors, s=150)

    if annotate_models:
        for i, key in enumerate(keys):
            if key in froi_models_data:
                plt.annotate(key, (mds_coords[i, 0], mds_coords[i, 1]), fontsize=8)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    # plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=human_color, markersize=10, label='Human'),
    #                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_color, markersize=10, label='Model')],
    #         loc='best')
    plt.title(f'{model_froi} and {fmri_froi}')
    plt.tight_layout()
    plt.savefig(f'results/plots/mds_univariate_{model_froi}_{fmri_froi}.pdf', dpi=300)
    plt.close()