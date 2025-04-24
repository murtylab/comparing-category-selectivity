import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_fn(loss_version, loss_temperature):
    """
    Get the loss function based on the version and temperature.
    """
    if loss_version == 'v1':
        return FaceContrastiveLossv1(temperature=loss_temperature)
    elif loss_version == 'v2':
        raise NotImplementedError("Loss version v2 is not implemented yet.")
    else:
        raise ValueError(f"Unknown loss version: {loss_version}")

class FaceContrastiveLossv1(nn.Module):
    """
    This loss brings closer the positive pairs (symmetrically) and pushes away the negative pairs.
    It is a hybrid of supervised contrastive loss and SimCLR.
    The loss is computed as the average of the cross-entropy losses for two sets of logits.
    The logits are computed from the cosine similarity between the normalized feature vectors.
    The temperature parameter controls the scaling of the logits.
    """
    def __init__(self, temperature=0.5, neg_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.neg_weight = neg_weight

    def forward(self, pos1, pos2, neg):
        """
        Args:
            pos1: Feature vectors for the first positive samples (shape: [B, D])
            pos2: Feature vectors for the second positive samples (shape: [B, D])
            neg: Feature vectors for the negative samples (shape: [B, N, D])
        Returns:
            loss: Computed contrastive loss (scalar)
        """
        # Normalize the feature vectors
        pos1, pos2, neg = map(lambda x: F.normalize(x, dim=1), (pos1, pos2, neg))

        # Compute the cosine similarity between positive pairs
        pos_sim = (pos1 * pos2).sum(dim=1) / self.temperature  # [B]

        # Compute the cosine similarity between positive and negative samples
        neg_sim_z1 = (pos1 @ neg.T) / self.temperature  # [B, N]
        neg_sim_z2 = (pos2 @ neg.T) / self.temperature  # [B, N]

        # Apply negative weight (only to negative logits)
        neg_sim_z1 *= self.neg_weight
        neg_sim_z2 *= self.neg_weight

        # Concatenate positive and negative logits
        logits_z1 = torch.cat([pos_sim.unsqueeze(1), neg_sim_z1], dim=1)  # [B, 1+N]
        logits_z2 = torch.cat([pos_sim.unsqueeze(1), neg_sim_z2], dim=1)  # [B, 1+N]

        # Compute the labels for the positive samples
        labels = torch.zeros(pos1.size(0), dtype=torch.long, device=pos1.device)

        # Compute the cross-entropy loss for both views
        loss_z1 = F.cross_entropy(logits_z1, labels)
        loss_z2 = F.cross_entropy(logits_z2, labels)

        return (loss_z1 + loss_z2) / 2
