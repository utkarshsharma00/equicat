"""
Contrastive Loss Implementation for Semi-Supervised Learning

This module implements a contrastive loss function for semi-supervised learning
in the context of molecular conformer analysis. The contrastive loss encourages
embeddings of the same molecule to be close together in the embedding space,
while pushing embeddings of different molecules apart.

The loss function is defined as:
L = ∑(i,j) [ y(i,j) * ||z(i) - z(j)||^2 + (1 - y(i,j)) * max(0, margin - ||z(i) - z(j)||)^2 ]

where:
- z(i) and z(j) are embeddings of two conformers
- y(i,j) is 1 if the conformers belong to the same molecule, 0 otherwise
- margin is a hyperparameter defining the minimum distance between embeddings of different molecules

This implementation is optimized for batch processing and uses efficient PyTorch operations.

Author: Utkarsh Sharma
Version: 1.0.0
Date: 07-11-2024 (MM-DD-YYYY)
License: MIT
"""

import torch
import torch.nn.functional as F

def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Compute the contrastive loss for semi-supervised pretraining of molecular conformers.

    This function calculates the contrastive loss for a batch of embeddings. It encourages
    embeddings of the same molecule to be close together while separating embeddings of
    different molecules by at least the specified margin.

    Args:
        embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim) containing
                                   the embeddings of the conformers.
        labels (torch.Tensor): Tensor of shape (batch_size,) containing integer labels
                               identifying which molecule each conformer belongs to.
        margin (float, optional): The margin for the contrastive loss. Defaults to 1.0.

    Returns:
        torch.Tensor: A scalar tensor containing the computed contrastive loss.

    Raises:
        ValueError: If the shapes of embeddings and labels are incompatible.

    Example:
        >>> embeddings = torch.randn(10, 32)  # 10 conformers, 32-dimensional embeddings
        >>> labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # 5 molecules, 2 conformers each
        >>> loss = contrastive_loss(embeddings, labels)
    """
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must match number of labels")

    # Compute pairwise Euclidean distances between all embeddings
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    
    # Create a binary matrix where entry (i,j) is 1 if labels[i] == labels[j], 0 otherwise
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    label_matrix = label_matrix.float()
    
    # Compute loss for positive pairs (same molecule)
    positive_loss = label_matrix * pairwise_distances.pow(2)
    
    # Compute loss for negative pairs (different molecules)
    negative_loss = (1 - label_matrix) * F.relu(margin - pairwise_distances).pow(2)
    
    # Combine positive and negative losses
    loss = positive_loss + negative_loss
    
    # Remove self-comparisons and compute mean loss
    loss.fill_diagonal_(0)
    loss = loss.sum() / (len(labels) * (len(labels) - 1))
    
    return loss

# Example usage
# if __name__ == "__main__":
#     # Generate random embeddings and labels for demonstration
#     batch_size, embedding_dim = 10, 32
#     embeddings = torch.randn(batch_size, embedding_dim)
#     labels = torch.randint(0, 5, (batch_size,))  # 5 different molecules
#
#     # Compute and print the loss
#     loss = contrastive_loss(embeddings, labels)
#     print(f"Contrastive Loss: {loss.item()}")