"""
Contrastive Loss Implementation for Semi-Supervised Learning of Molecular Conformers

This module implements an advanced contrastive loss function for semi-supervised learning
in the context of molecular conformer analysis. The contrastive loss encourages
embeddings of conformers from the same ensemble to be close together in the embedding space,
while pushing embeddings of conformers from different ensembles apart.

Key Features:
- Handles batch processing of conformer embeddings
- Supports variable number of atoms per conformer
- Utilizes efficient PyTorch operations for scalability
- Implements a margin-based loss for negative pairs
- Includes robust error handling and stability improvements

The loss function is defined as:
L = ∑(i,j) [ y(i,j) * ||z(i) - z(j)||^2 + (1 - y(i,j)) * max(0, margin - ||z(i) - z(j)||)^2 ]

where:
- z(i) and z(j) are flattened embeddings of two conformers
- y(i,j) is 1 if the conformers belong to the same ensemble, 0 otherwise
- margin is a hyperparameter defining the minimum distance between embeddings of different ensembles

Author: Utkarsh Sharma
Version: 1.2.0
Date: 07-22-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - Python 3.7+
    - PyTorch 1.9+
    - NumPy 1.20+ (for example usage)

Usage:
    from contrastive_loss import contrastive_loss

    # Prepare your data
    embeddings = torch.tensor(...)  # shape: (batch_size, num_atoms, embedding_dim)
    ensemble_ids = torch.tensor(...)  # shape: (batch_size,)

    # Compute the loss
    loss = contrastive_loss(embeddings, ensemble_ids, margin=1.0)

    # Use the loss in your training loop
    loss.backward()
    optimizer.step()

For a complete example, refer to the `main()` function in this file.

Change Log:
    - v1.2.0: Added robust error handling and stability improvements
    - v1.1.0: Updated to handle batch processing of conformer embeddings
    - v1.0.0: Initial implementation of contrastive loss for molecular conformers

TODO:
    - Implement an equivariant contrastive loss
    - Add support for custom distance metrics
    - Optimize performance for very large batch sizes
"""

import torch
import torch.nn.functional as F

EPSILON = 1e-8

def contrastive_loss(embeddings: torch.Tensor, ensemble_ids: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Compute the contrastive loss for semi-supervised pretraining of molecular conformers.

    This function calculates the contrastive loss for a batch of embeddings. It encourages
    embeddings of the same ensemble to be close together while separating embeddings of
    different ensembles by at least the specified margin.

    Args:
        embeddings (torch.Tensor): Tensor of shape (batch_size, num_atoms, embedding_dim) 
                                   containing the embeddings of the conformers.
        ensemble_ids (torch.Tensor): Tensor of shape (batch_size,) containing integer labels
                                     identifying which ensemble each conformer belongs to.
        margin (float, optional): The margin for the contrastive loss. Defaults to 1.0.

    Returns:
        torch.Tensor: A scalar tensor containing the computed contrastive loss.

    Raises:
        ValueError: If the shapes of embeddings and ensemble_ids are incompatible.
    """
    if embeddings.shape[0] != ensemble_ids.shape[0]:
        raise ValueError("Number of embeddings must match number of ensemble IDs")

    # Flatten embeddings to (batch_size, num_atoms * embedding_dim)
    flat_embeddings = embeddings.view(embeddings.shape[0], -1)

    # Compute pairwise Euclidean distances between all embeddings
    pairwise_distances = torch.cdist(flat_embeddings, flat_embeddings, p=2)
    
    # Create a binary matrix where entry (i,j) is 1 if ensemble_ids[i] == ensemble_ids[j], 0 otherwise
    ensemble_matrix = ensemble_ids.unsqueeze(0) == ensemble_ids.unsqueeze(1)
    ensemble_matrix = ensemble_matrix.float()
    
    # Compute loss for positive pairs (same ensemble)
    positive_loss = ensemble_matrix * pairwise_distances.pow(2)
    
    # Compute loss for negative pairs (different ensembles)
    negative_loss = (1 - ensemble_matrix) * F.relu(margin - pairwise_distances).pow(2)
    
    # Combine positive and negative losses
    loss = positive_loss + negative_loss
    
    # Remove self-comparisons and compute mean loss
    loss.fill_diagonal_(0)
    loss = loss.sum() / (len(ensemble_ids) * (len(ensemble_ids) - 1)) + EPSILON
    
    return loss

def main():
    """
    Example usage of the contrastive_loss function.
    
    This function demonstrates how to use the contrastive_loss function
    with synthetic data representing molecular conformer embeddings.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    batch_size = 10
    num_atoms = 5
    embedding_dim = 3
    embeddings = torch.randn(batch_size, num_atoms, embedding_dim)
    
    # Generate random ensemble IDs (0 to 2)
    ensemble_ids = torch.randint(0, 3, (batch_size,))
    
    # Compute contrastive loss
    loss = contrastive_loss(embeddings, ensemble_ids)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Ensemble IDs: {ensemble_ids}")
    print(f"Contrastive Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()