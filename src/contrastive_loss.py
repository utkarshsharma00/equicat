"""
Contrastive Loss Implementation for Semi-Supervised Learning of Molecular Conformers (GPU-enabled version)

This module implements an advanced contrastive loss function for semi-supervised learning
in the context of molecular conformer analysis. The contrastive loss encourages
embeddings of conformers from the same ensemble to be close together in the embedding space,
while pushing embeddings of conformers from different ensembles apart.

Key Features:
- GPU support for all operations
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
Version: 2.0.0
Date: 08-01-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - Python 3.10+
    - PyTorch 1.9+

Usage:
    from contrastive_loss import contrastive_loss

    # Prepare your data (assuming it's already on the correct device)
    embeddings = torch.tensor(...)  # shape: (batch_size, num_atoms, embedding_dim)
    ensemble_ids = torch.tensor(...)  # shape: (batch_size,)

    # Compute the loss
    loss = contrastive_loss(embeddings, ensemble_ids, margin=1.0)

    # Use the loss in your training loop
    loss.backward()
    optimizer.step()

For a complete example, refer to the `main()` function in this file.

Change Log:
    - v2.0.0: Added GPU support and ensured compatibility with updated equicat.py and train.py
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

    # Ensure inputs are on the same device
    device = embeddings.device
    ensemble_ids = ensemble_ids.to(device)

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

def move_to_device(obj, device):
    """
    Recursively moves an object to the specified device.

    Args:
        obj: The object to move (can be a tensor, list, tuple, or dict)
        device: The device to move the object to

    Returns:
        The object moved to the specified device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        return obj

# Example usage
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate some random embeddings and ensemble IDs
    embeddings = torch.randn(10, 5, 3).to(device)  # 10 conformers, 5 atoms, 3D embeddings
    ensemble_ids = torch.randint(0, 3, (10,)).to(device)  # 3 different ensembles
    
    # Compute the loss
    loss = contrastive_loss(embeddings, ensemble_ids)
    
    print(f"Computed loss: {loss.item()}")
    print(f"Loss device: {loss.device}")

if __name__ == "__main__":
    main()