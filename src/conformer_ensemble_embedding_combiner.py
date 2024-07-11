"""
Conformer Ensemble Embedding Combiner

This module provides advanced functionality to combine embeddings from multiple conformers
of a molecule into a single representation. It implements three different methods
for combining these embeddings: Mean Pooling, Deep Sets, and Self-Attention.

The module separates scalar and vector components of the embeddings and processes
them separately to maintain equivariance properties. It now includes an additional
step to average results across all batches of a single ensemble.

Author: Utkarsh Sharma
Version: 1.2.0
Date: 07-11-2024 (MM-DD-YYYY)
License: MIT

Classes:
    ConformerEnsembleEmbeddingCombiner: The main class that implements the
    combining methods.

Functions:
    process_conformer_ensemble: Processes a single batch of conformer embeddings.
    process_ensemble_batches: Processes all batches for a single ensemble and
    averages the results.

Dependencies:
    - torch (>=1.9.0)
    - torch_scatter (>=2.0.8)

Usage:
    from conformer_ensemble_embedding_combiner import process_ensemble_batches
    
    ensemble_embeddings = process_ensemble_batches(list_of_batch_embeddings)

For detailed usage instructions, please refer to the README.md file.

Change Log:
    - v1.2.0: Added process_ensemble_batches function for ensemble-level averaging
    - v1.1.0: Improved handling of scalar and vector components
    - v1.0.0: Initial release with Mean Pooling, Deep Sets, and Self-Attention methods

TODO:
    - Implement GPU acceleration for large ensemble processing
    - Add support for custom combining methods
"""

import torch
import torch.nn as nn

class ConformerEnsembleEmbeddingCombiner(nn.Module):
    """
    A module that combines embeddings from multiple conformers using various methods.
    """

    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the ConformerEnsembleEmbeddingCombiner.

        Args:
            scalar_dim (int): Dimension of the scalar part of the embedding.
            vector_dim (int): Dimension of the vector part of the embedding.
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

        # Deep Sets
        self.deep_sets_phi_scalar = self._create_mlp(scalar_dim, scalar_dim)
        self.deep_sets_phi_vector = self._create_mlp(vector_dim * 3, vector_dim * 3)
        self.deep_sets_rho_scalar = self._create_mlp(scalar_dim, scalar_dim)
        self.deep_sets_rho_vector = self._create_mlp(vector_dim * 3, vector_dim * 3)

        # Self-Attention
        self.attention_scalar = nn.Linear(scalar_dim, scalar_dim)
        self.attention_vector = nn.Linear(vector_dim * 3, vector_dim * 3)
        self.self_attention_phi_scalar = self._create_mlp(scalar_dim, scalar_dim)
        self.self_attention_phi_vector = self._create_mlp(vector_dim * 3, vector_dim * 3)
        self.self_attention_rho_scalar = self._create_mlp(scalar_dim, scalar_dim)
        self.self_attention_rho_vector = self._create_mlp(vector_dim * 3, vector_dim * 3)

    @staticmethod
    def _create_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
        """
        Create a simple MLP with SiLU activation.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.

        Returns:
            nn.Sequential: The created MLP.
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )

    def mean_pooling(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple:
        """
        Perform mean pooling on scalar and vector embeddings.

        Args:
            scalar (torch.Tensor): Scalar part of the embeddings.
            vector (torch.Tensor): Vector part of the embeddings.

        Returns:
            tuple: Mean-pooled scalar and vector embeddings.
        """
        scalar_result = scalar.mean(dim=0, keepdim=True)
        vector_result = vector.mean(dim=0, keepdim=True)
        return scalar_result, vector_result

    def deep_sets(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple:
        """
        Apply the Deep Sets method to combine embeddings.

        Args:
            scalar (torch.Tensor): Scalar part of the embeddings.
            vector (torch.Tensor): Vector part of the embeddings.

        Returns:
            tuple: Combined scalar and vector embeddings using Deep Sets.
        """
        scalar = self.deep_sets_phi_scalar(scalar)
        vector = self.deep_sets_phi_vector(vector.reshape(vector.shape[0], vector.shape[1], -1))
        scalar = scalar.mean(dim=0, keepdim=True)
        vector = vector.mean(dim=0, keepdim=True)
        scalar = self.deep_sets_rho_scalar(scalar)
        vector = self.deep_sets_rho_vector(vector)
        return scalar, vector.reshape(1, vector.shape[1], self.vector_dim, 3)

    def self_attention(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple:
        """
        Apply the Self-Attention method to combine embeddings.

        Args:
            scalar (torch.Tensor): Scalar part of the embeddings.
            vector (torch.Tensor): Vector part of the embeddings.

        Returns:
            tuple: Combined scalar and vector embeddings using Self-Attention.
        """
        scalar = self.self_attention_phi_scalar(scalar)
        vector = self.self_attention_phi_vector(vector.reshape(vector.shape[0], vector.shape[1], -1))
        scalar_scores = self.attention_scalar(scalar)
        vector_scores = self.attention_vector(vector)
        
        attention_weights = torch.softmax(
            torch.matmul(scalar_scores, scalar_scores.transpose(1, 2)) +
            torch.matmul(vector_scores, vector_scores.transpose(1, 2)),
            dim=-1
        )
        
        scalar_weighted = torch.matmul(attention_weights, scalar)
        vector_weighted = torch.matmul(attention_weights, vector)
        
        scalar_encoded = scalar_weighted.sum(dim=0, keepdim=True)
        vector_encoded = vector_weighted.sum(dim=0, keepdim=True)
        
        scalar_encoded = self.self_attention_rho_scalar(scalar_encoded)
        vector_encoded = self.self_attention_rho_vector(vector_encoded)
        return scalar_encoded, vector_encoded.reshape(1, vector_encoded.shape[1], self.vector_dim, 3)

    def forward(self, conformer_embeddings: torch.Tensor) -> dict:
        """
        Process the conformer embeddings using all three methods.

        Args:
            conformer_embeddings (torch.Tensor): The input conformer embeddings.

        Returns:
            dict: A dictionary containing the results of all three methods.
        """
        num_conformers, num_atoms, total_dim = conformer_embeddings.shape
        scalar = conformer_embeddings[:, :, :self.scalar_dim]
        vector = conformer_embeddings[:, :, self.scalar_dim:].reshape(num_conformers, num_atoms, self.vector_dim, 3)
        
        mean_pooled_scalar, mean_pooled_vector = self.mean_pooling(scalar, vector)
        deep_sets_scalar, deep_sets_vector = self.deep_sets(scalar, vector)
        self_attention_scalar, self_attention_vector = self.self_attention(scalar, vector)

        return {
            'mean_pooling': (mean_pooled_scalar, mean_pooled_vector),
            'deep_sets': (deep_sets_scalar, deep_sets_vector),
            'self_attention': (self_attention_scalar, self_attention_vector)
        }

def process_conformer_ensemble(conformer_embeddings: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Process a batch of conformer embeddings.

    Args:
        conformer_embeddings (torch.Tensor): A tensor of shape (num_conformers, num_atoms, total_dim) 
                                             containing the embeddings for each conformer.

    Returns:
        dict: A dictionary containing the combined embeddings using different methods.
    """
    print(f"process_conformer_ensemble input shape: {conformer_embeddings.shape}")
    
    num_conformers, num_atoms, total_dim = conformer_embeddings.shape
    scalar_dim = total_dim // 4
    vector_dim = scalar_dim
    
    print(f"Num conformers: {num_conformers}, Num atoms: {num_atoms}, Total dim: {total_dim}")
    print(f"Scalar dim: {scalar_dim}, Vector dim: {vector_dim}")
    
    combiner = ConformerEnsembleEmbeddingCombiner(scalar_dim, vector_dim)
    results = combiner(conformer_embeddings)

    # Reshape results to have separate entries for each conformer
    for method, (scalar, vector) in results.items():
        results[method] = (
            scalar.repeat(num_conformers, 1, 1),  # [num_conformers, num_atoms, scalar_dim]
            vector.repeat(num_conformers, 1, 1, 1)  # [num_conformers, num_atoms, vector_dim, 3]
        )

    return results

def process_ensemble_batches(batch_embeddings: list[torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Process all batches of a single ensemble and average the results.

    Args:
        batch_embeddings (List[torch.Tensor]): A list of tensors, each representing a batch of conformer embeddings.

    Returns:
        dict: A dictionary containing the averaged embeddings across all batches using different methods.
    """
    # Concatenate all batches
    all_conformers = torch.cat(batch_embeddings, dim=0)
    
    # Process the concatenated tensor
    results = process_conformer_ensemble(all_conformers)
    
    # Average the results
    final_results = {}
    for method, (scalar, vector) in results.items():
        avg_scalar = scalar.mean(dim=0, keepdim=True)  # [1, num_atoms, scalar_dim]
        avg_vector = vector.mean(dim=0, keepdim=True)  # [1, num_atoms, vector_dim, 3]
        final_results[method] = (avg_scalar, avg_vector)

    return final_results