"""
Conformer Ensemble Embedding Combiner

This module provides functionality to combine embeddings from multiple conformers
of a molecule into a single representation. It implements three different methods
for combining these embeddings: Mean Pooling, Deep Sets, and Self-Attention.

The module separates scalar and vector components of the embeddings and processes
them separately to maintain equivariance properties.

Author: Utkarsh Sharma
Version: 1.0.0
Date: 07-07-2024 (MM-DD-YYYY)
License: MIT

Classes:
    ConformerEnsembleEmbeddingCombiner: The main class that implements the
    combining methods.

Functions:
    process_conformer_ensemble: A utility function to prepare and process
    conformer embeddings using the ConformerEnsembleEmbeddingCombiner.

Dependencies:
    - torch (>=1.9.0)
    - torch_scatter (>=2.0.8)
    - e3nn (>=0.4.0)

Usage:
    from conformer_ensemble_embedding_combiner import process_conformer_ensemble
    
    ensemble_embeddings = process_conformer_ensemble(conformer_embeddings)

For detailed usage instructions, please refer to the README.md file.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3

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

    def mean_pooling(self, scalar: torch.Tensor, vector: torch.Tensor, batch: torch.Tensor) -> tuple:
        """
        Perform mean pooling on scalar and vector embeddings.

        Args:
            scalar (torch.Tensor): Scalar part of the embeddings.
            vector (torch.Tensor): Vector part of the embeddings.
            batch (torch.Tensor): Batch indices for each embedding.

        Returns:
            tuple: Mean-pooled scalar and vector embeddings.
        """
        scalar_result = scatter(scalar, batch, dim=0, reduce='mean')
        vector_result = scatter(vector, batch, dim=0, reduce='mean')
        return scalar_result, vector_result

    def deep_sets(self, scalar: torch.Tensor, vector: torch.Tensor, batch: torch.Tensor) -> tuple:
        """
        Apply the Deep Sets method to combine embeddings.

        Args:
            scalar (torch.Tensor): Scalar part of the embeddings.
            vector (torch.Tensor): Vector part of the embeddings.
            batch (torch.Tensor): Batch indices for each embedding.

        Returns:
            tuple: Combined scalar and vector embeddings using Deep Sets.
        """
        scalar = self.deep_sets_phi_scalar(scalar)
        vector = self.deep_sets_phi_vector(vector.view(vector.shape[0], -1))
        scalar = scatter(scalar, batch, dim=0, reduce='mean')
        vector = scatter(vector, batch, dim=0, reduce='mean')
        scalar = self.deep_sets_rho_scalar(scalar)
        vector = self.deep_sets_rho_vector(vector)
        return scalar, vector.view(vector.shape[0], -1, 3)

    def self_attention(self, scalar: torch.Tensor, vector: torch.Tensor, batch: torch.Tensor) -> tuple:
        """
        Apply the Self-Attention method to combine embeddings.

        Args:
            scalar (torch.Tensor): Scalar part of the embeddings.
            vector (torch.Tensor): Vector part of the embeddings.
            batch (torch.Tensor): Batch indices for each embedding.

        Returns:
            tuple: Combined scalar and vector embeddings using Self-Attention.
        """
        scalar = self.self_attention_phi_scalar(scalar)
        vector = self.self_attention_phi_vector(vector.view(vector.shape[0], -1))
        scalar_scores = self.attention_scalar(scalar)
        vector_scores = self.attention_vector(vector)
        
        unique_batches = batch.unique()
        scalar_outputs, vector_outputs = [], []
        
        for b in unique_batches:
            mask = (batch == b)
            scalar_batch, vector_batch = scalar[mask], vector[mask]
            scalar_scores_batch, vector_scores_batch = scalar_scores[mask], vector_scores[mask]
            
            attention_weights = torch.softmax(
                torch.matmul(scalar_scores_batch, scalar_scores_batch.transpose(0, 1)) +
                torch.matmul(vector_scores_batch, vector_scores_batch.transpose(0, 1)),
                dim=1
            )
            
            scalar_weighted = torch.matmul(attention_weights, scalar_batch)
            vector_weighted = torch.matmul(attention_weights, vector_batch)
            
            scalar_aggregated = scalar_weighted.sum(dim=0, keepdim=True)
            vector_aggregated = vector_weighted.sum(dim=0, keepdim=True)
            
            scalar_outputs.append(scalar_aggregated)
            vector_outputs.append(vector_aggregated)
        
        scalar_encoded = torch.cat(scalar_outputs, dim=0)
        vector_encoded = torch.cat(vector_outputs, dim=0)
        
        scalar_encoded = self.self_attention_rho_scalar(scalar_encoded)
        vector_encoded = self.self_attention_rho_vector(vector_encoded)
        return scalar_encoded, vector_encoded.view(vector_encoded.shape[0], -1, 3)

    def forward(self, conformer_embeddings: torch.Tensor, batch_indices: torch.Tensor) -> dict:
        """
        Process the conformer embeddings using all three methods.

        Args:
            conformer_embeddings (torch.Tensor): The input conformer embeddings.
            batch_indices (torch.Tensor): Batch indices for each embedding.

        Returns:
            dict: A dictionary containing the results of all three methods.
        """
        total_dim = conformer_embeddings.shape[1]
        scalar_dim = min(self.scalar_dim, total_dim // 4)
        vector_dim = min(self.vector_dim, (total_dim - scalar_dim) // 3)
        scalar = conformer_embeddings[:, :scalar_dim]
        vector = conformer_embeddings[:, scalar_dim:scalar_dim + vector_dim * 3].view(-1, vector_dim, 3)
        
        mean_pooled_scalar, mean_pooled_vector = self.mean_pooling(scalar, vector, batch_indices)
        deep_sets_scalar, deep_sets_vector = self.deep_sets(scalar, vector, batch_indices)
        self_attention_scalar, self_attention_vector = self.self_attention(scalar, vector, batch_indices)

        return {
            'mean_pooling': (mean_pooled_scalar, mean_pooled_vector),
            'deep_sets': (deep_sets_scalar, deep_sets_vector),
            'self_attention': (self_attention_scalar, self_attention_vector)
        }

def process_conformer_ensemble(conformer_embeddings: torch.Tensor) -> dict:
    """
    Process a batch of conformer embeddings.

    Args:
        conformer_embeddings (torch.Tensor): A tensor of shape (num_conformers, ...) 
                                             containing the embeddings for each conformer.

    Returns:
        dict: A dictionary containing the combined embeddings using different methods.
    """
    print(f"process_conformer_ensemble input shape: {conformer_embeddings.shape}")
    
    num_conformers = conformer_embeddings.shape[0]
    flattened_embeddings = conformer_embeddings.view(num_conformers, -1)
    total_dim = flattened_embeddings.shape[1]
    
    scalar_dim = total_dim // 4
    vector_dim = scalar_dim
    
    print(f"Num conformers: {num_conformers}, Total dim: {total_dim}")
    print(f"Scalar dim: {scalar_dim}, Vector dim: {vector_dim}")
    
    batch_indices = torch.arange(num_conformers)
    combiner = ConformerEnsembleEmbeddingCombiner(scalar_dim, vector_dim)
    results = combiner(flattened_embeddings, batch_indices)

    return results

if __name__ == "__main__":
    # Add any test or example code here
    pass