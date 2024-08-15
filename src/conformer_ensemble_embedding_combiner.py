"""
Conformer Ensemble Embedding Combiner with PCA Visualization

This module provides advanced functionality to combine embeddings from multiple conformers of
a molecule into a single representation, now with GPU support. 

It implements five different methods for combining these embeddings: Mean Pooling, Deep Sets, 
Self-Attention, Improved Deep Sets, and Improved Self-Attention. Additionally, it now includes 
PCA visualization for each method to understand how conformers are being clustered.

The module separates scalar and vector components of the embeddings and processes them separately 
to maintain equivariance properties. It includes an additional step to average results across all 
batches of a single ensemble.

New features:
- PCA visualization for each embedding combination method
- Enhanced debug printing for all five combination methods
- Improved error handling and input validation

Author: Utkarsh Sharma
Version: 2.1.0
Date: 08-13-2024 (MM-DD-YYYY)
License: MIT

Classes:
    DeepSets: Implementation of the Deep Sets method
    SelfAttention: Implementation of the Self-Attention method
    ImprovedDeepSets: Enhanced version of Deep Sets with increased complexity
    ImprovedSelfAttention: Enhanced version of Self-Attention with multi-head attention
    ConformerEnsembleEmbeddingCombiner: The main class that implements all
    combining methods.

Functions:
    process_conformer_ensemble: Processes a single batch of conformer embeddings.
    process_ensemble_batches: Processes all batches for a single ensemble and
    averages the results.
    visualize_embeddings: Performs PCA and visualizes the embeddings for each method.

Dependencies:
    - torch (>=1.9.0)
    - torch_scatter (>=2.0.8)
    - sklearn (>=0.24.0)
    - matplotlib (>=3.3.0)

Usage:
    from conformer_ensemble_embedding_combiner import process_ensemble_batches, visualize_embeddings
    
    ensemble_embeddings = process_ensemble_batches(list_of_batch_embeddings)
    visualize_embeddings(ensemble_embeddings)

For detailed usage instructions, please refer to the README.md file.

Change Log:
    - v2.1.0: Fixed averaging of ensemble embeddings and added dummy code for sanity checking
    - v2.0.0: Added GPU support
    - v1.3.0: Added PCA visualization for each embedding combination method
    - v1.2.0: Added process_ensemble_batches function for ensemble-level averaging
    - v1.1.0: Improved handling of scalar and vector components
    - v1.0.0: Initial release with Mean Pooling, Deep Sets, and Self-Attention methods

TODO:
    - Need to handle the case where the user only wants the scalar part as output after the CustomNonLinearReadout layer
    - Implement GPU acceleration for large ensemble processing
    - Add support for custom combining methods
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(0)

OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output/pca_visualizations"

class DeepSets(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the DeepSets module.

        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.

        Returns:
            None
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        
        # Phi networks for scalar and vector parts
        self.phi_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.ReLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.ReLU(),
        )
        self.phi_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.ReLU(),
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.ReLU(),
        )
        
        # Rho networks for scalar and vector parts
        self.rho_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.ReLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.rho_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.ReLU(),
            nn.Linear(vector_dim * 3, vector_dim * 3),
        )

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DeepSets module.

        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector tensors.
        """
        # Apply phi to each atom independently
        scalar = self.phi_scalar(scalar)
        vector_flat = vector.view(vector.shape[0], vector.shape[1], -1)
        vector_flat = self.phi_vector(vector_flat)
        
        # Aggregate across atoms (assuming dim=1 is the atom dimension)
        scalar_agg = scalar.mean(dim=1)
        vector_agg = vector_flat.mean(dim=1)
        
        # Apply rho to the aggregated results
        scalar_out = self.rho_scalar(scalar_agg)
        vector_out = self.rho_vector(vector_agg)
        
        # Reshape vector output
        vector_out = vector_out.view(vector_out.shape[0], self.vector_dim, 3)
        
        return scalar_out, vector_out

class SelfAttention(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the SelfAttention module.

        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.

        Returns:
            None
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        total_dim = scalar_dim + vector_dim * 3
        
        self.attention = nn.Linear(total_dim, total_dim)
        self.value = nn.Linear(total_dim, total_dim)
        
        self.output_scalar = nn.Sequential(
            nn.Linear(total_dim, scalar_dim),
            nn.ReLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.output_vector = nn.Sequential(
            nn.Linear(total_dim, vector_dim * 3),
            nn.ReLU(),
            nn.Linear(vector_dim * 3, vector_dim * 3),
        )

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SelfAttention module.

        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector tensors.
        """
        # Combine scalar and vector inputs
        vector_flat = vector.view(vector.shape[0], vector.shape[1], -1)
        combined = torch.cat([scalar, vector_flat], dim=-1)
        
        # Compute attention scores
        attention_scores = self.attention(combined)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention
        values = self.value(combined)
        attended = torch.sum(attention_weights * values, dim=1)
        
        # Split and reshape outputs
        scalar_out = self.output_scalar(attended)
        vector_out = self.output_vector(attended).view(vector.shape[0], self.vector_dim, 3)
        
        return scalar_out, vector_out

class ImprovedDeepSets(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the ImprovedDeepSets module.

        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.

        Returns:
            None
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        
        # Increase complexity of phi networks
        self.phi_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim * 2),
            nn.ReLU(),
            nn.Linear(scalar_dim * 2, scalar_dim * 2),
            nn.ReLU(),
            nn.Linear(scalar_dim * 2, scalar_dim),
            nn.ReLU(),
        )
        self.phi_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 6),
            nn.ReLU(),
            nn.Linear(vector_dim * 6, vector_dim * 6),
            nn.ReLU(),
            nn.Linear(vector_dim * 6, vector_dim * 3),
            nn.ReLU(),
        )
        
        # Increase complexity of rho networks
        self.rho_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim * 2),
            nn.ReLU(),
            nn.Linear(scalar_dim * 2, scalar_dim * 2),
            nn.ReLU(),
            nn.Linear(scalar_dim * 2, scalar_dim),
        )
        self.rho_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 6),
            nn.ReLU(),
            nn.Linear(vector_dim * 6, vector_dim * 6),
            nn.ReLU(),
            nn.Linear(vector_dim * 6, vector_dim * 3),
        )
        
        self.layer_norm_scalar = nn.LayerNorm(scalar_dim)
        self.layer_norm_vector = nn.LayerNorm(vector_dim * 3)

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ImprovedDeepSets module.

        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector tensors.
        """
        # Apply phi to each atom independently
        scalar = self.phi_scalar(scalar)
        vector_flat = vector.view(vector.shape[0], vector.shape[1], -1)
        vector_flat = self.phi_vector(vector_flat)
        
        # Apply LayerNorm
        scalar = self.layer_norm_scalar(scalar)
        vector_flat = self.layer_norm_vector(vector_flat)
        
        # Aggregate across atoms (assuming dim=1 is the atom dimension)
        scalar_agg = scalar.mean(dim=1)
        vector_agg = vector_flat.mean(dim=1)
        
        # Apply rho to the aggregated results
        scalar_out = self.rho_scalar(scalar_agg)
        vector_out = self.rho_vector(vector_agg)
        
        # Reshape vector output
        vector_out = vector_out.view(vector_out.shape[0], self.vector_dim, 3)
        
        return scalar_out, vector_out

class ImprovedSelfAttention(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int, num_heads: int = 4):
        """
        Initialize the ImprovedSelfAttention module.

        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.
            num_heads (int): Number of attention heads.

        Returns:
            None
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        total_dim = scalar_dim + vector_dim * 3
        
        self.attention = nn.MultiheadAttention(total_dim, num_heads)
        self.norm1 = nn.LayerNorm(total_dim)
        self.norm2 = nn.LayerNorm(total_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(total_dim, total_dim * 4),
            nn.ReLU(),
            nn.Linear(total_dim * 4, total_dim),
        )
        
        self.output_scalar = nn.Sequential(
            nn.Linear(total_dim, scalar_dim * 2),
            nn.ReLU(),
            nn.Linear(scalar_dim * 2, scalar_dim),
        )
        self.output_vector = nn.Sequential(
            nn.Linear(total_dim, vector_dim * 6),
            nn.ReLU(),
            nn.Linear(vector_dim * 6, vector_dim * 3),
        )

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ImprovedSelfAttention module.

        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector tensors.
        """
        # Combine scalar and vector inputs
        vector_flat = vector.view(vector.shape[0], vector.shape[1], -1)
        combined = torch.cat([scalar, vector_flat], dim=-1)
        
        # Multi-head attention
        attended = self.attention(combined, combined, combined)[0]
        attended = self.norm1(attended + combined)
        
        # Feed-forward network
        ffn_output = self.ffn(attended)
        ffn_output = self.norm2(ffn_output + attended)
        
        # Split and reshape outputs
        scalar_out = self.output_scalar(ffn_output.mean(dim=1))
        vector_out = self.output_vector(ffn_output.mean(dim=1)).view(vector.shape[0], self.vector_dim, 3)
        
        return scalar_out, vector_out

class ConformerEnsembleEmbeddingCombiner(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the ConformerEnsembleEmbeddingCombiner module.

        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.

        Returns:
            None
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

        self.deep_sets = DeepSets(scalar_dim, vector_dim)
        self.self_attention = SelfAttention(scalar_dim, vector_dim)
        self.improved_deep_sets = ImprovedDeepSets(scalar_dim, vector_dim)
        self.improved_self_attention = ImprovedSelfAttention(scalar_dim, vector_dim)

    def mean_pooling(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform mean pooling on scalar and vector inputs.

        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean-pooled scalar and vector tensors.
        """
        return scalar.mean(dim=1), vector.mean(dim=1)

    def forward(self, conformer_embeddings: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the ConformerEnsembleEmbeddingCombiner module.

        Args:
            conformer_embeddings (torch.Tensor): Input tensor of conformer embeddings.

        Returns:
            dict[str, tuple[torch.Tensor, torch.Tensor]]: Dictionary of processed embeddings for each method.
        """
        num_conformers, num_atoms, total_dim = conformer_embeddings.shape
        scalar = conformer_embeddings[:, :, :self.scalar_dim]
        vector = conformer_embeddings[:, :, self.scalar_dim:].reshape(num_conformers, num_atoms, self.vector_dim, 3)
        
        mean_pooled_scalar, mean_pooled_vector = self.mean_pooling(scalar, vector)
        deep_sets_scalar, deep_sets_vector = self.deep_sets(scalar, vector)
        self_attention_scalar, self_attention_vector = self.self_attention(scalar, vector)
        improved_deep_sets_scalar, improved_deep_sets_vector = self.improved_deep_sets(scalar, vector)
        improved_self_attention_scalar, improved_self_attention_vector = self.improved_self_attention(scalar, vector)

        return {
            'mean_pooling': (mean_pooled_scalar, mean_pooled_vector),
            'deep_sets': (deep_sets_scalar, deep_sets_vector),
            'self_attention': (self_attention_scalar, self_attention_vector),
            'improved_deep_sets': (improved_deep_sets_scalar, improved_deep_sets_vector),
            'improved_self_attention': (improved_self_attention_scalar, improved_self_attention_vector)
        }

def process_conformer_ensemble(conformer_embeddings: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Process a single batch of conformer embeddings.

    Args:
        conformer_embeddings (torch.Tensor): Input tensor of conformer embeddings.

    Returns:
        dict[str, tuple[torch.Tensor, torch.Tensor]]: Dictionary of processed embeddings for each method.
    """
    print(f"process_conformer_ensemble input shape: {conformer_embeddings.shape}")
    # print(f"Conformer embeddings dtype: {conformer_embeddings.dtype}")

    num_conformers, num_atoms, total_dim = conformer_embeddings.shape
    scalar_dim = total_dim // 4
    vector_dim = scalar_dim
    
    print(f"Num conformers: {num_conformers}, Num atoms: {num_atoms}, Total dim: {total_dim}")
    print(f"Scalar dim: {scalar_dim}, Vector dim: {vector_dim}")
    
    combiner = ConformerEnsembleEmbeddingCombiner(scalar_dim, vector_dim).to(conformer_embeddings.device)
    results = combiner(conformer_embeddings)

    return results

def process_ensemble_batches(batch_embeddings: list[torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Process all batches for a single ensemble and average the results.

    Args:
        batch_embeddings (list[torch.Tensor]): List of tensors containing conformer embeddings for each batch.

    Returns:
        dict[str, tuple[torch.Tensor, torch.Tensor]]: Dictionary of processed embeddings for each method.
    """
    device = batch_embeddings[0].device 
    all_conformers = torch.cat(batch_embeddings, dim=0).to(device)
    results = process_conformer_ensemble(all_conformers)
    
    # Average the results across all conformers
    averaged_results = {}

    for method, (scalar, vector) in results.items():
        averaged_scalar = scalar.mean(dim=0, keepdim=True)  # [1, scalar_dim]
        averaged_vector = vector.mean(dim=0, keepdim=True)  # [1, vector_dim, 3]
        averaged_results[method] = (averaged_scalar, averaged_vector)
    
    return averaged_results

def visualize_embeddings(embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """
    Visualize the embeddings for each method.
    If there's only one sample, we'll create a bar plot instead of PCA.

    Args:
        embeddings (dict[str, tuple[torch.Tensor, torch.Tensor]]): Dictionary of embeddings for each method.

    Returns:
        None
    """
    for method, (scalar, vector) in embeddings.items():
        scalar = scalar.cpu().detach().squeeze()
        vector = vector.cpu().detach().squeeze()

        # Combine scalar and vector embeddings
        combined = torch.cat([scalar, vector.reshape(-1)])
        
        plt.figure(figsize=(10, 6))
        
        if combined.dim() == 1:  # Only one sample
            plt.bar(range(len(combined)), combined.numpy())
            plt.title(f'Embedding Values for {method}')
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Value')
            
        else:  # Multiple samples
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined.numpy())
            
            plt.scatter(pca_result[:, 0], pca_result[:, 1])
            plt.title(f'PCA Visualization of {method} Embeddings')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
        
        plt.savefig(f'{OUTPUT_PATH}/{method}_visualization.png')
        plt.close()

        print(f"Visualization for {method} saved as '{method}_visualization.png'")

def move_to_device(obj, device):
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


