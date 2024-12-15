"""
Conformer Ensemble Embedding Combiner

This module provides advanced functionality for combining embeddings from multiple
conformers of a molecule into a single representation. It supports flexible
output configurations from the EquiCat model, including scalar-only, vector-only,
and combined scalar-vector outputs.

Key components:
1. DeepSets: Implements the Deep Sets algorithm for set-based embedding combination.
2. SelfAttention: Applies self-attention mechanism to process conformer embeddings.
3. ImprovedDeepSets: An enhanced version of Deep Sets with additional layers and normalization.
4. ImprovedSelfAttention: Incorporates multi-head attention and feed-forward networks.
5. ConformerEnsembleEmbeddingCombiner: Orchestrates multiple embedding combination methods.
6. Utility functions for processing and visualizing embeddings.

Key Features:
1. Advanced activation functions (SiLU) for better gradient flow
2. Comprehensive type hints and documentation
3. Flexible handling of scalar and vector inputs
4. GPU-accelerated processing support
5. Multiple embedding combination strategies
6. Visualization tools for embedding analysis

Author: Utkarsh Sharma
Version: 4.0.0
Date: 12-14-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
- torch (>=1.9.0)
- numpy (>=1.20.0)
- matplotlib (>=3.3.0)
- sklearn (>=0.24.0)
- e3nn (>=0.4.0)

Usage:
    # Initialize the combiner
    combiner = ConformerEnsembleEmbeddingCombiner(scalar_dim, vector_dim)
    
    # Process conformer embeddings
    results = process_conformer_ensemble(conformer_embeddings, irreps_str)
    
    # Process molecule conformers
    molecule_results = process_molecule_conformers(molecule_embeddings, irreps_str)
    
    # Visualize embeddings
    visualize_embeddings(results)

Change Log:
- v4.0.0 (12-14-2024):
  * Major version bump to align with EquiCat ecosystem
  * Replaced ReLU with SiLU activation functions for improved performance
  * Enhanced documentation and type hints
  * Standardized random seed initialization
  * Improved error handling and logging
- v3.1.1 (09-10-2024):
  * Improved detect_scalars_vectors() with better logging
  * Added process_molecule_conformers() for single molecule processing
  * Enhanced error handling and input validation
- v3.0.0 (08-22-2024):
  * Major refactor to support flexible output types from NonLinearReadout
  * Enhanced all combining methods to handle scalar-only, vector-only, and combined outputs
  * Improved PCA visualization to accommodate all output types
  * Optimized performance for large-scale conformer ensembles
  * Added comprehensive error handling and logging
- v2.0.0 (07-30-2024):
  * Introduced GPU support and basic handling of vector outputs
- v1.0.0 (07-01-2024):
  * Initial release with basic conformer embedding combination

ToDo:
    - Implement equivariant transformer architecture for improved embedding combination
    - Add support for custom combining methods defined by users
    - Enhance visualization with interactive plots and 3D representations
    - Optimize memory usage for very large conformer ensembles
    - Add unit tests and integration tests for robustness
    - Implement logging system for better debugging and monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from e3nn import o3

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(42)

OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output/pca_visualizations"

class DeepSets(nn.Module):
    """
    Implements the Deep Sets algorithm for processing sets of embeddings.

    This class can handle scalar-only, vector-only, or combined scalar-vector inputs.

    Args:
        scalar_dim (int): Dimension of scalar features.
        vector_dim (int): Dimension of vector features.

    Returns:
        None
    """
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the Deep Sets model for scalar and vector inputs.
    
        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.
    """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        
        if scalar_dim > 0:
            self.phi_scalar = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
                nn.SiLU(),
            )
            self.rho_scalar = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
            )
        
        if vector_dim > 0:
            self.phi_vector = nn.Sequential(
                nn.Linear(vector_dim * 3, vector_dim * 3),
                nn.SiLU(),
                nn.Linear(vector_dim * 3, vector_dim * 3),
                nn.SiLU(),
            )
            self.rho_vector = nn.Sequential(
                nn.Linear(vector_dim * 3, vector_dim * 3),
                nn.SiLU(),
                nn.Linear(vector_dim * 3, vector_dim * 3),
            )

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process scalar and vector inputs through the Deep Sets model.
        
        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector outputs.
    """    
        scalar_out, vector_out = None, None
    
        if self.scalar_dim > 0 and scalar is not None:
            scalar = self.phi_scalar(scalar)  # Apply phi network to scalar input
            scalar_agg = scalar.mean(dim=1)  # Aggregate across atoms
            scalar_out = self.rho_scalar(scalar_agg)  # Apply rho network to aggregated scalar
        
        if self.vector_dim > 0 and vector is not None:
            vector_flat = vector.view(vector.shape[0], vector.shape[1], -1)  # Reshape vector for processing
            vector_flat = self.phi_vector(vector_flat)  # Apply phi network to vector input
            vector_agg = vector_flat.mean(dim=1)  # Aggregate across atoms
            vector_out = self.rho_vector(vector_agg)  # Apply rho network to aggregated vector
            vector_out = vector_out.view(vector_out.shape[0], self.vector_dim, 3)  # Reshape back to 3D
        
        return scalar_out, vector_out

class SelfAttention(nn.Module):
    """
    Implements a Self-Attention mechanism for processing sets of embeddings.

    This class can handle scalar-only, vector-only, or combined scalar-vector inputs.

    Args:
        scalar_dim (int): Dimension of scalar features.
        vector_dim (int): Dimension of vector features.

    Returns:
        None
    """
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the Self-Attention model for scalar and vector inputs.
        
        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        total_dim = scalar_dim + vector_dim * 3
        
        self.attention = nn.Linear(total_dim, total_dim) # Attention mechanism
        self.value = nn.Linear(total_dim, total_dim) # Value projection
        
        if scalar_dim > 0:
            self.output_scalar = nn.Sequential(
                nn.Linear(total_dim, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
            )
        
        if vector_dim > 0:
            self.output_vector = nn.Sequential(
                nn.Linear(total_dim, vector_dim * 3),
                nn.SiLU(),
                nn.Linear(vector_dim * 3, vector_dim * 3),
            )

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process scalar and vector inputs through the Self-Attention model.
        
        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector outputs.
        """
        if self.scalar_dim > 0 and self.vector_dim > 0 and scalar is not None and vector is not None:
            vector_flat = vector.view(vector.shape[0], vector.shape[1], -1) # Flatten vector
            combined = torch.cat([scalar, vector_flat], dim=-1) # Combine scalar and vector
        elif self.scalar_dim > 0 and scalar is not None:
            combined = scalar
        elif self.vector_dim > 0 and vector is not None:
            combined = vector.view(vector.shape[0], vector.shape[1], -1)
        else:
            return None, None
        
        attention_scores = self.attention(combined) # Compute attention scores
        attention_weights = F.softmax(attention_scores, dim=1) # Apply softmax to get attention weights
        values = self.value(combined) # Compute value projections
        attended = torch.sum(attention_weights * values, dim=1)  # Apply attention
        
        scalar_out, vector_out = None, None
        
        if self.scalar_dim > 0 and scalar is not None:
            scalar_out = self.output_scalar(attended) # Process attended features for scalar output
        
        if self.vector_dim > 0 and vector is not None:
            vector_out = self.output_vector(attended).view(vector.shape[0], self.vector_dim, 3) # Process and reshape for vector output
        
        return scalar_out, vector_out

class ImprovedDeepSets(nn.Module):
    """
    Implements an improved version of the Deep Sets algorithm with additional layers and normalization.

    This class can handle scalar-only, vector-only, or combined scalar-vector inputs.

    Args:
        scalar_dim (int): Dimension of scalar features.
        vector_dim (int): Dimension of vector features.

    Returns:
        None
    """
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the Improved Deep Sets model with additional layers and normalization.
        
        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        
        if scalar_dim > 0:
            self.phi_scalar = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim * 2),
                nn.SiLU(),
                nn.Linear(scalar_dim * 2, scalar_dim * 2),
                nn.SiLU(),
                nn.Linear(scalar_dim * 2, scalar_dim),
                nn.SiLU(),
            )
            self.rho_scalar = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim * 2),
                nn.SiLU(),
                nn.Linear(scalar_dim * 2, scalar_dim * 2),
                nn.SiLU(),
                nn.Linear(scalar_dim * 2, scalar_dim),
            )
            self.layer_norm_scalar = nn.LayerNorm(scalar_dim)
        
        if vector_dim > 0:
            self.phi_vector = nn.Sequential(
                nn.Linear(vector_dim * 3, vector_dim * 6),
                nn.SiLU(),
                nn.Linear(vector_dim * 6, vector_dim * 6),
                nn.SiLU(),
                nn.Linear(vector_dim * 6, vector_dim * 3),
                nn.SiLU(),
            )
            self.rho_vector = nn.Sequential(
                nn.Linear(vector_dim * 3, vector_dim * 6),
                nn.SiLU(),
                nn.Linear(vector_dim * 6, vector_dim * 6),
                nn.SiLU(),
                nn.Linear(vector_dim * 6, vector_dim * 3),
            )
            self.layer_norm_vector = nn.LayerNorm(vector_dim * 3)

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process scalar and vector inputs through the Improved Deep Sets model.
        
        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector outputs.
        """
        scalar_out, vector_out = None, None
        
        if self.scalar_dim > 0 and scalar is not None:
            scalar = self.phi_scalar(scalar)  # Apply phi network to scalar input
            scalar = self.layer_norm_scalar(scalar)  # Apply layer normalization
            scalar_agg = scalar.mean(dim=1)  # Aggregate across atoms
            scalar_out = self.rho_scalar(scalar_agg)  # Apply rho network to aggregated scalar
        
        if self.vector_dim > 0 and vector is not None:
            vector_flat = vector.view(vector.shape[0], vector.shape[1], -1)  # Reshape vector for processing
            vector_flat = self.phi_vector(vector_flat)  # Apply phi network to vector input
            vector_flat = self.layer_norm_vector(vector_flat)  # Apply layer normalization
            vector_agg = vector_flat.mean(dim=1)  # Aggregate across atoms
            vector_out = self.rho_vector(vector_agg)  # Apply rho network to aggregated vector
            vector_out = vector_out.view(vector_out.shape[0], self.vector_dim, 3)  # Reshape back to 3D
    
        return scalar_out, vector_out

class ImprovedSelfAttention(nn.Module):
    """
    Implements an improved Self-Attention mechanism with multi-head attention and feed-forward networks.

    This class can handle scalar-only, vector-only, or combined scalar-vector inputs.

    Args:
        scalar_dim (int): Dimension of scalar features.
        vector_dim (int): Dimension of vector features.
        num_heads (int): Number of attention heads.

    Returns:
        None
    """
    def __init__(self, scalar_dim: int, vector_dim: int, num_heads: int = 4):
        """
        Initialize the Improved Self-Attention model with multi-head attention and feed-forward networks.
        
        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        total_dim = scalar_dim + vector_dim * 3
        
        #! RMSNorm layer
        self.attention = nn.MultiheadAttention(total_dim, num_heads) # Multi-head attention
        self.norm1 = nn.LayerNorm(total_dim) # Layer normalization 1
        self.norm2 = nn.LayerNorm(total_dim) # Layer normalization 2
        
        self.ffn = nn.Sequential(
            nn.Linear(total_dim, total_dim * 4),
            nn.SiLU(),
            nn.Linear(total_dim * 4, total_dim),
        )
        
        if scalar_dim > 0:
            self.output_scalar = nn.Sequential(
                nn.Linear(total_dim, scalar_dim * 2),
                nn.SiLU(),
                nn.Linear(scalar_dim * 2, scalar_dim),
            )
        
        if vector_dim > 0:
            self.output_vector = nn.Sequential(
                nn.Linear(total_dim, vector_dim * 6),
                nn.SiLU(),
                nn.Linear(vector_dim * 6, vector_dim * 3),
            )
    
    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process scalar and vector inputs through the Improved Self-Attention model.
        
        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed scalar and vector outputs.
        """
        if self.scalar_dim > 0 and self.vector_dim > 0 and scalar is not None and vector is not None:
            vector_flat = vector.view(vector.shape[0], vector.shape[1], -1)  # Flatten vector
            combined = torch.cat([scalar, vector_flat], dim=-1)  # Combine scalar and vector
        elif self.scalar_dim > 0 and scalar is not None:
            combined = scalar
        elif self.vector_dim > 0 and vector is not None:
            combined = vector.view(vector.shape[0], vector.shape[1], -1)
        else:
            return None, None
        
        attended = self.attention(combined, combined, combined)[0]  # Apply multi-head attention
        attended = self.norm1(attended + combined)  # Apply residual connection and normalization
        
        ffn_output = self.ffn(attended)  # Apply feed-forward network
        ffn_output = self.norm2(ffn_output + attended)  # Apply residual connection and normalization
        
        scalar_out, vector_out = None, None
        
        if self.scalar_dim > 0 and scalar is not None:
            scalar_out = self.output_scalar(ffn_output.mean(dim=1))  # Process for scalar output
        
        if self.vector_dim > 0 and vector is not None:
            vector_out = self.output_vector(ffn_output.mean(dim=1)).view(vector.shape[0], self.vector_dim, 3)  # Process and reshape for vector output
        
        return scalar_out, vector_out

class ConformerEnsembleEmbeddingCombiner(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int):
        """
        Initialize the Conformer Ensemble Embedding Combiner with multiple combination methods.
        
        Args:
            scalar_dim (int): Dimension of scalar features.
            vector_dim (int): Dimension of vector features.
        """
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

        self.deep_sets = DeepSets(scalar_dim, vector_dim)
        self.self_attention = SelfAttention(scalar_dim, vector_dim)
        self.improved_deep_sets = ImprovedDeepSets(scalar_dim, vector_dim)
        self.improved_self_attention = ImprovedSelfAttention(scalar_dim, vector_dim)

    def mean_pooling(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scalar_out = scalar.mean(dim=1) if scalar is not None else None
        vector_out = vector.mean(dim=1) if vector is not None else None
        return scalar_out, vector_out

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Process scalar and vector inputs through all combination methods.
        
        Args:
            scalar (torch.Tensor): Scalar input tensor.
            vector (torch.Tensor): Vector input tensor.
        
        Returns:
            dict[str, tuple[torch.Tensor, torch.Tensor]]: Results from each combination method.
        """
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

def detect_scalars_vectors(irreps_str):
    """
    Detects the number of scalar and vector dimensions from an irreps string.

    Args:
        irreps_str (str): String representation of irreducible representations.

    Returns:
        tuple: (scalar_dim, vector_dim)
    """
    irreps = o3.Irreps(irreps_str)
    scalar_dim = 0
    vector_dim = 0
    
    for mul, ir in irreps:
        if ir.l == 0:  # Scalar
            scalar_dim += mul
        elif ir.l == 1:  # Vector
            vector_dim += mul
    
    print(f"Detected dimensions - Scalar: {scalar_dim}, Vector: {vector_dim}")
    return scalar_dim, vector_dim

def process_conformer_ensemble(conformer_embeddings: torch.Tensor, irreps_str: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Processes a batch of conformer embeddings using various combination methods.

    Args:
        conformer_embeddings (torch.Tensor): Tensor of conformer embeddings.
        irreps_str (str): String representation of irreducible representations.

    Returns:
        dict: A dictionary containing the results of each combination method.
    """
    print(f"process_conformer_ensemble input shape: {conformer_embeddings.shape}")

    num_conformers, num_atoms, total_dim = conformer_embeddings.shape
    
    scalar_dim, vector_dim = detect_scalars_vectors(irreps_str)
    
    print(f"Num conformers: {num_conformers}, Num atoms: {num_atoms}, Total dim: {total_dim}")
    print(f"Using irreps: {irreps_str}")
    
    combiner = ConformerEnsembleEmbeddingCombiner(scalar_dim, vector_dim).to(conformer_embeddings.device)
    
    if scalar_dim > 0 and vector_dim > 0:
        scalar = conformer_embeddings[:, :, :scalar_dim]
        vector = conformer_embeddings[:, :, scalar_dim:].reshape(num_conformers, num_atoms, vector_dim, 3)
    elif scalar_dim > 0:
        scalar = conformer_embeddings
        vector = None
    else:
        scalar = None
        vector = conformer_embeddings.reshape(num_conformers, num_atoms, vector_dim, 3)
    
    results = combiner(scalar, vector)

    return results

def process_molecule_conformers(molecule_embeddings: torch.Tensor, irreps_str: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Processes conformers of a single molecule and combines their embeddings.

    Args:
        molecule_embeddings (torch.Tensor): Tensor of conformer embeddings for a single molecule.
        irreps_str (str): String representation of irreducible representations.

    Returns:
        dict: A dictionary containing the combined embeddings for the molecule.
    """
    results = process_conformer_ensemble(molecule_embeddings, irreps_str)
    
    # Average the results across all conformers of the molecule
    averaged_results = {}
    for method, (scalar, vector) in results.items():
        if scalar is not None:
            averaged_scalar = scalar.mean(dim=0, keepdim=True)
        else:
            averaged_scalar = None
        
        if vector is not None:
            averaged_vector = vector.mean(dim=0, keepdim=True)
        else:
            averaged_vector = None
        
        averaged_results[method] = (averaged_scalar, averaged_vector)
    
    return averaged_results

def visualize_embeddings(embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """
    Visualizes the embeddings using PCA and creates plots for each combination method.

    Args:
        embeddings (dict): Dictionary containing embeddings for different methods.

    Returns:
        None
    """
    for method, (scalar, vector) in embeddings.items():
        combined = []
        if scalar is not None:
            scalar = scalar.cpu().detach().squeeze()
            combined.append(scalar)
        if vector is not None:
            vector = vector.cpu().detach().squeeze()
            combined.append(vector.reshape(-1))
        
        if not combined:
            print(f"No data to visualize for method: {method}")
            continue
        
        combined = torch.cat(combined)
        
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
    """
    Recursively moves an object (tensor, list, tuple, or dict) to the specified device.

    Args:
        obj: The object to move.
        device: The target device (CPU or GPU).

    Returns:
        The object moved to the specified device.
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
