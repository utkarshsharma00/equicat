"""
EquiCat Model Implementation

This module implements the EquiCat model, a neural network architecture designed for 
equivariant learning on molecular systems. It leverages the MACE framework to create 
a model that respects the symmetries inherent in molecular data while providing 
efficient geometric message passing and feature aggregation.

Key components:
1. Model Architecture:
   - Neural network built on MACE framework
   - Multiple interaction and product layers
   - Equivariant message passing mechanism
   - Geometric feature processing pipeline

2. Processing Stages:
   - Node embedding and attribute computation
   - Radial basis function generation
   - Spherical harmonics computation
   - Geometric message passing
   - Feature aggregation and transformation

3. Error Handling:
   - Robust atomic number processing
   - Device management and tensor type checking
   - Shape consistency verification
   - Debug logging throughout forward pass

Key Features:
1. Equivariant molecular geometry processing
2. Multi-layer message passing architecture
3. Configurable interaction depth
4. GPU acceleration support
5. Comprehensive shape tracking
6. Flexible atomic number handling
7. Built-in error recovery
8. Debug-friendly tensor tracking
9. Device-agnostic computation
10. Memory-efficient processing

Author: Utkarsh Sharma
Version: 4.0.0
Date: 12-14-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
- torch (>=1.9.0)
- numpy (>=1.20.0)
- mace (custom package)
- torch_geometric (>=2.0.0)

Usage:
    # Initialize model with configuration
    model = EQUICAT(model_config, z_table)
    
    # Prepare input dictionary
    input_dict = {
        'positions': positions,          # [num_atoms, 3]
        'atomic_numbers': atomic_numbers, # [num_atoms]
        'edge_index': edge_index        # [2, num_edges]
    }
    
    # Forward pass
    output = model(input_dict)

For detailed usage instructions, please refer to the README.md file.

Change Log:
- v4.0.1 (12-14-2024):
  * Added skip connection tensor (sc) computation and passing to product layers
- v4.0.0 (12-14-2024):
  * Major version bump to align with EquiCat ecosystem
  * Major refactoring for improved modularity
  * Enhanced error handling and recovery
  * Improved shape consistency checking
  * Added comprehensive debug logging
  * Standardized tensor type management
  * Memory optimization for large systems
  * Enhanced device management
  * Improved documentation and type hints
- v2.0.0 (09-10-2024):
  * Added GPU support
  * Added compatibility updates
- v1.1.0 (08-15-2024):
  * Added multiple interaction layers
  * Enhanced debug printing
- v1.0.0 (07-01-2024):
  * Initial implementation

ToDo:
- Implement gradient checkpointing for memory efficiency
- Add support for custom activation functions
- Implement model parameter serialization
- Enhance debug visualization tools
- Add model architecture validation
- Implement memory usage profiling
- Add support for distributed training
"""

import torch
import torch.nn.functional
import numpy as np
from mace import modules, tools
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum
from torch_geometric.utils import to_dense_batch

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(42)

class EQUICAT(torch.nn.Module):
    def __init__(self, model_config, z_table):
        """
        Initialize the EQUICAT model.

        Args:
            model_config (dict): Configuration parameters for the MACE model.
            z_table (AtomicNumberTable): Table of atomic numbers.

        Returns:
            None
        """
        super(EQUICAT, self).__init__()
        model_config['atomic_numbers'] = torch.tensor(z_table.zs).cpu()  # Ensure it's on CPU
        self.model = modules.MACE(**model_config)
        self.z_table = z_table
        
        # Store the number of interactions
        self.num_interactions = model_config['num_interactions']
        
        # Create lists to store multiple interaction and product layers
        self.interaction_layers = torch.nn.ModuleList(self.model.interactions)
        self.product_layers = torch.nn.ModuleList(self.model.products)

    def get_forward_pass_summary(self):
        """
        Generate a summary of the forward pass of the EQUICAT model.

        Args:
            None

        Returns:
            str: A string containing a summary of the forward pass.
        """
        summary = [
            "EQUICAT Forward Pass Summary:",
            f"Number of interaction layers: {self.num_interactions}",
            f"Input processing: node embedding, radial embedding, spherical harmonics",
            "For each interaction layer:",
            "  - Linear up-projection",
            "  - Tensor product convolution",
            "  - Message passing and aggregation",
            "  - Linear projection",
            "  - Equivariant product basis",
            f"Output: Node features (shape determined by model configuration)"
        ]
        return "\n".join(summary)

    def forward(self, input_dict):
        """
        Forward pass of the EQUICAT model.

        Args:
            input_dict (dict): Input dictionary containing molecular data.
                               Expected keys: 'positions', 'atomic_numbers', 'edge_index'

        Returns:
            torch.Tensor: Processed node features.
        """
        torch.set_printoptions(profile="full")

        # Extract relevant information from the input dictionary
        positions = input_dict['positions']
        # print(f"EQUICAT input positions dtype: {positions.dtype}")
        atomic_numbers = input_dict['atomic_numbers']
        # print(f"EQUICAT atomic number dtype: {atomic_numbers.dtype}")
        edge_index = input_dict['edge_index']

        print("-" * 28)
        print(f"Processing a new conformer")
        print("-" * 28)
        print(f"Positions shape: {positions.shape}")
        print(f"Edge index shape: {edge_index.shape}")

        try:
            # Move atomic_numbers to CPU before conversion
            atomic_numbers_cpu = atomic_numbers.cpu()
            
            indices = tools.utils.atomic_numbers_to_indices(atomic_numbers_cpu, z_table=self.z_table)
            
            # Move indices back to the same device as the input
            indices = torch.tensor(indices, device=atomic_numbers.device)
            
        except ValueError as e:
            print(f"Warning: Unexpected atomic number encountered. Error: {e}")
            print(f"Unique atomic numbers in input: {torch.unique(atomic_numbers)}")
            print(f"Atomic numbers in z_table: {self.z_table.zs}")
            # Handle the error (e.g., by skipping this input or assigning a default index)
            indices = torch.zeros_like(atomic_numbers)  # Or some other default behavior

        # Set shifts to zero (assuming no periodic boundary conditions)
        shifts = torch.zeros((edge_index.shape[1], positions.shape[1]), dtype=positions.dtype, device=positions.device)
        sender, receiver = edge_index

        # Get edge vectors and lengths
        vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )

        # Compute edge attributes using spherical harmonics
        edge_attrs = self.model.spherical_harmonics(vectors)
        print("Edge attributes shape:", edge_attrs.shape)

        # Compute node attributes using one-hot encoding
        node_attrs = tools.torch_tools.to_one_hot(
            indices.unsqueeze(-1),
            num_classes=len(self.z_table),
        )
        print("Node attributes shape:", node_attrs.shape)

        # Compute radial embedding
        edge_feats = self.model.radial_embedding(lengths, node_attrs, edge_index, atomic_numbers)
        print("Edge features shape:", edge_feats.shape)

        # Compute initial node features using the node embedding block
        node_feats = self.model.node_embedding(node_attrs)
        print("Initial Node features shape:", node_feats.shape)

        # Process through multiple MACE layers
        for i in range(self.num_interactions):
            # Interaction layer
            node_feats_up = self.interaction_layers[i].linear_up(node_feats)
            print(f"Node features after linear up-projection in layer {i+1} shape:", node_feats_up.shape)
            
            tp_weights = self.interaction_layers[i].conv_tp_weights(edge_feats)
            print("tp_weights shape:", tp_weights.shape)
            
            mji = self.interaction_layers[i].conv_tp(node_feats_up[sender], edge_attrs, tp_weights)
            
            message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=node_feats.shape[0])
            print("Message shape:", message.shape)
            
            # Get skip connection
            sc = self.interaction_layers[i].skip_tp(node_feats, node_attrs)
            
            # Process message
            message = self.interaction_layers[i].linear(message)
            node_feats_processed = self.interaction_layers[i].reshape(message)
            
            # Product layer
            node_feats = self.product_layers[i](node_feats=node_feats_processed, sc=sc, node_attrs=node_attrs)
            
            print(f"Node features after layer {i+1} shape:", node_feats.shape)
        
        return node_feats   

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
