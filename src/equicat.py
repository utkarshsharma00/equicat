"""
EQUICAT Model Implementation

This module implements the EQUICAT model, a neural network architecture designed for 
equivariant learning on molecular systems. It leverages the MACE framework to create 
a model that respects the symmetries inherent in molecular data.

Key features:
1. GPU acceleration with CUDA support
2. Equivariant processing of molecular geometries
3. Handling of variable-sized molecular inputs
4. Incorporation of spherical harmonics for angular information
5. Use of radial basis functions for distance information
6. Implementation of symmetric contractions for feature aggregation
7. Multiple interaction and product layers for deep learning
8. Extensive debug printing throughout the forward pass

The EQUICAT class encapsulates the entire model, providing a forward method that 
processes input molecular data through various stages of the network.

Author: Utkarsh Sharma
Version: 2.0.0
Date: 08-01-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - mace (custom package)
    - torch_geometric (>=2.0.0)

Usage:
    model = EQUICAT(model_config, z_table)
    output = model(input_dict)

For detailed usage instructions, please refer to the README.md file.

Change Log:
    - v2.0.0: Added GPU support and ensured compatibility with updated equicat_plus_nonlinear.py and train.py
    - v1.2.0: Added extensive debug printing and sanity checks throughout the forward pass
    - v1.1.0: Added support for multiple interaction and product layers
    - v1.0.0: Initial implementation of EQUICAT model

TODO:
    - Implement checkpointing for large models
    - Add support for custom activation functions
    - Optimize memory usage for processing large molecular systems
"""

import torch
import torch.nn.functional
from mace import modules, tools
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum
from torch_geometric.utils import to_dense_batch

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
        atomic_numbers = input_dict['atomic_numbers']
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
            node_feats = self.interaction_layers[i].linear_up(node_feats)
            print(f"Node features after linear up-projection in layer {i+1} shape:", node_feats.shape)
            tp_weights = self.interaction_layers[i].conv_tp_weights(edge_feats)
            print("tp_weights shape:", tp_weights.shape)
            mji = self.interaction_layers[i].conv_tp(node_feats[sender], edge_attrs, tp_weights)
            message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=node_feats.shape[0])
            print("Message shape:", message.shape)
            node_feats = self.interaction_layers[i].linear(message)
            node_feats = self.interaction_layers[i].reshape(node_feats)

            # Product layer
            node_feats = self.product_layers[i](node_feats=node_feats, sc=None, node_attrs=node_attrs)

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