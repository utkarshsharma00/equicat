"""
EQUICAT Model Implementation

This module implements the EQUICAT model, a neural network architecture designed for 
equivariant learning on molecular systems. It leverages the MACE framework to create 
a model that respects the symmetries inherent in molecular data.

Key features:
1. Equivariant processing of molecular geometries
2. Handling of variable-sized molecular inputs
3. Incorporation of spherical harmonics for angular information
4. Use of radial basis functions for distance information
5. Implementation of symmetric contractions for feature aggregation
6. Multiple interaction and product layers for deep learning

The EQUICAT class encapsulates the entire model, providing a forward method that 
processes input molecular data through various stages of the network.

Author: Utkarsh Sharma
Version: 1.1.0
Date: 07-16-2024 (MM-DD-YYYY)
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
        """
        super(EQUICAT, self).__init__()
        model_config['atomic_numbers'] = torch.tensor(z_table.zs)
        self.model = modules.MACE(**model_config)
        self.z_table = z_table
        
        # Store the number of interactions
        self.num_interactions = model_config['num_interactions']
        
        # Create lists to store multiple interaction and product layers
        self.interaction_layers = torch.nn.ModuleList(self.model.interactions)
        self.product_layers = torch.nn.ModuleList(self.model.products)

    def forward(self, input_dict):
        """
        Forward pass of the EQUICAT model.

        Args:
            input_dict (dict): Input dictionary containing molecular data.

        Returns:
            torch.Tensor: Processed node features.
        """
        # Extract relevant information from the input dictionary
        positions = input_dict['positions'] 
        atomic_numbers = input_dict['atomic_numbers']
        try:
            indices = tools.utils.atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table)
        except ValueError as e:
            print(f"Warning: Unexpected atomic number encountered. Error: {e}")
            print(f"Unique atomic numbers in input: {torch.unique(atomic_numbers)}")
            print(f"Atomic numbers in z_table: {self.z_table.zs}")
            # Handle the error (e.g., by skipping this input or assigning a default index)
            indices = torch.zeros_like(atomic_numbers)  # Or some other default behavior
        edge_index = input_dict['edge_index']

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

        # Compute node attributes using one-hot encoding
        indices = tools.utils.atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table)
        node_attrs = tools.torch_tools.to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(self.z_table),
        )

        # Compute radial embedding
        edge_feats = self.model.radial_embedding(lengths, node_attrs, edge_index, atomic_numbers)

        # Compute initial node features using the node embedding block
        node_feats = self.model.node_embedding(node_attrs)

        # Process through multiple MACE layers
        for i in range(self.num_interactions):
            # Interaction layer
            node_feats = self.interaction_layers[i].linear_up(node_feats)
            tp_weights = self.interaction_layers[i].conv_tp_weights(edge_feats)
            mji = self.interaction_layers[i].conv_tp(node_feats[sender], edge_attrs, tp_weights)
            message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=node_feats.shape[0])
            node_feats = self.interaction_layers[i].linear(message)
            node_feats = self.interaction_layers[i].reshape(node_feats)

            # Product layer
            node_feats = self.product_layers[i](node_feats=node_feats, sc=None, node_attrs=node_attrs)

            print(f"Node features after layer {i+1} shape:", node_feats.shape)

        return node_feats