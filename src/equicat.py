"""
EQUICAT Model Implementation

This module implements the EQUICAT model, a neural network architecture designed
for equivariant learning on molecular systems. It leverages the MACE (Many-body 
Atomic Cluster Expansion) framework to create a model that respects the symmetries
inherent in molecular data.

Key features:
1. Equivariant processing of molecular geometries
2. Handling of variable-sized molecular inputs
3. Incorporation of spherical harmonics for angular information
4. Use of radial basis functions for distance information
5. Implementation of symmetric contractions for feature aggregation

The EQUICAT class encapsulates the entire model, providing a forward method
that processes input molecular data through various stages of the network.

Author: Utkarsh Sharma
Version: 1.0.0
Date: 07-07-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - mace (custom package)
    - torch_geometric (>=2.0.0)

Usage:
    model = EQUICAT(model_config, z_table)
    output = model(input_dict)

For detailed usage instructions, please refer to the README.md file.
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
        edge_index = input_dict['edge_index']

        # Set shifts to zero (assuming no periodic boundary conditions)
        shifts = torch.zeros((edge_index.shape[1], positions.shape[1]), dtype=positions.dtype, device=positions.device)
        sender, receiver = edge_index

        print("positions shape:", positions.shape)
        print("edge_index shape:", edge_index.shape)
        print("shifts shape:", shifts.shape)
        print("sender shape:", sender.shape)
        print("receiver shape:", receiver.shape)

        # Get edge vectors and lengths
        vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )

        # Compute edge attributes using spherical harmonics
        edge_attrs = self.model.spherical_harmonics(vectors)
        print("Edge attributes shape:", edge_attrs.shape)
        print("Edge attributes", edge_attrs)

        # Compute node attributes using one-hot encoding
        indices = tools.utils.atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table)
        node_attrs = tools.torch_tools.to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(self.z_table),
        )

        # Compute radial embedding
        edge_feats = self.model.radial_embedding(lengths, node_attrs, edge_index, atomic_numbers)
        print("Edge features shape:", edge_feats.shape)
        print("Edge features", edge_feats)

        print("Node attributes shape:", node_attrs.shape)
        print("Node attributes", node_attrs)
        print("Weights are internally flattened and have a shape", self.model.node_embedding.linear.__dict__['_parameters']['weight'].shape)

        # Compute initial node features using the node embedding block
        node_feats = self.model.node_embedding(node_attrs)
        print("Initial node features shape:", node_feats.shape)
        print("Initial node features", node_feats)

        # Linearly mixing the incoming node features
        node_feats = self.model.interactions[0].linear_up(node_feats)
        print("Node features after linear_up shape:", node_feats.shape)

        # Construct the learnable radial basis using the Bessel Basis and the radial MLP
        tp_weights = self.model.interactions[0].conv_tp_weights(edge_feats)
        print("Tensor product weights shape:", tp_weights.shape)

        # Formation of one particle basis
        # print("Dimensions of node_feats:", node_feats.shape)
        # print("Dimensions of sender:", sender.shape)
        # print("Sender indices:", sender)
        mji = self.model.interactions[0].conv_tp(node_feats[sender], edge_attrs, tp_weights)

        sender, receiver = edge_index

        # # Print the shapes of relevant tensors
        # print(f"Shape of node_feats: {node_feats.shape}")
        # print(f"Shape of sender indices: {sender.shape}")
        # print(f"Shape of receiver indices: {receiver.shape}")
        # print(f"Shape of node_feats[sender]: {node_feats[sender].shape}")
        # print(f"Shape of edge_attrs: {edge_attrs.shape}")
        # print(f"Shape of tp_weights: {tp_weights.shape}")

        mji = self.model.interactions[0].conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )
        print("Tensor Product Weights", tp_weights)

        #  The sum over the neighbors of atom i to form the atomic basis
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=node_feats.shape[0]
        )
        print("Message shape:", message.shape)
        print("Message", message)

        # Linear sketching to form the A basis, this step leaves the shape unchanged
        node_feats = self.model.interactions[0].linear(message)

        # Reshape node features
        node_feats = self.model.interactions[0].reshape(message)
        print("Input shape", node_feats.shape)

        # Apply symmetric contractions
        node_feats = self.model.products[0](node_feats=node_feats, sc=None, node_attrs=node_attrs)
        print("Node features shape", node_feats.shape)
        print("Node features", node_feats)
        #print("Output shape", message.shape)

        # print("nu = 3 :",self.model.products[0].symmetric_contractions.contractions[0].__dict__["_parameters"]["weights_max"].shape)
        # print("nu = 2 :",self.model.products[0].symmetric_contractions.contractions[0].weights[0].shape)
        # print("nu = 1 :",self.model.products[0].symmetric_contractions.contractions[0].weights[1].shape)

        # print(self.model.readouts[0])
        # readout = self.model.readouts[0](node_feats).squeeze(-1)
        # print("Readout shape:", readout.shape)
        # print("Readout", readout)

        return node_feats