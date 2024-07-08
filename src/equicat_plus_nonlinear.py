"""
EQUICAT Plus Non-Linear Readout Module

This module extends the EQUICAT (Equivariant Catalysis) model with a custom
non-linear readout layer. It provides two main classes:

1. CustomNonLinearReadout: A custom readout layer that applies a series of
   equivariant linear transformations and non-linear activations.
2. EQUICATPlusNonLinearReadout: Combines the EQUICAT model with the custom
   non-linear readout layer.

The module is designed to preserve equivariance while allowing for more
complex, non-linear processing of molecular representations.

Author: Utkarsh Sharma
Version: 1.0.0
Date: 07-07-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - e3nn (>=0.4.0)
    - equicat (custom package)

Usage:
    model = EQUICATPlusNonLinearReadout(model_config, z_table)
    output = model(input_dict)

For detailed usage instructions, please refer to the README.md file.
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn import nn as e3nn_nn
from equicat import EQUICAT

class CustomNonLinearReadout(nn.Module):
    """
    A custom non-linear readout layer that preserves equivariance.

    This layer applies a series of equivariant linear transformations
    interspersed with non-linear activations on scalar features.
    """

    def __init__(self, irreps_in, gate):
        """
        Initialize the CustomNonLinearReadout.

        Args:
            irreps_in (o3.Irreps): Input irreducible representations.
            gate (callable): Activation function for scalar features.
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        hidden_irreps = o3.Irreps("16x0e + 16x1o")
        output_irreps = o3.Irreps("8x0e + 8x1o")

        # First linear layer
        self.linear_1 = o3.Linear(irreps_in=self.irreps_in, irreps_out=hidden_irreps)
        
        # First non-linearity
        self.non_linearity_1 = e3nn_nn.Activation(
            irreps_in=hidden_irreps,
            acts=[gate if ir.is_scalar() else None for _, ir in hidden_irreps]
        )

        # Second linear layer
        self.linear_2 = o3.Linear(irreps_in=hidden_irreps, irreps_out=hidden_irreps)

        # Second non-linearity
        self.non_linearity_2 = e3nn_nn.Activation(
            irreps_in=hidden_irreps,
            acts=[gate if ir.is_scalar() else None for _, ir in hidden_irreps]
        )

        # Third linear layer
        self.linear_3 = o3.Linear(irreps_in=hidden_irreps, irreps_out=output_irreps)

    def forward(self, x):
        """
        Forward pass of the CustomNonLinearReadout.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed output tensor.
        """
        print(f"Input shape to CustomNonLinearReadout: {x.shape}")

        x = self.linear_1(x)
        x = self.non_linearity_1(x)
        x = self.linear_2(x)
        x = self.non_linearity_2(x)
        x = self.linear_3(x)
        
        # Debug prints
        # print(f"Output shape: {x.shape}")
        
        return x
    
class EQUICATPlusNonLinearReadout(nn.Module):
    """
    Combines the EQUICAT model with a custom non-linear readout layer.

    This class encapsulates the full pipeline of processing molecular data
    through the EQUICAT model and then applying a non-linear readout.
    """

    def __init__(self, model_config, z_table):
        """
        Initialize the EQUICATPlusNonLinearReadout model.

        Args:
            model_config (dict): Configuration for the EQUICAT model.
            z_table (AtomicNumberTable): Table of atomic numbers.
        """
        super().__init__()
        self.model_config = model_config
        self.equicat = EQUICAT(model_config, z_table)
        
        # Initialize CustomNonLinearReadout here
        input_irreps = o3.Irreps("32x0e + 32x1o")
        self.non_linear_readout = CustomNonLinearReadout(
            irreps_in=input_irreps,
            gate=self.model_config['gate']
        )
        print(f"Initialized CustomNonLinearReadout with input irreps: {input_irreps}")

    def forward(self, input_dict):
        """
        Forward pass of the EQUICATPlusNonLinearReadout model.

        Args:
            input_dict (dict): Input dictionary containing molecular data.

        Returns:
            torch.Tensor: Final output after non-linear readout.
        """
        # Get output from EQUICAT
        equicat_output = self.equicat(input_dict)
        print(f"EQUICAT output shape: {equicat_output.shape}")
        
        # Apply NonLinearReadout
        final_output = self.non_linear_readout(equicat_output)
        print(f"Final output shape after NonLinearReadout: {final_output.shape}")
        
        return final_output