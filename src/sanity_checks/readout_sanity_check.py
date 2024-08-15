"""
EQUICAT Readout Layer Sanity Check

This script performs sanity checks on the readout layers of the EQUICAT model:
1. Compares a vanilla readout layer with a custom non-linear readout layer
2. Checks their structures and outputs
3. Verifies equivariance properties of both layers

Author: Utkarsh Sharma
Version: 1.0.0
Date: 08-14-2024 (MM-DD-YYYY)
License: MIT

Usage:
    python readout_sanity_check.py

Dependencies:
    - torch (>=1.9.0)
    - e3nn (>=0.4.0)

TODO:
    - Add more comprehensive equivariance tests
    - Implement additional readout layer variations for comparison
    - Add command-line arguments for customizable testing parameters

Change Log:
    - v1.0.0: Initial implementation with vanilla and custom non-linear readout comparisons
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn import nn as e3nn_nn

class VanillaReadout(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, x):
        return self.linear(x)

    def __repr__(self):
        return self.detailed_repr()

    def detailed_repr(self):
        lines = [f"VanillaReadout("]
        lines.append(f"  (linear): Linear({self.irreps_in} -> {self.irreps_out} | {self.linear.weight.numel()} weights)")
        lines.append(")")
        return "\n".join(lines)

class CustomNonLinearReadout(nn.Module):
    def __init__(self, irreps_in, gate):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        hidden_irreps = o3.Irreps("4x0e + 4x1o")
        output_irreps = o3.Irreps("2x0e + 2x1o")

        self.linear_1 = o3.Linear(irreps_in=self.irreps_in, irreps_out=hidden_irreps)
        self.non_linearity_1 = e3nn_nn.Activation(
            irreps_in=hidden_irreps,
            acts=[gate if ir.is_scalar() else None for _, ir in hidden_irreps]
        )
        self.linear_2 = o3.Linear(irreps_in=hidden_irreps, irreps_out=hidden_irreps)
        self.non_linearity_2 = e3nn_nn.Activation(
            irreps_in=hidden_irreps,
            acts=[gate if ir.is_scalar() else None for _, ir in hidden_irreps]
        )
        self.linear_3 = o3.Linear(irreps_in=hidden_irreps, irreps_out=output_irreps)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.non_linearity_1(x)
        x = self.linear_2(x)
        x = self.non_linearity_2(x)
        x = self.linear_3(x)
        return x

    def __repr__(self):
        return self.detailed_repr()

    def detailed_repr(self):
        lines = [f"CustomNonLinearReadout("]
        lines.append(f"  (linear_1): Linear({self.irreps_in} -> {self.linear_1.irreps_out} | {self.linear_1.weight.numel()} weights)")
        lines.append(f"  (non_linearity_1): Activation [{'x' if self.non_linearity_1.acts[0] else ' '}] ({self.non_linearity_1.irreps_in} -> {self.non_linearity_1.irreps_out})")
        lines.append(f"  (linear_2): Linear({self.linear_2.irreps_in} -> {self.linear_2.irreps_out} | {self.linear_2.weight.numel()} weights)")
        lines.append(f"  (non_linearity_2): Activation [{'x' if self.non_linearity_2.acts[0] else ' '}] ({self.non_linearity_2.irreps_in} -> {self.non_linearity_2.irreps_out})")
        lines.append(f"  (linear_3): Linear({self.linear_3.irreps_in} -> {self.linear_3.irreps_out} | {self.linear_3.weight.numel()} weights)")
        lines.append(")")
        return "\n".join(lines)

def sanity_check():
    # Set up input
    irreps_in = o3.Irreps("4x0e + 4x1o")
    batch_size = 4
    input_tensor = torch.randn(batch_size, irreps_in.dim)
    
    # Initialize models
    vanilla_readout = VanillaReadout(irreps_in, o3.Irreps("2x0e + 2x1o"))
    custom_readout = CustomNonLinearReadout(irreps_in, torch.nn.functional.silu)

    # Print detailed representations
    print("Vanilla Readout Structure:")
    print(vanilla_readout)
    print("\nCustom NonLinear Readout Structure:")
    print(custom_readout)
    
    # Forward pass
    vanilla_output = vanilla_readout(input_tensor)
    custom_output = custom_readout(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Vanilla Readout output shape:", vanilla_output.shape)
    print("Custom NonLinear Readout output shape:", custom_output.shape)
    
    # Compare outputs
    print("\nVanilla Readout output:")
    print(vanilla_output)
    
    print("\nCustom NonLinear Readout output:")
    print(custom_output)
    
    # Check equivariance
    rotation = o3.rand_matrix()
    D_in = irreps_in.D_from_matrix(rotation)
    D_out = o3.Irreps("2x0e + 2x1o").D_from_matrix(rotation)
    
    # Rotate input
    rotated_input = torch.einsum("ij,bj->bi", D_in, input_tensor)
    
    # Forward pass with rotated input
    vanilla_rotated_output = vanilla_readout(rotated_input)
    custom_rotated_output = custom_readout(rotated_input)
    
    # Rotate output
    vanilla_output_rotated = torch.einsum("ij,bj->bi", D_out, vanilla_output)
    custom_output_rotated = torch.einsum("ij,bj->bi", D_out, custom_output)
    
    print("\nEquivariance check:")
    print("Vanilla Readout equivariance error:", torch.mean(torch.abs(vanilla_rotated_output - vanilla_output_rotated)))
    print("Custom NonLinear Readout equivariance error:", torch.mean(torch.abs(custom_rotated_output - custom_output_rotated)))

if __name__ == "__main__":
    sanity_check()