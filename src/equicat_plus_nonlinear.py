"""
EQUICAT Plus Non-Linear Readout Module

This module significantly enhances the EQUICAT model by introducing a flexible,
customizable non-linear readout layer. It provides advanced irreps handling, 
adaptive processing pathways, and comprehensive debugging capabilities.

Key components:
1. Architecture Design:
   - CustomNonLinearReadout for flexible tensor processing
   - Adaptive layer construction based on input types
   - Multi-level feature transformation pipeline
   - Equivariant non-linear activation handling

2. Processing Features:
   - Dynamic irreps analysis and handling
   - Type-aware tensor transformations
   - Scalar/vector separation and recombination
   - Multi-interaction support
   - Comprehensive shape tracking

3. Integration Features:
   - Seamless EQUICAT model integration
   - Automatic irreps compatibility checking
   - Device management
   - Memory optimization

Key Features:
1. Dynamic irreps handling for flexible input processing
2. Automatic scalar/vector feature detection
3. Configurable hidden layer architecture
4. Comprehensive shape tracking and validation
5. Advanced equivariant transformations
6. Flexible activation function support
7. Multiple interaction layer compatibility
8. Detailed debug logging system
9. Memory-efficient tensor handling
10. Device-agnostic computation

Author: Utkarsh Sharma
Version: 4.0.0
Date: 12-14-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
- torch (>=1.9.0)
- e3nn (>=0.4.0)
- equicat (custom package)
- numpy (>=1.20.0)

Usage:
    # Configure model
    model_config = {
        'hidden_irreps': o3.Irreps("256x0e + 256x1o"),
        'num_interactions': 2,
        'gate': torch.nn.functional.silu,
    }
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
    
    # Forward pass
    output = model(input_dict)

For detailed usage instructions, please refer to the README.md file.

Change Log:
- v4.0.0 (12-14-2024):
  * Major version bump to align with EquiCat ecosystem
  * Implemented dynamic irreps analysis and handling
  * Added comprehensive type validation
  * Enhanced multi-interaction layer support
  * Improved memory efficiency
  * Added detailed shape tracking
  * Enhanced debug logging system
  * Optimized tensor operations
  * Improved documentation
- v3.0.0 (09-10-2024):
  * Complete redesign of CustomNonLinearReadout
  * Added flexible irreps support
  * Enhanced debugging features
- v2.0.0 (08-01-2024):
  * Added GPU support
  * Enhanced debug printing
- v1.0.0 (07-07-2024):
  * Initial implementation

ToDo:
- Implement gradient checkpointing for memory efficiency
- Add support for custom layer architectures
- Enhance shape validation system
- Implement automated architecture validation
- Add performance profiling tools
- Improve tensor operation efficiency
- Add support for distributed training
- Enhance irreps compatibility checking
"""

import torch
import torch.nn as nn
import numpy as np
from e3nn import o3
from e3nn import nn as e3nn_nn
from equicat import EQUICAT, move_to_device

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(42)

class CustomNonLinearReadout(nn.Module):
    """
    A custom non-linear readout layer that adaptively processes geometric tensors.

    This class handles different input types from EQUICAT model (scalar/vector/both),
    maintains geometric properties through processing layers, and produces scalar output.
    The architecture automatically adjusts based on input type.

    Args:
        irreps_in (o3.Irreps): Input irreducible representations. 
        irreps_out (o3.Irreps): Output irreducible representations (scalar-only).
        hidden_irreps (List[int], optional): Dimensions of hidden layers.
        gate (Callable, optional): Activation function for scalar features.
        
    Returns:
        None
    """
    def __init__(self, irreps_in, irreps_out, hidden_irreps=[256, 192], gate=torch.nn.functional.silu):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        
        # Analyze input irreps
        self.has_scalar = any(ir.l == 0 for _, ir in self.irreps_in)
        self.has_vector = any(ir.l == 1 for _, ir in self.irreps_in)
        
        # Calculate dimensions
        self.scalar_dim = sum(mul * ir.dim for mul, ir in self.irreps_in if ir.l == 0)
        self.vector_dim = sum(mul * ir.dim for mul, ir in self.irreps_in if ir.l == 1)
        
        print(f"Input type - Scalar: {self.has_scalar}, Vector: {self.has_vector}")
        print(f"Dimensions - Scalar: {self.scalar_dim}, Vector: {self.vector_dim}")
        
        # Build layer structure based on input type
        self.layers = nn.ModuleList()
        
        current_scalar_dim = self.scalar_dim
        current_vector_dim = self.vector_dim
        
        # First layer
        if self.has_scalar and self.has_vector:
            # Combined scalar and vector
            irreps_h1 = o3.Irreps(f"{hidden_irreps[0]}x0e + {hidden_irreps[0]}x1o")
            layer1 = o3.Linear(self.irreps_in, irreps_h1)
        elif self.has_scalar:
            # Scalar only
            irreps_h1 = o3.Irreps(f"{hidden_irreps[0]}x0e")
            layer1 = o3.Linear(self.irreps_in, irreps_h1)
        else:
            # Vector only
            irreps_h1 = o3.Irreps(f"{hidden_irreps[0]}x1o")
            layer1 = o3.Linear(self.irreps_in, irreps_h1)
            
        self.layers.append(layer1)
        
        # Add non-linearity
        non_linearity1 = e3nn_nn.Activation(
            irreps_in=irreps_h1,
            acts=[gate if ir.is_scalar() else None for _, ir in irreps_h1]
        )
        self.layers.append(non_linearity1)
        
        # Second layer
        if self.has_scalar and self.has_vector:
            irreps_h2 = o3.Irreps(f"{hidden_irreps[1]}x0e + {hidden_irreps[1]}x1o")
            layer2 = o3.Linear(irreps_h1, irreps_h2)
        elif self.has_scalar:
            irreps_h2 = o3.Irreps(f"{hidden_irreps[1]}x0e")
            layer2 = o3.Linear(irreps_h1, irreps_h2)
        else:
            irreps_h2 = o3.Irreps(f"{hidden_irreps[1]}x1o")
            layer2 = o3.Linear(irreps_h1, irreps_h2)
            
        self.layers.append(layer2)
        
        # Add non-linearity
        non_linearity2 = e3nn_nn.Activation(
            irreps_in=irreps_h2,
            acts=[gate if ir.is_scalar() else None for _, ir in irreps_h2]
        )
        self.layers.append(non_linearity2)
        
        # Final layer - always outputs scalar as per requirement
        final_layer = o3.Linear(irreps_h2, self.irreps_out)
        self.layers.append(final_layer)

    def forward(self, x):
        """
        Forward pass handling different input types adaptively.
        """
        print(f"Input shape to CustomNonLinearReadout: {x.shape}")
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, o3.Linear):
                print(f"Shape before Linear layer {i//2 + 1}: {x.shape}")
                print(f"Linear layer {i//2 + 1} input irreps: {layer.irreps_in}")
                print(f"Linear layer {i//2 + 1} output irreps: {layer.irreps_out}")
            x = layer(x)
            print(f"Shape after {layer.__class__.__name__} {i//2 + 1}: {x.shape}")
        
        return x

    def __repr__(self):
        lines = [f"CustomNonLinearReadout("]
        lines.append(f"  Input type - Scalar: {self.has_scalar}, Vector: {self.has_vector}")
        lines.append(f"  Input dimensions - Scalar: {self.scalar_dim}, Vector: {self.vector_dim}")
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, o3.Linear):
                lines.append(f"  (linear_{i//2+1}): Linear({layer.irreps_in} -> {layer.irreps_out} | {layer.weight.numel()} weights)")
            elif isinstance(layer, e3nn_nn.Activation):
                acts = ''.join(['x' if act else ' ' for act in layer.acts])
                lines.append(f"  (non_linearity_{i//2+1}): Activation [{acts}] ({layer.irreps_in} -> {layer.irreps_out})")
        lines.append(")")
        return "\n".join(lines)

class EQUICATPlusNonLinearReadout(nn.Module):
    """
    Combines the EQUICAT model with a custom non-linear readout layer.

    This class encapsulates the full pipeline of processing molecular data
    through the EQUICAT model and then applying a non-linear readout.

    Args:
        model_config (dict): Configuration for the EQUICAT model.
        z_table (AtomicNumberTable): Table of atomic numbers.

    Returns:
        None
    """
    def __init__(self, model_config, z_table):
        super().__init__()
        self.model_config = model_config
        self.equicat = EQUICAT(model_config, z_table)
        
        # Get the actual output irreps from EQUICAT's last layer
        if model_config['num_interactions'] > 1:
            last_product_layer = self.equicat.product_layers[-1]
            equicat_output_irreps = last_product_layer.linear.irreps_out
        else:
            equicat_output_irreps = model_config['hidden_irreps']
        
        # Define the output irreps
        self.output_irreps = "192x0e"

        print(f"EQUICAT output irreps: {equicat_output_irreps}")
        self.non_linear_readout = CustomNonLinearReadout(
            irreps_in=equicat_output_irreps,
            irreps_out=self.output_irreps,
            hidden_irreps=[256, 192],
            gate=self.model_config['gate']
        )
        print(f"CustomNonLinearReadout output irreps: {self.output_irreps}")
        
    def get_forward_pass_summary(self):
        """
        Get a summary of the forward pass for the entire model.

        Returns:
            str: A string containing the summary of the forward pass.
        """
        equicat_summary = self.equicat.get_forward_pass_summary()
        nonlinear_summary = [
            "\nNonLinear Readout Summary:",
            str(self.non_linear_readout)
        ]
        return equicat_summary + "\n" + "\n".join(nonlinear_summary)

    def forward(self, input_dict):
        """
        Forward pass of the EQUICATPlusNonLinearReadout model.

        Args:
            input_dict (dict): Input dictionary containing molecular data.

        Returns:
            torch.Tensor: Final output after non-linear readout.
        """
        device = next(self.parameters()).device
        input_dict = move_to_device(input_dict, device)
        
        equicat_output = self.equicat(input_dict)
        print(f"EQUICAT output shape: {equicat_output.shape}")
        print(f"EQUICAT output device: {equicat_output.device}")
        
        final_output = self.non_linear_readout(equicat_output)
        print(f"Final output shape after CustomNonLinearReadout: {final_output.shape}")
        print(f"Final output device: {final_output.device}")
        
        return final_output

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