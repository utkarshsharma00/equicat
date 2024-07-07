# equicat_plus_nonlinear.py

import torch
import torch.nn as nn
from e3nn import o3
from e3nn import nn as e3nn_nn
from equicat import EQUICAT

class CustomNonLinearReadout(nn.Module):
    def __init__(self, irreps_in, gate):
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
    def __init__(self, model_config, z_table):
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
        # Get output from EQUICAT
        equicat_output = self.equicat(input_dict)
        print(f"EQUICAT output shape: {equicat_output.shape}")
        
        # Apply NonLinearReadout
        final_output = self.non_linear_readout(equicat_output)
        print(f"Final output shape after NonLinearReadout: {final_output.shape}")
        
        return final_output