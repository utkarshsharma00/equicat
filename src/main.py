"""
EQUICAT Conformer Analysis Pipeline

This script serves as the main entry point for the EQUICAT (Equivariant Catalysis) 
conformer analysis pipeline. It orchestrates the entire process of loading conformer 
data, configuring the EQUICAT model, generating embeddings, and applying various 
pooling methods to combine these embeddings.

The pipeline consists of the following key stages:
1. Data Loading: Retrieves conformer ensembles from the MOLLI library.
2. Model Configuration: Sets up the EQUICAT model with appropriate parameters.
3. Embedding Generation: Processes conformers through the EQUICAT model.
4. Ensemble Pooling: Combines conformer embeddings using multiple techniques.
5. Result Analysis: Outputs the processed embeddings for further use or analysis.

This script ties together various modules and functions to create a comprehensive
workflow for analyzing molecular conformers using equivariant neural networks.

Author: Utkarsh Sharma
Version: 1.1.0
Date: 07-11-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - numpy (>=1.20.0)
    - molli (>=0.1.0)
    - e3nn (>=0.4.0)
    - mace (custom package)

Usage:
    python main.py

For detailed usage instructions, please refer to the README.md file.

Change Log:
    - v1.1.0: Added ensemble-level processing and improved batch handling
    - v1.0.0: Initial implementation of EQUICAT pipeline

TODO:
    - Implement command-line arguments for configurable parameters
    - Add logging functionality for better debugging and monitoring
    - Optimize memory usage for large-scale conformer processing
"""

import torch
import numpy as np
import molli as ml
import torch.nn.functional as F
from e3nn import o3
from mace import data, modules, tools
from equicat import EQUICAT
from data_loader import process_data, ConformerDataset
from conformer_ensemble_embedding_combiner import process_conformer_ensemble, process_ensemble_batches
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout

# Constants
CUTOFF = 5.0
NUM_ENSEMBLES = 5
BATCH_SIZE = 16
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"

# Model configuration
model_config = {
    "atomic_numbers": None,
    "r_max": CUTOFF,
    "num_bessel": 8,
    "num_polynomial_cutoff": 6,
    "max_ell": 2,
    "num_interactions": 2,
    "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
    "correlation": 3,
    "MLP_irreps": o3.Irreps("16x0e"),
    "gate": torch.nn.functional.silu,
}

def main():
    # Load conformer data
    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
    conformer_dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES)

    current_ensemble_id = None
    ensemble_batches = []

    # Process each batch of conformers
    for batch_conformers, unique_atomic_numbers, avg_num_neighbors, ensemble_id in process_data(conformer_dataset, batch_size=BATCH_SIZE):
        print(f"Processing batch of {len(batch_conformers)} conformers from Ensemble {ensemble_id}")
        
        # If we've moved to a new ensemble, process the previous one
        if current_ensemble_id is not None and current_ensemble_id != ensemble_id:
            process_and_print_ensemble(ensemble_batches, current_ensemble_id)
            ensemble_batches = []

        # Initialize model components
        z_table = tools.AtomicNumberTable(unique_atomic_numbers)
        atomic_energies = np.zeros(len(z_table), dtype=float)

        # Update model configuration
        model_config.update({
            'num_elements': len(z_table),
            'atomic_energies': atomic_energies,
            'atomic_numbers': torch.tensor(z_table.zs),
            'avg_num_neighbors': avg_num_neighbors
        })

        # Initialize EQUICAT model
        equicat_model = EQUICATPlusNonLinearReadout(model_config, z_table)

        # Process conformers and generate embeddings
        conformer_embeddings = []
        for conformer in batch_conformers:
            input_dict = {
                'positions': conformer.positions,
                'atomic_numbers': conformer.atomic_numbers,
                'edge_index': conformer.edge_index
            }
            output = equicat_model(input_dict)
            conformer_embeddings.append(output)
            print(f"Conformer embeddings shape: {output.shape}")

        # Combine conformer embeddings for this batch
        conformer_embeddings = torch.stack(conformer_embeddings)
        ensemble_batches.append(conformer_embeddings)

        # Print individual conformer results for this batch
        batch_embeddings = process_conformer_ensemble(conformer_embeddings)
        print("\nIndividual Conformer Embeddings for current batch:")
        print_embeddings(batch_embeddings)

        current_ensemble_id = ensemble_id
        print("=" * 50)  # Separator between batches

    # Process the last ensemble
    if ensemble_batches:
        process_and_print_ensemble(ensemble_batches, current_ensemble_id)

    print("Finished processing all conformers in all ensembles.")

def process_and_print_ensemble(ensemble_batches, ensemble_id):
    print(f"\nProcessing complete Ensemble {ensemble_id}")
    ensemble_embeddings = process_ensemble_batches(ensemble_batches)
    print("\nEnsemble Average Embeddings:")
    print_embeddings(ensemble_embeddings)

def print_embeddings(embeddings):
    for method, (scalar, vector) in embeddings.items():
        print(f"{method}:")
        print(f"  Scalar shape: {scalar.shape}")
        print(f"  Vector shape: {vector.shape}")
        print(f"  Scalar (first 5 atoms):\n{scalar[0, :5]}")
        print(f"  Vector (first 5 atoms):\n{vector[0, :5]}")
        print("-" * 50)

if __name__ == "__main__":
    main()