"""
EQUICAT Conformer Analysis Pipeline (GPU-enabled version)

This script serves as the main entry point for the EQUICAT conformer analysis pipeline,
now with GPU support.

It orchestrates the entire process of loading conformer data, configuring the EQUICAT model, 
generating embeddings, and applying various pooling methods to combine these embeddings.

The pipeline consists of the following key stages:
1. Data Loading: Retrieves conformer ensembles from the MOLLI library.
2. Model Configuration: Sets up the EQUICAT model with appropriate parameters.
3. Embedding Generation: Processes conformers through the EQUICAT model.
4. Ensemble Pooling: Combines conformer embeddings using multiple techniques.
5. Result Analysis: Outputs the processed embeddings along with their PCA visualizations 
for downstream tasks.

This script ties together various modules and functions to create a comprehensive
workflow for analyzing molecular conformers using equivariant neural networks.

Author: Utkarsh Sharma
Version: 2.0.0
Date: 08-01-2024 (MM-DD-YYYY)
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
    - v2.0.0: Added GPU support
    - v1.2.0: Added extensive sanity checks and detailed embedding prints along with PCA visualizations
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
from e3nn import o3
from mace import data, modules, tools
from equicat import EQUICAT
from data_loader import process_data, ConformerDataset
from conformer_ensemble_embedding_combiner import process_conformer_ensemble, process_ensemble_batches, visualize_embeddings
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout

# Constants
CUTOFF = 5.0
NUM_ENSEMBLES = 2
BATCH_SIZE = 16
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"

# Model configuration
model_config = {
    "atomic_numbers": None,
    "r_max": CUTOFF,
    "num_bessel": 8,
    "num_polynomial_cutoff": 6,
    "max_ell": 1,
    "num_interactions": 2,
    "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
    "correlation": 3,
    "MLP_irreps": o3.Irreps("16x0e"),
    "gate": torch.nn.functional.silu,
}

def sanity_check_conformer(conformer):
    """
    Perform sanity checks on a single conformer.

    Args:
        conformer (torch_geometric.data.Data): A single conformer data object.

    Raises:
        AssertionError: If any of the sanity checks fail.
    """
    print(f"Conformer sanity check:")
    print(f"  Positions shape: {conformer.positions.shape}")
    print(f"  Atomic numbers shape: {conformer.atomic_numbers.shape}")
    print(f"  Edge index shape: {conformer.edge_index.shape}")
    print(f"  Number of nodes: {conformer.num_nodes}")
    print(f"  Number of edges: {conformer.edge_index.shape[1]}")
    
    assert conformer.positions.shape[0] == conformer.num_nodes, "Number of positions doesn't match number of nodes"
    assert conformer.atomic_numbers.shape[0] == conformer.num_nodes, "Number of atomic numbers doesn't match number of nodes"
    assert torch.max(conformer.edge_index) < conformer.num_nodes, "Edge index contains invalid node references"

def sanity_check_embeddings(embeddings):
    """
    Perform sanity checks on the embeddings.

    Args:
        embeddings (dict): A dictionary containing embeddings for different methods.

    Raises:
        AssertionError: If any of the sanity checks fail.
    """
    print("Embeddings sanity check:")
    for method, (scalar, vector) in embeddings.items():
        print(f"  {method}:")
        print(f"    Scalar shape: {scalar.shape}")
        print(f"    Vector shape: {vector.shape}")
        print(f"    Scalar mean: {scalar.mean().item():.4f}, std: {scalar.std().item():.4f}")
        print(f"    Vector mean: {vector.mean().item():.4f}, std: {vector.std().item():.4f}")
        
        assert not torch.isnan(scalar).any(), f"NaN values in {method} scalar embeddings"
        assert not torch.isnan(vector).any(), f"NaN values in {method} vector embeddings"
        assert scalar.shape[0] == vector.shape[0], f"Mismatch in number of atoms between scalar and vector for {method}"
        
        # Check for differences between conformers
        if scalar.shape[0] > 1:
            num_comparisons = min(10, scalar.shape[0] * (scalar.shape[0] - 1) // 2)
            scalar_diffs = []
            vector_diffs = []
            
            for _ in range(num_comparisons):
                i, j = torch.randint(0, scalar.shape[0], (2,))
                while i == j:
                    j = torch.randint(0, scalar.shape[0], (1,))
                
                scalar_diff = (scalar[i] - scalar[j]).abs().mean()
                vector_diff = (vector[i] - vector[j]).abs().mean()
                scalar_diffs.append(scalar_diff.item())
                vector_diffs.append(vector_diff.item())
            
            print(f"    Conformer difference statistics (over {num_comparisons} random pairs):")
            print(f"      Scalar - mean: {np.mean(scalar_diffs):.4f}, std: {np.std(scalar_diffs):.4f}, min: {np.min(scalar_diffs):.4f}, max: {np.max(scalar_diffs):.4f}")
            print(f"      Vector - mean: {np.mean(vector_diffs):.4f}, std: {np.std(vector_diffs):.4f}, min: {np.min(vector_diffs):.4f}, max: {np.max(vector_diffs):.4f}")

def print_detailed_embeddings(embeddings, level="Batch"):
    """
    Print detailed information about the embeddings.

    Args:
        embeddings (dict): A dictionary containing embeddings for different methods.
        level (str): The level of embeddings being printed (e.g., "Batch" or "Ensemble").
    """
    print(f"\n{level} Detailed Embeddings:")
    for method, (scalar, vector) in embeddings.items():
        print(f"  {method}:")
        print(f"    Scalar embeddings shape: {scalar.shape}")
        print(f"    Vector embeddings shape: {vector.shape}")
        
        # Print first few scalar embeddings for each conformer
        print(f"    Scalar embeddings (first 3 conformers, first 5 features):")
        for i in range(min(3, scalar.shape[0])):
            print(f"      Conformer {i}:")
            print(scalar[i, :5])
        
        # Print first few vector embeddings for each conformer
        print(f"    Vector embeddings (first 3 conformers, first 3 features):")
        for i in range(min(3, vector.shape[0])):
            print(f"      Conformer {i}:")
            print(vector[i, :3])

def process_and_print_ensemble(ensemble_batches, ensemble_id, device):
    """
    Process and print embeddings for a complete ensemble.

    Args:
        ensemble_batches (list): List of batches for the ensemble.
        ensemble_id (int): ID of the ensemble being processed.
    """
    print(f"\nProcessing complete Ensemble {ensemble_id}")
    ensemble_embeddings = process_ensemble_batches(ensemble_batches)
    print("\nEnsemble Average Embeddings:")
    sanity_check_embeddings(ensemble_embeddings)
    print_detailed_embeddings(ensemble_embeddings, level="Ensemble")
    visualize_embeddings(ensemble_embeddings)

def main():
    """
    Main function to run the EQUICAT conformer analysis pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
    conformer_dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES)

    current_ensemble_id = None
    ensemble_batches = []

    for batch_conformers, unique_atomic_numbers, avg_num_neighbors, ensemble_id in process_data(conformer_dataset, batch_size=BATCH_SIZE, device=device):
        print(f"\nProcessing batch of {len(batch_conformers)} conformers from Ensemble {ensemble_id}")
        
        # Sanity check each conformer in the batch
        for i, conformer in enumerate(batch_conformers):
            print(f"\nSanity check for conformer {i} in batch:")
            sanity_check_conformer(conformer)

        if current_ensemble_id is not None and current_ensemble_id != ensemble_id:
            process_and_print_ensemble(ensemble_batches, current_ensemble_id, device)
            ensemble_batches = []

        z_table = tools.AtomicNumberTable(unique_atomic_numbers)
        atomic_energies = np.zeros(len(z_table), dtype=float)

        model_config.update({
            'num_elements': len(z_table),
            'atomic_energies': atomic_energies,
            'atomic_numbers': torch.tensor(z_table.zs, device=device),
            'avg_num_neighbors': avg_num_neighbors
        })

        equicat_model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
        print(equicat_model.get_forward_pass_summary())

        conformer_embeddings = []
        for i, conformer in enumerate(batch_conformers):
            input_dict = {
                'positions': conformer.positions,
                'atomic_numbers': conformer.atomic_numbers,
                'edge_index': conformer.edge_index
            }
            output = equicat_model(input_dict)
            conformer_embeddings.append(output)
            print(f"Conformer {i} embeddings shape: {output.shape}")
            
            # Sanity check the output
            assert not torch.isnan(output).any(), f"NaN values in output for conformer {i}"
            assert output.shape[0] == conformer.num_nodes, f"Output shape mismatch for conformer {i}"

        conformer_embeddings = torch.stack(conformer_embeddings)
        ensemble_batches.append(conformer_embeddings)

        batch_embeddings = process_conformer_ensemble(conformer_embeddings)
        print("\nSanity check for batch embeddings:")
        sanity_check_embeddings(batch_embeddings)
        print_detailed_embeddings(batch_embeddings, level="Batch")

        current_ensemble_id = ensemble_id
        print("=" * 50)

    if ensemble_batches:
        process_and_print_ensemble(ensemble_batches, current_ensemble_id, device)

    print("Finished processing all conformers in all ensembles.")

if __name__ == "__main__":
    main()



