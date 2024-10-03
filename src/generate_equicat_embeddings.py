"""
EQUICAT Final Embedding Generator

This script generates final embeddings for molecules using a trained EQUICAT model. 
It processes all conformers of each molecule and combines them into a single embedding.

Key components:
1. ConformerDataset: Custom dataset for handling molecular conformers
2. EQUICATPlusNonLinearReadout: The EQUICAT model with non-linear readout
3. Checkpoint loading: Loads a trained model from a checkpoint
4. Embedding generation: Processes all conformers and generates embeddings
5. Embedding combination: Uses 'improved_self_attention' to combine conformer embeddings
6. Output: Saves final embeddings for all molecules in a JSON file

Author: Utkarsh Sharma
Version: 1.0.0
Date: 10-03-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - mace (custom package)
    - e3nn (>=0.4.0)
    - molli (>=0.1.0)
    - numpy (>=1.20.0)

Usage:
    1. Ensure all dependencies are installed
    2. Set the constants at the top of the script (paths, cutoff, etc.)
    3. Run the script:
       python generate_equicat_embeddings.py

TODO:
    - Implement parallel processing for faster embedding generation
    - Add command-line arguments for easier configuration
    - Optimize memory usage for very large molecular systems
    - Implement error handling and recovery for long-running processes
"""

import torch
import os
import json
import logging
from mace import data, tools, modules
from e3nn import o3
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from conformer_ensemble_embedding_combiner import process_molecule_conformers, move_to_device
import molli as ml
from data_loader import ConformerDataset, compute_avg_num_neighbors

# Constants
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"
CHECKPOINT_PATH = "/Users/utkarsh/MMLI/saved-outputs/sophia/less-conformers/checkpoints/checkpoint_epoch_30.pt"
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output/final_embeddings"
CUTOFF = 6.0
NUM_ENSEMBLES = 806  # Total number of molecules to process
EMBEDDING_TYPE = 'improved_self_attention'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDED_MOLECULES = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']
SAMPLE_SIZE = 10

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    setup_logging(f"{OUTPUT_PATH}/final_embedding_generation.log")
    logging.info("Starting final embedding generation process")

    # Load conformer ensemble
    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
    logging.info(f"Loaded conformer ensemble from {CONFORMER_LIBRARY_PATH}")

    # Create dataset
    dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES, 
                               sample_size=SAMPLE_SIZE, exclude_molecules=EXCLUDED_MOLECULES)  
    
    # Calculate average number of neighbors and unique atomic numbers
    avg_num_neighbors = 0
    unique_atomic_numbers = set()
    for sample in range(len(dataset)):
        sample_data = dataset.get_next_sample()
        if sample_data is None:
            break
        for atomic_data_list, _ in sample_data:
            for conformer in atomic_data_list:
                avg_num_neighbors += compute_avg_num_neighbors(conformer)
                unique_atomic_numbers.update(conformer.atomic_numbers.tolist())
    avg_num_neighbors /= (len(dataset) * dataset.sample_size)
    unique_atomic_numbers = list(unique_atomic_numbers)
    
    z_table = tools.AtomicNumberTable(unique_atomic_numbers)

    # Load model configuration (adjust based on your saved configuration)
    model_config = {
        "r_max": CUTOFF,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 2,
        "num_interactions": 1,
        "num_elements": len(z_table),
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "hidden_irreps": o3.Irreps("256x0e + 256x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "atomic_energies": tools.to_numpy(torch.zeros(len(z_table), dtype=torch.float64)),
        "correlation": 3,
        "gate": torch.nn.functional.silu,
        "avg_num_neighbors": avg_num_neighbors,
    }

    # Initialize and load model
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(DEVICE)
    model = load_checkpoint(CHECKPOINT_PATH, model, DEVICE)
    logging.info(f"Loaded model from checkpoint: {CHECKPOINT_PATH}")

    # Process molecules and generate embeddings
    final_embeddings = {}
    dataset.reset()

    for molecule_idx in range(len(dataset)):
        sample_data = dataset.get_next_sample()
        if sample_data is None:
            break

        for atomic_data_list, key in sample_data:
            logging.info(f"Processing molecule {key} with {len(atomic_data_list)} conformers")
            
            molecule_embeddings = []
            for conformer in atomic_data_list:
                conformer = move_to_device(conformer, DEVICE)
                input_dict = {
                    'positions': conformer.positions,
                    'atomic_numbers': conformer.atomic_numbers,
                    'edge_index': conformer.edge_index
                }
                with torch.no_grad():
                    output = model(input_dict)
                molecule_embeddings.append(output)
            
            molecule_embeddings = torch.stack(molecule_embeddings)
            averaged_embeddings = process_molecule_conformers(molecule_embeddings, model.non_linear_readout.irreps_out)
            
            scalar, vector = averaged_embeddings[EMBEDDING_TYPE]
            if scalar is not None and vector is not None:
                combined = torch.cat([scalar.view(-1), vector.view(-1)])
            elif scalar is not None:
                combined = scalar.view(-1)
            else:
                combined = vector.view(-1)
            
            final_embeddings[key] = combined.detach().cpu().numpy().tolist()
        
        logging.info(f"Processed molecule {molecule_idx + 1}/{len(dataset)}")

    # Save final embeddings
    with open(f"{OUTPUT_PATH}/embeddings.json", 'w') as f:
        json.dump(final_embeddings, f)

    logging.info(f"Saved final embeddings for {len(final_embeddings)} molecules")
    logging.info("Final embedding generation process completed")

if __name__ == "__main__":
    main()
