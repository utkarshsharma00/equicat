"""
conformer_averaging_sanity_check.py

This script performs sanity checks to verify if the averaging of multiple conformers
is happening correctly in the EQUICAT model training process.

It uses a small sample of the dataset and logs detailed information about
the conformer processing and averaging steps.

Author: Utkarsh Sharma
Version: 1.0.0
Date: 10-03-2024 (MM-DD-YYYY)
License: MIT
"""

import torch
import logging
import os
import random
import molli as ml
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import MultiFamilyConformerDataset
from train import calculate_avg_num_neighbors_and_unique_atomic_numbers
from conformer_ensemble_embedding_combiner import process_molecule_conformers, move_to_device
from mace import tools

logger = logging.getLogger('sanity_check')

# Constants (adjust as needed)
CONFORMER_LIBRARY_PATHS = {
    "family1": "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib",
    "family2": "/Users/utkarsh/MMLI/molli-data/00-libraries/molnet.clib",
}
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/develop_op/"
SAMPLE_SIZE = 2
MAX_CONFORMERS = 2
CUTOFF = 6.0
NUM_FAMILIES = 2
ENSEMBLES_PER_FAMILY = 2

def setup_logging(log_file):
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_model_config(z_table, avg_num_neighbors):
    from mace import modules
    from e3nn import o3
    return {
        "r_max": CUTOFF,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 2,
        "num_interactions": 1,
        "num_elements": len(z_table),
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "hidden_irreps": o3.Irreps("32x0e + 32x1o"),  # Reduced size for sanity check
        "MLP_irreps": o3.Irreps("16x0e"),
        "atomic_energies": tools.to_numpy(torch.zeros(len(z_table), dtype=torch.float64)),
        "correlation": 3,
        "gate": torch.nn.functional.silu,
        "avg_num_neighbors": avg_num_neighbors,
    }

def process_single_molecule(model, atomic_data_list, key, family, device, embedding_type):
    logging.info(f"Processing molecule {key} from family {family}")
    logging.info(f"Number of conformers: {len(atomic_data_list)}")

    molecule_embeddings = []
    for i, conformer in enumerate(atomic_data_list):
        conformer = move_to_device(conformer, device)
        input_dict = {
            'positions': conformer.positions,
            'atomic_numbers': conformer.atomic_numbers,
            'edge_index': conformer.edge_index
        }
        output = model(input_dict)
        molecule_embeddings.append(output)
        logging.info(f"Conformer {i+1} embedding:")
        logging.info(f"  Shape: {output.shape}")
        logging.info(f"  Mean: {output.mean().item():.6f}")
        logging.info(f"  Std: {output.std().item():.6f}")
        logging.info(f"  Min: {output.min().item():.6f}")
        logging.info(f"  Max: {output.max().item():.6f}")
        logging.info(f"  First 5 values: {output[:5].tolist()}")

    molecule_embeddings = torch.stack(molecule_embeddings)
    logging.info(f"Stacked molecule embeddings shape: {molecule_embeddings.shape}")

    averaged_embeddings = process_molecule_conformers(molecule_embeddings, model.non_linear_readout.irreps_out)
    
    scalar, vector = averaged_embeddings[embedding_type]
    
    if scalar is not None and vector is not None:
        combined = torch.cat([scalar.view(-1), vector.view(-1)])
    elif scalar is not None:
        combined = scalar.view(-1)
    else:
        combined = vector.view(-1)

    logging.info(f"Averaged embedding:")
    logging.info(f"  Shape: {combined.shape}")
    logging.info(f"  Mean: {combined.mean().item():.6f}")
    logging.info(f"  Std: {combined.std().item():.6f}")
    logging.info(f"  Min: {combined.min().item():.6f}")
    logging.info(f"  Max: {combined.max().item():.6f}")
    logging.info(f"  First 5 values: {combined[:5].tolist()}")

    return combined

def main():
    log_file = f"{OUTPUT_PATH}/conformer_averaging_sanity_check.log"
    setup_logging(log_file)
    logging.info("Starting conformer averaging sanity check")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create dataset
    logging.info("Creating dataset")
    conformer_libraries = {family: ml.ConformerLibrary(path) for family, path in CONFORMER_LIBRARY_PATHS.items()}
    dataset = MultiFamilyConformerDataset(
        conformer_libraries=conformer_libraries,
        cutoff=CUTOFF,
        sample_size=SAMPLE_SIZE,
        max_conformers=MAX_CONFORMERS,
        num_families=NUM_FAMILIES,
        ensembles_per_family=ENSEMBLES_PER_FAMILY
    )
    logging.info("Dataset created successfully")

    # Calculate average neighbors and unique atomic numbers
    logging.info("Calculating average neighbors and unique atomic numbers")
    avg_num_neighbors, unique_atomic_numbers = calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device)
    logging.info(f"Average number of neighbors: {avg_num_neighbors}")
    logging.info(f"Unique atomic numbers: {unique_atomic_numbers}")

    z_table = tools.AtomicNumberTable(unique_atomic_numbers)

    # Create model
    logging.info("Creating model")
    model_config = get_model_config(z_table, avg_num_neighbors)
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
    logging.info("Model created successfully")

    # Process a small sample
    logging.info("Processing a small sample of molecules")
    embedding_type = 'mean_pooling'
    
    for sample_idx, sample in enumerate(dataset):
        logging.info(f"Processing sample {sample_idx + 1}")
        for atomic_data_list, key, family in sample:
            combined_embedding = process_single_molecule(model, atomic_data_list, key, family, device, embedding_type)
            logging.info(f"Processed molecule {key} from family {family}")
        
        if sample_idx >= 1:  # Process only 2 samples for this sanity check
            break

    logging.info("Conformer averaging sanity check completed")

if __name__ == "__main__":
    main()