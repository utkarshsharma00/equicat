"""
EQUICAT Model Training Script (GPU-enabled version with Optional Conformer Padding)

This script implements the training pipeline for the EQUICAT model, a neural network 
designed for molecular conformer analysis. It includes data loading, model initialization, 
training loop, logging functionality, early stopping, and robust error handling.
It now supports GPU acceleration using CUDA and optional handling of variable-sized conformer ensembles.

Key features:
1. GPU acceleration with CUDA support
2. Optional Conformer Padding: Handles variable-sized conformer ensembles when enabled
3. Customizable logging setup with detailed gradient and loss tracking
4. Dynamic calculation of average neighbors and unique atomic numbers
5. Contrastive loss for semi-supervised learning
6. Gradient clipping and comprehensive logging during training
7. Configurable model parameters and training hyperparameters
8. Total runtime measurement
9. Early stopping to prevent overfitting
10. Robust error handling and NaN detection
11. GPU memory tracking

New Feature:
- Optional Conformer Padding: The training process now accommodates optional padding of
  conformer batches, ensuring consistent batch sizes across all molecules and epochs when enabled.
  This feature can be controlled via a command-line argument.

Author: Utkarsh Sharma
Version: 2.2.0
Date: 08-08-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - e3nn (>=0.4.0)
    - mace (custom package)
    - molli (custom package) 

Usage:
    python train.py [--pad_batches]

For detailed usage instructions, please refer to the README.md file.

Change Log: 
    - v2.2.0: Added optional conformer padding via command-line argument
    - v2.1.0: Added support for training with padded conformer batches
    - v2.0.0: Added GPU support with CUDA
    - v1.4.0: Added robust error handling, gradient clipping, and NaN detection
    - v1.3.0: Implemented early stopping
    - v1.2.0: Added total runtime measurement and made constants global
    - v1.1.0: Added gradient checking and improved logging
    - v1.0.0: Initial implementation of EQUICAT training pipeline

TODO:
    - Implement learning rate scheduling
    - Add support for distributed training
    - Implement checkpointing for resuming training
    - Implement weighted loss calculation to account for conformer duplications
"""

import torch
import molli as ml  
import logging
import sys
import numpy as np
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from e3nn import o3
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
from mace import data, modules, tools
from mace.tools import to_numpy
from collections import OrderedDict
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import ConformerDataset, process_data
from contrastive_loss import contrastive_loss
from conformer_ensemble_embedding_combiner import process_conformer_ensemble

# Global constants
# CONFORMER_LIBRARY_PATH = "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib"
# OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/output"
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib" 
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output"
NUM_ENSEMBLES = 2
CUTOFF = 5.0
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 1e-4
GRADIENT_CLIP_VALUE = 1.0
EPSILON = 1e-8
VISUALIZATION_INTERVAL = 1

def get_device():
    """
    Select the best available device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def setup_logging(log_file):
    """
    Set up logging configuration for the training process.

    Args:
        log_file (str): Path to the log file.

    Returns:
        None
    """
    # Clear existing log file
    with open(log_file, 'w') as f:
        f.write("")  # This will clear the file
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 'w' mode overwrites the file
            logging.StreamHandler(sys.stdout)
        ]
    )

def calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device):
    """
    Calculate the average number of neighbors and unique atomic numbers in the dataset.

    Args:
        dataset (ConformerDataset): The dataset to analyze.
        device (torch.device): The device to perform calculations on.

    Returns:
        tuple: A tuple containing:
            - float: Average number of neighbors per atom.
            - list: List of unique atomic numbers in the dataset.
    """
    total_neighbors = 0
    total_atoms = 0
    unique_atomic_numbers = OrderedDict()
    
    for batch_conformers, _, _, _, _ in process_data(dataset, batch_size=BATCH_SIZE, device=device): #! added num_added over here
        for conformer in batch_conformers:
            conformer = conformer.to(device)
            total_neighbors += conformer.edge_index.shape[1]
            total_atoms += conformer.positions.shape[0]
            for atomic_number in conformer.atomic_numbers.cpu().tolist():
                unique_atomic_numbers[atomic_number] = None  # Using OrderedDict to maintain order
    
    avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
    return avg_neighbors, list(unique_atomic_numbers.keys())

def train_equicat(model_config, z_table, conformer_ensemble, cutoff, device, pad_batches):
    """
    Train the EQUICAT model using contrastive loss with early stopping and optional batch padding.

    Args:
        model_config (dict): Configuration for the EQUICAT model.
        z_table (AtomicNumberTable): Table of atomic numbers.
        conformer_ensemble (ConformerLibrary): Library of conformers for training.
        cutoff (float): Cutoff distance for atomic interactions.
        device (torch.device): The device to perform training on.
        pad_batches (bool): Whether to pad batches to a consistent size.

    Returns:
        EQUICATPlusNonLinearReadout: The trained EQUICAT model.

    Note:
        When pad_batches is True, the function will pad smaller batches to the specified
        batch size. The loss calculation will only consider the original, non-padded
        conformers to maintain training integrity.
    """
    print(f"train_equicat called with pad_batches={pad_batches}")
    # Initialize model and move to device
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
    print(model)
    logging.info(f"Model initialized and moved to {device}")

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    logging.info(f"Optimizer initialized with learning rate: {LEARNING_RATE}")

    # Initialize dataset
    dataset = ConformerDataset(conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES)
    logging.info(f"Dataset initialized with {NUM_ENSEMBLES} ensembles")

    # Early stopping variables
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        batch_count = 0

        with detect_anomaly():
            for batch_conformers, _, _, ensemble_id, num_added in process_data(dataset, batch_size=BATCH_SIZE, device=device, pad_batches=pad_batches):
                print(f"Batch received: {len(batch_conformers)} conformers, num_added: {num_added}")  # Debug print
                optimizer.zero_grad()

                embeddings = []

                for conformer in batch_conformers:
                    input_dict = {
                        'positions': conformer.positions,
                        'atomic_numbers': conformer.atomic_numbers,
                        'edge_index': conformer.edge_index
                    }
                    output = model(input_dict)
                    embeddings.append(output)

                embeddings = torch.stack(embeddings).to(device)
                ensemble_ids = torch.full((len(batch_conformers),), ensemble_id, device=device)

                if num_added > 0:
                    original_batch_size = len(batch_conformers) - num_added
                    loss = contrastive_loss(embeddings[:original_batch_size], ensemble_ids[:original_batch_size])
                else:
                    loss = contrastive_loss(embeddings, ensemble_ids)

                logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_count+1}]")
                logging.info(f"Embeddings shape: {embeddings.shape}, Loss: {loss.item():.6f}")
                logging.info(f"Ensemble IDs: {ensemble_ids}")
                if pad_batches:
                    logging.info(f"Number of randomly added conformers: {num_added}")

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"NaN or Inf loss detected: {loss.item()}")
                    continue

                loss.backward()

                # Gradient clipping
                clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)

                # Gradient check
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm == 0:
                            logging.warning(f"Gradient for {name} is zero!")
                        elif torch.isnan(param.grad).any():
                            logging.warning(f"Gradient for {name} contains NaN values!")
                        logging.info(f"Gradient norm for {name}: {grad_norm:.6f}")

                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_count}], Loss: {loss.item():.6f}")

                # Log GPU memory usage
                if device.type == 'cuda':
                    logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
                    logging.info(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB")

        avg_loss = total_loss / (batch_count + EPSILON) # Add epsilon to prevent division by zero
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.6f}")

        # Early stopping check
        if avg_loss < best_loss - EARLY_STOPPING_DELTA:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve == EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            model.load_state_dict(best_model)
            break

    logging.info("Training completed")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EQUICAT Model Training")
    parser.add_argument('--pad_batches', action='store_true', help='Enable batch padding')
    args = parser.parse_args()
    print(f"Parsed args.pad_batches: {args.pad_batches}")
    start_time = time.time()

    setup_logging(f"{OUTPUT_PATH}/training.log")
    logging.info("Starting EQUICAT training")

    device = get_device()
    logging.info(f"Using device: {device}")
    logging.info(f"Padding batches: {args.pad_batches}")

    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
    logging.info(f"Loaded conformer ensemble from {CONFORMER_LIBRARY_PATH}")

    temp_dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES)
    
    avg_num_neighbors, unique_atomic_numbers = calculate_avg_num_neighbors_and_unique_atomic_numbers(temp_dataset, device)
    np.save(f'{OUTPUT_PATH}/unique_atomic_numbers.npy', np.array(unique_atomic_numbers))
    logging.info("Initializing model configuration...")
    logging.info(f"Unique atomic numbers in dataset: {unique_atomic_numbers}")
    logging.info(f"Average number of neighbors across dataset: {avg_num_neighbors}")
    logging.info(f"Initialized dataset with {NUM_ENSEMBLES} conformer ensembles")
    logging.info("Model initialization complete. Starting training...")

    z_table = tools.AtomicNumberTable(unique_atomic_numbers)

    model_config = {
        "r_max": CUTOFF,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 1,
        "num_interactions": 2,
        "num_elements": len(z_table),
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "atomic_energies": to_numpy(torch.zeros(len(z_table))),
        "correlation": 3,
        "gate": torch.nn.functional.silu,
        "avg_num_neighbors": avg_num_neighbors,
    }
    logging.info(f"Model configuration: {model_config}")

    trained_model = train_equicat(model_config, z_table, conformer_ensemble, CUTOFF, device, args.pad_batches)

    torch.save(trained_model.state_dict(), f"{OUTPUT_PATH}/trained_equicat_model.pt")
    logging.info("Model training completed and saved.")

    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(f"Total runtime: {total_runtime:.2f} seconds")