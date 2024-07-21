"""
EQUICAT Model Training Script

This script implements the training pipeline for the EQUICAT model, a neural network 
designed for molecular conformer analysis. It includes data loading, model initialization, 
training loop, logging functionality, and early stopping.

Key features:
1. Customizable logging setup
2. Dynamic calculation of average neighbors and unique atomic numbers
3. Contrastive loss for semi-supervised learning
4. Gradient checking and comprehensive logging during training
5. Configurable model parameters and training hyperparameters
6. Total runtime measurement
7. Early stopping to prevent overfitting

Author: Utkarsh Sharma
Version: 1.3.0
Date: 07-21-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - e3nn (>=0.4.0)
    - mace (custom package)
    - molli (custom package) 

Usage:
    python train.py

For detailed usage instructions, please refer to the README.md file.

Change Log: 
    - v1.3.0: Implemented early stopping
    - v1.2.0: Added total runtime measurement and made constants global
    - v1.1.0: Added gradient checking and improved logging
    - v1.0.0: Initial implementation of EQUICAT training pipeline

TODO:
    - Add support for model checkpointing
    - Implement learning rate scheduling
"""

import torch
import molli as ml  
import logging
import sys
import numpy as np
import time
from torch.optim import Adam
from e3nn import o3
from mace import data, modules, tools
from mace.tools import to_numpy
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import ConformerDataset, process_data
from contrastive_loss import contrastive_loss
from collections import OrderedDict

# Global constants
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib" 
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output"
NUM_ENSEMBLES = 1
CUTOFF = 5.0
NUM_EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 1e-4

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

def calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset):
    """
    Calculate the average number of neighbors and unique atomic numbers in the dataset.

    Args:
        dataset (ConformerDataset): The dataset to analyze.

    Returns:
        tuple: A tuple containing:
            - float: Average number of neighbors per atom.
            - list: List of unique atomic numbers in the dataset.
    """
    total_neighbors = 0
    total_atoms = 0
    unique_atomic_numbers = OrderedDict()
    
    for batch_conformers, _, _, _ in process_data(dataset, batch_size=BATCH_SIZE):
        for conformer in batch_conformers:
            total_neighbors += conformer.edge_index.shape[1]
            total_atoms += conformer.positions.shape[0]
            for atomic_number in conformer.atomic_numbers.tolist():
                unique_atomic_numbers[atomic_number] = None  # Using OrderedDict to maintain order
    
    avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
    return avg_neighbors, list(unique_atomic_numbers.keys())

def train_equicat(model_config, z_table, conformer_ensemble, cutoff):
    """
    Train the EQUICAT model using contrastive loss with early stopping.

    Args:
        model_config (dict): Configuration for the EQUICAT model.
        z_table (AtomicNumberTable): Table of atomic numbers.
        conformer_ensemble (ConformerLibrary): Library of conformers for training.
        cutoff (float): Cutoff distance for atomic interactions.

    Returns:
        EQUICATPlusNonLinearReadout: The trained EQUICAT model.
    """
    # Initialize model
    model = EQUICATPlusNonLinearReadout(model_config, z_table)
    print(model)
    logging.info("Model initialized")

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

        for batch_conformers, _, _, ensemble_id in process_data(dataset, batch_size=BATCH_SIZE):
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

            embeddings = torch.stack(embeddings)
            batch_ensemble_ids = torch.full((len(batch_conformers),), ensemble_id, device=embeddings.device)

            loss = contrastive_loss(embeddings, batch_ensemble_ids)

            logging.info(f"Embeddings shape: {embeddings.shape}, Loss: {loss.item()}")

            if loss.item() == 0:
                logging.warning("Loss is zero! Check embeddings and loss computation.")

            loss.backward()

            # Gradient check
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm == 0:
                        logging.warning(f"Gradient for {name} is zero!")
                    elif torch.isnan(param.grad).any():
                        logging.warning(f"Gradient for {name} contains NaN values!")

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Add this logging statement
            logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_count}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        # Add this logging statement
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

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
    start_time = time.time()

    # Set up logging
    setup_logging(f"{OUTPUT_PATH}/training.log")
    logging.info("Starting EQUICAT training")

    # Load your conformer ensemble
    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
    logging.info(f"Loaded conformer ensemble from {CONFORMER_LIBRARY_PATH}")

    # Create a dataset to get unique atomic numbers and calculate avg_num_neighbors
    temp_dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES)
    
    # Calculate average number of neighbors and get unique atomic numbers
    avg_num_neighbors, unique_atomic_numbers = calculate_avg_num_neighbors_and_unique_atomic_numbers(temp_dataset)
    np.save(f'{OUTPUT_PATH}/unique_atomic_numbers.npy', np.array(unique_atomic_numbers))
    logging.info("Initializing model configuration...")
    logging.info(f"Unique atomic numbers in dataset: {unique_atomic_numbers}")
    logging.info(f"Average number of neighbors across dataset: {avg_num_neighbors}")
    logging.info(f"Initialized dataset with {NUM_ENSEMBLES} conformer ensembles")
    logging.info("Model initialization complete. Starting training...")

    # Create z_table with unique atomic numbers
    z_table = tools.AtomicNumberTable(unique_atomic_numbers)

    # Model configuration
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

    # Train the model
    trained_model = train_equicat(model_config, z_table, conformer_ensemble, CUTOFF)

    # Save the trained model
    torch.save(trained_model.state_dict(), f"{OUTPUT_PATH}/trained_equicat_model.pt")
    logging.info("Model training completed and saved.")

    # Calculate and log total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(f"Total runtime: {total_runtime:.2f} seconds")
