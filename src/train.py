"""
EQUICAT Model Training Script

This script implements the training pipeline for the EQUICAT model, a neural network 
designed for molecular conformer analysis. It includes data loading, model initialization, 
training loop, and logging functionality.

Key features:
1. Customizable logging setup
2. Dynamic calculation of average neighbors and unique atomic numbers
3. Contrastive loss for semi-supervised learning
4. Gradient checking and comprehensive logging during training
5. Configurable model parameters and training hyperparameters

Author: Utkarsh Sharma
Version: 1.1.0
Date: 07-17-2024 (MM-DD-YYYY)
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
    - v1.1.0: Added gradient checking and improved logging
    - v1.0.0: Initial implementation of EQUICAT training pipeline

TODO:
    - Implement early stopping
    - Add support for model checkpointing
    - Implement learning rate scheduling
"""

import torch
import molli as ml  
import logging
import sys
import numpy as np
from torch.optim import Adam
from e3nn import o3
from mace import data, modules, tools
from mace.tools import to_numpy
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import ConformerDataset, process_data
from contrastive_loss import contrastive_loss
from collections import OrderedDict

# Define NUM_ENSEMBLES here
NUM_ENSEMBLES = 1
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib" 
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output"

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
    
    for batch_conformers, _, _, _ in process_data(dataset, batch_size=16):
        for conformer in batch_conformers:
            total_neighbors += conformer.edge_index.shape[1]
            total_atoms += conformer.positions.shape[0]
            for atomic_number in conformer.atomic_numbers.tolist():
                unique_atomic_numbers[atomic_number] = None  # Using OrderedDict to maintain order
    
    avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
    return avg_neighbors, list(unique_atomic_numbers.keys())

def train_equicat(model_config, z_table, conformer_ensemble, cutoff, num_epochs=5, batch_size=16, learning_rate=1e-3):
    """
    Train the EQUICAT model using contrastive loss.

    Args:
        model_config (dict): Configuration for the EQUICAT model.
        z_table (AtomicNumberTable): Table of atomic numbers.
        conformer_ensemble (ConformerLibrary): Library of conformers for training.
        cutoff (float): Cutoff distance for atomic interactions.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        EQUICATPlusNonLinearReadout: The trained EQUICAT model.
    """
    # Initialize model
    model = EQUICATPlusNonLinearReadout(model_config, z_table)
    print(model)
    logging.info("Model initialized")

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    logging.info(f"Optimizer initialized with learning rate: {learning_rate}")

    # Initialize dataset
    dataset = ConformerDataset(conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES)
    logging.info(f"Dataset initialized with {NUM_ENSEMBLES} ensembles")

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0

        for batch_conformers, _, _, ensemble_id in process_data(dataset, batch_size=batch_size):
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

            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    logging.info("Training completed")
    return model

if __name__ == "__main__":
    # Set up logging
    setup_logging(f"{OUTPUT_PATH}/training.log")
    logging.info("Starting EQUICAT training")

    # Load your conformer ensemble
    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
    logging.info(f"Loaded conformer ensemble from {CONFORMER_LIBRARY_PATH}")

    # Create a dataset to get unique atomic numbers and calculate avg_num_neighbors
    cutoff = 5.0  # Your cutoff value
    temp_dataset = ConformerDataset(conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES)
    
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
        "r_max": 5.0,  # Your cutoff radius
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 2,
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
    trained_model = train_equicat(model_config, z_table, conformer_ensemble, cutoff)

    # Save the trained model
    torch.save(trained_model.state_dict(), f"{OUTPUT_PATH}/trained_equicat_model.pt")
    logging.info("Model training completed and saved.")














# """
# EQUICAT Model Training Script with Single and Ensemble Learning Options

# This script implements the training pipeline for the EQUICAT model, offering two approaches:
# 1. Training a single model on the entire dataset
# 2. Training separate models for each molecular ensemble (ensemble learning)

# Key features:
# 1. Customizable logging setup
# 2. Dynamic calculation of average neighbors and unique atomic numbers
# 3. Contrastive loss for semi-supervised learning
# 4. Gradient checking and comprehensive logging during training
# 5. Option to choose between single model and ensemble learning

# Author: Utkarsh Sharma
# Version: 1.3.0
# Date: 07-19-2024 (MM-DD-YYYY)
# License: MIT

# Dependencies:
#     - torch (>=1.9.0)
#     - e3nn (>=0.4.0)
#     - mace (custom package)
#     - molli (custom package)

# Usage:
#     python train.py --mode [single|ensemble]

# For detailed usage instructions, please refer to the README.md file.

# Change Log:
#     - v1.3.0: Added option to choose between single model and ensemble learning
#     - v1.2.0: Implemented ensemble learning approach
#     - v1.1.0: Added gradient checking and improved logging
#     - v1.0.0: Initial implementation of EQUICAT training pipeline

# TODO:
#     - Implement early stopping
#     - Add support for model checkpointing
#     - Implement learning rate scheduling
# """

# import torch
# import molli as ml
# import logging
# import sys
# import numpy as np
# import json
# import argparse
# from torch.optim import Adam
# from e3nn import o3
# from mace import data, modules, tools
# from mace.tools import to_numpy
# from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
# from data_loader import ConformerDataset, process_data
# from contrastive_loss import contrastive_loss
# from collections import OrderedDict

# # Constants
# NUM_ENSEMBLES = 1
# CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib" 
# OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output"
# CUTOFF = 5.0
# NUM_EPOCHS = 5
# BATCH_SIZE = 16
# LEARNING_RATE = 1e-3

# def setup_logging(log_file):
#     """
#     Set up logging configuration for the training process.

#     Args:
#         log_file (str): Path to the log file.

#     Returns:
#         None
#     """
#     # Clear existing log file
#     with open(log_file, 'w') as f:
#         f.write("")  # This will clear the file
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file, mode='w'),  # 'w' mode overwrites the file
#             logging.StreamHandler(sys.stdout)
#         ]
#     )

# def calculate_stats_for_dataset(dataset):
#     total_neighbors = 0
#     total_atoms = 0
#     unique_atomic_numbers = OrderedDict()
    
#     for batch_conformers, _, _, _ in process_data(dataset, batch_size=BATCH_SIZE):
#         for conformer in batch_conformers:
#             total_neighbors += conformer.edge_index.shape[1]
#             total_atoms += conformer.positions.shape[0]
#             for atomic_number in conformer.atomic_numbers.tolist():
#                 unique_atomic_numbers[atomic_number] = None

#     avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
#     return avg_neighbors, list(unique_atomic_numbers.keys())

# def calculate_stats_for_ensemble(dataset, ensemble_id):
#     total_neighbors = 0
#     total_atoms = 0
#     unique_atomic_numbers = OrderedDict()
    
#     for batch_conformers, _, _, batch_ensemble_id in process_data(dataset, batch_size=BATCH_SIZE):
#         if batch_ensemble_id != ensemble_id:
#             continue
#         for conformer in batch_conformers:
#             total_neighbors += conformer.edge_index.shape[1]
#             total_atoms += conformer.positions.shape[0]
#             for atomic_number in conformer.atomic_numbers.tolist():
#                 unique_atomic_numbers[atomic_number] = None

#     avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
#     return avg_neighbors, list(unique_atomic_numbers.keys())

# def train_equicat(model, dataset, optimizer, ensemble_id=None):
#     for epoch in range(NUM_EPOCHS):
#         total_loss = 0.0
#         batch_count = 0

#         for batch_conformers, _, _, batch_ensemble_id in process_data(dataset, batch_size=BATCH_SIZE):
#             if ensemble_id is not None and batch_ensemble_id != ensemble_id:
#                 continue

#             optimizer.zero_grad()

#             embeddings = []
#             for conformer in batch_conformers:
#                 input_dict = {
#                     'positions': conformer.positions,
#                     'atomic_numbers': conformer.atomic_numbers,
#                     'edge_index': conformer.edge_index
#                 }
#                 output = model(input_dict)
#                 embeddings.append(output)

#             embeddings = torch.stack(embeddings)
#             batch_labels = torch.arange(len(batch_conformers), device=embeddings.device)

#             loss = contrastive_loss(embeddings, batch_labels)

#             if loss.item() == 0:
#                 logging.warning(f"{'Ensemble ' + str(ensemble_id) if ensemble_id is not None else 'Global'}: Loss is zero! Check embeddings and loss computation.")

#             loss.backward()

#             for name, param in model.named_parameters():
#                 if param.grad is not None:
#                     grad_norm = param.grad.norm().item()
#                     if grad_norm == 0:
#                         logging.warning(f"{'Ensemble ' + str(ensemble_id) if ensemble_id is not None else 'Global'}: Gradient for {name} is zero!")
#                     elif torch.isnan(param.grad).any():
#                         logging.warning(f"{'Ensemble ' + str(ensemble_id) if ensemble_id is not None else 'Global'}: Gradient for {name} contains NaN values!")

#             optimizer.step()

#             total_loss += loss.item()
#             batch_count += 1

#         avg_loss = total_loss / batch_count if batch_count > 0 else 0
#         logging.info(f"{'Ensemble ' + str(ensemble_id) if ensemble_id is not None else 'Global'}, Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

#     return model

# def get_object_from_string(object_string):
#     if object_string == "RealAgnosticResidualInteractionBlock":
#         return modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
#     elif object_string == "torch.nn.functional.silu":
#         return torch.nn.functional.silu
#     else:
#         raise ValueError(f"Unknown object string: {object_string}")

# def create_model_config(avg_num_neighbors, z_table):
#     return {
#         "r_max": CUTOFF,
#         "num_bessel": 8,
#         "num_polynomial_cutoff": 6,
#         "max_ell": 2,
#         "num_interactions": 2,
#         "num_elements": len(z_table),
#         "interaction_cls": "RealAgnosticResidualInteractionBlock",
#         "interaction_cls_first": "RealAgnosticResidualInteractionBlock",
#         "hidden_irreps": str(o3.Irreps("32x0e + 32x1o")),
#         "MLP_irreps": str(o3.Irreps("16x0e")),
#         "atomic_energies": to_numpy(torch.zeros(len(z_table))).tolist(),
#         "correlation": 3,
#         "gate": "torch.nn.functional.silu",
#         "avg_num_neighbors": avg_num_neighbors,
#     }

# def prepare_model_config(config):
#     model_config = config.copy()
#     model_config["interaction_cls"] = get_object_from_string(config["interaction_cls"])
#     model_config["interaction_cls_first"] = get_object_from_string(config["interaction_cls_first"])
#     model_config["hidden_irreps"] = o3.Irreps(config["hidden_irreps"])
#     model_config["MLP_irreps"] = o3.Irreps(config["MLP_irreps"])
#     model_config["atomic_energies"] = torch.tensor(config["atomic_energies"])
#     model_config["gate"] = get_object_from_string(config["gate"])
#     return model_config

# def main(mode):
#     setup_logging(f"{OUTPUT_PATH}/training_{mode}.log")
#     logging.info(f"Starting EQUICAT training in {mode} mode")

#     conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
#     logging.info(f"Loaded conformer ensemble from {CONFORMER_LIBRARY_PATH}")

#     dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES)

#     if mode == 'single':
#         avg_num_neighbors, unique_atomic_numbers = calculate_stats_for_dataset(dataset)
#         z_table = tools.AtomicNumberTable(unique_atomic_numbers)

#         model_config = create_model_config(avg_num_neighbors, z_table)
#         logging.info(f"Global model configuration: {model_config}")

#         model_config_for_model = prepare_model_config(model_config)
#         model = EQUICATPlusNonLinearReadout(model_config_for_model, z_table)
#         optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

#         model = train_equicat(model, dataset, optimizer)
#         torch.save(model.state_dict(), f"{OUTPUT_PATH}/trained_equicat_model_single.pt")

#         np.save(f'{OUTPUT_PATH}/unique_atomic_numbers.npy', np.array(unique_atomic_numbers))
#         with open(f"{OUTPUT_PATH}/model_config_single.json", "w") as f:
#             json.dump(model_config, f, indent=2)

#     elif mode == 'ensemble':
#         ensemble_models = []
#         ensemble_stats = {}

#         for ensemble_id in range(NUM_ENSEMBLES):
#             logging.info(f"Processing ensemble {ensemble_id}")
            
#             avg_num_neighbors, unique_atomic_numbers = calculate_stats_for_ensemble(dataset, ensemble_id)
#             ensemble_stats[ensemble_id] = {
#                 "avg_num_neighbors": avg_num_neighbors,
#                 "unique_atomic_numbers": unique_atomic_numbers
#             }
            
#             z_table = tools.AtomicNumberTable(unique_atomic_numbers)
#             model_config = create_model_config(avg_num_neighbors, z_table)
#             logging.info(f"Ensemble {ensemble_id} configuration: {model_config}")

#             model_config_for_model = prepare_model_config(model_config)
#             model = EQUICATPlusNonLinearReadout(model_config_for_model, z_table)
#             optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

#             model = train_equicat(model, dataset, optimizer, ensemble_id)
#             ensemble_models.append(model)

#             torch.save(model.state_dict(), f"{OUTPUT_PATH}/trained_equicat_model_ensemble_{ensemble_id}.pt")
            
#             # Save config for each ensemble
#             with open(f"{OUTPUT_PATH}/model_config_ensemble_{ensemble_id}.json", "w") as f:
#                 json.dump(model_config, f, indent=2)

#         # Save ensemble statistics
#         with open(f"{OUTPUT_PATH}/ensemble_stats.json", "w") as f:
#             json.dump(ensemble_stats, f, indent=2)

#     logging.info(f"Training completed in {mode} mode and model(s) saved.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="EQUICAT Model Training")
#     parser.add_argument('--mode', type=str, choices=['single', 'ensemble'], required=True,
#                         help='Training mode: single model or ensemble learning')
#     args = parser.parse_args()

#     main(args.mode)