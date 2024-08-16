"""
EQUICAT Model Training Script

This script implements the training pipeline for the EQUICAT model, a neural network 
designed for molecular conformer analysis. It includes data loading, model initialization, 
training loop, logging functionality, early stopping, robust error handling, and features
model checkpointing for easy resumption of training.

Key features:
1. GPU acceleration with CUDA support
2. Optional Conformer Padding: Handles variable-sized conformer ensembles when enabled
3. Customizable logging setup with detailed gradient and loss tracking
4. Dynamic calculation of average neighbors and unique atomic numbers
5. Integrated contrastive loss calculation for semi-supervised learning
6. Gradient clipping and comprehensive logging during training
7. Configurable model parameters and training hyperparameters
8. Total runtime measurement
9. Early stopping to prevent overfitting
10. Robust error handling and NaN detection
11. GPU memory tracking
12. Model checkpointing for training resumption
13. Normalized embedding comparison in loss calculations
14. Graceful handling of single-conformer batches

New Features:
- Graceful Single-Conformer Handling: The training process now gracefully handles batches
  with single conformers by skipping the backward pass and keeping the positive loss at zero.
  This ensures that the model continues to learn from multi-conformer batches and negative
  examples without introducing errors or halting training due to single-conformer cases.

Author: Utkarsh Sharma
Version: 2.5.0
Date: 08-15-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - e3nn (>=0.4.0)
    - mace (custom package)
    - molli (custom package) 
    - matplotlib (>=3.3.0)
    - sklearn (>=0.24.0)

Usage:
    python train.py [--pad_batches] [--embedding_type {mean_pooling,deep_sets,self_attention,improved_deep_sets,improved_self_attention,all}] [--resume_from_checkpoint CHECKPOINT_PATH]

For detailed usage instructions, please refer to the README.md file.

Change Log: 
    - v2.5.0: Added graceful handling of single-conformer batches
    - v2.4.0: Added advanced learning rate scheduling options
    - v2.3.0: Added model checkpointing and integrated contrastive loss calculation
    - v2.2.0: Added optional conformer padding via command-line argument
    - v2.1.0: Added support for training with padded conformer batches
    - v2.0.0: Added GPU support with CUDA
    - v1.4.0: Added robust error handling, gradient clipping, and NaN detection
    - v1.3.0: Implemented early stopping
    - v1.2.0: Added total runtime measurement and made constants global
    - v1.1.0: Added gradient checking and improved logging
    - v1.0.0: Initial implementation of EQUICAT training pipeline

TODO:
    - Add support for distributed training
    - Implement weighted loss calculation to account for conformer duplications
"""

import torch
import torch.nn.functional as F
import molli as ml  
import logging
import sys
import numpy as np
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
from e3nn import o3
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
from mace import data, modules, tools
from mace.tools import to_numpy
from collections import OrderedDict
from sklearn.decomposition import PCA
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import ConformerDataset, process_data
from conformer_ensemble_embedding_combiner import process_conformer_ensemble
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(0)

# Global constants
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib" # Path to the conformer library file containing molecular structures.
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output" # Directory where training outputs, such as logs, models, and visualizations, will be saved.
# CONFORMER_LIBRARY_PATH = "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib"
# OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/output"
NUM_ENSEMBLES = 2  # Number of ensemble models to be trained and averaged.
CUTOFF = 5.0  # Cutoff distance for interaction calculations or neighborhood definition.
NUM_EPOCHS = 2  # Total number of training epochs.
BATCH_SIZE = 6  # Number of samples per batch during training.
LEARNING_RATE = 1e-3  # Initial learning rate for the optimizer.
EARLY_STOPPING_PATIENCE = 10  # Number of epochs with no improvement after which training stops early.
EARLY_STOPPING_DELTA = 1e-4  # Minimum change in monitored metric to qualify as an improvement.
GRADIENT_CLIP_VALUE = 1.0  # Maximum allowed value for gradients to prevent exploding gradients.
EPSILON = 1e-8  # Small constant to avoid division by zero in numerical computations.
VISUALIZATION_INTERVAL = 1  # Number of epochs between visualizations or logging of results.
CHECKPOINT_INTERVAL = 5  # Number of epochs between saving model checkpoints.
SCHEDULER_STEP_SIZE = 10  # Number of epochs after which the learning rate is reduced in StepLR.
SCHEDULER_GAMMA = 0.1  # Multiplicative factor by which the learning rate is reduced in schedulers.
SCHEDULER_PATIENCE = 5  # Number of epochs with no improvement before reducing the learning rate in ReduceLROnPlateau.
SCHEDULER_T_MAX = NUM_EPOCHS  # Maximum number of epochs before restarting the learning rate schedule in CosineAnnealingLR.
SCHEDULER_PCT_START = 0.3  # Percentage of the total training duration during which the learning rate increases in OneCycleLR.

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
    
    for batch_conformers, _, _, _, _ in process_data(dataset, batch_size=BATCH_SIZE, device=device):
        for conformer in batch_conformers:
            conformer = conformer.to(device)
            total_neighbors += conformer.edge_index.shape[1]
            total_atoms += conformer.positions.shape[0]
            for atomic_number in conformer.atomic_numbers.cpu().tolist():
                unique_atomic_numbers[atomic_number] = None  # Using OrderedDict to maintain order
    
    avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
    return avg_neighbors, list(unique_atomic_numbers.keys())

def plot_embeddings_pca(embeddings, epoch, OUTPUT_PATH):
    """
    Perform PCA on the embeddings and create a visualization.

    Args:
        embeddings (dict): Dictionary containing embeddings for each ensemble and method.
        epoch (int): Current training epoch.
        OUTPUT_PATH (str): Directory to save the plot.

    Returns:
        None
    """
    # Combine all embeddings
    all_embeddings = []
    ensemble_labels = []
    for ensemble_id, methods in embeddings.items():
        for method, (scalar, vector) in methods.items():
            combined = torch.cat([scalar.view(-1), vector.view(-1)])
            all_embeddings.append(combined.detach().cpu().numpy())
            ensemble_labels.append(ensemble_id)

    all_embeddings = np.array(all_embeddings)

    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    for ensemble_id in set(ensemble_labels):
        mask = np.array(ensemble_labels) == ensemble_id
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Ensemble {ensemble_id}')

    plt.title(f'PCA of Embeddings - Epoch {epoch}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    
    # Save the plot
    plt.savefig(f'{OUTPUT_PATH}/pca_visualizations/embeddings_pca_epoch_{epoch}.png')
    plt.close()

def get_scheduler(scheduler_type, optimizer, num_epochs, steps_per_epoch):
    """
    Get the specified learning rate scheduler.

    Args:
        scheduler_type (str): Type of scheduler to use.
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        num_epochs (int): Total number of epochs for training.
        steps_per_epoch (int): Number of steps (batches) per epoch.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
    """
    if scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_GAMMA, patience=SCHEDULER_PATIENCE, verbose=True)
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)
    elif scheduler_type == 'onecycle':
        return OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=num_epochs, steps_per_epoch=steps_per_epoch, pct_start=SCHEDULER_PCT_START)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def train_equicat(model_config, z_table, conformer_ensemble, cutoff, device, pad_batches, embedding_type, scheduler_type, resume_from=None):
    """
    Train the EQUICAT model using contrastive loss with early stopping, checkpointing, and learning rate scheduling.

    Args:
        model_config (dict): Configuration for the EQUICAT model.
        z_table (AtomicNumberTable): Table of atomic numbers.
        conformer_ensemble (ConformerLibrary): Library of conformers for training.
        cutoff (float): Cutoff distance for atomic interactions.
        device (torch.device): Device to run the training on.
        pad_batches (bool): Whether to pad batches to handle variable-sized conformer ensembles.
        embedding_type (str): Type of embedding to use for loss computation.
        scheduler_type (str): Type of learning rate scheduler to use.
        resume_from (str, optional): Path to checkpoint file to resume training from.

    Returns:
        EQUICATPlusNonLinearReadout: The trained EQUICAT model.
    """
    # Initialize model and move it to the specified device
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
    print(model)
    logging.info(f"Model initialized and moved to {device}")

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    logging.info(f"Optimizer initialized with learning rate: {LEARNING_RATE}")

    # Create dataset and calculate steps per epoch
    dataset = ConformerDataset(conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES)
    steps_per_epoch = len(dataset) // BATCH_SIZE + (1 if len(dataset) % BATCH_SIZE != 0 else 0)
    
    # Initialize learning rate scheduler
    scheduler = get_scheduler(scheduler_type, optimizer, NUM_EPOCHS, steps_per_epoch)
    logging.info(f"Learning rate scheduler initialized: {scheduler.__class__.__name__}")

    # Resume training from checkpoint if specified
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info(f"Resuming training from epoch {start_epoch}")

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    ensemble_embeddings = {}

    for epoch in range(start_epoch, NUM_EPOCHS):
        total_loss = 0.0
        batch_count = 0

        with detect_anomaly():
            for batch_conformers, _, _, ensemble_id, num_added in process_data(dataset, batch_size=BATCH_SIZE, device=device, pad_batches=pad_batches):
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
                
                # Log ensemble ID and embedding shape
                logging.info(f"Ensemble ID: {ensemble_id}, Embedding shape: {embeddings.shape}")
                
                processed_embeddings = process_conformer_ensemble(embeddings)
                
                print(f"\nBatch-level detailed embeddings for Ensemble {ensemble_id}:")
                print_detailed_embeddings(processed_embeddings, level="Batch")

                # Accumulate embeddings for ensemble-level averaging
                if ensemble_id not in ensemble_embeddings:
                    ensemble_embeddings[ensemble_id] = {method: [] for method in processed_embeddings.keys()}
                for method, (scalar, vector) in processed_embeddings.items():
                    ensemble_embeddings[ensemble_id][method].append((scalar, vector))

                # Compute positive loss
                positive_loss = compute_positive_loss(processed_embeddings, embedding_type, normalize=False)

                # Handle padded batches
                if num_added > 0:
                    original_batch_size = len(batch_conformers) - num_added
                    positive_loss = positive_loss[:original_batch_size]

                # Perform backward pass only if we have a valid positive loss
                if torch.is_tensor(positive_loss) and positive_loss.nelement() > 0 and positive_loss.requires_grad:
                    loss = positive_loss.mean()

                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.error(f"NaN or Inf loss detected: {loss.item()}")
                        continue

                    loss.backward()

                    clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if grad_norm == 0:
                                logging.warning(f"Gradient for {name} is zero!")
                            elif torch.isnan(param.grad).any():
                                logging.warning(f"Gradient for {name} contains NaN values!")
                            logging.info(f"Gradient norm for {name}: {grad_norm:.6f}")

                    optimizer.step()
                    
                    if scheduler_type != 'plateau':
                        scheduler.step()
                    
                    total_loss += loss.item()
                    batch_count += 1

                    logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_count}], Positive Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    logging.info(f"Skipping backward pass for batch with single conformer (Ensemble {ensemble_id})")

                if device.type == 'cuda':
                    logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
                    logging.info(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB")

        # Compute negative loss and total loss
        if len(ensemble_embeddings) > 1:
            avg_embeddings = compute_average_ensemble_embeddings(ensemble_embeddings)
            # plot_embeddings_pca(avg_embeddings, epoch+1, OUTPUT_PATH)

            print("\nEnsemble Average Embeddings:")
            for ensemble_id, methods in avg_embeddings.items():
                print(f"Ensemble {ensemble_id}:")
                for method, (scalar, vector) in methods.items():
                    print(f"  {method}:")
                    print(f"    Scalar shape: {scalar.shape}")
                    print(f"    Vector shape: {vector.shape}")
                    print(f"    Scalar: {scalar.squeeze().tolist()}")
                    print(f"    Vector: {vector.squeeze().tolist()}")

            negative_loss = compute_negative_loss(avg_embeddings, embedding_type, normalize=False)
            
            logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Negative Loss: {negative_loss.item():.6f}")

            total_loss = (total_loss / batch_count if batch_count > 0 else 0) + negative_loss
        else:
            logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Not enough ensembles for negative loss")
            total_loss = total_loss / batch_count if batch_count > 0 else 0

        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Total Loss: {total_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Update plateau scheduler if used
        if scheduler_type == 'plateau':
            scheduler.step(total_loss)

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f'{OUTPUT_PATH}/checkpoints/checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

        # Early stopping check
        if total_loss < best_loss - EARLY_STOPPING_DELTA:
            best_loss = total_loss
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

def print_detailed_embeddings(embeddings, level="Batch"):
    """
    Print detailed information about the embeddings.

    Args:
        embeddings (dict): A dictionary containing embeddings for different methods.
        level (str): The level of embeddings being printed (e.g., "Batch" or "Ensemble").

    Returns:
        None
    """
    print(f"\n{level} Detailed Embeddings:")
    for method, (scalar, vector) in embeddings.items():
        print(f"  {method}:")
        print(f"    Scalar embeddings shape: {scalar.shape}")
        print(f"    Vector embeddings shape: {vector.shape}")
        
        print(f"    Scalar embeddings (all conformers, all features):")
        for i in range(scalar.shape[0]):
            print(f"      Conformer {i}:")
            print(scalar[i])
        
        print(f"    Vector embeddings (all conformers, all features):")
        for i in range(vector.shape[0]):
            print(f"      Conformer {i}:")
            print(vector[i])
        
        print("-" * 65)  # Separator between methods

def compute_positive_loss(embeddings, embedding_type, normalize=False):
    """
    Compute the positive loss for the contrastive learning.

    Args:
        embeddings (dict): A dictionary containing embeddings for different methods.
        embedding_type (str): The type of embedding to use for loss computation.
        normalize (bool): Whether to apply L2 normalization to the embeddings.

    Returns:
        torch.Tensor: The computed positive loss.
    """
    if embedding_type == 'all':
        loss = 0
        for method, (scalar, vector) in embeddings.items():
            combined = torch.cat([scalar, vector.view(vector.shape[0], -1)], dim=1)
            if normalize:
                combined = F.normalize(combined, p=2, dim=-1)
            if combined.shape[0] == 1:  # Single conformer case
                loss += torch.tensor(0.0, device=combined.device, requires_grad=True)  # No positive loss for single conformer
            else:
                pairwise_distances = torch.cdist(combined, combined)
                print(f"Positive pairwise distances for {method}: \n{pairwise_distances}")
                print(f"Positive combined shape for {method}: {combined.shape}")
                print(f"Positive combined stats for {method}: mean={combined.mean():.4f}, std={combined.std():.4f}")
                loss += pairwise_distances.mean(dim=1)
        return loss / len(embeddings) if len(embeddings) > 0 else torch.tensor(0.0, device=combined.device, requires_grad=True)
    else:
        scalar, vector = embeddings[embedding_type]
        combined = torch.cat([scalar, vector.view(vector.shape[0], -1)], dim=1)
        if normalize:
            combined = F.normalize(combined, p=2, dim=-1)
        if combined.shape[0] == 1:  # Single conformer case
            return torch.tensor(0.0, device=combined.device, requires_grad=True)
        pairwise_distances = torch.cdist(combined, combined)
        print(f"Positive pairwise distances for {embedding_type}: \n{pairwise_distances}")
        print(f"Positive combined shape for {embedding_type}: {combined.shape}")
        print(f"Positive combined stats for {embedding_type}: mean={combined.mean():.4f}, std={combined.std():.4f}")
        return pairwise_distances.mean(dim=1)

def compute_average_ensemble_embeddings(ensemble_embeddings):
    """
    Compute average embeddings for each ensemble.

    Args:
        ensemble_embeddings (dict): A dictionary containing embeddings for each ensemble.

    Returns:
        dict: A dictionary containing average embeddings for each ensemble.
    """
    avg_embeddings = {}
    for ensemble_id, ensemble_methods in ensemble_embeddings.items():
        avg_embeddings[ensemble_id] = {}
        for method, batches in ensemble_methods.items():
            all_scalars = torch.cat([scalar for scalar, _ in batches], dim=0)
            all_vectors = torch.cat([vector for _, vector in batches], dim=0)
            avg_scalar = all_scalars.mean(dim=0, keepdim=True)
            avg_vector = all_vectors.mean(dim=0, keepdim=True)
            avg_embeddings[ensemble_id][method] = (avg_scalar, avg_vector)
    return avg_embeddings

def compute_negative_loss(avg_embeddings, embedding_type, normalize=False):
    """
    Compute the negative loss for the contrastive learning.

    Args:
        avg_embeddings (dict): A dictionary containing average embeddings for each ensemble.
        embedding_type (str): The type of embedding to use for loss computation.
        normalize (bool): Whether to apply L2 normalization to the embeddings.

    Returns:
        torch.Tensor: The computed negative loss.
    """
    if embedding_type == 'all':
        loss = 0
        for method in avg_embeddings[list(avg_embeddings.keys())[0]].keys():
            combined = torch.stack([
                torch.cat([
                    embeddings[method][0].view(-1),
                    embeddings[method][1].view(-1)
                ])
                for embeddings in avg_embeddings.values()
            ])
            if normalize:
                combined = F.normalize(combined, p=2, dim=-1)
            pairwise_distances = torch.cdist(combined, combined)
            print(f"Pairwise distances for {method}: \n{pairwise_distances}")
            print(f"Combined shape for {method}: {combined.shape}")
            print(f"Combined stats for {method}: mean={combined.mean():.4f}, std={combined.std():.4f}")
            loss += torch.relu(1.0 - pairwise_distances).sum()
        return loss / len(avg_embeddings[list(avg_embeddings.keys())[0]])
    else:
        combined = torch.stack([
            torch.cat([
                embeddings[embedding_type][0].view(-1),
                embeddings[embedding_type][1].view(-1)
            ])
            for embeddings in avg_embeddings.values()
        ])
        if normalize:
            combined = F.normalize(combined, p=2, dim=-1)
        pairwise_distances = torch.cdist(combined, combined)
        print(f"Pairwise distances for {embedding_type}: \n{pairwise_distances}")
        print(f"Combined shape for {embedding_type}: {combined.shape}")
        print(f"Combined stats for {embedding_type}: mean={combined.mean():.4f}, std={combined.std():.4f}")
        return torch.relu(1.0 - pairwise_distances).sum()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EQUICAT Model Training")
    parser.add_argument('--pad_batches', action='store_true', help='Enable batch padding')
    parser.add_argument('--embedding_type', type=str, default='improved_self_attention',
                        choices=['mean_pooling', 'deep_sets', 'self_attention', 'improved_deep_sets', 'improved_self_attention', 'all'],
                        help='Type of embedding to use for loss computation')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['plateau', 'step', 'cosine', 'onecycle'],
                        help='Type of learning rate scheduler to use')
    args = parser.parse_args()

    start_time = time.time()

    setup_logging(f"{OUTPUT_PATH}/training.log")
    logging.info("Starting EQUICAT training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Padding batches: {args.pad_batches}")
    logging.info(f"Embedding type: {args.embedding_type}")
    logging.info(f"Learning rate scheduler: {args.scheduler}")

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
        "max_ell": 2,
        "num_interactions": 2,
        "num_elements": len(z_table),
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "atomic_energies": to_numpy(torch.zeros(len(z_table), dtype=torch.float64)),
        "correlation": 3,
        "gate": torch.nn.functional.silu,
        "avg_num_neighbors": avg_num_neighbors,
    }
    logging.info(f"Model configuration: {model_config}")

    trained_model = train_equicat(model_config, z_table, conformer_ensemble, CUTOFF, device, args.pad_batches, args.embedding_type, args.scheduler, args.resume_from_checkpoint)

    torch.save(trained_model.state_dict(), f"{OUTPUT_PATH}/trained_equicat_model.pt")
    logging.info("Model training completed and saved.")

    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(f"Total runtime: {total_runtime:.2f} seconds")