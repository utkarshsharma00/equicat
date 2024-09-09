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
import random
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
from conformer_ensemble_embedding_combiner import process_conformer_ensemble, process_ensemble_batches
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast

torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=15)
np.random.seed(0)

# Global constants
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib" # Path to the conformer library file containing molecular structures.
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output" # Directory where training outputs, such as logs, models, and visualizations, will be saved.
# CONFORMER_LIBRARY_PATH = "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib"
# OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/output"
NUM_ENSEMBLES = 20  # Number of ensemble models to be trained and averaged.
CUTOFF = 6.0  # Cutoff distance for interaction calculations or neighborhood definition.
NUM_EPOCHS = 25  # Total number of training epochs.
BATCH_SIZE = 6  # Number of samples per batch during training.
SAMPLE_SIZE = 4 
MAX_PREVIOUS_SAMPLES = 10  # Maximum number of previous samples to consider for cross-sample loss.
LEARNING_RATE = 1e-3  # Initial learning rate for the optimizer.
EARLY_STOPPING_PATIENCE = 10  # Number of epochs with no improvement after which training stops early.
EARLY_STOPPING_DELTA = 1e-4  # Minimum change in monitored metric to qualify as an improvement.
GRADIENT_CLIP_VALUE = 1.0  # Maximum allowed value for gradients to prevent exploding gradients.
EPSILON = 1e-8  # Small constant to avoid division by zero in numerical computations.
VISUALIZATION_INTERVAL = 1  # Number of epochs between visualizations or logging of results.
CHECKPOINT_INTERVAL = 5  # Number of epochs between saving model checkpoints.
SCHEDULER_STEP_SIZE = 5  # Number of epochs after which the learning rate is reduced in StepLR.
SCHEDULER_GAMMA = 0.1  # Multiplicative factor by which the learning rate is reduced in schedulers.
SCHEDULER_PATIENCE = 5  # Number of epochs with no improvement before reducing the learning rate in ReduceLROnPlateau.
SCHEDULER_T_MAX = NUM_EPOCHS  # Maximum number of epochs before restarting the learning rate schedule in CosineAnnealingLR.
SCHEDULER_PCT_START = 0.3  # Percentage of the total training duration during which the learning rate increases in OneCycleLR.
ACCUMULATION_STEPS = 4

def print_gpu_memory():
    if torch.cuda.is_available():
        logging.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory()

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
    total_neighbors = 0
    total_atoms = 0
    unique_atomic_numbers = OrderedDict()
    
    for batch_data in process_data(dataset, batch_size=BATCH_SIZE, device=device):
        if batch_data is None:
            continue
        batch_conformers, _, _, _, _, key = batch_data  # Unpack all 6 values
        for conformer in batch_conformers:
            conformer = conformer.to(device)
            total_neighbors += conformer.edge_index.shape[1]
            total_atoms += conformer.positions.shape[0]
            for atomic_number in conformer.atomic_numbers.cpu().tolist():
                unique_atomic_numbers[atomic_number] = None

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

def get_scheduler(scheduler_type, optimizer, epochs_per_sample, steps_per_epoch):
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs_per_sample * steps_per_epoch
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_GAMMA, 
            patience=SCHEDULER_PATIENCE, verbose=True
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
        )
    elif scheduler_type == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, 
            epochs=epochs_per_sample, 
            steps_per_epoch=steps_per_epoch,
            pct_start=SCHEDULER_PCT_START
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def train_equicat(model_config, z_table, conformer_ensemble, cutoff, device, pad_batches, embedding_type, scheduler_type, resume_from=None):
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
    print(model)
    logging.info(f"Model initialized and moved to {device}")

    dataset = ConformerDataset(conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES, sample_size=SAMPLE_SIZE)
    total_samples = len(dataset)
    print("Total samples: ", total_samples)
    epochs_per_sample = 5
    total_epochs = total_samples * epochs_per_sample
    print("Total epochs: ", total_epochs)

    best_loss = float('inf')
    best_model = None
    previous_samples_embeddings = []
    sample_ids = []

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(scheduler_type, optimizer, total_epochs, dataset.sample_size)

    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, total_epochs):
        sample_index = epoch // epochs_per_sample
        epoch_in_sample = epoch % epochs_per_sample

        if epoch_in_sample == 0:
            sample = dataset.get_next_sample()
            if sample is None:
                dataset.reset()
                sample = dataset.get_next_sample()
            
            sample_id = random.randint(1, 1000000)
            sample_ids.append(sample_id)
            logging.info(f"Loaded new sample (ID: {sample_id}) with {len(sample)} ensembles")

        logging.info(f"Starting epoch {epoch+1}/{total_epochs} (Sample {sample_index+1}, Epoch {epoch_in_sample+1}/{epochs_per_sample})")

        optimizer.zero_grad()

        with detect_anomaly():
            all_averaged_embeddings = process_sample(model, sample, device)

            # Compute within-sample loss
            current_loss = compute_contrastive_loss(all_averaged_embeddings, embedding_type, is_cross_sample=False)
            
            total_loss = current_loss

            # Compute cross-sample loss with previous samples
            if previous_samples_embeddings:
                combined_embeddings = {**all_averaged_embeddings}
                for i, prev_emb in enumerate(previous_samples_embeddings):
                    combined_embeddings.update({f"prev_{i}_{k}": v for k, v in prev_emb.items()})
                
                cross_sample_loss = compute_contrastive_loss(combined_embeddings, embedding_type, is_cross_sample=True)
                total_loss += cross_sample_loss
                logging.info(f"Total loss (within-sample + cross-sample): {total_loss.item()}")
            else:
                logging.info(f"Total loss (within-sample only): {total_loss.item()}")

            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)

            # Update model parameters
            optimizer.step()
            scheduler.step()

        # Log gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                logging.info(f"Gradient stats for {name}: mean={param.grad.mean().item():.6f}, std={param.grad.std().item():.6f}, max={param.grad.max().item():.6f}, min={param.grad.min().item():.6f}")
            else:
                logging.info(f"No gradient for parameter: {name}")

        logging.info(f"Epoch {epoch+1}, Total Loss: {total_loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Update best model if needed
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model = model.state_dict()

        # Checkpointing
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f'{OUTPUT_PATH}/checkpoints/checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss.item(),
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

        # Update previous samples embeddings
        if epoch_in_sample == epochs_per_sample - 1:
            previous_samples_embeddings.append({k: {method: (s.detach() if s is not None else None, 
                                                             v.detach() if v is not None else None) 
                                                    for method, (s, v) in emb.items()}
                                                for k, emb in all_averaged_embeddings.items()})
            if len(previous_samples_embeddings) > MAX_PREVIOUS_SAMPLES:
                previous_samples_embeddings.pop(0)

    logging.info("Training completed")
    logging.info(f"Samples processed in order: {sample_ids}")
    
    # Load the best model
    model.load_state_dict(best_model)
    
    return model

def process_sample(model, sample, device):
    all_averaged_embeddings = {}
    for ensemble_id, (atomic_data_list, key) in enumerate(sample):
        batch_embeddings = []
        for i in range(0, len(atomic_data_list), BATCH_SIZE):
            batch_conformers = atomic_data_list[i:i+BATCH_SIZE]
            batch_conformers = [conformer.to(device) for conformer in batch_conformers]

            embeddings = []
            for conformer in batch_conformers:
                input_dict = {
                    'positions': conformer.positions.clone(),
                    'atomic_numbers': conformer.atomic_numbers.clone(),
                    'edge_index': conformer.edge_index.clone()
                }
                output = model(input_dict)
                embeddings.append(output)

            embeddings = torch.stack(embeddings)
            batch_embeddings.append(embeddings)

        ensemble_result = process_ensemble_batches(batch_embeddings, model.non_linear_readout.irreps_out)
        all_averaged_embeddings[key] = ensemble_result

    logging.info(f"Processed sample with {len(all_averaged_embeddings)} ensembles")
    logging.info(f"Averaged embeddings: {all_averaged_embeddings}")

    return all_averaged_embeddings

def compute_contrastive_loss(all_averaged_embeddings, embedding_type, is_cross_sample=False):
    """
    Compute contrastive loss across all averaged embeddings.
    
    Args:
        all_averaged_embeddings (dict): Dictionary of averaged embeddings for each molecule/ensemble.
        embedding_type (str): Type of embedding to use for loss computation.
        is_cross_sample (bool): Whether this is a cross-sample computation.
    
    Returns:
        torch.Tensor: Computed contrastive loss
    """
    embeddings = []
    keys = list(all_averaged_embeddings.keys())

    for key in keys:
        if embedding_type == 'all':
            combined = []
            for method, (scalar, vector) in all_averaged_embeddings[key].items():
                if scalar is not None:
                    combined.append(scalar.view(-1))
                if vector is not None:
                    combined.append(vector.view(-1))
            embeddings.append(torch.cat(combined))
        else:
            scalar, vector = all_averaged_embeddings[key][embedding_type]
            combined = []
            if scalar is not None:
                combined.append(scalar.view(-1))
            if vector is not None:
                combined.append(vector.view(-1))
            embeddings.append(torch.cat(combined))

    embeddings = torch.stack(embeddings)
    
    # Check for NaN values
    if torch.isnan(embeddings).any():
        logging.warning("NaN values detected in embeddings. Returning zero loss.")
        return torch.tensor(0.0, device=embeddings.device)

    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings)
    
    # Create a mask to exclude self-comparisons
    mask = torch.eye(distances.shape[0], device=distances.device).bool()
    
    # Compute the number of comparisons
    n = distances.shape[0]
    num_comparisons = n * (n - 1) / 2
    
    # Compute contrastive loss
    # We want to maximize the distances between different embeddings
    loss = -torch.sum(distances[~mask]) / (2 * num_comparisons)

    loss_type = "Cross-sample" if is_cross_sample else "Within-sample"
    logging.info(f"{loss_type} contrastive loss computation:")
    logging.info(f"  Number of ensemble embeddings: {len(embeddings)}")
    logging.info(f"  Average pairwise distance: {distances[~mask].mean().item()}")
    logging.info(f"  Max pairwise distance: {distances[~mask].max().item()}")
    logging.info(f"  Min pairwise distance: {distances[~mask].min().item()}")
    logging.info(f"  Contrastive loss: {loss.item()}")
     
    return loss

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

    #* temp_dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES)
    dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES, sample_size=SAMPLE_SIZE)

    logging.info(f"Temporary dataset created with {len(dataset.keys)} ensembles and sample size {dataset.sample_size}") #! temp_dataset
    
    avg_num_neighbors, unique_atomic_numbers = calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device) #! temp_dataset

    # avg_neighbors, unique_atomic_numbers, ensemble_avg_neighbors = calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device)
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
        "num_interactions": 1,
        "num_elements": len(z_table),
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "hidden_irreps": o3.Irreps("256x0e + 256x1o"),
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