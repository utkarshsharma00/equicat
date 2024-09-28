"""
EQUICAT Model Training Script

This script implements the training pipeline for the EQUICAT model, a neural network 
designed for molecular conformer analysis. It handles multi-molecule samples, performs
contrastive learning, and includes advanced logging and visualization features.
After training, it saves the final embeddings for each molecule for downstream tasks.

Key components:
1. GPU acceleration with CUDA support
2. Customizable logging setup with detailed gradient and loss tracking
3. Dynamic calculation of average neighbors and unique atomic numbers
4. Contrastive loss calculation for multi-molecule samples
5. Gradient clipping and comprehensive logging during training
6. Configurable model parameters and training hyperparameters
7. Learning rate scheduling with multiple options
8. Model checkpointing for training resumption
9. Embedding tracking and visualization across epochs
10. Saving final molecule embeddings for downstream tasks

New functionalities in v3.1:
1. Saving final molecule embeddings: After training, the script saves the final
   embeddings for each molecule along with their keys for downstream tasks.
2. Enhanced logging: Includes information about the saved embeddings.

Training process:
1. Loads X molecules, divided into X1, X2, X3, ... Xn samples of Y conformers each
2. For each epoch:
   a. Processes all X1, X2, X3, ... Xn samples sequentially
   b. For each sample:
      - Performs forward pass for all molecules in the sample
      - Calculates contrastive loss for the sample
      - Performs backward propagation
      - Updates model parameters
   c. Logs gradients, loss, and embeddings
   d. Adjusts learning rate based on chosen scheduler
3. After training, saves final embeddings for each molecule

Author: Utkarsh Sharma
Version: 3.1.0
Date: 09-11-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - e3nn (>=0.4.0)
    - mace (custom package)
    - molli (custom package) 
    - matplotlib (>=3.3.0)
    - sklearn (>=0.24.0)

Usage:
    python train.py [--embedding_type {mean_pooling,deep_sets,self_attention,improved_deep_sets,improved_self_attention,all}] 
                    [--resume_from_checkpoint CHECKPOINT_PATH]
                    [--scheduler {plateau,step,cosine,onecycle}]

For detailed usage instructions, please refer to the README.md file.

Change Log: 
    - v3.1.0: Added functionality to save final molecule embeddings for downstream tasks
    - v3.0.0: Implemented multi-molecule sample processing and contrastive learning
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
    - Implement distributed training for multi-GPU setups
    - Add support for dynamic sample sizes and ensemble counts
    - Implement more advanced contrastive learning techniques
    - Enhance visualization with interactive embedding plots
    - Optimize memory usage for processing larger molecule sets
    - Implement adaptive learning rate techniques
    - Add support for transfer learning from pre-trained models
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
import json
from e3nn import o3
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
from mace import data, modules, tools
from mace.tools import to_numpy
from collections import OrderedDict
from sklearn.decomposition import PCA
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import ConformerDataset
from conformer_ensemble_embedding_combiner import process_molecule_conformers
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

# Constants
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output"
# CONFORMER_LIBRARY_PATH = "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib"
# OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/output"
NUM_ENSEMBLES = 50  # Total number of molecules
SAMPLE_SIZE = 10  # Number of molecules per sample
MAX_CONFORMERS = 20  # Maximum number of conformers per molecule
CUTOFF = 6.0
LEARNING_RATE = 1e-3
EPOCHS = 5  # Total number of epochs
GRADIENT_CLIP_VALUE = 1.0
CHECKPOINT_INTERVAL = 10
EXCLUDED_MOLECULES = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(0)

def setup_logging(log_file):
    """
    Set up logging configuration for the training process.

    Args:
        log_file (str): Path to the log file.
    """
    with open(log_file, 'w') as f:
        f.write("")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_lr(optimizer):
    """
    Get the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer object.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
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
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 25 * steps_per_epoch)
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_type == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, epochs=num_epochs, steps_per_epoch=steps_per_epoch
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device):
    """
    Calculate the average number of neighbors and unique atomic numbers in the dataset.

    Args:
        dataset (ConformerDataset): The dataset to analyze.
        device (torch.device): The device to use for computations.

    Returns:
        tuple: (average number of neighbors, list of unique atomic numbers)
    """
    total_neighbors = 0
    total_atoms = 0
    unique_atomic_numbers = OrderedDict()
    
    for _ in range(len(dataset)):
        sample = dataset.get_next_sample()
        if sample is None:
            break
        
        for atomic_data_list, _ in sample:
            for conformer in atomic_data_list:
                conformer = conformer.to(device)
                total_neighbors += conformer.edge_index.shape[1]
                total_atoms += conformer.positions.shape[0]
                for atomic_number in conformer.atomic_numbers.cpu().tolist():
                    unique_atomic_numbers[atomic_number] = None

    avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
    return avg_neighbors, list(unique_atomic_numbers.keys())

def process_sample(model, sample, device, embedding_type):
    """
    Process a sample of molecules through the model and combine their embeddings.

    Args:
        model (EQUICATPlusNonLinearReadout): The EQUICAT model.
        sample (list): List of molecule data.
        device (torch.device): The device to use for computations.
        embedding_type (str): Type of embedding combination to use.

    Returns:
        list: List of tuples containing combined embeddings and molecule keys.
    """
    sample_embeddings = []
    for molecule_id, (atomic_data_list, key) in enumerate(sample):
        molecule_embeddings = [] # For each molecule, initialize a list to store embeddings of its conformers.
        for conformer in atomic_data_list:
            conformer = conformer.to(device)
            input_dict = {
                'positions': conformer.positions,
                'atomic_numbers': conformer.atomic_numbers,
                'edge_index': conformer.edge_index
            }
            output = model(input_dict)
            molecule_embeddings.append(output)
        
        molecule_embeddings = torch.stack(molecule_embeddings)
        averaged_embeddings = process_molecule_conformers(molecule_embeddings, model.non_linear_readout.irreps_out)
        
        scalar, vector = averaged_embeddings[embedding_type]
        
        if scalar is not None and vector is not None:
            combined = torch.cat([scalar.view(-1), vector.view(-1)])
        elif scalar is not None:
            combined = scalar.view(-1)
        else:
            combined = vector.view(-1)
        
        sample_embeddings.append((combined, key))

    return sample_embeddings

def compute_contrastive_loss(sample_embeddings):
    """
    Compute the contrastive loss for a sample of molecule embeddings.

    Args:
        sample_embeddings (list): List of tuples containing embeddings and keys.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    embeddings = []
    keys = []
    for emb, key in sample_embeddings:
        embeddings.append(emb)
        keys.append(key)
    
    embeddings = torch.stack(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    print("Embeddings shape: ", embeddings.shape)
    print("Molecules in this sample:")
    for i, (key, embedding) in enumerate(zip(keys, embeddings)):
        print(f"Molecule {i+1} (Key: {key}):")
        print(f"  Embedding: {embedding}")
        print(f"  Mean: {embedding.mean().item():.4f}")
        print(f"  Std: {embedding.std().item():.4f}")
        print(f"  Min: {embedding.min().item():.4f}")
        print(f"  Max: {embedding.max().item():.4f}")

    epsilon = 1e-8
    distances = torch.cdist(embeddings, embeddings) + epsilon

    mask = torch.eye(distances.shape[0], device=distances.device).bool()
    n = distances.shape[0]
    num_comparisons = n * (n - 1) / 2

    loss = -torch.sum(distances[~mask]) / (2 * num_comparisons)
    return loss

def log_gradients(model):
    """
    Log gradient statistics for model parameters.

    Args:
        model (nn.Module): The model whose gradients are to be logged.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            logging.info(f"Gradient stats for {name}:")
            logging.info(f"  Norm: {grad_norm:.6f}, Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")

def log_all_molecule_embeddings(all_molecule_embeddings, epoch):
    """
    Log statistics of molecule embeddings across epochs.

    Args:
        all_molecule_embeddings (dict): Dictionary containing embeddings for each molecule across epochs.
        epoch (int): The current epoch number.
    """
    logging.info(f"Averaged embeddings after epoch {epoch}:")
    for key, embeddings in all_molecule_embeddings.items():
        current_embedding = embeddings[-1]  # Get the latest embedding
        emb_mean = current_embedding.mean().item()
        emb_std = current_embedding.std().item()
        emb_norm = current_embedding.norm().item()
        logging.info(f"  Molecule {key}:")
        logging.info(f"    Mean: {emb_mean:.6f}, Std: {emb_std:.6f}, Norm: {emb_norm:.6f}")
        
        # If there are multiple epochs, compute change from previous epoch
        if len(embeddings) > 1:
            prev_embedding = embeddings[-2]
            change = (current_embedding - prev_embedding).norm().item()
            logging.info(f"    Change from previous epoch: {change:.6f}")

def train_equicat(model_config, z_table, conformer_ensemble, cutoff, device, embedding_type, scheduler_type, resume_from=None):
    """
    Train the EQUICAT model and save final molecule embeddings.

    Args:
        model_config (dict): Configuration for the EQUICAT model.
        z_table (AtomicNumberTable): Table of atomic numbers.
        conformer_ensemble (ConformerLibrary): Library of conformers.
        cutoff (float): Cutoff distance for atomic interactions.
        device (torch.device): Device to use for training.
        embedding_type (str): Type of embedding to use.
        scheduler_type (str): Type of learning rate scheduler to use.
        resume_from (str, optional): Path to checkpoint to resume training from.

    Returns:
        nn.Module: The trained EQUICAT model.
    """
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
    print(model)
    logging.info(f"Model initialized and moved to {device}")

    dataset = ConformerDataset(conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES, 
                               sample_size=SAMPLE_SIZE, exclude_molecules=EXCLUDED_MOLECULES)
    
    num_samples = dataset.total_samples
    steps_per_epoch = num_samples // SAMPLE_SIZE + (1 if num_samples % SAMPLE_SIZE != 0 else 0)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(scheduler_type, optimizer, EPOCHS, steps_per_epoch)

    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info(f"Resuming training from epoch {start_epoch}")

    best_loss = float('inf')
    best_model = None
    patience = 30  # Number of epochs to wait for improvement before stopping
    patience_counter = 0
    logging.info(f"Early stopping patience set to {patience} epochs")

    # Dictionary to store all molecule embeddings across epochs
    all_molecule_embeddings = {}

    # Dictionary to store final molecule embeddings
    final_molecule_embeddings = {}

    for epoch in range(start_epoch, EPOCHS):
        logging.info(f"Starting epoch {epoch+1}/{EPOCHS}")
        logging.info(f"Current learning rate: {get_lr(optimizer):.6f}")
        epoch_loss = 0.0
        dataset.reset()  # Reset dataset at the start of each epoch

        # Dictionary to store molecule embeddings for this epoch
        epoch_embeddings = {}

        for sample_idx in range(num_samples):
            sample = dataset.get_next_sample()
            if sample is None:
                break

            optimizer.zero_grad()

            with detect_anomaly():
                sample_embeddings = process_sample(model, sample, device, embedding_type)
                loss = compute_contrastive_loss(sample_embeddings)

                loss.backward()
                clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
                optimizer.step()

            epoch_loss += loss.item()
            logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Sample [{sample_idx+1}/{num_samples}], "
                         f"Loss: {loss.item():.6f}, LR: {get_lr(optimizer):.6f}")

            # Log gradients after each sample
            log_gradients(model)

            # Store embeddings for each molecule in this sample
            for embedding, key in sample_embeddings:
                if key not in epoch_embeddings:
                    epoch_embeddings[key] = []
                epoch_embeddings[key].append(embedding.detach().cpu())

        avg_epoch_loss = epoch_loss / num_samples
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.6f}, "
                     f"Final LR: {get_lr(optimizer):.6f}")

        # Compute average embeddings for each molecule in this epoch
        for key, embeddings in epoch_embeddings.items():
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            if key not in all_molecule_embeddings:
                all_molecule_embeddings[key] = []
            all_molecule_embeddings[key].append(avg_embedding)

        # Log embeddings summary for all molecules after each epoch
        log_all_molecule_embeddings(all_molecule_embeddings, epoch+1)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_epoch_loss)
        else:
            scheduler.step()

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model = model.state_dict()
            patience_counter = 0
            logging.info(f"New best model found with loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement in loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, OUTPUT_PATH)

    logging.info("Training completed")
    logging.info(f"Best model had a loss of {best_loss:.6f}")
    model.load_state_dict(best_model)

    # Compute final embeddings using the best model
    logging.info("Computing final embeddings using the best model")
    model.eval()
    dataset.reset()
    with torch.no_grad():
        for sample_idx in range(num_samples):
            sample = dataset.get_next_sample()
            if sample is None:
                break
            sample_embeddings = process_sample(model, sample, device, embedding_type)
            for embedding, key in sample_embeddings:
                final_molecule_embeddings[key] = embedding.cpu().numpy()
    
    logging.info(f"Computed final embeddings for {len(final_molecule_embeddings)} molecules")

    # Save the final molecule embeddings
    save_final_embeddings(final_molecule_embeddings, OUTPUT_PATH)

    return model

def save_final_embeddings(embeddings, output_path):
    """
    Save the final molecule embeddings to a file.

    Args:
        embeddings (dict): Dictionary containing molecule keys and their embeddings.
        output_path (str): Directory to save the embeddings.
    """
    embeddings_file = f'{output_path}/final_molecule_embeddings.json'
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_embeddings = {key: emb.tolist() for key, emb in embeddings.items()}
    
    with open(embeddings_file, 'w') as f:
        json.dump(serializable_embeddings, f)
    
    logging.info(f"Final molecule embeddings saved to {embeddings_file}")
    logging.info(f"Number of molecules with saved embeddings: {len(embeddings)}")       

def save_checkpoint(epoch, model, optimizer, scheduler, loss, output_path):
    """
    Save a checkpoint of the model's state.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save.
        loss (float): Current loss value.
        output_path (str): Directory to save the checkpoint.
    """
    checkpoint_path = f'{output_path}/checkpoints/checkpoint_epoch_{epoch+1}.pt'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EQUICAT Model Training")
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
    logging.info(f"Embedding type: {args.embedding_type}")
    logging.info(f"Learning rate scheduler: {args.scheduler}")

    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)
    logging.info(f"Loaded conformer ensemble from {CONFORMER_LIBRARY_PATH}")

    dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES, 
                               sample_size=SAMPLE_SIZE, exclude_molecules=EXCLUDED_MOLECULES)    
    
    avg_num_neighbors, unique_atomic_numbers = calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device)
    np.save(f'{OUTPUT_PATH}/unique_atomic_numbers.npy', np.array(unique_atomic_numbers))
    logging.info(f"Average number of neighbors: {avg_num_neighbors}")
    logging.info(f"Unique atomic numbers: {unique_atomic_numbers}")

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

    trained_model = train_equicat(model_config, z_table, conformer_ensemble, CUTOFF, device, args.embedding_type, args.scheduler, args.resume_from_checkpoint)

    torch.save(trained_model.state_dict(), f"{OUTPUT_PATH}/trained_equicat_model.pt")
    logging.info("Model training completed and saved.")

    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(f"Total runtime: {total_runtime:.2f} seconds")