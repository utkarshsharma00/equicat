"""
train.py

Author: Utkarsh Sharma
Version: 1.0.0
Date: 10-03-2024 (MM-DD-YYYY)
License: MIT
"""
import torch
import torch.nn.functional as F
import logging
import sys
import numpy as np
import time
import argparse
import os
import json
from e3nn import o3
import molli as ml
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from mace import data, modules, tools
from mace.tools import to_numpy
from collections import OrderedDict
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import MultiFamilyConformerDataset
from conformer_ensemble_embedding_combiner import process_molecule_conformers, move_to_device
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(42)

logger = logging.getLogger('train')

# Constants
CONFORMER_LIBRARY_PATHS = {
    # "family1": "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib",
    # "family2": "/Users/utkarsh/MMLI/molli-data/00-libraries/thiol_confs.clib",
    # "family3": "/Users/utkarsh/MMLI/molli-data/00-libraries/imine_confs.clib",
    # "family4": "/Users/utkarsh/MMLI/molli-data/00-libraries/product_confs.clib",

    "family1": "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib",
    "family2": "/eagle/FOUND4CHEM/utkarsh/dataset/thiol_confs.clib",
    "family3": "/eagle/FOUND4CHEM/utkarsh/dataset/imine_confs.clib",
    "family4": "/eagle/FOUND4CHEM/utkarsh/dataset/product_confs.clib",
}

OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/develop_op"
SAMPLE_SIZE = 10
MAX_CONFORMERS = 8
CUTOFF = 6.0
LEARNING_RATE = 1e-3
EPOCHS = 50
GRADIENT_CLIP_VALUE = 1.0
CHECKPOINT_INTERVAL = 5
EXCLUDED_MOLECULES = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']

def setup_logging(log_file):
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_scheduler(scheduler_type, optimizer, num_epochs, steps_per_epoch):
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=25 * steps_per_epoch)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_type == 'onecycle':
        return OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device):
    logger.info("Calculating average neighbors and unique atomic numbers")
    total_neighbors = 0
    total_atoms = 0
    unique_atomic_numbers = OrderedDict()
    
    for sample in dataset:
        for atomic_data_list, _, _ in sample:
            for conformer in atomic_data_list:
                conformer = conformer.to(device)
                total_neighbors += conformer.edge_index.shape[1]
                total_atoms += conformer.positions.shape[0]
                for atomic_number in conformer.atomic_numbers.cpu().tolist():
                    unique_atomic_numbers[atomic_number] = None

    avg_neighbors = total_neighbors / total_atoms if total_atoms > 0 else 0
    logger.info(f"Calculation complete. Average neighbors: {avg_neighbors}")
    return avg_neighbors, list(unique_atomic_numbers.keys())

def process_sample(model, sample, device, embedding_type):
    sample_embeddings = []
    for atomic_data_list, key, family in sample:
        molecule_embeddings = []
        for conformer in atomic_data_list:
            conformer = move_to_device(conformer, device)
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
        
        sample_embeddings.append((combined, key, family))

    return sample_embeddings

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
            logger.info(f"Gradient stats for {name}:")
            logger.info(f"  Norm: {grad_norm:.6f}, Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")

def compute_contrastive_loss(sample_embeddings, temperature=0.1):
    embeddings = []
    families = []
    for emb, _, family in sample_embeddings:
        embeddings.append(emb)
        families.append(family)
    
    if len(embeddings) < 2:
        logging.warning(f"Not enough embeddings to compute contrastive loss. Only {len(embeddings)} embeddings available.")
        return torch.tensor(0.0, requires_grad=True)  # Return a dummy loss

    embeddings = torch.stack(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    labels = torch.tensor([families.index(f) for f in families], device=embeddings.device)
    
    # Create mask for positive pairs
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    # We don't want to consider the similarity of a sample with itself as a positive pair
    pos_mask.fill_diagonal_(0)
    
    # Negative mask is just the inverse of the positive mask
    neg_mask = 1 - pos_mask
    
    # Compute loss
    exp_sim = torch.exp(similarity_matrix)
    
    # Compute positive and negative scores
    pos_scores = torch.sum(exp_sim * pos_mask, dim=1)
    neg_scores = torch.sum(exp_sim * neg_mask, dim=1)
    
    # Compute loss
    loss = -torch.log(pos_scores / (pos_scores + neg_scores))
    
    return loss.mean()

def train_equicat(model, dataset, device, embedding_type, scheduler_type, args):
    logger.info("Starting train_equicat function")
    
    num_samples = len(dataset)
    steps_per_epoch = num_samples
    logger.info(f"Number of samples: {num_samples}, Steps per epoch: {steps_per_epoch}")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(scheduler_type, optimizer, EPOCHS, steps_per_epoch)
    logger.info("Optimizer and scheduler initialized")

    best_loss = float('inf')
    best_model = None
    patience = 30
    patience_counter = 0
    logger.info(f"Early stopping patience set to {patience} epochs")

    all_molecule_embeddings = {}
    final_molecule_embeddings = {}

    for epoch in range(EPOCHS):
        logger.info(f"Starting epoch {epoch+1}/{EPOCHS}")
        model.train()
        epoch_loss = 0.0
        dataset.reset()

        epoch_embeddings = {}

        for sample_idx, sample in enumerate(dataset):
            logger.info(f"Processing sample {sample_idx+1}/{num_samples}")
            optimizer.zero_grad()

            sample_embeddings = process_sample(model, sample, device, embedding_type)
            loss = compute_contrastive_loss(sample_embeddings)

            loss.backward()
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            optimizer.step()

            epoch_loss += loss.item()
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Sample [{sample_idx+1}/{num_samples}], "
                         f"Loss: {loss.item():.6f}, LR: {get_lr(optimizer):.6f}")

            for embedding, key, _ in sample_embeddings:
                if key not in epoch_embeddings:
                    epoch_embeddings[key] = []
                epoch_embeddings[key].append(embedding.detach().cpu())

        avg_epoch_loss = epoch_loss / num_samples
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.6f}, "
                     f"Final LR: {get_lr(optimizer):.6f}")

        for key, embeddings in epoch_embeddings.items():
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            if key not in all_molecule_embeddings:
                all_molecule_embeddings[key] = []
            all_molecule_embeddings[key].append(avg_embedding)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_epoch_loss)
        else:
            scheduler.step()

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model = model.state_dict()
            patience_counter = 0
            logger.info(f"New best model found with loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, OUTPUT_PATH)

    logger.info("Training completed")
    logger.info(f"Best model had a loss of {best_loss:.6f}")
    
    # Load the best model for final embeddings computation
    model.load_state_dict(best_model)

    # Compute final embeddings using the best model
    logger.info("Computing final embeddings using the best model")
    
    model.eval()
    final_molecule_embeddings = {}
    with torch.no_grad():
        for family, keys in dataset.family_keys.items():
            for key in keys:
                atomic_data_list = dataset.get_molecule_data(key)
                sample_embeddings = process_sample(model, [(atomic_data_list, key, family)], device, embedding_type)
                embedding, _, _ = sample_embeddings[0]
                final_molecule_embeddings[key] = embedding.cpu().numpy()

    logger.info(f"Computed final embeddings for {len(final_molecule_embeddings)} molecules")

    # Save the final molecule embeddings
    save_final_embeddings(final_molecule_embeddings, OUTPUT_PATH)

def save_final_embeddings(embeddings, output_path):
    embeddings_file = f'{output_path}/final_molecule_embeddings.json'
    
    serializable_embeddings = {key: emb.tolist() for key, emb in embeddings.items()}
    
    with open(embeddings_file, 'w') as f:
        json.dump(serializable_embeddings, f)
    
    logger.info(f"Final molecule embeddings saved to {embeddings_file}")
    logger.info(f"Number of molecules with saved embeddings: {len(embeddings)}")       

def save_checkpoint(epoch, model, optimizer, scheduler, loss, output_path):
    checkpoint_path = f'{output_path}/checkpoints/checkpoint_epoch_{epoch+1}.pt'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def main(args):
    logger.info("Starting main function")
    
    setup_logging(f"{OUTPUT_PATH}/training.log")
    logger.info("Starting EQUICAT training")
    logger.info(f"Embedding type: {args.embedding_type}")
    logger.info(f"Learning rate scheduler: {args.scheduler}")
    logger.info(f"Number of families: {args.num_families}")
    logger.info(f"Ensembles per family: {args.ensembles_per_family}")

    logger.info("Loading conformer libraries")
    conformer_libraries = {
        family: ml.ConformerLibrary(path)
        for family, path in CONFORMER_LIBRARY_PATHS.items()
    }
    logger.info(f"Loaded conformer libraries: {', '.join(conformer_libraries.keys())}")

    logger.info("Creating dataset")
    dataset = MultiFamilyConformerDataset(
        conformer_libraries=conformer_libraries,
        cutoff=CUTOFF,
        sample_size=SAMPLE_SIZE,
        max_conformers=MAX_CONFORMERS,
        exclude_molecules=EXCLUDED_MOLECULES,
        num_families=args.num_families,
        ensembles_per_family=args.ensembles_per_family
    )
    logger.info("Dataset created successfully")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    avg_num_neighbors, unique_atomic_numbers = calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device)
    np.save(f'{OUTPUT_PATH}/unique_atomic_numbers.npy', np.array(unique_atomic_numbers))
    logger.info(f"Average number of neighbors: {avg_num_neighbors}")
    logger.info(f"Unique atomic numbers: {unique_atomic_numbers}")

    z_table = tools.AtomicNumberTable(unique_atomic_numbers)

    logger.info("Creating model config")
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
    logger.info(f"Model configuration: {model_config}")

    logger.info("Initializing model")
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(device)
    print(model)
    logger.info(f"Model initialized and moved to {device}")

    logger.info("Starting training process")
    train_equicat(model, dataset, device, args.embedding_type, args.scheduler, args)
    logger.info("Training process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EQUICAT Model Training")
    parser.add_argument('--embedding_type', type=str, default='improved_self_attention',
                        choices=['mean_pooling', 'deep_sets', 'self_attention', 'improved_deep_sets', 'improved_self_attention', 'all'],
                        help='Type of embedding to use for loss computation')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['plateau', 'step', 'cosine', 'onecycle'],
                        help='Type of learning rate scheduler to use')
    parser.add_argument('--num_families', type=int, default=None, help='Number of molecule families to use')
    parser.add_argument('--ensembles_per_family', type=int, default=None, help='Number of ensembles to use per family')
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    total_runtime = end_time - start_time
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")
                        