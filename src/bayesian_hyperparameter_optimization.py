"""
EQUICAT Hyperparameter Optimization

This script performs hyperparameter tuning for the EQUICAT model using Bayesian optimization.
It processes molecular conformer data, trains the EQUICAT model with various hyperparameter configurations,
and identifies the best performing set of hyperparameters.

Key components:
1. Data loading and preprocessing with conformer limitation
2. Hyperparameter space definition
3. Objective function for model training and evaluation
4. Bayesian optimization using hyperopt

Flow:
1. Load conformer ensemble
2. Define hyperparameter search space
3. For each trial in the optimization process:
   a. Create dataset with current hyperparameters
   b. Train EQUICAT model
   c. Evaluate model performance
   d. Return loss for optimization
4. Select best hyperparameters based on lowest loss
5. Save optimization results

Author: Utkarsh Sharma
Version: 1.0.0
Date: 09-27-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - numpy (>=1.20.0)
    - hyperopt (>=0.2.5)
    - mace (custom package)
    - e3nn (>=0.4.0)
    - molli (custom package)
    - sklearn (>=0.24.0)

Usage:
    python hyperparameter_optimization.py

TODO:
    - Implement multi-GPU support for faster optimization
    - Add more sophisticated early stopping criteria
    - Explore more hyperparameter combinations
    - Implement cross-validation for more robust results
"""
import os
import logging
import torch
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mace import data, tools, modules
from e3nn import o3
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import ConformerDataset, compute_avg_num_neighbors
from train import train_equicat, compute_contrastive_loss, process_sample
import molli as ml
from sklearn.cluster import KMeans

# Constants
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output"
# CONFORMER_LIBRARY_PATH = "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib"
# OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/output"
NUM_ENSEMBLES = 256  # Number of ensembles for optimization
SAMPLE_SIZE = 10  # Number of molecules per sample
MAX_CONFORMERS = 5  # Maximum number of conformers per molecule
MAX_EVALS = 10  # Number of iterations for Bayesian Optimization
EXCLUDED_MOLECULES = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']
RSTATE = np.random.default_rng(0)

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)

# Setup logging
LOG_FILE = os.path.join(OUTPUT_PATH, 'hyperparameter_optimization.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=LOG_FILE,
                    filemode='w')

# Add a stream handler to print to console as well
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Load conformer ensemble
conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)

def select_diverse_conformers(conformers, max_conformers):
    """
    Select diverse conformers using K-means clustering.

    Args:
        conformers (list): List of conformer coordinates.
        max_conformers (int): Maximum number of conformers to select.

    Returns:
        list: Selected diverse conformers.
    """
    if len(conformers) <= max_conformers:
        return conformers
    
    features = [np.mean(conf, axis=0) for conf in conformers]
    kmeans = KMeans(n_clusters=max_conformers, random_state=42)
    kmeans.fit(features)
    
    selected_indices = []
    for i in range(max_conformers):
        cluster_members = np.where(kmeans.labels_ == i)[0]
        center = kmeans.cluster_centers_[i]
        distances = [np.linalg.norm(features[j] - center) for j in cluster_members]
        selected_indices.append(cluster_members[np.argmin(distances)])
    
    return [conformers[i] for i in selected_indices]

class LimitedConformerDataset(ConformerDataset):
    """
    A dataset class that limits the number of conformers per molecule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit_conformers()

    def limit_conformers(self):
        """
        Limit the number of conformers for each molecule in the dataset.
        """
        for key in self.keys:
            with self.conformer_ensemble.reading():
                conformer = self.conformer_ensemble[key]
                limited_conformers = select_diverse_conformers(conformer.coords, MAX_CONFORMERS)
                self.conformer_ensemble[key].coords = limited_conformers

# Define the hyperparameter search space
space = {
    'hidden_layer_dim': hp.choice('hidden_layer_dim', [64, 128, 192, 256]),
    'max_ell': hp.choice('max_ell', [1, 2, 3]),
    'embedding_dim': hp.choice('embedding_dim', [16, 32, 64, 128, 192, 256]),
    'epochs': hp.quniform('epochs', 10, 100, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-1)),    
    'cutoff': hp.choice('cutoff', [4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
}

def create_dataset(cutoff):
    """
    Create a dataset with the given cutoff.

    Args:
        cutoff (float): Cutoff distance for atomic interactions.

    Returns:
        ConformerDataset: The created dataset.
    """
    return ConformerDataset(conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES, 
                            sample_size=SAMPLE_SIZE, exclude_molecules=EXCLUDED_MOLECULES)

def calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device):
    """
    Calculate the average number of neighbors and unique atomic numbers in the dataset.

    Args:
        dataset (ConformerDataset): The dataset to analyze.
        device (torch.device): The device to use for computations.

    Returns:
        tuple: Average number of neighbors and list of unique atomic numbers.
    """
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
    return avg_num_neighbors, list(unique_atomic_numbers)

def objective(params):
    """
    Objective function for hyperparameter optimization.

    Args:
        params (dict): Hyperparameters to evaluate.

    Returns:
        dict: A dictionary containing the loss and status of the evaluation.
    """
    # Extract hyperparameters
    hidden_layer_dim = params['hidden_layer_dim']
    max_ell = params['max_ell']
    embedding_dim = params['embedding_dim']
    epochs = int(params['epochs'])
    learning_rate = params['learning_rate']
    cutoff = params['cutoff']
    print("Params", params)

    # Create dataset
    dataset = create_dataset(cutoff)
    logging.info(f"Created dataset with {NUM_ENSEMBLES} molecules and sample size {SAMPLE_SIZE}")

    # Calculate average number of neighbors and unique atomic numbers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_num_neighbors, unique_atomic_numbers = calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device)
    
    z_table = tools.AtomicNumberTable(unique_atomic_numbers)

    # Define model configuration
    model_config = {
        "r_max": cutoff,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": max_ell,
        "num_interactions": 1,
        "num_elements": len(z_table),
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "hidden_irreps": o3.Irreps(f"{hidden_layer_dim}x0e + {hidden_layer_dim}x1o"),
        "MLP_irreps": o3.Irreps(f"{embedding_dim}x0e"),
        "atomic_energies": tools.to_numpy(torch.zeros(len(z_table), dtype=torch.float64)),
        "correlation": 3,
        "gate": torch.nn.functional.silu,
        "avg_num_neighbors": avg_num_neighbors,
    }

    # Set global constants in train.py
    import train
    train.EPOCHS = epochs
    train.LEARNING_RATE = learning_rate

    # Train model
    model = train_equicat(model_config, z_table, conformer_ensemble, cutoff, device, 'improved_self_attention', 'cosine')

    # Evaluate model
    dataset.reset()
    total_loss = 0
    num_samples = 0
    
    for sample_idx in range(len(dataset)):
        sample = dataset.get_next_sample()
        if sample is None:
            break

        sample_embeddings = process_sample(model, sample, device, 'improved_self_attention')
        loss = compute_contrastive_loss(sample_embeddings)
        total_loss += loss.item()
        num_samples += 1

    avg_loss = total_loss / num_samples

    logging.info(f"Hyperparameters: {params}")
    logging.info(f"Average Loss Trials: {avg_loss}")
    print("Average Loss Trials", avg_loss)
    return {'loss': avg_loss, 'status': STATUS_OK}

def run_optimization():
    """
    Run the hyperparameter optimization process.
    """
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=MAX_EVALS,
                trials=trials, rstate=RSTATE)

    logging.info(f"Best hyperparameters: {best}")
    logging.info(f"Best loss: {trials.best_trial['result']['loss']}")

    # Save results
    np.save(os.path.join(OUTPUT_PATH, 'optimization_results.npy'), trials)

if __name__ == "__main__":
    run_optimization()

