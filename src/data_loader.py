"""
Conformer Data Loader and Processor for EQUICAT

This module provides comprehensive functionality for loading and processing conformer data
for use with the EQUICAT model. It includes a custom dataset class, data loading utilities, 
and efficient processing functions, now with GPU support and optional conformer padding.

Key components:
1. ConformerDataset: A custom PyTorch dataset class for handling conformer ensembles.
2. compute_avg_num_neighbors: Utility function to calculate average neighbors in a batch.
3. custom_collate: Custom collation function for batching data.
4. process_data: Generator function for processing conformer data in batches with optional padding.
5. pad_batch: Function to pad batches to a consistent size using random sampling when enabled.

This module is optimized to work seamlessly with the MACE framework and PyTorch Geometric,
providing efficient and scalable data handling for molecular conformer analysis on both CPU and GPU.

New Feature:
- Optional Conformer Padding: Addresses the issue of variable conformer counts in molecular ensembles.
  When enabled and a molecule has fewer conformers than the batch size, or when the last batch of an
  ensemble is smaller than the desired batch size, the function pads the batch by randomly
  sampling from the available conformers. This ensures consistent batch sizes across all
  molecules and batches when needed, which is crucial for stable training and accurate ensemble embeddings.

Author: Utkarsh Sharma
Version: 2.2.0
Date: 08-08-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - numpy (>=1.20.0)
    - molli (>=0.1.0)
    - mace (custom package)
    - torch_geometric (>=2.0.0)

Usage:
    from data_loader import ConformerDataset, process_data
    
    dataset = ConformerDataset(conformer_ensemble, cutoff)
    for batch_data in process_data(dataset, batch_size=32, device=device, pad_batches=True):
        # Process batch_data

For detailed usage instructions, please refer to the README.md file.

Change Log:
    - v2.2.0: Added optional conformer padding via pad_batches parameter
    - v2.1.0: Added support for processing padded conformer batches
    - v2.0.0: Added GPU support
    - v1.2.0: Fixed critical edge_index generation issue
    - v1.1.0: Added ensemble_id to process_data output
    - v1.0.0: Initial release

TODO:
    - Implement weighted sampling for padding to further improve ensemble representation
    - Add option for deterministic padding for reproducibility
"""

import torch
import random
import logging
import molli as ml
from typing import List, Optional, Tuple
from mace import data, tools
from torch_geometric.data import Data, Batch
from sklearn.cluster import KMeans
import numpy as np

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(0)

# Constants
CONFORMER_LIBRARY_PATH = "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"
CUTOFF = 6.0
NUM_ENSEMBLES = 40
SAMPLE_SIZE = 10
BATCH_SIZE = 6
LOG_FILE = "/Users/utkarsh/MMLI/equicat/output/data_loader.log"
MAX_CONFORMERS = 20  # New constant for maximum number of conformers per ensemble

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

def select_diverse_conformers(conformers: List[np.ndarray], max_conformers: int) -> List[int]:
    """
    Select diverse conformers using K-means clustering.
    
    Args:
        conformers (List[np.ndarray]): List of conformer coordinates.
        max_conformers (int): Maximum number of conformers to select.
    
    Returns:
        List[int]: Indices of selected diverse conformers.
    """
    n_conformers = len(conformers)
    if n_conformers <= max_conformers:
        return list(range(n_conformers))
    
    # Flatten conformers for clustering
    flattened_conformers = [conf.flatten() for conf in conformers]
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=max_conformers, random_state=42)
    kmeans.fit(flattened_conformers)
    
    # Select the conformer closest to each cluster center
    selected_indices = []
    for cluster in range(max_conformers):
        cluster_conformers = [i for i, label in enumerate(kmeans.labels_) if label == cluster]
        cluster_center = kmeans.cluster_centers_[cluster]
        distances = [np.linalg.norm(flattened_conformers[i] - cluster_center) for i in cluster_conformers]
        selected_indices.append(cluster_conformers[np.argmin(distances)])
    
    return selected_indices

class ConformerDataset:
    def __init__(self, conformer_ensemble, cutoff, num_ensembles=NUM_ENSEMBLES, sample_size=SAMPLE_SIZE):
        self.conformer_ensemble = conformer_ensemble
        self.cutoff = cutoff
        self.sample_size = sample_size

        with self.conformer_ensemble.reading():
            self.keys = list(self.conformer_ensemble.keys())[:num_ensembles]

        self.total_samples = len(self.keys) // self.sample_size
        self.current_sample = 0
        self.sampled_keys = set()

        # Pre-select diverse conformers for each ensemble
        self.selected_conformers = {}
        for key in self.keys:
            with self.conformer_ensemble.reading():
                conformer = self.conformer_ensemble[key]
                coords = conformer.coords
                selected_indices = select_diverse_conformers(coords, MAX_CONFORMERS)
                self.selected_conformers[key] = selected_indices

        logging.info(f"ConformerDataset initialized with {len(self.keys)} total ensembles, "
                     f"sample size {self.sample_size}, and {self.total_samples} total samples")

    def get_next_sample(self):
        if self.current_sample >= self.total_samples:
            return None

        start_idx = self.current_sample * self.sample_size
        end_idx = start_idx + self.sample_size
        sample_keys = self.keys[start_idx:end_idx]

        sample_data = []
        for key in sample_keys:
            with self.conformer_ensemble.reading():
                conformer = self.conformer_ensemble[key]
                coords = torch.tensor(conformer.coords, dtype=torch.float64)
                atomic_numbers = torch.tensor([atom.element for atom in conformer.atoms], dtype=torch.long)
                z_table = tools.AtomicNumberTable(torch.unique(atomic_numbers).tolist())

                selected_indices = self.selected_conformers[key]
                atomic_data_list = []
                for i in selected_indices:
                    config = data.Configuration(
                        atomic_numbers=atomic_numbers.numpy(),
                        positions=coords[i].numpy()
                    )
                    atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=self.cutoff)
                    torch_geo_data = Data(
                        x=torch.tensor(atomic_data.node_attrs, dtype=torch.float64),
                        positions=atomic_data.positions,
                        edge_index=atomic_data.edge_index,
                        atomic_numbers=atomic_numbers,
                        key=key
                    )
                    atomic_data_list.append(torch_geo_data)
                sample_data.append((atomic_data_list, key))

        self.current_sample += 1
        logging.info(f"Loaded sample {self.current_sample} with {len(sample_data)} ensembles")
        return sample_data

    def reset(self):
        self.current_sample = 0
        random.shuffle(self.keys)

    def __len__(self):
        return self.total_samples

def custom_collate(batch):
    """
    Custom collation function for batching conformer data.

    Args:
        batch: A list of tuples (atomic_data_list, key).

    Returns:
        tuple: Batched conformer data and keys.
    """
    all_conformers = [item for sublist, _ in batch for item in sublist]
    keys = [key for _, key in batch]
    return Batch.from_data_list(all_conformers), keys

def pad_batch(batch: List[torch.Tensor], full_ensemble: List[torch.Tensor], batch_size: int, pad_batches: bool) -> Tuple[List[torch.Tensor], int]:
    """
    Pad a batch to the desired size by randomly sampling from the full ensemble if padding is enabled.
    
    Args:
        batch: The current batch of conformers.
        full_ensemble: The full list of conformers for the current molecule.
        batch_size: The desired batch size.
        pad_batches: Whether to pad batches or not.
    
    Returns:
        Tuple of padded batch and number of added conformers.
    """
    if not pad_batches:
        return batch, 0
    
    num_to_add = batch_size - len(batch)
    if num_to_add <= 0:
        return batch, 0
    
    # If we need more conformers than available, we'll sample with replacement
    if num_to_add > len(full_ensemble):
        added_conformers = random.choices(full_ensemble, k=num_to_add)
    else:
        added_conformers = random.sample(full_ensemble, num_to_add)
    
    return batch + added_conformers, num_to_add

def compute_avg_num_neighbors(batch):
    _, receivers = batch.edge_index
    _, counts = torch.unique(receivers, return_counts=True)
    avg_num_neighbors = torch.mean(counts.float())
    return avg_num_neighbors.item()

def process_data(conformer_dataset, batch_size=BATCH_SIZE, device=None, pad_batches=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Using device: {device}")

    sample_count = 0
    while True:
        sample = conformer_dataset.get_next_sample()
        if sample is None:
            break  # End of dataset

        sample_count += 1
        logging.info(f"Processing sample {sample_count}")

        for ensemble_id, (atomic_data_list, key) in enumerate(sample):
            num_conformers = len(atomic_data_list)
            logging.info(f"Processing ensemble {key} with {num_conformers} conformers")

            for i in range(0, num_conformers, batch_size):
                batch_conformers = atomic_data_list[i:i+batch_size]
                original_batch_size = len(batch_conformers)

                if pad_batches:
                    batch_conformers, num_added = pad_batch(batch_conformers, atomic_data_list, batch_size, pad_batches)
                else:
                    num_added = 0

                logging.info(f"Batch for ensemble {key}: {original_batch_size} conformers, {num_added} added")

                try:
                    batch_conformers = [conformer.to(device) for conformer in batch_conformers]
                except AssertionError as e:
                    logging.warning(f"Failed to move tensors to {device}. Error: {str(e)}")
                    logging.info("Falling back to CPU")
                    device = torch.device("cpu")
                    batch_conformers = [conformer.to(device) for conformer in batch_conformers]

                unique_atomic_numbers = []
                for conformer in batch_conformers:
                    for atomic_number in conformer.atomic_numbers.cpu():
                        if atomic_number.item() not in unique_atomic_numbers:
                            unique_atomic_numbers.append(atomic_number.item())

                avg_num_neighbors = sum(compute_avg_num_neighbors(conformer) for conformer in batch_conformers) / len(batch_conformers)

                yield batch_conformers, unique_atomic_numbers, avg_num_neighbors, ensemble_id, num_added, key

        logging.info(f"Finished processing sample {sample_count}")
        yield None  # Indicate end of current sample

def main():
    setup_logging()

    logging.info(f"Loading conformer ensemble from {CONFORMER_LIBRARY_PATH}")
    conformer_ensemble = ml.ConformerLibrary(CONFORMER_LIBRARY_PATH)

    dataset = ConformerDataset(conformer_ensemble, CUTOFF, num_ensembles=NUM_ENSEMBLES, sample_size=SAMPLE_SIZE)

    logging.info(f"Dataset created with {len(dataset)} total ensembles")

    for i, data in enumerate(process_data(dataset, batch_size=BATCH_SIZE, pad_batches=False)):
        if data is None:
            logging.info("End of sample reached")
            continue

        batch_conformers, unique_atomic_numbers, avg_num_neighbors, ensemble_id, num_added, key = data

        logging.info(f"\nProcessing batch {i+1}")
        logging.info(f"  Ensemble ID: {ensemble_id}")
        logging.info(f"  Ensemble Key: {key}")
        logging.info(f"  Batch size: {len(batch_conformers)}")
        logging.info(f"  Unique atomic numbers: {unique_atomic_numbers}")
        logging.info(f"  Average number of neighbors: {avg_num_neighbors:.2f}")
        logging.info(f"  Number of added conformers: {num_added}")

        for j, conformer in enumerate(batch_conformers):
            logging.info(f"  Conformer {j+1} details:")
            logging.info(f"    Number of atoms: {conformer.num_nodes}")
            logging.info(f"    Number of edges: {conformer.num_edges}")
            logging.info(f"    Atomic numbers: {conformer.atomic_numbers}")
            logging.info(f"    Positions shape: {conformer.positions.shape}")
            logging.info(f"    Edge index shape: {conformer.edge_index.shape}")

    logging.info("Data processing test completed.")

if __name__ == "__main__":
    main()