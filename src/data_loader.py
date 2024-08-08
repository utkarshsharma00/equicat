"""
Conformer Data Loader and Processor for EQUICAT (GPU-enabled version with Optional Conformer Padding)

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
import molli as ml
import torch.utils.data
import argparse
from typing import List, Tuple
from mace import data, tools
from torch_geometric.data import Data, Batch

class ConformerDataset:
    """
    A custom dataset class for handling conformer ensembles.
    
    Attributes:
        conformer_ensemble: The source conformer ensemble.
        cutoff: Cutoff distance for atomic interactions.
        keys: List of keys for conformer ensembles.
    """

    def __init__(self, conformer_ensemble, cutoff, num_ensembles=5):
        """
        Initialize the ConformerDataset.

        Args:
            conformer_ensemble: Source conformer ensemble.
            cutoff (float): Cutoff distance for atomic interactions.
            num_ensembles (int): Number of ensembles to process.
        """
        self.conformer_ensemble = conformer_ensemble
        self.cutoff = cutoff
        with self.conformer_ensemble.reading():
            self.keys = list(self.conformer_ensemble.keys())[:num_ensembles]
        print(f"Initialized ConformerDataset with {len(self.keys)} conformer ensembles")

    def __len__(self):
        """Return the number of conformer ensembles in the dataset."""
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Retrieve a specific conformer ensemble by index.

        Args:
            idx (int): Index of the conformer ensemble to retrieve.

        Returns:
            tuple: List of atomic data and the ensemble key.
        """
        key = self.keys[idx]
        with self.conformer_ensemble.reading():
            conformer = self.conformer_ensemble[key]
            coords = torch.tensor(conformer.coords, dtype=torch.float32)
            atomic_numbers = torch.tensor([atom.element for atom in conformer.atoms], dtype=torch.long)
            print(f"Retrieved conformer ensemble {key} with {coords.shape[0]} conformers")
            print(f"Atomic Numbers: {atomic_numbers}")
            print(f"Coords shape: {coords.shape}") # Should be (num_conformers, num_atoms, 3)

            z_table = tools.AtomicNumberTable(torch.unique(atomic_numbers).tolist())

            atomic_data_list = []
            for i in range(coords.shape[0]):
                # Create a Configuration object for each conformer
                config = data.Configuration(
                    atomic_numbers=atomic_numbers.numpy(),
                    positions=coords[i].numpy()
                )
                
                # Use MACE's AtomicData.from_config to generate atomic data with correct edge_index
                atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=self.cutoff)
                
                # Create PyTorch Geometric Data object with the correct edge_index
                torch_geo_data = Data(
                    x=torch.tensor(atomic_data.node_attrs, dtype=torch.float32),
                    positions=atomic_data.positions,
                    edge_index=atomic_data.edge_index,
                    atomic_numbers=atomic_numbers,
                    key=key
                )
                
                atomic_data_list.append(torch_geo_data)

            print(f"Number of conformers in atomic_data_list: {len(atomic_data_list)}")

            return atomic_data_list, key

def compute_avg_num_neighbors(batch):
    """
    Compute the average number of neighbors for atoms in a batch.

    Args:
        batch: A batch of conformer data.

    Returns:
        float: Average number of neighbors.
    """
    _, receivers = batch.edge_index
    _, counts = torch.unique(receivers, return_counts=True)
    avg_num_neighbors = torch.mean(counts.float())
    return avg_num_neighbors.item()

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

def process_data(conformer_dataset, batch_size=32, device=torch.device("cuda"), pad_batches=False):
    """
    Process conformer data in batches, with support for GPU processing and optional consistent batch sizes.

    Args:
        conformer_dataset: The ConformerDataset to process.
        batch_size (int): Number of conformers to process in each batch.
        device (torch.device): The device to move the data to (CPU or GPU).
        pad_batches (bool): Whether to pad batches to ensure consistent size.

    Yields:
        tuple: Batch of conformers, unique atomic numbers, average number of neighbors, ensemble id, and number of added conformers.
    """
    total_batches = 0
    total_conformers = 0

    data_loader = torch.utils.data.DataLoader(
        dataset=conformer_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0]
    )

    for ensemble_id, (atomic_data_list, key) in enumerate(data_loader):
        num_conformers = len(atomic_data_list)
        total_conformers += num_conformers

        print(f"\nProcessing Conformer Ensemble: {key}")
        print(f"Number of conformers in this ensemble: {num_conformers}")
        print(f"Batch size: {batch_size}")
        print(f"Padding enabled: {pad_batches}")

        for i in range(0, num_conformers, batch_size):
            batch_conformers = atomic_data_list[i:i+batch_size]
            original_batch_size = len(batch_conformers)
            
            if pad_batches:
                batch_conformers, num_added = pad_batch(batch_conformers, atomic_data_list, batch_size, pad_batches)
                print(f"Padded batch: original size {original_batch_size}, padded size {len(batch_conformers)}, num_added {num_added}")
            else:
                num_added = 0
                # print(f"Unpadded batch: size {len(batch_conformers)}")
            
            total_batches += 1

            print(f"\nBatch {total_batches} in Ensemble: {key}")
            print(f"Number of conformers in this batch before padding: {original_batch_size}")
            print(f"Number of conformers in this batch after padding: {len(batch_conformers)}")
            if pad_batches:
                print(f"Number of randomly added conformers: {num_added}")

            # Move batch_conformers to the specified device
            batch_conformers = [conformer.to(device) for conformer in batch_conformers]

            unique_atomic_numbers = []
            for conformer in batch_conformers:
                for atomic_number in conformer.atomic_numbers.cpu():  # Move to CPU for processing
                    if atomic_number.item() not in unique_atomic_numbers:
                        unique_atomic_numbers.append(atomic_number.item())

            avg_num_neighbors = sum(compute_avg_num_neighbors(conformer) for conformer in batch_conformers) / len(batch_conformers)
        
            print(f"Unique Atomic Numbers: {unique_atomic_numbers}")
            print(f"Average number of neighbors: {avg_num_neighbors:.2f}")

            yield batch_conformers, unique_atomic_numbers, avg_num_neighbors, ensemble_id, num_added

        print(f"\nFinished processing Conformer Ensemble: {key}")
        print("=" * 50)

    print(f"\nTotal number of batches processed: {total_batches}")
    print(f"Total number of conformers processed: {total_conformers}")

def move_to_device(obj, device):
    """
    Recursively moves an object to the specified device.

    Args:
        obj: The object to move (can be a tensor, list, tuple, or dict)
        device: The device to move the object to

    Returns:
        The object moved to the specified device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        return obj