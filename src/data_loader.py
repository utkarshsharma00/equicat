"""
Conformer Data Loader and Processor for EQUICAT

This module provides comprehensive functionality for loading and processing conformer data
for use with the EQUICAT model. It includes a custom dataset class, data loading utilities, 
and efficient processing functions.

Key components:
1. ConformerDataset: A custom PyTorch dataset class for handling conformer ensembles.
2. compute_avg_num_neighbors: Utility function to calculate average neighbors in a batch.
3. custom_collate: Custom collation function for batching data.
4. process_data: Generator function for processing conformer data in batches.

This module is optimized to work seamlessly with the MACE framework and PyTorch Geometric,
providing efficient and scalable data handling for molecular conformer analysis.

Important Note:
During sanity checking, it was discovered that the edge_index remained the same even when
the positions (xyz coordinates) of the conformers were changing. This was a critical error
as the edge_index should reflect the unique connectivity of each conformer based on its
specific atomic positions and the cutoff. The issue was resolved by using MACE's 
AtomicData.from_config method to generate unique edge_index for each conformer, 
ensuring that the connectivity is correctly updated for each set of atomic positions.

Fix Implementation:
1. In the ConformerDataset.__getitem__ method, we now use data.AtomicData.from_config
   to create atomic data for each conformer separately.
2. The edge_index is now directly obtained from the atomic_data object for each conformer,
   ensuring unique connectivity information for each set of atomic positions.

Author: Utkarsh Sharma
Version: 1.2.0
Date: 07-24-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - molli (>=0.1.0)
    - mace (custom package)
    - torch_geometric (>=2.0.0)

Usage:
    from data_loader import ConformerDataset, process_data
    
    dataset = ConformerDataset(conformer_ensemble, cutoff)
    for batch_data in process_data(dataset, batch_size=32):
        # Process batch_data

For detailed usage instructions, please refer to the README.md file.

Change Log:
    - v1.2.0: Fixed critical edge_index generation issue, ensuring unique connectivity for each conformer
    - v1.1.0: Added ensemble_id to process_data output
    - v1.0.0: Initial release
"""

import torch
import molli as ml
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
            print(f"Coords shape: {coords.shape}")  # Should be (num_conformers, num_atoms, 3)

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

def process_data(conformer_dataset, batch_size=32):
    """
    Process conformer data in batches.

    Args:
        conformer_dataset: The ConformerDataset to process.
        batch_size (int): Number of conformers to process in each batch.

    Yields:
        tuple: Batch of conformers, unique atomic numbers, average number of neighbors and ensemble id.
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

        for i in range(0, num_conformers, batch_size):
            batch_conformers = atomic_data_list[i:i+batch_size]
            total_batches += 1

            print(f"\nBatch {total_batches} in Ensemble: {key}")
            print(f"Number of conformers in this batch: {len(batch_conformers)}")
            
            #! Sanity checks
            # for j, conformer in enumerate(batch_conformers):
            #     print(f"Conformer {j} positions shape: {conformer.positions.shape}")
            #     print(f"Conformer {j} positions:\n {conformer.positions}")

            unique_atomic_numbers = []
            for conformer in batch_conformers:
                for atomic_number in conformer.atomic_numbers:
                    if atomic_number.item() not in unique_atomic_numbers:
                        unique_atomic_numbers.append(atomic_number.item())

            avg_num_neighbors = sum(compute_avg_num_neighbors(conformer) for conformer in batch_conformers) / len(batch_conformers)
        
            print(f"Unique Atomic Numbers: {unique_atomic_numbers}")
            print(f"Average number of neighbors: {avg_num_neighbors:.2f}")

            yield batch_conformers, unique_atomic_numbers, avg_num_neighbors, ensemble_id

        print(f"\nFinished processing Conformer Ensemble: {key}")
        print("=" * 50)

    print(f"\nTotal number of batches processed: {total_batches}")
    print(f"Total number of conformers processed: {total_conformers}")