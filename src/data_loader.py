"""
data_loader.py

This module provides a custom dataset class for handling multiple families of molecular conformers.
It is designed to work with the EQUICAT model and supports contrastive learning by providing
samples that include molecules from different families.

Key features:
1. Supports multiple conformer libraries (molecule families)
2. Processes all molecules within each family
3. Implements efficient sampling across all molecule families for each batch
4. Selects diverse conformers using K-means clustering
5. Provides family information for each molecule, enabling contrastive learning
6. Supports exclusion of specific molecules
7. Implements batch sampling for training
8. Comprehensive logging for debugging and monitoring

Author: Utkarsh Sharma
Version: 1.0.0
Date: 10-03-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
- torch
- numpy
- molli
- mace
- torch_geometric
- sklearn
- logging
"""

import random
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
import molli as ml
from mace import data, tools
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger('data_loader')

# Constants
CUTOFF = 6.0
MAX_CONFORMERS = 5
SAMPLE_SIZE = 10
LOG_FILE = "/Users/utkarsh/MMLI/equicat/develop_op/data_loader.log"

def setup_logging(log_file):
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class MultiFamilyConformerDataset(Dataset):
    def __init__(
        self,
        conformer_libraries: Dict[str, ml.ConformerLibrary],
        cutoff: float,
        sample_size: int,
        max_conformers: int,
        exclude_molecules: Optional[List[str]] = None,
        num_families: Optional[int] = None,
        ensembles_per_family: Optional[int] = None
    ):
        logger.info("Initializing MultiFamilyConformerDataset")
        self.conformer_libraries = conformer_libraries
        self.cutoff = cutoff
        self.sample_size = sample_size
        self.max_conformers = max_conformers

        self.family_keys = defaultdict(list)
        self.total_molecules = 0
        self.molecule_to_family = {}

        # Limit the number of families if specified
        families = list(conformer_libraries.keys())
        if num_families is not None:
            families = families[:num_families]
        logger.info(f"Processing {len(families)} families")
        
        # Process each family and its molecules
        for family in families:
            logger.info(f"Processing family: {family}")
            library = conformer_libraries[family]
            with library.reading():
                family_keys = list(library.keys())
                logger.info(f"Total molecules in {family}: {len(family_keys)}")
                if exclude_molecules:
                    family_keys = [key for key in family_keys if key not in exclude_molecules]
                    logger.info(f"Molecules after exclusion: {len(family_keys)}")
                if ensembles_per_family is not None:
                    family_keys = family_keys[:ensembles_per_family]
                    logger.info(f"Molecules after limiting to {ensembles_per_family}: {len(family_keys)}")
                self.family_keys[family] = family_keys
                self.total_molecules += len(family_keys)
                for key in family_keys:
                    self.molecule_to_family[key] = family
            logger.info(f"Family {family}: {len(family_keys)} molecules")

        self.all_keys = [key for keys in self.family_keys.values() for key in keys]
        
        logger.info("Preselecting diverse conformers")
        self.selected_conformers = self._preselect_conformers()
        logger.info("Conformer preselection completed")
        
        # Calculate total number of samples
        self.total_samples = self.total_molecules // sample_size
        self.current_sample = 0

        logger.info(f"Dataset initialized with {self.total_molecules} molecules from {len(self.family_keys)} families")
        logger.info(f"Total samples: {self.total_samples}, Sample size: {sample_size}")

    def _preselect_conformers(self) -> Dict[str, List[int]]:
        """
        Preselect diverse conformers for each molecule in the dataset.
    
        This method iterates through all molecules in all families and calls
        _select_diverse_conformers for each molecule to select a diverse subset
        of conformers.

        Returns:
            Dict[str, List[int]]: Dictionary mapping molecule keys to lists of 
            selected conformer indices.
        """
        logger.info("Preselecting diverse conformers")
        selected_conformers = {}
        for family, library in self.conformer_libraries.items():
            for key in self.family_keys[family]:
                with library.reading():
                    conformer = library[key]
                    coords = conformer.coords
                    selected_indices = self._select_diverse_conformers(coords, self.max_conformers)
                    selected_conformers[key] = selected_indices
                logger.info(f"Selected {len(selected_indices)} conformers for molecule {key} from {family}")
        logger.info("Conformer preselection completed")
        return selected_conformers

    @staticmethod
    def _select_diverse_conformers(conformers: List[np.ndarray], max_conformers: int) -> List[int]:
        """
        Select diverse conformers for a single molecule using K-means clustering.

        This method applies K-means clustering to the conformers of a single molecule
        to select a diverse subset. If the number of conformers is less than or equal
        to max_conformers, all conformers are selected.

        Args:
            conformers (List[np.ndarray]): List of conformer coordinates for a single molecule.
            max_conformers (int): Maximum number of conformers to select.

        Returns:
            List[int]: Indices of selected diverse conformers.
        """
        n_conformers = len(conformers)
        if n_conformers <= max_conformers:
            return list(range(n_conformers))
        
        flattened_conformers = [conf.flatten() for conf in conformers]
        kmeans = KMeans(n_clusters=max_conformers, random_state=42)
        kmeans.fit(flattened_conformers)
        
        selected_indices = []
        for cluster in range(max_conformers):
            cluster_conformers = [i for i, label in enumerate(kmeans.labels_) if label == cluster]
            cluster_center = kmeans.cluster_centers_[cluster]
            distances = [np.linalg.norm(flattened_conformers[i] - cluster_center) for i in cluster_conformers]
            selected_indices.append(cluster_conformers[np.argmin(distances)])
        
        return selected_indices

    def get_molecule_data(self, key):
        family = self.molecule_to_family[key]
        library = self.conformer_libraries[family]
        with library.reading():
            conformer = library[key]
            coords = torch.tensor(conformer.coords, dtype=torch.float32)
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
                    x=torch.tensor(atomic_data.node_attrs, dtype=torch.float32),
                    positions=atomic_data.positions,
                    edge_index=atomic_data.edge_index,
                    atomic_numbers=atomic_numbers,
                    key=key,
                    family=family
                )
                atomic_data_list.append(torch_geo_data)
        return atomic_data_list
    
    def _sample_across_families(self) -> List[str]:
        """
        Samples molecules from different families for a batch.
        
        Returns:
            List[str]: List of molecule keys sampled from various families.
        """
        sampled_molecules = []
        families = list(self.family_keys.keys())
        family_keys = {family: self.family_keys[family].copy() for family in families}
        
        while len(sampled_molecules) < self.sample_size:
            for family in families:
                if family_keys[family] and len(sampled_molecules) < self.sample_size:
                    sampled_molecules.append(family_keys[family].pop(random.randint(0, len(family_keys[family]) - 1)))
            if not any(family_keys.values()):  # If all families are empty, break
                break
        
        # If we couldn't fill the batch, we'll just repeat some molecules
        while len(sampled_molecules) < self.sample_size:
            sampled_molecules.append(random.choice(self.all_keys))
        
        # Shuffle to ensure random order in the batch
        random.shuffle(sampled_molecules)
        return sampled_molecules

    def get_next_sample(self) -> Optional[List[Tuple[List[Data], str, str]]]:
        """
        Get the next sample of molecules.

        Returns:
            Optional[List[Tuple[List[Data], str, str]]]: List of (atomic_data_list, key, family) tuples for each molecule in the sample,
                                                         or None if all samples have been processed.
        """
        if self.current_sample >= self.total_samples:
            logger.info("All samples processed, returning None")
            return None

        sample_keys = self._sample_across_families()
        sample_data = []
        
        logger.info(f"Processing sample {self.current_sample + 1}/{self.total_samples}")
        for key in sample_keys:
            family = self.molecule_to_family[key]
            library = self.conformer_libraries[family]
            with library.reading():
                conformer = library[key]
                coords = torch.tensor(conformer.coords, dtype=torch.float32)
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
                        x=torch.tensor(atomic_data.node_attrs, dtype=torch.float32),
                        positions=atomic_data.positions,
                        edge_index=atomic_data.edge_index,
                        atomic_numbers=atomic_numbers,
                        key=key,
                        family=family
                    )
                    atomic_data_list.append(torch_geo_data)
                sample_data.append((atomic_data_list, key, family))
                logger.info(f"Processed molecule {key} from {family} with {len(atomic_data_list)} conformers")

        self.current_sample += 1
        logger.info(f"Sample {self.current_sample} processed with {len(sample_data)} molecules")
        return sample_data

    def reset(self):
        """
        Reset the dataset for a new epoch.
        """
        logger.info("Resetting dataset for a new epoch")
        self.current_sample = 0
        # Shuffle the keys within each family
        for family in self.family_keys:
            random.shuffle(self.family_keys[family])

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.total_samples

    def __iter__(self):
        """
        Iterator for the dataset.

        Yields:
            List[Tuple[List[Data], str, str]]: Next sample of molecules.
        """
        self.reset()
        while True:
            sample = self.get_next_sample()
            if sample is None:
                break
            yield sample

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

def get_unique_atomic_numbers(conformer_libraries: Dict[str, ml.ConformerLibrary]) -> List[int]:
    """
    Get a list of unique atomic numbers across all molecules in all families.

    Args:
        conformer_libraries (Dict[str, ml.ConformerLibrary]): Dictionary of conformer libraries for each family.

    Returns:
        List[int]: List of unique atomic numbers.
    """
    unique_atomic_numbers = set()
    for family, library in conformer_libraries.items():
        with library.reading():
            for key in library.keys():
                conformer = library[key]
                unique_atomic_numbers.update([atom.element for atom in conformer.atoms])
    return sorted(list(unique_atomic_numbers))

def main():
    """
    Main function to demonstrate and test the MultiFamilyConformerDataset.
    """
    logger.info("Starting main function for testing data_loader.py")

    # Example usage with actual data
    conformer_libraries = {
        "family1": ml.ConformerLibrary("/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"),
        "family2": ml.ConformerLibrary("/Users/utkarsh/MMLI/molli-data/00-libraries/molnet.clib"),
    }
    
    excluded_molecules = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']
    
    dataset = MultiFamilyConformerDataset(
        conformer_libraries=conformer_libraries,
        cutoff=CUTOFF,
        sample_size=SAMPLE_SIZE,
        max_conformers=MAX_CONFORMERS,
        exclude_molecules=excluded_molecules
    )

    logger.info("Testing dataset iteration")
    for i, sample in enumerate(dataset):
        logger.info(f"Processing sample {i+1}/{len(dataset)}")
        families_in_sample = set()
        for j, (atomic_data_list, key, family) in enumerate(sample):
            families_in_sample.add(family)
            logger.info(f"\nProcessing batch {j+1}")
            logger.info(f"  Ensemble ID: {j}")
            logger.info(f"  Ensemble Key: {key}")
            logger.info(f"  Family: {family}")
            logger.info(f"  Batch size: {len(atomic_data_list)}")
            
            unique_atomic_numbers = set()
            total_neighbors = 0
            total_atoms = 0
            
            for k, conformer in enumerate(atomic_data_list):
                unique_atomic_numbers.update(conformer.atomic_numbers.tolist())
                total_neighbors += conformer.num_edges
                total_atoms += conformer.num_nodes
                
                logger.info(f"  Conformer {k+1} details:")
                logger.info(f"    Number of atoms: {conformer.num_nodes}")
                logger.info(f"    Number of edges: {conformer.num_edges}")
                logger.info(f"    Atomic numbers: {conformer.atomic_numbers.tolist()}")
                logger.info(f"    Positions shape: {conformer.positions.shape}")
                logger.info(f"    Edge index shape: {conformer.edge_index.shape}")
                
                avg_neighbors = compute_avg_num_neighbors(conformer)
                logger.info(f"    Average neighbors: {avg_neighbors:.2f}")
            
            logger.info(f"  Unique atomic numbers: {sorted(list(unique_atomic_numbers))}")
            logger.info(f"  Overall average number of neighbors: {total_neighbors/total_atoms:.2f}")
        
        logger.info(f"Families in this sample: {', '.join(families_in_sample)}")

        if i >= 5:  # Process only first 5 samples for this test
            break

    unique_atomic_numbers = get_unique_atomic_numbers(conformer_libraries)
    logger.info(f"Unique atomic numbers across all families: {unique_atomic_numbers}")

    logger.info("Data loader test completed successfully")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Run the main function
    main()