"""
EquiCat Molecule Data Loader and Processor

This module provides comprehensive functionality for loading and processing molecular conformer data
for use with the EquiCat model. It implements multi-family support, efficient sampling strategies,
and performance profiling, with GPU acceleration and advanced tracing capabilities.

Key components:
1. MultiFamilyConformerDataset: Advanced dataset class for handling multiple conformer families
2. Smart Sampling: Cross-family sampling mechanism for contrastive learning
3. K-means Conformer Selection: Intelligent selection of diverse conformers 
4. Performance Profiling: Built-in PyTorch profiling and tracing
5. Family-based Organization: Structured handling of molecular families
6. Comprehensive Logging: Detailed tracking of data processing
7. Memory Management: Efficient handling of large molecular datasets
8. Chrome Tracing: Performance visualization and analysis tools

Key Features:
1. Multi-family data handling with balanced sampling
2. Performance optimization with PyTorch profiling 
3. K-means based conformer selection
4. Comprehensive logging and error handling
5. GPU acceleration support
6. Cross-family contrastive learning support
7. Efficient memory management
8. Flexible molecule exclusion
9. Advanced profiling and tracing
10. Chrome trace export for performance analysis

Author: Utkarsh Sharma
Version: 4.0.0
Date: 12-14-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
- torch (>=1.9.0)
- numpy (>=1.20.0)
- molli (>=0.1.0)
- mace (custom package)
- torch_geometric (>=2.0.0)
- sklearn (>=0.24.0)
- torch.profiler
- json

Usage:
   from data_loader import MultiFamilyConformerDataset
   
   # Initialize conformer libraries
   conformer_libraries = {
       "family1": ml.ConformerLibrary(path1),
       "family2": ml.ConformerLibrary(path2)
   }
   
   # Create dataset with multiple families
   dataset = MultiFamilyConformerDataset(
       conformer_libraries=conformer_libraries,
       cutoff=CUTOFF,
       sample_size=SAMPLE_SIZE,
       max_conformers=MAX_CONFORMERS,
       exclude_molecules=EXCLUDED_MOLECULES
   )
   
   # Iterate through samples
   for sample in dataset:
       # Process sample data for training

For detailed usage instructions, please refer to the README.md file.

Change Log:
- v4.0.0 (12-14-2024):
 * Major architectural change to support multiple molecular families
 * Implemented PyTorch profiling and tracing for performance monitoring
 * Added cross-family sampling for contrastive learning
 * Enhanced logging with comprehensive event tracking
 * Improved conformer selection with K-means clustering
 * Removed conformer padding functionality
 * Changed from single to multi-library handling
 * Restructured dataset class for family-based organization
 * Added performance profiling with Chrome trace export
- v3.0.0 (09-10-2024):
 * Added conformer capping with K-means selection
 * Implemented sample-based processing
 * Introduced direct molecule handling
 * Updated logging for better tracking
- v2.0.0 (08-01-2024):
 * Added GPU support and optimization
 * Enhanced memory management
 * Added conformer padding capability (removed in v4.0.0)
- v1.0.0 (07-01-2024):
 * Initial release with basic conformer handling

ToDo:
- Implement adaptive sampling strategies for better family balance
- Add support for custom conformer selection methods
- Optimize memory usage for very large molecular systems
- Add caching mechanism for frequently accessed data
- Implement advanced data augmentation techniques
- Add comprehensive test suite for all components
- Enhance profiling visualization tools
- Support for parallel data loading
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
from torch.profiler import profile, record_function, ProfilerActivity
import json

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(42)

# Logger setup
logger = logging.getLogger('data_loader')

# Constants
CUTOFF = 6.0
MAX_CONFORMERS = 10
SAMPLE_SIZE = 30
LOG_FILE = "/eagle/FOUND4CHEM/utkarsh/project/equicat/epoch_large/data_loader.log"
PROFILE_OUTPUT_FILE = "/eagle/FOUND4CHEM/utkarsh/project/equicat/epoch_large/profiler_output.json"
# LOG_FILE = "/Users/utkarsh/MMLI/equicat/epoch_large/data_loader_profiler.log"
# PROFILE_OUTPUT_FILE = "/Users/utkarsh/MMLI/equicat/epoch_large/profiler_output.json"

def setup_logging(log_file):
    """
    Set up logging configuration for the data loader.

    Args:
        log_file (str): Path to the log file where messages will be written.

    Returns:
        None
    """
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

setup_logging(LOG_FILE)
logger.info("Logging initialized for data_loader")

class MultiFamilyConformerDataset(Dataset):
    """
    A custom dataset class for handling multiple families of molecular conformers.
    Implements sample-based processing and supports contrastive learning across families.

    Args:
        conformer_libraries (Dict[str, ml.ConformerLibrary]): Dictionary mapping family names to conformer libraries.
        cutoff (float): Cutoff distance for atomic interactions.
        sample_size (int): Number of molecules per sample.
        max_conformers (int): Maximum number of conformers to select per molecule.
        exclude_molecules (Optional[List[str]]): List of molecule keys to exclude.
        num_families (Optional[int]): Number of families to use (if None, uses all).
        ensembles_per_family (Optional[int]): Number of molecules to use per family (if None, uses all).

    Returns:
        None
    """
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
        with record_function("MultiFamilyConformerDataset.__init__"):
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
        Preselect diverse conformers for each molecule using K-means clustering.
        Improves efficiency by doing selection once during initialization.

        Args:
            None (uses class attributes)

        Returns:
            Dict[str, List[int]]: Dictionary mapping molecule keys to lists of selected conformer indices.
        """
        with record_function("MultiFamilyConformerDataset._preselect_conformers"):
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
        Select diverse conformers from a set using K-means clustering.
        Chooses conformers closest to cluster centers.

        Args:
            conformers (List[np.ndarray]): List of conformer coordinates.
            max_conformers (int): Maximum number of conformers to select.

        Returns:
            List[int]: Indices of selected diverse conformers.
        """
        with record_function("MultiFamilyConformerDataset._select_diverse_conformers"):
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

    def get_molecule_data(self, key, use_all_conformers=False):
        """
        Get atomic data for a specific molecule.

        Args:
            key (str): Molecule identifier.
            use_all_conformers (bool): If True, uses all conformers instead of preselected ones.

        Returns:
            List[Data]: List of PyTorch Geometric Data objects for each conformer.
        """
        with record_function("MultiFamilyConformerDataset.get_molecule_data"):
            family = self.molecule_to_family[key]
            library = self.conformer_libraries[family]
            with library.reading():
                conformer = library[key]
                coords = torch.tensor(conformer.coords, dtype=torch.float64)
                atomic_numbers = torch.tensor([atom.element for atom in conformer.atoms], dtype=torch.long)
                z_table = tools.AtomicNumberTable(torch.unique(atomic_numbers).tolist())

                if use_all_conformers:
                    indices = range(len(coords))
                else:
                    indices = self.selected_conformers[key]

                atomic_data_list = []
                for i in indices:
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
                        key=key,
                        family=family
                    )
                    atomic_data_list.append(torch_geo_data)
            return atomic_data_list
    
    def _sample_across_families(self) -> List[str]:
        """
        Sample molecules across different families for balanced representation.
        Ensures each sample contains molecules from multiple families.

        Args:
            None (uses class attributes)

        Returns:
            List[str]: List of selected molecule keys.
        """
        with record_function("MultiFamilyConformerDataset._sample_across_families"):
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
        Get the next sample of molecules for processing.

        Args:
            None (uses class state)

        Returns:
            Optional[List[Tuple[List[Data], str, str]]]: 
                List of tuples containing (atomic_data_list, molecule_key, family_name),
                or None if all samples have been processed.
        """
        with record_function("MultiFamilyConformerDataset.get_next_sample"):
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
        Reset the dataset for a new epoch, shuffling family keys.

        Args:
            None

        Returns:
            None
        """
        with record_function("MultiFamilyConformerDataset.reset"):
            logger.info("Resetting dataset for a new epoch")
            self.current_sample = 0
            # Shuffle the keys within each family
            for family in self.family_keys:
                random.shuffle(self.family_keys[family])

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self):
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
        batch (Data): PyTorch Geometric batch object containing edge indices.

    Returns:
        float: Average number of neighbors per atom.
    """
    with record_function("compute_avg_num_neighbors"):
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        avg_num_neighbors = torch.mean(counts.float())
        return avg_num_neighbors.item()

def get_unique_atomic_numbers(conformer_libraries: Dict[str, ml.ConformerLibrary]) -> List[int]:
    """
    Get a sorted list of unique atomic numbers across all conformer libraries.

    Args:
        conformer_libraries (Dict[str, ml.ConformerLibrary]): Dictionary of conformer libraries.

    Returns:
        List[int]: Sorted list of unique atomic numbers.
    """
    with record_function("get_unique_atomic_numbers"):
        unique_atomic_numbers = set()
        for family, library in conformer_libraries.items():
            with library.reading():
                for key in library.keys():
                    conformer = library[key]
                    unique_atomic_numbers.update([atom.element for atom in conformer.atoms])
        return sorted(list(unique_atomic_numbers))

def main():
    """
    Main function to demonstrate and test the data loader functionality.
    Sets up profiling, initializes dataset, and processes test samples.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting main function for testing data_loader.py")

    conformer_libraries = {

        # "family1": ml.ConformerLibrary("/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"),
        # "family2": ml.ConformerLibrary("/Users/utkarsh/MMLI/molli-data/00-libraries/imine_confs.clib"),
        # "family3": ml.ConformerLibrary("/Users/utkarsh/MMLI/molli-data/00-libraries/thiols.clib"),
        # "family4": ml.ConformerLibrary("/Users/utkarsh/MMLI/molli-data/00-libraries/product_confs.clib"),

        "family1": ml.ConformerLibrary("/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib"),
        "family2": ml.ConformerLibrary("/eagle/FOUND4CHEM/utkarsh/dataset/imine_confs.clib"),
        "family3": ml.ConformerLibrary("/eagle/FOUND4CHEM/utkarsh/dataset/thiols.clib"),
        "family4": ml.ConformerLibrary("/eagle/FOUND4CHEM/utkarsh/dataset/product_confs.clib"),
    }
    
    excluded_molecules = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 use_cuda=torch.cuda.is_available()) as prof:
        
        with record_function("dataset_initialization"):
            dataset = MultiFamilyConformerDataset(
                conformer_libraries=conformer_libraries,
                cutoff=CUTOFF,
                sample_size=SAMPLE_SIZE,
                max_conformers=MAX_CONFORMERS,
                exclude_molecules=excluded_molecules
            )

        logger.info("Testing dataset iteration")
        for i, sample in enumerate(dataset):
            with record_function(f"process_sample_{i}"):
                logger.info(f"Processing sample {i+1}/{len(dataset)}")
                families_in_sample = set()
                for j, (atomic_data_list, key, family) in enumerate(sample):
                    with record_function(f"process_batch_{j}"):
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
                            with record_function(f"process_conformer_{k}"):
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

            if i >= 83:  # Process only first 5 samples for this test
                break

        with record_function("get_unique_atomic_numbers"):
            unique_atomic_numbers = get_unique_atomic_numbers(conformer_libraries)
            logger.info(f"Unique atomic numbers across all families: {unique_atomic_numbers}")

    logger.info("Data loader test completed successfully")

    # Export profiler results to JSON
    prof.export_chrome_trace(PROFILE_OUTPUT_FILE)
    logger.info(f"Profiler results exported to {PROFILE_OUTPUT_FILE}")

    # Print profiler summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Run the main function
    main()