"""
EquiCat Model Training Pipeline

This module implements an advanced training pipeline for the EquiCat model, featuring 
cluster-aware contrastive learning, multi-family handling, and comprehensive molecule 
analysis. It includes molecular clustering, loss computation, and detailed training 
progress tracking.

Key components:
1. Training Architecture:
   - Cluster-aware contrastive learning mechanism with adaptive weighting
   - Multi-family molecule handling
   - Advanced embedding processing
   - Hierarchical similarity computation
   - Gradient management and optimization
   - Comprehensive checkpoint system

2. Clustering Features:
   - Family-scoped molecule clustering
   - Hierarchical relationship weighting
   - Cluster-based similarity metrics
   - Adaptive cluster processing
   - Dynamic cluster validation

3. Learning Components:
   - Multi-stage learning rate scheduling
   - Advanced gradient clipping
   - Early stopping with patience
   - Loss computation with adaptive temperature and margin
   - Detailed progress tracking
   - Memory-efficient batch processing

Key Features:
1. Cluster-aware contrastive learning with dynamic adaptation
2. Family-based molecular organization
3. Hierarchical similarity computation
4. Comprehensive logging system
5. Multiple scheduler options
6. Advanced checkpoint management
7. Gradient monitoring and control
8. Memory optimization
9. Multi-family support
10. Cluster-based training with adaptive weighting

Author: Utkarsh Sharma
Version: 4.0.2
Date: 02-08-2025 (MM-DD-YYYY)
License: MIT

Dependencies:
- torch (>=1.9.0)
- numpy (>=1.20.0)
- e3nn (>=0.4.0)
- mace (custom package)
- molli (custom package)
- sklearn (>=0.24.0)

Usage:
    python train.py [--embedding_type {mean_pooling,deep_sets,self_attention,
                    improved_deep_sets,improved_self_attention,all}]
                   [--scheduler {plateau,step,cosine,cosine_restart,onecycle}]
                   [--num_families NUM_FAMILIES]
                   [--ensembles_per_family ENSEMBLES_PER_FAMILY]
                   [--resume_from_checkpoint CHECKPOINT_PATH]

For detailed usage instructions, please refer to the README.md file.

Change Log:
- v4.0.2 (02-08-2025):
  * Improved contrastive loss stability with bounded parameters
  * Enhanced temperature annealing with minimum threshold
  * Added running loss meter for better tracking
  * Adjusted relationship weights for stability
  * Enhanced batch statistics logging
  * Improved gradient handling
  * Added comprehensive loss component analysis
- v4.0.1 (02-03-2025):
  * Fixed contrastive loss stability issues
  * Improved temperature annealing with exponential decay (0.95^epoch)
  * Enhanced margin calculation with training progress awareness
  * Fixed negative relationship handling using magnitude-based scoring
  * Improved class balancing with inverse frequency weights
  * Added robust handling for unpaired samples
  * Enhanced numerical stability in loss computation
  * Added comprehensive loss component logging
- v4.0.0 (12-14-2024):
  * Implemented family-scoped molecule clustering
  * Added initial cluster-aware contrastive loss
  * Added molecular relationship weighting
  * Enhanced batch processing with cluster awareness
  * Added detailed relationship logging
  * Removed conformer padding functionality
  * Restructured training pipeline
  * Removed basic sampling approach
- v3.1.0 (09-11-2024):
  * Added final embeddings saving
  * Enhanced logging capabilities
- v3.0.0 (09-01-2024):
  * Added multi-molecule processing
  * Initial contrastive learning implementation

- v2.0.0 (08-01-2024):
  * Added GPU support
  * Added conformer padding (removed in v4.0.0)

- v1.0.0 (07-01-2024):
  * Initial implementation

ToDo:
- Implement dynamic relationship weighting
- Add support for custom clustering metrics
- Enhance memory efficiency for large datasets
- Implement automated hyperparameter tuning
- Add support for custom relationship definitions
- Enhance visualization of cluster relationships
- Implement distributed training support
- Add comprehensive testing framework
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
from collections import OrderedDict, defaultdict
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from data_loader import MultiFamilyConformerDataset
from conformer_ensemble_embedding_combiner import process_molecule_conformers, move_to_device
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
from molecular_clustering import MolecularClusterProcessor, save_cluster_data
from typing import Dict, Optional, List, Tuple, Any
from io import StringIO

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(42)

# Logger setup
logger = logging.getLogger('train')

# Constants
CONFORMER_LIBRARY_PATHS = {
    # "family1": "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib",
    # "family2": "/Users/utkarsh/MMLI/molli-data/00-libraries/imine_confs.clib",
    # "family3": "/Users/utkarsh/MMLI/molli-data/00-libraries/thiols.clib",
    # "family4": "/Users/utkarsh/MMLI/molli-data/00-libraries/product_confs.clib",

    # "family1": "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib",
    # "family2": "/eagle/FOUND4CHEM/utkarsh/dataset/imine_confs.clib",
    # "family3": "/eagle/FOUND4CHEM/utkarsh/dataset/thiols.clib",
    # "family4": "/eagle/FOUND4CHEM/utkarsh/dataset/product_confs.clib",

    # "family1": "/Users/utkarsh/MMLI/bdsi/catalysts.clib",
    # "family2": "/Users/utkarsh/MMLI/bdsi/substrates.clib",
    # "family3": "/Users/utkarsh/MMLI/bdsi/products.clib",

    "family1": "/eagle/FOUND4CHEM/utkarsh/dataset/bdsi/catalysts.clib",
    "family2": "/eagle/FOUND4CHEM/utkarsh/dataset/bdsi/substrates.clib",
    "family3": "/eagle/FOUND4CHEM/utkarsh/dataset/bdsi/products.clib",
}
# OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/bdsi_large"
# CLUSTERING_RESULTS_DIR = "/Users/utkarsh/MMLI/equicat/src/clustering_results"
OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/bdsi_large"
CLUSTERING_RESULTS_DIR = "/eagle/FOUND4CHEM/utkarsh/project/equicat/src/clustering_results"
SAMPLE_SIZE = 10
MAX_CONFORMERS = 8
CUTOFF = 6.0
LEARNING_RATE = 1e-4
EPOCHS = 500
GRADIENT_CLIP_VALUE = 1.0
CHECKPOINT_INTERVAL = 25
# EXCLUDED_MOLECULES = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']
EXCLUDED_MOLECULES = []

def setup_logging(log_file):
    """
    Set up logging configuration for the trainig.

    Args:
        log_file (str): Path to the log file where messages will be written.

    Returns:
        None
    """
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_lr(optimizer):
    """
    Gets current learning rate from optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer

    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_scheduler(scheduler_type, optimizer, num_epochs, steps_per_epoch):
    """
    Creates learning rate scheduler based on specified type.

    Args:
        scheduler_type (str): Type of scheduler to use
        optimizer (torch.optim.Optimizer): The optimizer to schedule
        num_epochs (int): Total number of epochs
        steps_per_epoch (int): Number of steps per epoch

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler
    """
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=steps_per_epoch)
    elif scheduler_type == 'cosine_restart':
        return CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_type == 'onecycle':
        return OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def calculate_avg_num_neighbors_and_unique_atomic_numbers(dataset, device):
    """
    Calculates average number of neighbors per atom and finds unique atomic numbers.

    Args:
        dataset (MultiFamilyConformerDataset): Dataset to analyze
        device (torch.device): Device to perform calculations on

    Returns:
        Tuple[float, List[int]]: Average neighbors per atom and list of unique atomic numbers
    """
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
    """
    Processes a batch of molecules through the model to generate embeddings.

    Args:
        model (nn.Module): EQUICAT model
        sample (List): Batch of molecule data
        device (torch.device): Device to process on
        embedding_type (str): Type of embedding to generate

    Returns:
        List[Tuple[torch.Tensor, str, str]]: List of (embedding, key, family) tuples
    """
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
    Logs statistics about gradients for model parameters.

    Args:
        model (nn.Module): Model to analyze gradients for

    Returns:
        None
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            logger.info(f"Gradient stats for {name}:")
            logger.info(f"  Norm: {grad_norm:.6f}, Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")

def initialize_clustering(conformer_libraries):
    """
    Initializes molecular clustering before training starts.

    Args:
        conformer_libraries (Dict[str, ConformerLibrary]): Libraries of molecular conformers

    Returns:
        None
    """
    logger.info("Initializing molecular clustering")
    processor = MolecularClusterProcessor(
        conformer_libraries,
        clustering_cutoff=0.2,
        output_dir=CLUSTERING_RESULTS_DIR
    )
    processor.process_all_families()
    logger.info("Molecular clustering completed")

# Configure logger
logger = logging.getLogger(__name__)

class AverageMeter:
    """Tracks running average of a quantity."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ClusterAwareContrastiveLoss:
    """
    Implements cluster-aware contrastive learning with family-scoped clustering and hierarchical relationships.

    Computes loss based on molecule relationships within and across families and clusters,
    using weighted attraction/repulsion between samples.

    Args:
        clustering_results_dir (str): Directory containing clustering results files
        weights (Optional[Dict[str, float]]): Custom weights for different relationship types:
            - same_family_same_cluster: Weight for molecules in same family and cluster 
            - same_family_diff_cluster: Weight for molecules in same family but different clusters
            - diff_family: Weight for molecules from different families
        temperature (float): Temperature parameter for scaling similarities (default: 0.25)

    Returns:
        None
    """
    def __init__(
        self,
        clustering_results_dir: str,
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 0.25,
        max_epochs: int = EPOCHS,
    ):
        self.clustering_results_dir = clustering_results_dir
        self.temperature = temperature
        self.max_epochs = max_epochs
        
        # Default weights with hierarchical structure
        default_weights = {
            'same_family_same_cluster': 1.0,  # Strong attraction within family+cluster
            'same_family_diff_cluster': -0.15,  # Medium attraction within family
            'diff_family': -0.35  # Strong repulsion between families
        }
        
        self.weights = weights if weights is not None else default_weights
        logger.info(f"Initialized contrastive loss with weights: {self.weights}")
        
        # Validate weights
        self._validate_weights()
        
        # Load cluster data
        self.cluster_data = self._load_cluster_data()
        
        # Create family-scoped cluster lookup
        self.molecule_clusters = self._create_family_scoped_clusters()
        
        logger.info(f"Initialized loss with {len(self.molecule_clusters)} molecule mappings")

    def _validate_weights(self):
        """
        Validates weight configuration and logs analysis of relative strengths between relationship types.

        Checks for required weights and validates their relationships to ensure proper hierarchical structure.

        Args: 
            None

        Returns:
            None

        Raises:
            ValueError: If required weights are missing
        """
        required_weights = ['same_family_same_cluster', 'same_family_diff_cluster', 'diff_family']
        for weight_name in required_weights:
            if weight_name not in self.weights:
                raise ValueError(f"Missing required weight: {weight_name}")
        
        logger.info("\nWeight configuration analysis:")
        logger.info(f"Same family+cluster attraction: {self.weights['same_family_same_cluster']:.2f}")
        logger.info(f"Same family attraction: {self.weights['same_family_diff_cluster']:.2f}")
        logger.info(f"Different family repulsion: {self.weights['diff_family']:.2f}")
        
        # Log relative weight strengths
        cluster_to_family_ratio = self.weights['same_family_same_cluster'] / self.weights['same_family_diff_cluster']
        logger.info(f"Cluster vs Family strength ratio: {cluster_to_family_ratio:.2f}")
        
        if self.weights['diff_family'] > 0:
            logger.warning("Different family weight is positive - this may lead to undesired attraction between families")

    def _load_cluster_data(self) -> Dict:
        """
        Loads and validates clustering results from saved files.

        Reads cluster mappings and metadata, validates data structure and format.

        Args:
            None

        Returns:
            Dict: Containing cluster mappings, family data and metadata

        Raises:
            FileNotFoundError: If cluster data file not found
            ValueError: If cluster data format is invalid
        """
        # Try to load cluster data file
        cluster_data_path = os.path.join(self.clustering_results_dir, 'butina', 'cluster_data.pt')
        
        if not os.path.exists(cluster_data_path):
            error_msg = f"Cluster data not found at {cluster_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            data = torch.load(cluster_data_path)
            
            # Validate data structure
            required_keys = ['cluster_mappings', 'family_data', 'metadata']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' in cluster data")
            
            # Log cluster data summary
            logger.info("\nLoaded cluster data summary:")
            logger.info(f"Number of families: {len(data['cluster_mappings'])}")
            
            total_molecules = sum(len(mappings) for mappings in data['cluster_mappings'].values())
            logger.info(f"Total molecules: {total_molecules}")
            
            # Log per-family statistics
            for family, mappings in data['cluster_mappings'].items():
                num_clusters = len(set(info['cluster_id'] for info in mappings.values()))
                logger.info(f"{family}: {len(mappings)} molecules in {num_clusters} clusters")
            
            return data
            
        except Exception as e:
            error_msg = f"Error loading cluster data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _create_family_scoped_clusters(self) -> Dict[str, Dict[str, Any]]:
        """
        Creates mapping of molecules to their family-scoped clusters.

        Generates unique cluster IDs scoped to each family and tracks cluster statistics.

        Args:
            None

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of molecules to their family and cluster information
        """
        molecule_clusters = {}
        family_cluster_counts = defaultdict(int)
        
        for family, family_mappings in self.cluster_data['cluster_mappings'].items():
            # Track clusters per family
            family_clusters = set()
            
            for mol_key, info in family_mappings.items():
                cluster_id = info['cluster_id']
                family_clusters.add(cluster_id)
                
                # Create unique family-scoped cluster ID
                scoped_cluster_id = f"{family}_cluster_{cluster_id}"
                
                full_key = f"{family}_{mol_key}"
                molecule_clusters[full_key] = {
                    'family': family,
                    'cluster_id': cluster_id,
                    'scoped_cluster_id': scoped_cluster_id
                }
            
            family_cluster_counts[family] = len(family_clusters)
        
        # Log family-cluster statistics
        logger.info("\nFamily-Cluster Statistics:")
        for family, count in family_cluster_counts.items():
            logger.info(f"{family}: {count} unique clusters")
            
        return molecule_clusters

    def _compute_relationship_matrix(self, molecules: List[Tuple[str, str]]) -> torch.Tensor:
        """
        Computes relationship matrix between molecules encoding family and cluster relationships.

        Args:
            molecules (List[Tuple[str, str]]): List of (molecule_key, family) tuples

        Returns:
            torch.Tensor: Matrix of relationship weights between all molecule pairs
        """
        n = len(molecules)
        relationships = torch.zeros((n, n))
        
        for i, (key_i, family_i) in enumerate(molecules):
            mol_i = self.molecule_clusters.get(f"{family_i}_{key_i}")
            if mol_i is None:
                continue
                
            for j, (key_j, family_j) in enumerate(molecules):
                if i == j:
                    continue
                    
                mol_j = self.molecule_clusters.get(f"{family_j}_{key_j}")
                if mol_j is None:
                    continue
                
                # Check family relationship
                if mol_i['family'] == mol_j['family']:
                    # Same family - check cluster
                    if mol_i['scoped_cluster_id'] == mol_j['scoped_cluster_id']:
                        relationships[i, j] = self.weights['same_family_same_cluster']
                    else:
                        relationships[i, j] = self.weights['same_family_diff_cluster']
                else:
                    # Different family
                    relationships[i, j] = self.weights['diff_family']
        
        return relationships

    def _log_batch_relationships(self, relationships: torch.Tensor, molecules: List[Tuple[str, str]]):
        """
        Logs detailed analysis of relationships between molecules in current batch.

        Args:
            relationships (torch.Tensor): Matrix of relationship weights
            molecules (List[Tuple[str, str]]): List of (molecule_key, family) tuples

        Returns:
            None
        """
        n = len(molecules)
        counts = {
            'same_family_same_cluster': 0,
            'same_family_diff_cluster': 0,
            'diff_family': 0
        }
        
        logger.info("\nBatch Relationship Analysis:")
        logger.info("Molecule List:")
        for i, (key, family) in enumerate(molecules):
            mol = self.molecule_clusters.get(f"{family}_{key}")
            if mol:
                logger.info(f"  {i}: {key} (Family: {family}, Cluster: {mol['scoped_cluster_id']})")
        
        for i in range(n):
            for j in range(i+1, n):  # Only count unique pairs
                rel_strength = relationships[i, j].item()
                if abs(rel_strength - self.weights['same_family_same_cluster']) < 1e-6:
                    counts['same_family_same_cluster'] += 1
                elif abs(rel_strength - self.weights['same_family_diff_cluster']) < 1e-6:
                    counts['same_family_diff_cluster'] += 1
                elif abs(rel_strength - self.weights['diff_family']) < 1e-6:
                    counts['diff_family'] += 1
        
        logger.info("\nRelationship Statistics:")
        total_pairs = sum(counts.values())
        for rel_type, count in counts.items():
            percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
            logger.info(f"{rel_type}: {count} pairs ({percentage:.1f}%)")

    def __call__(self, sample_embeddings, current_epoch: int) -> torch.Tensor:
        """
        Computes contrastive loss with proper handling of positive and negative relationships.
        
        Args:
            sample_embeddings: List of (embedding, key, family) tuples
            current_epoch: Current epoch number for temperature annealing
        """
        embeddings = []
        molecules = []
        eps = 1e-8

        # Dynamic margin that increases with epochs
        margin = min(0.4, 0.1 + 0.1 * (current_epoch / self.max_epochs))  # Gentler margin increase
        
        # Exponential decay for temperature
        current_temp = max(0.15, self.temperature * (0.95 ** (current_epoch // 15)))  # Slower decay

        for emb, key, family in sample_embeddings:
            embeddings.append(emb)
            molecules.append((key, family))
                
        if len(embeddings) < 2:
            logger.warning("Not enough embeddings for contrastive loss")
            return torch.tensor(0.0, requires_grad=True, device=embeddings[0].device)
                    
        embeddings = torch.stack(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / current_temp
        exp_sim = torch.exp(similarity_matrix)
        
        # Get relationship matrix and create masks
        relationships = self._compute_relationship_matrix(molecules).to(embeddings.device)
        same_cluster_mask = (relationships == self.weights['same_family_same_cluster']).float()
        same_cluster_mask.fill_diagonal_(0)
        other_rels_mask = (relationships < 0).float()
        other_rels_mask.fill_diagonal_(0)
        
        # Compute positive and negative scores
        pos_scores = torch.sum(exp_sim * same_cluster_mask, dim=1)
        neg_scores = torch.sum(exp_sim * other_rels_mask * (-relationships), dim=1)
        
        # Initialize loss tensor
        loss = torch.zeros_like(pos_scores)
        
        # Handle samples with positive pairs
        has_pos = (pos_scores > 0)
        if has_pos.any():
            pos_term = torch.log(pos_scores[has_pos] + eps)
            neg_term = torch.log(1 + neg_scores[has_pos] + margin)
            loss[has_pos] = -(pos_term - neg_term)
        
        # Handle samples without positive pairs
        no_pos = ~has_pos
        if no_pos.any():
            loss[no_pos] = torch.log(1 + neg_scores[no_pos] + margin)
        
        final_loss = loss.mean()
        
        # Enhanced logging
        logger.info("\nStep 1: Training Parameters")
        logger.info(f"Epoch: {current_epoch}/{self.max_epochs}")
        logger.info(f"Current temperature: {current_temp:.4f}")
        logger.info(f"Current margin: {margin:.4f}")
        
        logger.info("\nStep 2: Batch Statistics")
        logger.info(f"Number of molecules in a batch: {len(embeddings)}")
        logger.info(f"Average similarity: {similarity_matrix.mean().item():.4f}")
        
        logger.info("\nStep 3: Relationship Analysis")
        self._log_batch_relationships(relationships, molecules)
        logger.info(f"Samples with positive pairs: {has_pos.sum().item()}")
        logger.info(f"Samples without positive pairs: {no_pos.sum().item()}")
        
        logger.info("\nStep 4: Score Analysis")
        logger.info(f"Positive scores range: [{pos_scores.min().item():.4f}, {pos_scores.max().item():.4f}]")
        logger.info(f"Negative scores range: [{neg_scores.min().item():.4f}, {neg_scores.max().item():.4f}]")
        
        logger.info("\nStep 5: Loss Analysis")
        if has_pos.any():
            logger.info(f"Loss for samples with positives: {loss[has_pos].mean().item():.4f}")
        if no_pos.any():
            logger.info(f"Loss for samples without positives: {loss[no_pos].mean().item():.4f}")
        logger.info(f"Final combined loss: {final_loss.item():.4f}")
        
        return final_loss
    
def train_equicat(model, dataset, device, embedding_type, scheduler_type, args, start_epoch, contrastive_loss_fn):
    """
    Main training loop for EQUICAT model.

    Args:
        model (nn.Module): EQUICAT model
        dataset (MultiFamilyConformerDataset): Training dataset
        device (torch.device): Device to train on
        embedding_type (str): Type of embedding to use
        scheduler_type (str): Type of learning rate scheduler
        args (argparse.Namespace): Command line arguments
        start_epoch (int): Epoch to start/resume from
        contrastive_loss_fn (ClusterAwareContrastiveLoss): Loss function

    Returns:
        nn.Module: Trained model
    """
    checkpoint_dir = f'{OUTPUT_PATH}/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info("Starting train_equicat function")
    
    num_samples = len(dataset)
    steps_per_epoch = num_samples
    logger.info(f"Number of samples: {num_samples}, Steps per epoch: {steps_per_epoch}")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(scheduler_type, optimizer, EPOCHS, steps_per_epoch)
    logger.info("Optimizer and scheduler initialized")

    logger.info(f"Loaded cluster data: {contrastive_loss_fn.cluster_data}")

    best_loss = float('inf')
    best_model = None
    patience = 75
    patience_counter = 0
    logger.info(f"Early stopping patience set to {patience} epochs")

    all_molecule_embeddings = {}
    
    previous_gradients = {}                 # For tracking gradient changes between epochs
    gradient_check_epoch_frequency = 50     # Check every 50 epochs
    gradient_check_samples = 2              # Check 2 random samples per check

    running_loss = AverageMeter()

    for epoch in range(start_epoch, EPOCHS):
        logger.info(f"Starting epoch {epoch+1}/{EPOCHS}")
        model.train()
        epoch_loss = 0.0
        dataset.reset()
        running_loss.reset()  # Reset at start of epoch

        epoch_embeddings = {}

        for sample_idx, sample in enumerate(dataset):
            logger.info(f"Processing sample {sample_idx+1}/{num_samples}")
            optimizer.zero_grad()

            sample_embeddings = process_sample(model, sample, device, embedding_type)
            # loss = compute_contrastive_loss(sample_embeddings)
            loss = contrastive_loss_fn(sample_embeddings, current_epoch=epoch + 1) 
            running_loss.update(loss.item())

            loss.backward()

            if epoch % gradient_check_epoch_frequency == 0 or sample_idx < gradient_check_samples:
                logger.info(f"=== Gradient Analysis (Epoch {epoch+1}, Sample {sample_idx+1}) ===")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_mean = param.grad.mean().item()
                        grad_min = param.grad.min().item()
                        grad_max = param.grad.max().item()
                        grad_std = param.grad.std().item()
                        
                        # Check if we have previous gradients to compare
                        if name in previous_gradients:
                            prev_norm = previous_gradients[name]['norm']
                            norm_change = grad_norm - prev_norm
                            logger.info(f"  {name}: norm={grad_norm:.6f} (Δ={norm_change:.6f}), mean={grad_mean:.6f}, min={grad_min:.6f}, max={grad_max:.6f}, std={grad_std:.6f}")
                        else:
                            logger.info(f"  {name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}, min={grad_min:.6f}, max={grad_max:.6f}, std={grad_std:.6f}")
                        
                        # Store current gradients for next comparison
                        previous_gradients[name] = {'norm': grad_norm, 'mean': grad_mean}
                    else:
                        logger.info(f"  {name}: No gradient")
                logger.info("========================")

            clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            optimizer.step()

            epoch_loss += loss.item()
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Sample [{sample_idx+1}/{num_samples}], "
            f"Loss: {loss.item():.6f}, Running Loss: {running_loss.avg:.6f}, "
            f"LR: {get_lr(optimizer):.8f}")

            for embedding, key, _ in sample_embeddings:
                if key not in epoch_embeddings:
                    epoch_embeddings[key] = []
                epoch_embeddings[key].append(embedding.detach().cpu())

        avg_epoch_loss = epoch_loss / num_samples
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.6f}, "
                     f"Final LR: {get_lr(optimizer):.8f}")

        for key, embeddings in epoch_embeddings.items():
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            if key not in all_molecule_embeddings:
                all_molecule_embeddings[key] = []
            all_molecule_embeddings[key].append(avg_embedding)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_epoch_loss)
        else:
            scheduler.step()

        is_best = False
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model = model.state_dict()
            patience_counter = 0
            is_best = True
            logger.info(f"New best model found with loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in loss. Patience counter: {patience_counter}/{patience}")

        # Save checkpoint every CHECKPOINT_INTERVAL epochs
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or is_best:
            save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, OUTPUT_PATH, is_best)

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    logger.info("Training completed")
    logger.info(f"Best model had a loss of {best_loss:.6f}")
    
    # Load the best model for final embeddings computation
    best_model_path = f'{checkpoint_dir}/best_model.pt'
    logger.info(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    logger.info("Best model loaded successfully")

    # Compute final embeddings using the best model
    logger.info("Computing final embeddings using the best model")

    model.eval()
    final_molecule_embeddings = {}
    total_molecules = sum(len(keys) for keys in dataset.family_keys.values())
    processed_molecules = 0
    total_conformers = 0

    with torch.no_grad():
        for family, keys in dataset.family_keys.items():
            logger.info(f"Processing family: {family}")
            for key in keys:
                atomic_data_list = dataset.get_molecule_data(key, use_all_conformers=True)
                num_conformers = len(atomic_data_list)
                total_conformers += num_conformers
                logger.info(f"Processing molecule {key} from {family} with {num_conformers} conformers")
                
                molecule_embeddings = []
                for conformer_idx, conformer in enumerate(atomic_data_list):
                    conformer = move_to_device(conformer, device)
                    input_dict = {
                        'positions': conformer.positions,
                        'atomic_numbers': conformer.atomic_numbers,
                        'edge_index': conformer.edge_index
                    }
                    output = model(input_dict)
                    molecule_embeddings.append(output)
                    logger.debug(f"  Processed conformer {conformer_idx + 1}/{num_conformers} for molecule {key}")
                
                molecule_embeddings = torch.stack(molecule_embeddings)
                logger.info(f"Stacked embeddings shape for {key}: {molecule_embeddings.shape}")
                
                averaged_embedding = process_molecule_conformers(molecule_embeddings, model.non_linear_readout.irreps_out)
                
                scalar, vector = averaged_embedding[embedding_type]
                if scalar is not None and vector is not None:
                    combined = torch.cat([scalar.view(-1), vector.view(-1)])
                    logger.info(f"Combined embedding shape for {key}: {combined.shape}")
                elif scalar is not None:
                    combined = scalar.view(-1)
                    logger.info(f"Scalar-only embedding shape for {key}: {combined.shape}")
                else:
                    combined = vector.view(-1)
                    logger.info(f"Vector-only embedding shape for {key}: {combined.shape}")
                
                final_molecule_embeddings[f"{family}_{key}"] = combined.cpu().numpy()
                
                processed_molecules += 1
                logger.info(f"Processed {processed_molecules}/{total_molecules} molecules")

    logger.info(f"Computed final embeddings for {len(final_molecule_embeddings)} molecules")
    logger.info(f"Total conformers processed: {total_conformers}")

    # Save the final molecule embeddings
    save_final_embeddings(final_molecule_embeddings, OUTPUT_PATH)
    logger.info("Final embeddings saved successfully")

    return model

def save_final_embeddings(embeddings, output_path):
    """
    Saves final molecule embeddings to JSON file.

    Args:
        embeddings (Dict[str, np.ndarray]): Molecule embeddings
        output_path (str): Directory to save embeddings

    Returns:
        None
    """
    embeddings_file = f'{output_path}/final_molecule_embeddings.json'
    
    serializable_embeddings = {key: emb.tolist() for key, emb in embeddings.items()}
    
    with open(embeddings_file, 'w') as f:
        json.dump(serializable_embeddings, f)
    
    logger.info(f"Final molecule embeddings saved to {embeddings_file}")
    logger.info(f"Number of molecules with saved embeddings: {len(embeddings)}")       

def save_checkpoint(epoch, model, optimizer, scheduler, loss, output_path, is_best=False):
    """
    Saves training checkpoint.

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model state
        optimizer (torch.optim.Optimizer): Optimizer state
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler state
        loss (float): Current loss value
        output_path (str): Directory to save checkpoint
        is_best (bool): Whether this is the best model so far

    Returns:
        None
    """
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    if is_best:
        checkpoint_path = f'{output_path}/checkpoints/best_model.pt'
    else:
        checkpoint_path = f'{output_path}/checkpoints/checkpoint_epoch_{epoch+1}.pt'
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    if epoch + 1 == EPOCHS:
        final_model_path = f'{output_path}/checkpoints/final_model.pt'
        torch.save(checkpoint, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")


def initialize_training(conformer_libraries, clustering_results_dir):
    """
    Initializes clustering and validates results before training begins.

    Args:
        conformer_libraries (Dict[str, str]): Paths to conformer libraries
        clustering_results_dir (str): Directory to save clustering results

    Returns:
        Dict: Clustering metadata
    """
    logger.info("Initializing molecular clustering")
    
    # Create clustering processor
    processor = MolecularClusterProcessor(
        library_paths=conformer_libraries,
        clustering_cutoff=0.2,
        output_dir=clustering_results_dir
    )
    
    # Process families and save results
    processor.process_all_families()
    
    # Save cluster data with new format
    save_cluster_data(processor, clustering_results_dir)
    
    # Validate saved data
    cluster_data_path = os.path.join(clustering_results_dir, 'butina', 'cluster_data.pt')
    if not os.path.exists(cluster_data_path):
        raise FileNotFoundError("Clustering failed to save results")
        
    data = torch.load(cluster_data_path)
    logger.info(f"Clustering completed with {len(data['cluster_mappings'])} families")
    
    # Return clustering metadata for reference
    return data['metadata']

def main(args):
    """
    Main function controlling training pipeline flow.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        None
    """
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

    # Initialize clustering with validation
    clustering_metadata = initialize_training(
        conformer_libraries=CONFORMER_LIBRARY_PATHS,
        clustering_results_dir=CLUSTERING_RESULTS_DIR
    )
    logger.info(f"Clustering initialized with {clustering_metadata['num_families']} families")
    
    # Create contrastive loss with new implementation
    contrastive_loss_fn = ClusterAwareContrastiveLoss(
    clustering_results_dir=CLUSTERING_RESULTS_DIR,
    weights={
        'same_family_same_cluster': 1.0,
        'same_family_diff_cluster': -0.15,
        'diff_family': -0.35
    },
    temperature=0.25,
    max_epochs=EPOCHS
)

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
        "num_interactions": 2,
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

    old_stdout = sys.stdout
    string_buffer = StringIO()
    sys.stdout = string_buffer
    print(model)
    # Restore standard output
    sys.stdout = old_stdout
    # Log the captured output
    model_str = string_buffer.getvalue()
    logger.info("Model Architecture:")
    for line in model_str.split('\n'):
        logger.info(line)

    logger.info(f"Model initialized and moved to {device}")

    start_epoch = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed from epoch {start_epoch}")

    logger.info("Starting training process")
    trained_model = train_equicat(
        model=model, 
        dataset=dataset, 
        device=device, 
        embedding_type=args.embedding_type, 
        scheduler_type=args.scheduler, 
        args=args, 
        start_epoch=start_epoch,
        contrastive_loss_fn=contrastive_loss_fn 
    )
    logger.info("Training process completed")

    # Final model is already saved in train_equicat function

    logger.info("Embedding generation process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EQUICAT Model Training")
    parser.add_argument('--embedding_type', type=str, default='improved_self_attention',
                        choices=['mean_pooling', 'deep_sets', 'self_attention', 'improved_deep_sets', 'improved_self_attention', 'all'],
                        help='Type of embedding to use for loss computation')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--scheduler', type=str, default='cosine_restart',
                        choices=['plateau', 'step', 'cosine', 'cosine_restart', 'onecycle'],
                        help='Type of learning rate scheduler to use')
    parser.add_argument('--num_families', type=int, default=None, help='Number of molecule families to use')
    parser.add_argument('--ensembles_per_family', type=int, default=None, help='Number of ensembles to use per family')
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    total_runtime = end_time - start_time
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")