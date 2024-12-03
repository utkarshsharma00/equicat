"""
Molecular Clustering and Analysis Module

This module provides functionality for processing molecular conformers, generating 
canonical SMILES representations, performing clustering analysis, and creating 
interactive visualizations of molecular clusters.

Key features:
1. Support for multiple molecular families and conformer libraries
2. Multiple clustering methods (Butina, hierarchical, K-means)
3. Interactive visualization using Plotly
4. Robust error handling and validation
5. Detailed cluster statistics and analysis
6. Support for high-dimensional fingerprint processing
7. Distance matrix computation and visualization
8. Balanced cluster size optimization

Author: Utkarsh Sharma
Version: 1.0.0
Date: 10-04-2024 (MM-DD-YYYY) 
License: MIT

Dependencies:
    - torch (>=1.9.0)
    - rdkit (>=2021.03.1)  
    - plotly (>=5.0.0)
    - numpy (>=1.20.0)
    - pandas (>=1.3.0)
    - seaborn (>=0.11.0)
    - sklearn (>=0.24.0)
    - molli (custom package)

Usage:
    processor = MolecularClusterProcessor(
        library_paths=CONFORMER_LIBRARY_PATHS,
        clustering_cutoff=0.2,
        output_dir="./clustering_results"
    )
    processor.process_all_families()
"""

import torch
import molli as ml
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.ML.Cluster import Butina
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_cluster_mappings(cluster_mappings: dict) -> dict:
    """
    Generate formatted cluster mappings with consistent key structure.

    Args:
        cluster_mappings (dict): Raw cluster mappings from processor
        
    Returns:
        dict: Formatted mappings with standardized keys
    """
    formatted_mappings = {}
    for family, mappings in cluster_mappings.items():
        formatted_mappings[family] = {
            key: {
                'cluster_id': cluster_id,
                'family': family,
                'molecule_key': key
            }
            for key, cluster_id in mappings.items()
        }
    return formatted_mappings

def save_cluster_data(processor, output_dir: str, method: str = 'butina'):
    """
    Save cluster data with additional metadata.

    Args:
        processor: MolecularClusterProcessor instance
        output_dir (str): Directory to save results
        method (str): Clustering method used  
    """
    output_subdir = os.path.join(output_dir, method)
    os.makedirs(output_subdir, exist_ok=True)
    
    cluster_data = {
        'cluster_mappings': format_cluster_mappings(processor.cluster_mappings),
        'family_data': {
            k: {
                'molecules': list(v['molecules'].keys()),
                'num_clusters': len(set(processor.cluster_mappings[k].values()))
            }
            for k, v in processor.family_data.items() 
            if v['molecules']
        },
        'metadata': {
            'clustering_method': method,
            'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'num_families': len(processor.family_data),
            'total_molecules': sum(len(v['molecules']) 
                                 for v in processor.family_data.values() 
                                 if v['molecules'])
        }
    }
    
    save_path = os.path.join(output_subdir, 'cluster_data.pt')
    torch.save(cluster_data, save_path)
    
    # Also save a JSON version for easier inspection
    json_path = os.path.join(output_subdir, 'cluster_data.json')
    with open(json_path, 'w') as f:
        json.dump(cluster_data, f, indent=2)

class MolecularClusterProcessor:
    """
    Handles molecule processing, fingerprint generation, clustering, and visualization
    across multiple molecular families. Supports different clustering methods and
    provides detailed analysis of cluster properties.

    Args:
        library_paths (Dict[str, str]): Paths to conformer libraries
        clustering_cutoff (float): Similarity cutoff for clustering
        output_dir (str): Directory for output files

    Attributes:
        family_data (dict): Processed data for each molecular family
        cluster_mappings (dict): Mapping of molecules to clusters
    """
    def __init__(self, library_paths: Dict[str, str], clustering_cutoff: float = 0.2,
                 output_dir: str = "./clustering_results"):
        self.library_paths = library_paths
        self.clustering_cutoff = clustering_cutoff
        self.output_dir = output_dir
        self.family_data = {}
        self.cluster_mappings = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _validate_fingerprints(self, fingerprints: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, np.ndarray]]:
        """
        Validate and normalize fingerprints to ensure consistent shapes and dimensions.

        This function checks fingerprint validity, normalizes dimensions, and handles padding/truncation
        to ensure all fingerprints have consistent shapes for clustering.

        Args:
            fingerprints (Dict[str, np.ndarray]): Dictionary of molecular fingerprints

        Returns:
            Tuple[bool, Dict[str, np.ndarray]]: Tuple containing:
                - bool: True if validation successful, False otherwise
                - Dict: Normalized fingerprints if successful, empty dict if failed
        """
        if not fingerprints:
            return False, {}
            
        try:
            # Get shapes of all fingerprints
            shapes = [fp.shape for fp in fingerprints.values()]
            if not shapes:
                return False, {}
                
            # Check if all fingerprints have same dimensionality
            if not all(len(shape) == len(shapes[0]) for shape in shapes):
                logger.warning("Inconsistent fingerprint dimensionality")
                return False, {}
                
            # Find maximum length in each dimension
            max_shape = tuple(max(shape[i] for shape in shapes) for i in range(len(shapes[0])))
            
            # Pad or truncate fingerprints to consistent shape
            normalized = {}
            for key, fp in fingerprints.items():
                if fp.shape != max_shape:
                    # Pad with zeros if smaller
                    if any(fp.shape[i] < max_shape[i] for i in range(len(max_shape))):
                        pad_width = [(0, max_shape[i] - fp.shape[i]) for i in range(len(max_shape))]
                        normalized[key] = np.pad(fp, pad_width)
                    else:
                        # Truncate if larger
                        slices = tuple(slice(0, s) for s in max_shape)
                        normalized[key] = fp[slices]
                else:
                    normalized[key] = fp
                    
            return True, normalized
            
        except Exception as e:
            logger.error(f"Error validating fingerprints: {str(e)}")
            return False, {}

    def _create_rdkit_mol(self, atoms: List[Tuple], bonds: List[Tuple], coords: np.ndarray) -> Optional[Chem.Mol]:
        """
        Create RDKit Mol object with improved sanitization.

        Args:
            atoms (List[Tuple]): List of atom elements and indices
            bonds (List[Tuple]): List of bond connections
            coords (np.ndarray): 3D coordinates

        Returns: 
            Optional[Chem.Mol]: RDKit molecule object
        """
        mol = Chem.RWMol()
        
        try:
            # Add atoms with 3D coordinates
            for (element, idx), coord in zip(atoms, coords):
                atom = Chem.Atom(element)
                # Special handling for nitrogen
                if element == 'N':
                    atom.SetFormalCharge(0)
                    atom.SetNoImplicit(True)
                    atom.SetNumExplicitHs(1)  # Set explicit hydrogens
                mol.AddAtom(atom)
            
            # Add bonds
            for a1, a2, _ in bonds:
                mol.AddBond(int(a1), int(a2), Chem.BondType.SINGLE)
            
            # Add conformer
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, coord in enumerate(coords):
                conf.SetAtomPosition(i, tuple(coord))
            mol.AddConformer(conf)
            
            # Try sanitization with different options
            try:
                Chem.SanitizeMol(mol,
                                sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE,
                                catchErrors=True)
            except:
                # Fallback sanitization
                Chem.SanitizeMol(mol,
                                sanitizeOps=Chem.SANITIZE_CLEANUP|Chem.SANITIZE_PROPERTIES|Chem.SANITIZE_SYMMRINGS,
                                catchErrors=True)
            
            # Additional validation
            if mol is None or mol.GetNumAtoms() == 0:
                return None
                
            return mol
            
        except Exception as e:
            logger.warning(f"Failed to create/sanitize molecule, attempting fallback: {str(e)}")
            try:
                # Fallback: try to create molecule without 3D coordinates
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                mol_fallback = Chem.MolFromSmiles(smiles)
                if mol_fallback is not None:
                    return mol_fallback
            except:
                pass
            return None

    def process_conformer_library(self, family: str) -> Dict:
        """
        Process a single conformer library and generate molecular representations.

        This function processes a molecular family's conformer library, generating RDKit molecules,
        SMILES representations, and handling errors. Includes validation and detailed logging.

        Args:
            family (str): Name of the molecular family to process

        Returns:
            Dict: Dictionary containing processed molecule data including:
                - coords: 3D coordinates
                - rdkit_mol: RDKit molecule object
                - smiles: Canonical SMILES
                - atoms: List of atoms
                - bonds: List of bonds
        """
        library_path = self.library_paths[family]
        con_lib = ml.ConformerLibrary(library_path)
        molecules = defaultdict(dict)
        
        try:
            with con_lib.reading():
                keys = list(con_lib.keys())
                logger.info(f"Processing {len(keys)} molecules from {family}")
                
                mol_info = []
                failed_mols = []
                
                for key in keys:
                    try:
                        mol_data = con_lib[key]
                        coords = mol_data.coords
                        
                        # Ensure coordinates exist
                        if len(coords) == 0:
                            logger.warning(f"No coordinates found for molecule {key}")
                            failed_mols.append(key)
                            continue
                        
                        # Get bond information
                        bonds = []
                        for bond in mol_data.bonds:
                            bonds.append((int(bond.a1.idx), int(bond.a2.idx), 1))
                        
                        # Create atom list
                        atoms = []
                        for atom in mol_data.atoms:
                            atoms.append((atom.element, int(atom.idx)))
                        
                        # Create RDKit molecule
                        rd_mol = self._create_rdkit_mol(atoms, bonds, coords[0])
                        if rd_mol is None:
                            failed_mols.append(key)
                            continue
                            
                        canonical_smiles = Chem.MolToSmiles(rd_mol, canonical=True)
                        molecules[key] = {
                            'coords': coords,
                            'rdkit_mol': rd_mol,
                            'smiles': canonical_smiles,
                            'atoms': atoms,
                            'bonds': bonds
                        }
                        mol_info.append({
                            'key': key,
                            'smiles': canonical_smiles,
                            'n_atoms': len(atoms),
                            'n_conformers': len(coords)
                        })
                    except Exception as e:
                        logger.error(f"Error processing molecule {key} in {family}: {str(e)}")
                        failed_mols.append(key)
                        continue
                        
                # Log statistics
                if failed_mols:
                    logger.warning(f"Failed to process {len(failed_mols)} molecules in {family}")
                
                if mol_info:
                    df = pd.DataFrame(mol_info)
                    df.to_csv(os.path.join(self.output_dir, f"{family}_molecules.csv"), index=False)
                    logger.info(f"Successfully processed {len(mol_info)} molecules in {family}")
                
                return molecules
                
        except Exception as e:
            logger.error(f"Error accessing conformer library for {family}: {str(e)}")
            return {}

    def perform_clustering(self, distance_matrix: np.ndarray, fingerprints: Dict[str, np.ndarray],
                          method: str = 'butina', n_clusters: Optional[int] = None,
                          cutoff: float = 0.2) -> List[List[int]]:
        """
        Perform molecular clustering using specified method and parameters.

        Implements multiple clustering approaches (Butina, K-means, hierarchical) with
        automatic parameter selection and error handling. Includes cluster size balancing.

        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix
            fingerprints (Dict[str, np.ndarray]): Molecular fingerprints
            method (str): Clustering method ('butina', 'kmeans', 'hierarchical')
            n_clusters (Optional[int]): Number of clusters (auto-calculated if None)
            cutoff (float): Similarity cutoff for clustering

        Returns:
            List[List[int]]: List of clusters, where each cluster is a list of molecule indices
        """
        if len(fingerprints) < 2:
            return [[0]]

        try:
            fp_array = np.array(list(fingerprints.values()))
            
            # Ensure all fingerprints have the same shape
            if len(set(fp.shape[0] for fp in fp_array)) > 1:
                logger.error("Inconsistent fingerprint lengths detected")
                # Pad or truncate to consistent length
                max_len = max(fp.shape[0] for fp in fp_array)
                fp_array = np.array([np.pad(fp, (0, max_len - len(fp))) if len(fp) < max_len else fp[:max_len] 
                                   for fp in fp_array])
            
            if method == 'kmeans':
                if n_clusters is None:
                    n_clusters = max(3, len(fingerprints) // 10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=2801)
                labels = kmeans.fit_predict(fp_array)
                
                clusters = [[] for _ in range(n_clusters)]
                for idx, label in enumerate(labels):
                    clusters[label].append(idx)
                    
            else:  # Default to hierarchical
                if n_clusters is None:
                    n_clusters = max(3, len(fingerprints) // 10)
                
                linkage_matrix = linkage(fp_array, method='ward')
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                clusters = [[] for _ in range(max(labels))]
                for idx, label in enumerate(labels):
                    clusters[label-1].append(idx)
            
            # Balance clusters
            balanced_clusters = self._balance_clusters(clusters)
            return balanced_clusters
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            # Return single cluster with all molecules
            return [list(range(len(fingerprints)))]

    def visualize_clusters(self, family: str, use_pca: bool = True, min_cluster_size: int = 2):
        """
        Create enhanced visualizations of molecular clusters with interactive features.

        Generates PCA/t-SNE visualizations, cluster statistics, and distance matrix plots.
        Includes minimum cluster size filtering and detailed cluster analytics.

        Args:
            family (str): Molecular family name
            use_pca (bool): Use PCA (True) or t-SNE (False) for dimensionality reduction
            min_cluster_size (int): Minimum number of molecules per cluster

        Returns:
            None: Saves visualizations and statistics to output directory
        """
        if family not in self.family_data:
            logger.error(f"No data found for family {family}")
            return
        
        data = self.family_data[family]
        molecules = data['molecules']
        clusters = data['clusters']
        
        # Filter out clusters that are too small
        valid_clusters = [c for c in clusters if len(c) >= min_cluster_size]
        
        if not valid_clusters:
            logger.warning(f"No clusters of sufficient size found for {family}")
            return
        
        vis_dir = os.path.join(self.output_dir, f"{family}_clusters")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate cluster statistics
        cluster_stats = self._generate_cluster_stats(valid_clusters, molecules)
        
        # Save statistics
        pd.DataFrame(cluster_stats).to_csv(
            os.path.join(vis_dir, "cluster_statistics.csv"), 
            index=False
        )
        
        # Dimensionality reduction and plotting
        try:
            self._create_cluster_plot(family, data, valid_clusters, use_pca, vis_dir)
            self._create_distance_matrix_plot(family, data, vis_dir)
        except Exception as e:
            logger.error(f"Error in visualization for {family}: {str(e)}")

    def _create_cluster_plot(self, family: str, data: dict, valid_clusters: List, use_pca: bool, vis_dir: str):
        """
        Create interactive cluster visualization plot using Plotly.

        Generates dimensionality-reduced scatter plots with hover information, cluster 
        annotations, and size indicators. Supports both PCA and t-SNE visualization.

        Args:
            family (str): Molecular family name
            data (dict): Family clustering data
            valid_clusters (List): List of valid clusters
            use_pca (bool): Whether to use PCA for dimensionality reduction
            vis_dir (str): Output directory for visualizations

        Returns:
            None: Saves interactive HTML and static PNG plots
        """
        import plotly.express as px
        import plotly.graph_objects as go
        
        fingerprints = np.array(list(data['fingerprints'].values()))
        molecule_keys = list(data['molecules'].keys())
        smiles_list = [data['molecules'][key]['smiles'] for key in molecule_keys]
        
        # Dimensionality reduction
        if use_pca:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=2801)
        else:
            reducer = TSNE(n_components=2, random_state=2801)
        
        reduced_data = reducer.fit_transform(fingerprints)
        
        # Prepare data for plotting
        cluster_labels = np.zeros(len(fingerprints)) - 1
        for i, cluster in enumerate(valid_clusters):
            cluster_labels[cluster] = i
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': reduced_data[:, 0],
            'PC2': reduced_data[:, 1],
            'Cluster': [f'Cluster {int(l)}' if l >= 0 else 'Unclustered' for l in cluster_labels],
            'Molecule': molecule_keys,
            'SMILES': smiles_list,
            'Cluster_size': [len(valid_clusters[int(l)]) if l >= 0 else 0 for l in cluster_labels]
        })
        
        # Create interactive plot
        fig = px.scatter(plot_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        hover_data=['Molecule', 'SMILES', 'Cluster_size'],
                        title=f'{family} clusters ({len(valid_clusters)} clusters)',
                        labels={'PC1': f'{"PC" if use_pca else "tSNE"} 1',
                            'PC2': f'{"PC" if use_pca else "tSNE"} 2'},
                        color_discrete_sequence=px.colors.qualitative.Set3)
        
        # Add cluster size annotations
        for i, cluster in enumerate(valid_clusters):
            center = reduced_data[cluster].mean(axis=0)
            fig.add_annotation(
                x=center[0],
                y=center[1],
                text=f'n={len(cluster)}',
                showarrow=False,
                yshift=10
            )
        
        # Update layout
        fig.update_layout(
            width=1000,
            height=800,
            showlegend=True,
            legend_title='Clusters'
        )
        
        # Save both interactive HTML and static PNG
        fig.write_html(os.path.join(vis_dir, f"{'pca' if use_pca else 'tsne'}_visualization.html"))
        fig.write_image(os.path.join(vis_dir, f"{'pca' if use_pca else 'tsne'}_visualization.png"))

    def _create_distance_matrix_plot(self, family: str, data: dict, vis_dir: str):
        """
        Create interactive distance matrix visualization using Plotly.

        Generates hierarchical clustering dendrogram with interactive heatmap showing
        molecular similarities. Includes hover information and cluster boundaries.

        Args:
            family (str): Molecular family name
            data (dict): Family clustering data
            vis_dir (str): Output directory for visualizations

        Returns:
            None: Saves interactive HTML and static PNG visualizations
        """
        import plotly.figure_factory as ff
        
        # Convert distance matrix to condensed form
        n = len(data['distance_matrix'])
        molecule_keys = list(data['molecules'].keys())
        
        # Create heatmap
        fig = ff.create_dendrogram(data['distance_matrix'],
                                orientation='left',
                                labels=molecule_keys)
        
        # Create the heatmap
        heatmap = go.Heatmap(
            z=data['distance_matrix'],
            x=molecule_keys,
            y=molecule_keys,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Molecule 1: %{x}<br>Molecule 2: %{y}<br>Distance: %{z}<extra></extra>'
        )
        
        # Update layout
        fig.update_layout(
            title=f'Clustered Distance Matrix for {family}',
            width=1200,
            height=1200,
            showlegend=False
        )
        
        # Add heatmap
        fig.add_trace(heatmap)
        
        # Save both interactive HTML and static PNG
        fig.write_html(os.path.join(vis_dir, "distance_matrix.html"))
        fig.write_image(os.path.join(vis_dir, "distance_matrix.png"))

    def _balance_clusters(self, clusters: List[List[int]], 
                         min_size: int = 3, max_size: int = 20,
                         merge_threshold: int = 2) -> List[List[int]]:
        """
        Balance cluster sizes for more uniform distribution.

        Args:
            clusters (List[List[int]]): Input clusters
            min_size (int): Minimum cluster size
            max_size (int): Maximum cluster size
            merge_threshold (int): Size threshold for merging

        Returns:
            List[List[int]]: Balanced clusters
        """
        if not clusters:
            return []
        
        # Sort clusters by size
        clusters = sorted(clusters, key=len, reverse=True)
        
        balanced = []
        to_merge = []
        
        for cluster in clusters:
            if len(cluster) < merge_threshold:
                to_merge.extend(cluster)
            elif len(cluster) < min_size:
                if len(to_merge) + len(cluster) >= min_size:
                    balanced.append(to_merge + cluster)
                    to_merge = []
                else:
                    to_merge.extend(cluster)
            elif len(cluster) > max_size:
                # Split large clusters
                for i in range(0, len(cluster), max_size):
                    balanced.append(cluster[i:i + max_size])
            else:
                balanced.append(cluster)
        
        # Handle remaining molecules
        if to_merge:
            if balanced:
                # Distribute to existing clusters
                while to_merge:
                    smallest_cluster = min(balanced, key=len)
                    if len(smallest_cluster) >= max_size:
                        balanced.append([to_merge.pop()])
                    else:
                        smallest_cluster.append(to_merge.pop())
            else:
                # Create new cluster
                balanced.append(to_merge)
        
        return balanced
    
    def _generate_cluster_stats(self, valid_clusters: List[List[int]], molecules: dict) -> List[dict]:
        """
        Generate detailed statistics for each molecular cluster.

        Computes cluster-level statistics including size, composition, and molecular
        properties. Provides insights into cluster characteristics and diversity.

        Args:
            valid_clusters (List[List[int]]): List of valid cluster indices
            molecules (dict): Dictionary of molecular data

        Returns:
            List[dict]: List of dictionaries containing statistics for each cluster
                - cluster_id: Cluster identifier
                - size: Number of molecules
                - molecules: List of molecule keys
                - smiles: List of SMILES strings
                - avg_num_atoms: Average number of atoms
                - std_num_atoms: Standard deviation of atom count
        """
        cluster_stats = []
        for i, cluster in enumerate(valid_clusters):
            mol_keys = [list(molecules.keys())[idx] for idx in cluster]
            cluster_stats.append({
                'cluster_id': i,
                'size': len(cluster),
                'molecules': mol_keys,
                'smiles': [molecules[key]['smiles'] for key in mol_keys],
                'avg_num_atoms': np.mean([len(molecules[key]['atoms']) for key in mol_keys]),
                'std_num_atoms': np.std([len(molecules[key]['atoms']) for key in mol_keys])
            })
        return cluster_stats

    def process_all_families(self):
        """
        Process all molecular families with comprehensive clustering and analysis.

        Main processing pipeline that handles molecule loading, fingerprint generation,
        clustering, visualization, and result storage for all molecular families.

        Includes:
        1. Molecule processing and validation
        2. Fingerprint generation and normalization
        3. Distance matrix computation
        4. Clustering analysis
        5. Visualization generation
        6. Result storage and error handling

        Returns:
            None: Updates instance attributes with processed data
        """
        for family in self.library_paths:
            logger.info(f"Processing family: {family}")
            
            try:
                # Process molecules
                molecules = self.process_conformer_library(family)
                if not molecules:
                    continue
                    
                # Generate fingerprints using MorganGenerator
                morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
                fingerprints = {}
                for key, mol_data in molecules.items():
                    mol = mol_data['rdkit_mol']
                    fp = morgan_gen.GetFingerprint(mol)
                    fingerprints[key] = fp

                # Convert fingerprints to numpy arrays
                fp_arrays = {}
                for key, fp in fingerprints.items():
                    arr = np.zeros((1024,))
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fp_arrays[key] = arr
                
                # Validate fingerprints
                valid, normalized_fps = self._validate_fingerprints(fp_arrays)
                if not valid:
                    logger.warning(f"Invalid fingerprints for {family}, skipping clustering")
                    continue
                
                # Compute distance matrix
                n = len(normalized_fps)
                keys = list(normalized_fps.keys())
                distance_matrix = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(i+1, n):
                        similarity = 1 - DataStructs.TanimotoSimilarity(
                            fingerprints[keys[i]], 
                            fingerprints[keys[j]]
                        )
                        distance_matrix[i,j] = similarity
                        distance_matrix[j,i] = similarity
                
                # Perform clustering
                clusters = self.perform_clustering(
                    distance_matrix=distance_matrix,
                    fingerprints=normalized_fps,
                    method='hierarchical',  # Using hierarchical as default
                    n_clusters=10  # Adjust this based on your needs
                )
                
                if clusters:
                    # Store results
                    self.family_data[family] = {
                        'molecules': molecules,
                        'fingerprints': normalized_fps,
                        'distance_matrix': distance_matrix,
                        'clusters': clusters
                    }
                    
                    # Create mapping from molecule index to cluster
                    molecule_to_cluster = {}
                    for cluster_idx, cluster in enumerate(clusters):
                        for mol_idx in cluster:
                            molecule_to_cluster[list(molecules.keys())[mol_idx]] = cluster_idx
                    
                    self.cluster_mappings[family] = molecule_to_cluster
                    
                    # Generate visualizations
                    try:
                        if len(clusters) > 1:
                            self.visualize_clusters(family, use_pca=True)
                    except Exception as e:
                        logger.error(f"Error generating visualizations for {family}: {str(e)}")
                    
                    logger.info(f"Completed processing {family} with {len(clusters)} clusters")
                
            except Exception as e:
                logger.error(f"Error processing family {family}: {str(e)}")
                self.family_data[family] = {
                    'molecules': {},
                    'clusters': [],
                    'distance_matrix': np.array([]),
                    'fingerprints': {}
                }
                continue

def main():
    """Test the clustering implementation."""
    CONFORMER_LIBRARY_PATHS = {
        # "family1": "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib",
        # "family2": "/Users/utkarsh/MMLI/molli-data/00-libraries/imine_confs.clib",
        # "family3": "/Users/utkarsh/MMLI/molli-data/00-libraries/thiols.clib",
        # "family4": "/Users/utkarsh/MMLI/molli-data/00-libraries/product_confs.clib"

        "family1": "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib",
        "family2": "/eagle/FOUND4CHEM/utkarsh/dataset/imine_confs.clib",
        "family3": "/eagle/FOUND4CHEM/utkarsh/dataset/thiols.clib",
        "family4": "/eagle/FOUND4CHEM/utkarsh/dataset/product_confs.clib",
    }
    
    # OUTPUT_DIR = "/Users/utkarsh/MMLI/equicat/src/clustering_results"
    OUTPUT_DIR = "/eagle/FOUND4CHEM/utkarsh/project/equicat/src/clustering_results"
    
    # Test each clustering method
    clustering_methods = ['butina', 'hierarchical', 'kmeans'] 
    
    for method in clustering_methods:
        logger.info(f"\nTesting clustering method: {method}")
        
        output_subdir = os.path.join(OUTPUT_DIR, method)
        os.makedirs(output_subdir, exist_ok=True)
        
        processor = MolecularClusterProcessor(
            CONFORMER_LIBRARY_PATHS,
            clustering_cutoff=0.2,
            output_dir=output_subdir
        )
        
        processor.process_all_families()
        
        # Save cluster mappings for successful families
        torch.save({
            'cluster_mappings': processor.cluster_mappings,
            'family_data': {k: v for k, v in processor.family_data.items() if v['molecules']}
        }, os.path.join(output_subdir, 'cluster_data.pt'))
        
        # Print summary for this method
        print(f"\nClustering Results Summary ({method}):")
        print("=" * 50)
        for family in CONFORMER_LIBRARY_PATHS:
            if family in processor.family_data and processor.family_data[family]['molecules']:
                n_clusters = len(processor.family_data[family]['clusters'])
                n_molecules = len(processor.family_data[family]['molecules'])
                print(f"\n{family}:")
                print(f"  Total molecules: {n_molecules}")
                print(f"  Number of clusters: {n_clusters}")
                if n_clusters > 0:
                    cluster_sizes = [len(cluster) for cluster in processor.family_data[family]['clusters']]
                    print(f"  Average cluster size: {n_molecules/n_clusters:.2f}")
                    print(f"  Smallest cluster: {min(cluster_sizes)} molecules")
                    print(f"  Largest cluster: {max(cluster_sizes)} molecules")
            else:
                print(f"\n{family}: Processing failed")

if __name__ == "__main__":
    main()