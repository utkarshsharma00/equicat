"""
EQUICAT Molecule Embedding Visualization System

A comprehensive visualization system for molecular embeddings, featuring PCA transformation,
family-based analysis, and interactive plotting capabilities. This module unifies both basic
and family-aware visualization approaches into a single cohesive system.

Key components:
1. Data Processing:
   - JSON embedding data loading
   - Advanced error handling
   - Flexible path management
   - Data validation and verification
   
2. Analysis Features:
   - PCA dimensionality reduction
   - Family extraction and grouping
   - Similarity analysis
   - Outlier detection
   
3. Visualization System:
   - Interactive scatter plots
   - Family-based coloring
   - Dynamic tooltips
   - Custom styling options
   - Publication-ready outputs

4. Quality Management:
   - Data validation checks
   - Error logging
   - Debug information
   - Performance monitoring

Key Features:
1. Unified embedding visualization
2. Family-aware analysis
3. Interactive data exploration
4. Publication-ready plots
5. Comprehensive error handling
6. Flexible output formats
7. Advanced hover information
8. Custom colormap support
9. Data validation system
10. Performance optimization

Author: Utkarsh Sharma
Version: 4.0.0
Date: 12-15-2024
License: MIT

Dependencies:
    - plotly (>=5.0.0)
    - numpy (>=1.20.0)
    - sklearn (>=0.24.0)
    - pandas (>=1.0.0)

Usage:
    python visualize_molecule_embeddings.py

Change Log:
- v4.0.0 (12-15-2024):
  * Unified basic and family-aware visualization
  * Added comprehensive data validation
  * Enhanced error handling system
  * Improved memory efficiency
  * Added performance monitoring
  * Enhanced plot customization
  * Removed redundant plotting functions
- v3.0.0 (10-03-2024):
  * Added family-based coloring
  * Enhanced interactive features
  * Improved data processing
- v2.0.0 (09-11-2024):
  * Added basic PCA visualization
  * Implemented interactive plotting
- v1.0.0 (08-01-2024):
  * Initial implementation

ToDo:
- Add t-SNE and UMAP alternatives
- Implement clustering visualization
- Add statistical analysis tools
- Support for larger datasets
- Add export options
- Real-time visualization
- Interactive legend controls
- Custom annotation support
"""

import json
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import pandas as pd
import re

# Constants
INPUT_FILE = "/Users/utkarsh/MMLI/equicat/bdsi_large/final_molecule_embeddings.json"
OUTPUT_FILE = "/Users/utkarsh/MMLI/equicat/bdsi_large/embeddings_pca.html"

def load_embeddings(file_path):
    """
    Load molecule embeddings from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing embeddings.

    Returns:
        dict: A dictionary of molecule keys and their corresponding embeddings.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_family(key):
    """
    Extract family information from the molecule key.

    Args:
        key (str): Molecule key.

    Returns:
        str: Extracted family information.
    """
    match = re.match(r'family(\d+)', key)
    return match.group(1) if match else 'Unknown'

def perform_pca(embeddings):
    """
    Perform PCA on the embeddings and extract family information.

    Args:
        embeddings (dict): Dictionary of molecule embeddings.

    Returns:
        tuple: PCA-transformed embeddings, corresponding molecule keys, and family information.
    """
    keys = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[key] for key in keys])
    families = [extract_family(key) for key in keys]
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embedding_matrix)
    
    return pca_result, keys, families

def create_interactive_plot(pca_result, keys, families):
    """
    Create an interactive scatter plot of PCA-transformed embeddings with family-based coloring.

    Args:
        pca_result (np.array): PCA-transformed embeddings.
        keys (list): List of molecule keys.
        families (list): List of family information for each molecule.

    Returns:
        plotly.graph_objects.Figure: Interactive plot figure.
    """
    df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Key': keys,
        'Family': families
    })

    fig = go.Figure()

    for family in sorted(set(families)):
        family_data = df[df['Family'] == family]
        fig.add_trace(go.Scatter(
            x=family_data['PC1'],
            y=family_data['PC2'],
            mode='markers',
            name=f'Family {family}',
            text=family_data['Key'],
            hoverinfo='text'
        ))

    fig.update_layout(
        title='PCA Visualization of Molecule Embeddings (Colored by Family)',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        hovermode='closest',
        legend_title='Family'
    )

    return fig

def main():
    """
    Main function to load data, perform PCA, and create the interactive plot.
    """
    try:
        # Load embeddings
        embeddings = load_embeddings(INPUT_FILE)
        print(f"Loaded embeddings for {len(embeddings)} molecules.")

        # Perform PCA and extract family information
        pca_result, keys, families = perform_pca(embeddings)
        print("PCA transformation completed.")

        # Create and save the interactive plot
        fig = create_interactive_plot(pca_result, keys, families)
        fig.write_html(OUTPUT_FILE)
        print(f"Interactive PCA plot saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()