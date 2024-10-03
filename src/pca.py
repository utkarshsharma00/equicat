"""
EQUICAT Molecule Embedding Visualizer

This script loads molecule embeddings from a JSON file, performs PCA,
and creates an interactive plot to visualize the embeddings.

Key components:
1. JSON data loading
2. PCA transformation
3. Interactive plot generation using Plotly

Author: Utkarsh Sharma
Version: 1.0.0
Date: 10-03-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - plotly (>=5.0.0)
    - numpy (>=1.20.0)
    - sklearn (>=0.24.0)

Usage:
    python visualize_molecule_embeddings.py

TODO:
    - Add command-line arguments for input/output file paths
    - Implement error handling for missing or corrupted JSON files
    - Add option to customize plot appearance (colors, marker sizes, etc.)
    - Implement additional dimensionality reduction techniques (t-SNE, UMAP)
"""

import json
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

# Constants
INPUT_FILE = "/Users/utkarsh/MMLI/equicat/develop_op/final_molecule_embeddings.json"
OUTPUT_FILE = "/Users/utkarsh/MMLI/equicat/develop_op/embeddings_pca.html"

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

def perform_pca(embeddings):
    """
    Perform PCA on the embeddings.

    Args:
        embeddings (dict): Dictionary of molecule embeddings.

    Returns:
        tuple: PCA-transformed embeddings and corresponding molecule keys.
    """
    keys = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[key] for key in keys])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embedding_matrix)
    
    return pca_result, keys

def create_interactive_plot(pca_result, keys):
    """
    Create an interactive scatter plot of PCA-transformed embeddings.

    Args:
        pca_result (np.array): PCA-transformed embeddings.
        keys (list): List of molecule keys.

    Returns:
        plotly.graph_objects.Figure: Interactive plot figure.
    """
    fig = go.Figure(data=go.Scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=pca_result[:, 0],  # Color by the first principal component
            colorscale='Viridis',
            showscale=True
        ),
        text=keys,  # This will show the molecule key on hover
        hoverinfo='text'
    ))

    fig.update_layout(
        title='PCA Visualization of Molecule Embeddings',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        hovermode='closest'
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

        # Perform PCA
        pca_result, keys = perform_pca(embeddings)
        print("PCA transformation completed.")

        # Create and save the interactive plot
        fig = create_interactive_plot(pca_result, keys)
        fig.write_html(OUTPUT_FILE)
        print(f"Interactive PCA plot saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()