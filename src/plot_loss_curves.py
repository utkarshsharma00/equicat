"""
EQUICAT Plot Loss Curves

This script generates interactive loss curves for analyzing the training process of the EQUICAT model. 
It focuses on creating a combined plot of average loss and learning rate across training epochs.

Key components:
1. Data extraction from training log file
2. Interactive plot generation using Plotly
3. Dual-axis visualization for loss and learning rate

New functionality in v3.0.0:
- plot_average_loss_and_learning_rate(): Creates a combined plot of average loss and learning rate
  across epochs, providing a comprehensive view of the training progress.

Author: Utkarsh Sharma
Version: 1.0.0
Date: 10-03-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - plotly (>=5.0.0)
    - pandas (>=1.0.0)

Usage:
    python plot_loss_curves.py

Change Log:
    - v3.0.0: Implemented plot_average_loss_and_learning_rate()
              Removed other plotting functions to focus on average loss and learning rate
    - v2.0.0: (Version jump to align with major changes)
    - v1.5.0: Added combined plot for Total, Positive, and Negative losses per epoch
    - v1.4.0: Ensured the batch_plots directory is completely cleared and recreated for each run
    - v1.3.0: Refactored to handle large datasets and organize output in subdirectories
    - v1.2.0: Added individual batch loss plots across epochs
    - v1.1.0: Added average batch loss across epochs plot
    - v1.0.0: Initial implementation with average loss per epoch plot

TODO:
    - Implement command-line arguments for configurable input/output paths
    - Add option to export plots as static images (PNG/SVG)
    - Implement error handling for missing or corrupted log files
    - Add functionality to compare multiple training runs
    - Implement smoothing option for loss curves
    - Add option to customize plot appearance (colors, line styles, etc.)
"""

import plotly.graph_objects as go
import numpy as np
import re
import os
from plotly.subplots import make_subplots

np.set_printoptions(precision=15)
np.random.seed(0)

# Constants
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/develop_op/loss_curves"
LOG_FILE_PATH = "/Users/utkarsh/MMLI/equicat/develop_op/"
LOG_FILE = os.path.join(LOG_FILE_PATH, "data_loader.log")

def plot_average_loss_and_learning_rate(log_file):
    """
    Generate a combined plot of average loss and learning rate across epochs.

    This function reads the training log file, extracts epoch, average loss,
    and learning rate information, and creates an interactive Plotly graph
    with dual y-axes for loss and learning rate.

    Args:
        log_file (str): Path to the training log file.

    Returns:
        None. Saves the generated plot as an HTML file.
    """
    # Initialize lists to store extracted data
    epochs = []
    avg_losses = []
    learning_rates = []
    
    # Define regex patterns to match relevant log lines
    epoch_pattern = re.compile(r"Starting epoch (\d+)/(\d+)")
    avg_loss_pattern = re.compile(r"Epoch \[(\d+)/\d+\], Average Loss: ([-\d.]+), Final LR: ([\d.e-]+)")
    
    current_epoch = 0
    
    # Read the log file and extract data
    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            avg_loss_match = avg_loss_pattern.search(line)
            
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            if avg_loss_match:
                epoch = int(avg_loss_match.group(1))
                avg_loss = float(avg_loss_match.group(2))
                final_lr = float(avg_loss_match.group(3))
                epochs.append(epoch)
                avg_losses.append(avg_loss)
                learning_rates.append(final_lr)
    
    # Create a subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add average loss trace
    fig.add_trace(
        go.Scatter(x=epochs, y=avg_losses, mode='lines+markers', name='Average Loss'),
        secondary_y=False
    )
    
    # Add learning rate trace
    fig.add_trace(
        go.Scatter(x=epochs, y=learning_rates, mode='lines+markers', name='Learning Rate'),
        secondary_y=True
    )
    
    # Update layout with titles and labels
    fig.update_layout(
        title_text="Average Loss and Learning Rate",
        xaxis_title="Epoch",
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Average Loss", secondary_y=False)
    fig.update_yaxes(title_text="Learning Rate", secondary_y=True)
    
    # Save the plot as an HTML file
    output_file = os.path.join(OUTPUT_PATH, 'average_loss_and_learning_rate.html')
    fig.write_html(output_file)
    print(f"Average Loss and Learning Rate plot saved to {output_file}")

def main():
    """
    Main function to execute the plotting process.

    This function calls the plotting function and handles any exceptions
    that may occur during the process.
    """
    try:
        plot_average_loss_and_learning_rate(LOG_FILE)
        print("Visualization complete. Check the output directory for the generated HTML plot.")
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")
        print("Please check if the log file exists and contains valid data.")

        # Debug: Print more information about the error
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()