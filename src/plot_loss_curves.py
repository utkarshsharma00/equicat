"""
EQUICAT Plot Loss Curves

This script provides interactive loss curves for analyzing the training process of the EQUICAT model. 
It generates three types of interactive HTML plots:
1. Average Loss per Epoch: Shows the trend of average loss across training epochs.
2. Average Batch Loss Across Epochs: Displays the average loss for each batch, computed across all epochs.
3. Individual Batch Loss Across Epochs: Shows separate plots for each batch's loss across all epochs.

The script reads training data from a log file, processes it, and creates interactive Plotly visualizations.
These visualizations allow users to hover over data points for detailed information and zoom/pan for closer inspection.

Author: Utkarsh Sharma
Version: 1.3.0
Date: 07-22-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - plotly (>=5.0.0)
    - pandas (>=1.0.0)

Usage:
    python plot_loss_curves.py

Change Log:
    - v1.3.0: Refactored to handle large datasets and organize output in subdirectories
    - v1.2.0: Added individual batch loss plots across epochs
    - v1.1.0: Added average batch loss across epochs plot
    - v1.0.0: Initial implementation with average loss per epoch plot

TODO:
    - Implement command-line arguments for configurable input/output paths
    - Add option to export plots as static images (PNG/SVG)
    - Implement error handling for missing or corrupted log files
    - Add functionality to compare multiple training runs
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import re
from collections import defaultdict
import os

# Constants
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output"
LOG_FILE = os.path.join(OUTPUT_PATH, "training.log")

def plot_average_loss_per_epoch(log_file):
    """
    Plot the average loss per epoch from the training log.
    
    Args:
        log_file (str): Path to the training log file.
    
    Returns:
        None
    """
    epochs = []
    avg_losses = []
    
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Average Loss: ([\d.]+)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                avg_loss = float(match.group(2))
                epochs.append(epoch)
                avg_losses.append(avg_loss)
    
    fig = go.Figure(data=go.Scatter(
        x=epochs,
        y=avg_losses,
        mode='lines+markers',
        marker=dict(size=10),
        hovertemplate='Epoch: %{x}<br>Average Loss: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Loss per Epoch',
        xaxis_title='Epoch',
        yaxis_title='Average Loss',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        hovermode='closest'
    )
    
    pio.write_html(fig, file=os.path.join(OUTPUT_PATH, 'avg_loss_per_epoch.html'), auto_open=False)

def plot_average_batch_loss_across_epochs(log_file):
    """
    Plot the average batch loss across all epochs from the training log.
    
    Args:
        log_file (str): Path to the training log file.
    
    Returns:
        None
    """
    batch_losses = defaultdict(list)
    
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Batch \[(\d+)\], Loss: ([\d.]+)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                batch = int(match.group(2))
                loss = float(match.group(3))
                batch_losses[batch].append(loss)
    
    avg_batch_losses = {batch: sum(losses) / len(losses) for batch, losses in batch_losses.items()}
    
    df = pd.DataFrame(list(avg_batch_losses.items()), columns=['Batch', 'Average Loss'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Batch'],
            y=df['Average Loss'],
            hovertemplate='Batch: %{x}<br>Average Loss: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Average Batch Loss Across Epochs',
        xaxis_title='Batch Number',
        yaxis_title='Average Loss',
        hovermode='closest'
    )
    
    pio.write_html(fig, file=os.path.join(OUTPUT_PATH, 'avg_batch_loss_across_epochs.html'), auto_open=False)

def plot_individual_batch_loss_across_epochs(log_file):
    """
    Plot individual batch losses across all epochs from the training log.
    
    Args:
        log_file (str): Path to the training log file.
    
    Returns:
        None
    """
    batch_losses = defaultdict(lambda: defaultdict(float))
    max_epoch = 0
    max_batch = 0
    
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Batch \[(\d+)\], Loss: ([\d.]+)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                loss = float(match.group(3))
                batch_losses[batch][epoch] = loss
                max_epoch = max(max_epoch, epoch)
                max_batch = max(max_batch, batch)
    
    if not batch_losses:
        print("No batch loss data found in the log file.")
        return
    
    # Create a subdirectory for batch plots
    batch_plots_dir = os.path.join(OUTPUT_PATH, "batch_plots")
    os.makedirs(batch_plots_dir, exist_ok=True)
    
    # Create a separate plot for each batch
    for batch, losses in sorted(batch_losses.items()):
        epochs = sorted(losses.keys())
        loss_values = [losses[epoch] for epoch in epochs]
        
        fig = go.Figure(data=go.Scatter(
            x=epochs,
            y=loss_values,
            mode='lines+markers',
            name=f'Batch {batch}',
            hovertemplate='Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Loss for Batch {batch} Across Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='closest'
        )
        
        # Save the plot in the batch_plots subdirectory
        pio.write_html(fig, file=os.path.join(batch_plots_dir, f'batch_{batch}_loss.html'), auto_open=False)
    
    print(f"Generated individual batch loss plots for {len(batch_losses)} batches in {batch_plots_dir}")

def main():
    """
    Main function to generate and save the loss curve plots.
    """
    try:
        plot_average_loss_per_epoch(LOG_FILE)
        plot_average_batch_loss_across_epochs(LOG_FILE)
        plot_individual_batch_loss_across_epochs(LOG_FILE)
        print("Visualization complete. Check the output directory for the generated HTML plots.")
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")
        print("Please check if the log file exists and contains valid data.")

if __name__ == "__main__":
    main()