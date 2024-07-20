"""
EQUICAT Plot Loss Curves

This script provides interactive loss curves for analyzing the training process of the EQUICAT model. 
It generates two interactive HTML plots:
1. Average Loss per Epoch: Shows the trend of average loss across training epochs.
2. Average Batch Loss Across Epochs: Displays the average loss for each batch, computed across all epochs.

The script reads training data from a log file, processes it, and creates interactive Plotly visualizations.
These visualizations allow users to hover over data points for detailed information and zoom/pan for closer inspection.

Author: Utkarsh Sharma
Version: 1.0.0
Date: 07-20-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - plotly (>=5.0.0)
    - pandas (>=1.0.0)

Usage:
    python visualize_training.py
"""

import plotly.graph_objects as go
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

def main():
    """
    Main function to generate and save the loss curve plots.
    """
    plot_average_loss_per_epoch(LOG_FILE)
    plot_average_batch_loss_across_epochs(LOG_FILE)
    print("Visualization complete. Check the output directory for the generated HTML plots.")

if __name__ == "__main__":
    main()