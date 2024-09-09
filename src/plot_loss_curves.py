"""
EQUICAT Plot Loss Curves

This script provides interactive loss curves for analyzing the training process of the EQUICAT model. 
It generates four types of interactive HTML plots:
1. Total Loss, Positive Loss, and Negative Loss per Epoch: Shows the trend of these losses across training epochs.
2. Average Batch Loss Across Epochs: Displays the average loss for each batch, computed across all epochs.
3. Individual Batch Loss Across Epochs: Shows separate plots for each batch's loss across all epochs.

The script reads training data from a log file, processes it, and creates interactive Plotly visualizations.
These visualizations allow users to hover over data points for detailed information and zoom/pan for closer inspection.

Author: Utkarsh Sharma
Version: 1.5.0
Date: 08-13-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
    - plotly (>=5.0.0)
    - pandas (>=1.0.0)

Usage:
    python plot_loss_curves.py

Change Log:
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
    - Add smoothing option for loss curves
"""

import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import pandas as pd
import re
import os
import shutil
from collections import defaultdict
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

np.set_printoptions(precision=15)
np.random.seed(0)

# Constants
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/output/loss_curves"
LOG_FILE_PATH = "/Users/utkarsh/MMLI/equicat/output/"
# OUTPUT_PATH = "/Users/utkarsh/MMLI/saved-outputs/gpu/"
# LOG_FILE_PATH = "/Users/utkarsh/MMLI/saved-outputs/gpu/"
LOG_FILE = os.path.join(LOG_FILE_PATH, "training.log")

def plot_loss_curves(log_file):
    epochs = []
    total_losses = []
    learning_rates = []

    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch" in line and "Total Loss:" in line:
                match = re.search(r"Epoch (\d+), Total Loss: ([-\d.]+), LR: ([\d.e-]+)", line)
                if match:
                    epochs.append(int(match.group(1)))
                    total_losses.append(float(match.group(2)))
                    learning_rates.append(float(match.group(3)))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=epochs, y=total_losses, mode='lines+markers', name='Total Loss'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=learning_rates, mode='lines+markers', name='Learning Rate'),
        secondary_y=True,
    )

    fig.update_layout(
        title='Total Loss and Learning Rate Across Epochs',
        xaxis_title='Epoch',
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Total Loss", secondary_y=False)
    fig.update_yaxes(title_text="Learning Rate", secondary_y=True)

    pio.write_html(fig, file=f'{OUTPUT_PATH}/loss_curves.html', auto_open=False)
    print(f"Loss curves plot saved to {OUTPUT_PATH}/loss_curves.html")

def plot_gradient_flow(log_file):
    epochs = []
    gradient_data = {}

    with open(log_file, 'r') as f:
        current_epoch = None
        for line in f:
            if "Epoch" in line and "Total Loss:" in line:
                match = re.search(r"Epoch (\d+)", line)
                if match:
                    current_epoch = int(match.group(1))
                    epochs.append(current_epoch)
            elif "Gradient stats for" in line and current_epoch is not None:
                match = re.search(r"Gradient stats for (.*): mean=([-\d.e]+), std=([-\d.e]+), max=([-\d.e]+), min=([-\d.e]+)", line)
                if match:
                    layer = match.group(1)
                    mean = float(match.group(2))
                    std = float(match.group(3))
                    max_val = float(match.group(4))
                    min_val = float(match.group(5))
                    
                    if layer not in gradient_data:
                        gradient_data[layer] = {'mean': [], 'std': [], 'max': [], 'min': []}
                    
                    gradient_data[layer]['mean'].append(mean)
                    gradient_data[layer]['std'].append(std)
                    gradient_data[layer]['max'].append(max_val)
                    gradient_data[layer]['min'].append(min_val)

    fig = make_subplots(rows=2, cols=2, subplot_titles=("Mean", "Standard Deviation", "Max", "Min"))

    for i, (title, index) in enumerate([("Mean", 'mean'), ("Standard Deviation", 'std'), ("Max", 'max'), ("Min", 'min')]):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for layer, data in gradient_data.items():
            if len(data[index]) == len(epochs):
                fig.add_trace(
                    go.Scatter(x=epochs, y=data[index], mode='lines+markers', name=layer),
                    row=row, col=col
                )

        fig.update_xaxes(title_text="Epoch", row=row, col=col)
        fig.update_yaxes(title_text=title, row=row, col=col)

    fig.update_layout(
        height=800,
        width=1200,
        title_text="Gradient Statistics Across Layers",
        showlegend=False,
    )

    fig.write_html(f'{OUTPUT_PATH}/gradient_flow.html')
    print(f"Gradient flow plot saved to {OUTPUT_PATH}/gradient_flow.html")

def plot_overall_loss_across_epochs(log_file):
    """
    Plot the overall loss across all epochs from the training log.
    
    Args:
        log_file (str): Path to the training log file.
    
    Returns:
        None
    """
    epochs = []
    losses = []
    learning_rates = []
    
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Loss: ([-\d.]+), LR: ([\d.e-]+)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                lr = float(match.group(3))
                epochs.append(epoch)
                losses.append(loss)
                learning_rates.append(lr)
    
    if not epochs:
        print("No loss data found in the log file.")
        return
    
    fig = go.Figure()
    
    # Plot loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode='lines+markers',
        name='Loss',
        yaxis='y1',
        line=dict(color='blue', width=2),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
    ))
    
    # Plot learning rate on secondary y-axis
    fig.add_trace(go.Scatter(
        x=epochs,
        y=learning_rates,
        mode='lines+markers',
        name='Learning Rate',
        yaxis='y2',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=8, symbol='square'),
        hovertemplate='Epoch: %{x}<br>LR: %{y:.6f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Overall Loss and Learning Rate Across Epochs',
        xaxis_title='Epoch',
        yaxis=dict(
            title='Loss',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Learning Rate',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    output_file = os.path.join(OUTPUT_PATH, 'overall_loss_across_epochs.html')
    fig.write_html(output_file)
    print(f"Overall loss across epochs plot saved to {output_file}")

    # Print some statistics
    print(f"Total epochs plotted: {len(epochs)}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Initial learning rate: {learning_rates[0]:.6f}")
    print(f"Final learning rate: {learning_rates[-1]:.6f}")

def plot_sample_loss_across_epochs(log_file):
    """
    Plot the loss of each sample across all epochs from the training log.
    
    Args:
        log_file (str): Path to the training log file.
    
    Returns:
        None
    """
    sample_losses = defaultdict(lambda: defaultdict(float))
    max_epoch = 0
    max_sample = 0
    
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Sample \[(\d+)/\d+\], Loss: ([-\d.]+)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                sample = int(match.group(2))
                loss = float(match.group(3))
                sample_losses[sample][epoch] = loss
                max_epoch = max(max_epoch, epoch)
                max_sample = max(max_sample, sample)
    
    if not sample_losses:
        print("No sample loss data found in the log file.")
        return
    
    fig = go.Figure()
    
    for sample, losses in sorted(sample_losses.items()):
        epochs = sorted(losses.keys())
        loss_values = [losses[epoch] for epoch in epochs]
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=loss_values,
            mode='lines+markers',
            name=f'Sample {sample}',
            hovertemplate='Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Sample Loss Across Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='closest',
        legend_title='Samples',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    pio.write_html(fig, file=os.path.join(OUTPUT_PATH, 'sample_loss_across_epochs.html'), auto_open=False)
    print(f"Sample loss across epochs plot saved to {os.path.join(OUTPUT_PATH, 'sample_loss_across_epochs.html')}")

def plot_combined_losses_per_epoch(log_file):
    """
    Plot the total, positive, and negative losses per epoch from the training log.
    
    Args:
        log_file (str): Path to the training log file.
    
    Returns:
        None
    """
    epochs = []
    total_losses = []
    positive_losses = defaultdict(list)
    negative_losses = []
    
    total_pattern = re.compile(r"Epoch \[(\d+)/\d+\], Total Loss: ([\d.]+)")
    negative_pattern = re.compile(r"Epoch \[(\d+)/\d+\], Negative Loss: ([\d.]+)")
    positive_pattern = re.compile(r"Positive Loss: ([\d.]+)")
    
    with open(log_file, 'r') as f:
        current_epoch = 0
        for line in f:
            total_match = total_pattern.search(line)
            negative_match = negative_pattern.search(line)
            positive_match = positive_pattern.search(line)
            
            if total_match:
                epoch = int(total_match.group(1))
                total_loss = float(total_match.group(2))
                epochs.append(epoch)
                total_losses.append(total_loss)
                current_epoch = epoch
            
            if negative_match:
                negative_losses.append(float(negative_match.group(2)))
            
            if positive_match:
                positive_losses[current_epoch].append(float(positive_match.group(1)))

    # Calculate average positive loss for each epoch
    avg_positive_losses = [
        np.mean(positive_losses[epoch-1]) if epoch-1 in positive_losses else np.nan
        for epoch in range(1, max(epochs) + 1)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=total_losses,
        mode='lines+markers', name='Total Loss',
        line=dict(color='blue', width=2, dash='solid'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Epoch: %{x}<br>Total Loss: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=avg_positive_losses,
        mode='lines+markers', name='Avg Positive Loss',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=8, symbol='square'),
        hovertemplate='Epoch: %{x}<br>Avg Positive Loss: %{y:.6f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=negative_losses,
        mode='lines+markers', name='Negative Loss',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='Epoch: %{x}<br>Negative Loss: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Losses per Epoch',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    pio.write_html(fig, file=os.path.join(OUTPUT_PATH, 'combined_losses_per_epoch.html'), auto_open=False)
    print(f"Combined losses plot saved to {os.path.join(OUTPUT_PATH, 'combined_losses_per_epoch.html')}")
    
    # Print debug information
    print(f"Total Loss data points: {len(total_losses)}")
    print(f"Avg Positive Loss data points: {len(avg_positive_losses)}")
    print(f"Negative Loss data points: {len(negative_losses)}")
    
    # Print average positive loss for each epoch
    for epoch, avg_loss in enumerate(avg_positive_losses, start=1):
        print(f"Epoch {epoch}: Average Positive Loss = {avg_loss:.6f}")

    # Print total number of positive loss entries
    total_positive_entries = sum(len(losses) for losses in positive_losses.values())
    print(f"Total number of positive loss entries: {total_positive_entries}")

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
    
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Batch \[(\d+)\], Positive Loss: ([\d.]+)")
    
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

def clear_output_directory(directory):
    """
    Clear all HTML files in the specified directory.
    If it's the batch_plots directory, remove it entirely and recreate.
    
    Args:
        directory (str): Path to the directory to clear.
    """
    if os.path.basename(directory) == "batch_plots":
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".html"):
                os.remove(os.path.join(directory, filename))

def main():
    """
    Main function to generate and save the loss curve plots.
    """
    try:
        # Clear existing plots
        clear_output_directory(OUTPUT_PATH)
        clear_output_directory(os.path.join(OUTPUT_PATH, "batch_plots"))

        # plot_combined_losses_per_epoch(LOG_FILE)
        # plot_average_batch_loss_across_epochs(LOG_FILE)
        # plot_individual_batch_loss_across_epochs(LOG_FILE)
        # plot_sample_loss_across_epochs(LOG_FILE)
        # plot_overall_loss_across_epochs(LOG_FILE)
        plot_loss_curves(LOG_FILE)
        plot_gradient_flow(LOG_FILE)
        print("Visualization complete. Check the output directory for the generated HTML plots.")
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")
        print("Please check if the log file exists and contains valid data.")

        # Debug: Print more information about the error
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()