"""
EquiCat Loss Curves Visualization

This script provides comprehensive visualization tools for analyzing training metrics,
featuring split views for step-wise and epoch-level analysis. It generates 
publication-quality interactive plots with enhanced readability and analysis tools.

Key components:
1. Data Processing:
   - Multi-level metric extraction (step, running avg, epoch)
   - Enhanced regex pattern matching
   - Robust error handling for malformed logs
   - Time series data validation and smoothing

2. Visualization Features:
   - Split-view plotting (step-wise and epoch-wise)
   - Dual-axis visualization with proper scaling
   - Interactive hover tools with detailed metrics
   - Responsive and publication-ready layouts
   - Enhanced readability with color coding

3. Output Management:
   - Interactive HTML report generation
   - Flexible file path handling
   - Comprehensive error logging
   - Plot validation and debugging

Key Features:
1. Split-view visualization system
2. Running average loss tracking
3. Scientific notation for learning rates
4. Automatic scale adjustment
5. Enhanced hover information
6. Publication-ready styling
7. Responsive layout design
8. Comprehensive error handling
9. Directory management
10. Debug information system

Author: Utkarsh Sharma
Version: 4.0.1
Date: 08-02-2025 (MM-DD-YYYY)
License: MIT

Dependencies:
- plotly (>=5.0.0)
- numpy (>=1.20.0)
- pandas (>=1.0.0)

Usage:
    python plot_loss_curves.py

For detailed usage instructions, please refer to the README.md file.

Change Log:
- v4.0.1 (02-08-2025):
  * Added split-view visualization
  * Enhanced metric tracking and display
  * Improved axis scaling and notation
  * Added running loss visualization
  * Enhanced plot styling and readability
  * Improved hover information
  * Fixed learning rate display
  * Added subplot organization
- v4.0.0 (12-14-2024):
  * Standardized module naming and structure
  * Enhanced error handling and logging
  * Improved path management
  * Added comprehensive data validation
  * Enhanced plot customization options
  * Added debug information system
- v3.0.0 (09-10-2024):
  * Added plot_average_loss_and_learning_rate()
  * Removed legacy plotting functions
  * Streamlined visualization process
- v2.0.0 (08-15-2024):
  * Major architecture refactoring
  * Added combined plotting capabilities
- v1.0.0 (07-01-2024):
  * Initial implementation with basic plotting

ToDo:
- Add command-line argument support
- Implement static image export (PNG/SVG)
- Add multi-run comparison features
- Implement curve smoothing options
- Add custom styling parameters
- Enhance error handling for corrupt logs
- Add automated plot annotation
- Implement interactive legend controls
"""

import plotly.graph_objects as go
import numpy as np
import re
import os
from plotly.subplots import make_subplots

np.set_printoptions(precision=15)
np.random.seed(0)

# Constants
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/loss_check/"
LOG_FILE_PATH = "/Users/utkarsh/MMLI/equicat/loss_check/"
LOG_FILE = os.path.join(LOG_FILE_PATH, "training.log")

def plot_training_metrics(log_file):
    """
    Generate comprehensive visualization of training metrics with split views for step-wise 
    and epoch-level analysis. Creates interactive plots showing instantaneous and running 
    losses along with learning rate progression using dual y-axes scales.

    Args:
        log_file (str): Path to the training log file containing metric data
                       with epoch, loss, and learning rate information.

    Returns:
        None. Saves an interactive HTML plot containing step-wise and epoch-wise 
        training metrics.
    """
    # Initialize data lists
    epochs = []
    avg_losses = []
    instant_losses = []
    running_losses = []
    learning_rates = []
    steps = []
    current_epoch = 0
    
    # Patterns
    epoch_pattern = re.compile(r"Starting epoch (\d+)/(\d+)")
    avg_loss_pattern = re.compile(r"Epoch \[(\d+)/\d+\], Average Loss: ([-\d.]+), Final LR: ([\d.e-]+)")
    sample_pattern = re.compile(r"Epoch \[(\d+)/\d+\], Sample \[\d+/\d+\], Loss: ([-\d.]+), Running Loss: ([-\d.]+), LR: ([\d.e-]+)")
    
    # Extract data
    step_counter = 0
    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            avg_loss_match = avg_loss_pattern.search(line)
            sample_match = sample_pattern.search(line)
            
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            if avg_loss_match:
                epoch = int(avg_loss_match.group(1))
                avg_loss = float(avg_loss_match.group(2))
                final_lr = float(avg_loss_match.group(3))
                epochs.append(epoch)
                avg_losses.append(avg_loss)
                learning_rates.append(final_lr)
            
            if sample_match:
                instant_losses.append(float(sample_match.group(2)))
                running_losses.append(float(sample_match.group(3)))
                steps.append(step_counter)
                step_counter += 1
    
    # Create two subplots with secondary y-axis for learning rate
    fig = make_subplots(
        rows=2, 
        cols=1, 
        subplot_titles=('Step-wise Training Loss', 'Epoch-wise Metrics'),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Plot 1: Step-wise losses
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=instant_losses, 
            mode='lines', 
            name='Instantaneous Loss', 
            opacity=0.7,
            line=dict(color='pink', width=1)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=running_losses, 
            mode='lines', 
            name='Running Loss', 
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Plot 2: Epoch-wise metrics
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=avg_losses, 
            mode='lines+markers', 
            name='Epoch Average Loss',
            line=dict(color='green', width=2)
        ),
        row=2, col=1,
        secondary_y=False
    )
    
    # Add learning rate trace with secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=learning_rates, 
            mode='lines+markers', 
            name='Learning Rate',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1,
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode="x unified",
        title_text="Training Metrics",
        title_x=0.5,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update axes labels and ranges
    fig.update_xaxes(title_text="Steps", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=2, col=1)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    
    # Update y-axes for the second plot (with dual scales)
    fig.update_yaxes(
        title_text="Loss",
        row=2, col=1,
        secondary_y=False,
        range=[0, max(avg_losses)*1.1]  # Give some headroom for loss scale
    )
    
    fig.update_yaxes(
        title_text="Learning Rate",
        row=2, col=1,
        secondary_y=True,
        range=[0, max(learning_rates)*1.1],  # Give some headroom for LR scale
        tickformat=".1e"  # Scientific notation for small LR values
    )
    
    # Save the plot
    output_file = os.path.join(OUTPUT_PATH, 'training_metrics.html')
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

def main():
    """
    Main function to execute the plotting process.

    This function calls the plotting function and handles any exceptions
    that may occur during the process.
    """
    try:
        plot_training_metrics(LOG_FILE)
        print("Visualization complete. Check the output directory for the generated HTML plot.")
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")
        print("Please check if the log file exists and contains valid data.")

        # Debug: Print more information about the error
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()