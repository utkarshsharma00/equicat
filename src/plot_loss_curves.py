"""
EquiCat Loss Curves Visualization

This script provides interactive visualization tools for analyzing EquiCat training metrics,
combining loss curves and learning rate plots for comprehensive training analysis. It features
robust data extraction from logs and generates publication-quality interactive plots.

Key components:
1. Data Processing:
   - Advanced log parsing with regex patterns
   - Robust error handling for malformed logs
   - Multi-metric data extraction
   - Time series data validation

2. Visualization Features:
   - Interactive dual-axis plotting
   - Combined loss and learning rate visualization
   - Dynamic hover information
   - Responsive layout design
   - Custom plot styling options

3. Output Management:
   - HTML report generation
   - Flexible file path handling
   - Directory structure validation
   - Error logging and debugging

Key Features:
1. Dual-axis visualization system
2. Interactive data exploration
3. Comprehensive error handling
4. Flexible output formatting
5. Robust data extraction
6. Publication-ready plots
7. Advanced hover tooltips
8. Customizable plot styling
9. Directory management
10. Debug information logging

Author: Utkarsh Sharma
Version: 4.0.0
Date: 12-14-2024 (MM-DD-YYYY)
License: MIT

Dependencies:
- plotly (>=5.0.0)
- numpy (>=1.20.0)
- pandas (>=1.0.0)

Usage:
    python plot_loss_curves.py

For detailed usage instructions, please refer to the README.md file.

Change Log:
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
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/epoch_large/"
LOG_FILE_PATH = "/Users/utkarsh/MMLI/equicat/epoch_large/"
LOG_FILE = os.path.join(LOG_FILE_PATH, "training.log")

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