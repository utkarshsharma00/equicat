# [EquiCat: An Equivariant Neural Network Architecture for Predicting Enantioselectivity in Asymmetric Catalysis](https://drive.google.com/file/d/1cG3APwV34jZuRNw6rlpyb1Jwp3ME4ay5/view?usp=drive_link)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/equicat/badge/?version=latest)](https://equicat.readthedocs.io/en/latest/?badge=latest)

## Overview

EquiCat is an advanced framework for analyzing molecular conformers using equivariant neural networks. Built upon the MACE (Multi Atomic Cluster Expansion) layers, it processes molecular geometries while preserving their inherent symmetries. The framework introduces novel techniques for multi-family molecular analysis, cluster-aware contrastive learning, and sophisticated conformer embedding combination strategies.

### Key Features

- **Multi-Family Support**: Process multiple families of molecular conformers simultaneously with intelligent family-based organization
- **Advanced Clustering**: K-means based conformer selection with family-aware processing and hierarchical analysis
- **Flexible Embedding**: Multiple sophisticated strategies for combining conformer embeddings (Deep Sets, Self-Attention, etc.)
- **Contrastive Learning**: Family-scoped cluster-aware contrastive learning with relationship weighting
- **Performance Optimization**: Built-in profiling and Chrome tracing for performance analysis
- **Interactive Visualization**: Advanced PCA and plotting tools for embedding analysis
- **Memory Efficiency**: Smart memory management for large molecular systems
- **Comprehensive Logging**: Detailed tracking and visualization of training progress

## Project Structure

```
equicat/
├── src/
│   ├── experiments/                    # Experimental studies and research
│   │   ├── study1/
│   │   ├── study2/
│   │   ├── study3/
│   │   └── equicat_regression_studies.md
│   ├── sanity_checks/                  # Validation and testing
│   │   ├── conformer_embedding_combiner_sanity_check.py
│   │   └── readout_sanity_check.py
│   ├── conformer_ensemble_embedding_combiner.py   # Embedding combination
│   ├── data_loader.py                  # Data processing and loading
│   ├── equicat.py                      # Core model architecture
│   ├── equicat_plus_nonlinear.py       # Enhanced model with non-linear readout
│   ├── generate_equicat_embeddings.py  # Embedding generation
│   ├── molecular_clustering.py         # Molecular clustering tools
│   ├── pca.py                         # Dimensionality reduction and visualization
│   ├── plot_loss_curves.py            # Training visualization
│   └── train.py                       # Training pipeline
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM recommended for large molecular systems

### Dependencies
```
torch>=1.9.0
torch-geometric>=2.0.0
e3nn>=0.4.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
plotly>=5.0.0
h5py>=3.0.0
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/equicat.git
cd equicat
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Core Components

### 1. Molecular Data Processing (`data_loader.py`)
The data processing pipeline handles complex molecular data with efficiency and flexibility:

- **MultiFamilyConformerDataset**:
  - Cross-family sampling for balanced representation
  - GPU-accelerated processing pipelines
  - Smart batching strategies
  - Built-in profiling capabilities
  - Automatic data validation
  - Memory usage optimization

- **Features**:
  - Supports multiple molecular file formats
  - Efficient memory management
  - Comprehensive error handling
  - Chrome trace generation for performance analysis

### 2. Model Architecture
The model is structured in two main components:

#### Base Model (`equicat.py`)
- **Core Features**:
  - Equivariant message passing layers
  - Geometric feature processing
  - Symmetry-preserving operations
  - Advanced radial basis functions
  - Spherical harmonics computations
  - Edge feature generation

#### Enhanced Model (`equicat_plus_nonlinear.py`)
- **Advanced Features**:
  - Non-linear readout mechanism
  - Flexible activation functions
  - Scalar/vector output support
  - Residual connections
  - Advanced normalization layers
  - Enhanced feature combination strategies

### 3. Molecular Clustering (`molecular_clustering.py`)
Sophisticated clustering system for molecular analysis:

- **Clustering Capabilities**:
  - Family-aware molecular organization
  - K-means based conformer selection
  - Hierarchical relationship analysis
  - Automatic cluster validation
  - Dynamic cluster size adaptation

- **Visualization**:
  - Interactive cluster plots
  - Distance matrix visualization
  - Dendrogram generation
  - Family distribution analysis

### 4. Analysis and Visualization
Comprehensive tools for data analysis and visualization:

#### PCA Analysis (`pca.py`)
- **Core Features**:
  - Principal Component Analysis
  - Interactive visualization
  - Family-based coloring
  - Custom plot styling
  - Dynamic tooltips
  - Export capabilities

#### Training Visualization (`plot_loss_curves.py`)
- **Plotting Features**:
  - Combined loss/learning rate plots
  - Interactive Plotly graphs
  - Multiple visualization types
  - Customizable styling
  - Progress tracking
  - Export functionality

### 5. Embedding Generation (`generate_equicat_embeddings.py`)
Flexible system for generating molecular embeddings:

- **Key Features**:
  - Multiple embedding strategies
  - Efficient batch processing
  - GPU acceleration
  - Memory-optimized operations
  - Comprehensive error handling

### 6. Training Pipeline (`train.py`)
Advanced training system with multiple optimization strategies:

- **Training Features**:
  - Cluster-aware contrastive learning
  - Multiple scheduler options
  - Comprehensive checkpointing
  - Early stopping mechanism
  - Learning rate scheduling
  - Gradient clipping

- **Monitoring**:
  - Performance profiling
  - Memory tracking
  - Progress logging
  - Loss visualization

## Usage

### Basic Usage

1. Prepare your molecular data:
```python
from data_loader import MultiFamilyConformerDataset

dataset = MultiFamilyConformerDataset(
    conformer_libraries=libraries,
    cutoff=6.0,
    sample_size=30,
    max_conformers=10
)
```

2. Initialize and train the model:
```python
from train import train_equicat

model = train_equicat(
    model=model,
    dataset=dataset,
    device=device,
    embedding_type='improved_self_attention',
    scheduler_type='cosine',
    contrastive_loss_fn=contrastive_loss_fn
)
```

3. Generate embeddings:
```python
from generate_equicat_embeddings import main as generate_embeddings

generate_embeddings()
```

### Advanced Usage

#### Molecular Clustering and Analysis
```python
from molecular_clustering import MolecularClusterProcessor
from pca import create_interactive_plot

# Initialize clustering
processor = MolecularClusterProcessor(
    library_paths=CONFORMER_LIBRARY_PATHS,
    clustering_cutoff=0.2,
    output_dir='clustering_results'
)

# Process and visualize
processor.process_all_families()
processor.visualize_clusters(family="family1", use_pca=True)

# Generate PCA visualization
create_interactive_plot(pca_result, keys, families)
```

#### Training Visualization
```python
from plot_loss_curves import plot_average_loss_and_learning_rate
from pca import visualize_embeddings

# Plot training progress
plot_average_loss_and_learning_rate('training.log')

# Visualize embeddings
visualize_embeddings(embeddings)
```

## Performance Optimization

### GPU Acceleration
EquiCat automatically utilizes available GPU resources:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Memory Management
- Control memory usage with `max_conformers`
- Enable profiling for monitoring:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True) as prof:
    # Your code here
    pass
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Add comprehensive docstrings to all new functions and classes
- Maintain test coverage
- Follow PEP 8 style guidelines
- Update documentation for significant changes
- Add appropriate unit tests for new features


## Citation

If you use EquiCat in your research, please cite:

```bibtex
@software{equicat2024,
  author = {Utkarsh Sharma, Elena Burlova, Alexander Shved, Scott Denmark, and Ganesh Sivaraman },
  title = {EquiCat: An Equivariant Neural Network Architecture for Predicting Enantioselectivity in Asymmetric Catalysis},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/equicat}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
