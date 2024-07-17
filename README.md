
```markdown
# EQUICAT: Equivariant Conformer Analysis Tool

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Module Descriptions](#module-descriptions)
   - [main.py](#mainpy)
   - [data_loader.py](#data_loaderpy)
   - [equicat.py](#equicatpy)
   - [equicat_plus_nonlinear.py](#equicat_plus_nonlinearpy)
   - [conformer_ensemble_embedding_combiner.py](#conformer_ensemble_embedding_combinerpy)
   - [contrastive_loss.py](#contrastive_losspy)
   - [train.py](#trainpy)
6. [Conformer Embedding Creation Process](#conformer-embedding-creation-process)
7. [Data Flow](#data-flow)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

EQUICAT (Equivariant Conformer Analysis Tool) is an advanced Python-based project designed for the analysis of molecular conformers using equivariant neural networks. It builds upon the MACE (Many-body Atomic Cluster Expansion) framework to process molecular geometries while preserving their inherent symmetries.

The primary goal of EQUICAT is to generate and combine embeddings from multiple conformers of a molecule, providing a unified representation that can be leveraged for various downstream tasks in computational chemistry and drug discovery. By utilizing equivariant neural networks, EQUICAT ensures that the learned representations respect the fundamental symmetries of molecular systems, leading to more accurate and physically meaningful predictions.

## Installation

To use EQUICAT, you need to install the following dependencies:

```bash
pip install torch torch_geometric e3nn mace molli numpy
```

Clone the repository:

```bash
git clone https://github.com/yourusername/equicat.git
cd equicat
```

## Project Structure

```
equicat/
│
├── main.py
├── data_loader.py
├── equicat.py
├── equicat_plus_nonlinear.py
├── conformer_ensemble_embedding_combiner.py
├── contrastive_loss.py
├── train.py
├── README.md
└── requirements.txt
```

## Usage

To run the EQUICAT pipeline:

```bash
python main.py
```

To train the EQUICAT model:

```bash
python train.py
```

## Module Descriptions

### main.py

`main.py` serves as the entry point for the EQUICAT pipeline. It orchestrates the entire process of loading conformer data, configuring the EQUICAT model, generating embeddings, and applying various pooling methods to combine these embeddings.

Key functionalities:
- Loads conformer data from the MOLLI library
- Configures and initializes the EQUICAT model
- Processes conformers through the model to generate embeddings
- Applies ensemble pooling techniques to combine conformer embeddings
- Manages the overall workflow of the EQUICAT pipeline

Usage example:

```python
from main import main

if __name__ == "__main__":
    main()
```

### data_loader.py

`data_loader.py` is responsible for loading and processing conformer data. It provides a custom dataset class, data loading utilities, and efficient processing functions to handle molecular conformer data.

Key components:
- `ConformerDataset`: A custom PyTorch dataset class for handling conformer ensembles
- `compute_avg_num_neighbors`: Utility function to calculate average neighbors in a batch
- `custom_collate`: Custom collation function for batching data
- `process_data`: Generator function for processing conformer data in batches

Usage example:

```python
from data_loader import ConformerDataset, process_data

dataset = ConformerDataset(conformer_ensemble, cutoff)
for batch_data in process_data(dataset, batch_size=32):
    # Process batch_data
```

### equicat.py

`equicat.py` implements the core EQUICAT model, a neural network architecture designed for equivariant learning on molecular systems.

Key features:
- Equivariant processing of molecular geometries
- Handling of variable-sized molecular inputs
- Incorporation of spherical harmonics for angular information
- Use of radial basis functions for distance information
- Implementation of symmetric contractions for feature aggregation

Usage example:

```python
from equicat import EQUICAT

model = EQUICAT(model_config, z_table)
output = model(input_dict)
```

### equicat_plus_nonlinear.py

`equicat_plus_nonlinear.py` extends the EQUICAT model with a custom non-linear readout layer, allowing for more complex processing of molecular representations while preserving equivariance.

Key components:
- `CustomNonLinearReadout`: A custom readout layer with equivariant linear transformations and non-linear activations
- `EQUICATPlusNonLinearReadout`: Combines the EQUICAT model with the custom non-linear readout layer

Usage example:

```python
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout

model = EQUICATPlusNonLinearReadout(model_config, z_table)
output = model(input_dict)
```

### conformer_ensemble_embedding_combiner.py

`conformer_ensemble_embedding_combiner.py` provides advanced functionality to combine embeddings from multiple conformers of a molecule into a single representation. It implements three different methods: Mean Pooling, Deep Sets, and Self-Attention.

Key components:
- `ConformerEnsembleEmbeddingCombiner`: The main class implementing the combining methods
- `process_conformer_ensemble`: Processes a single batch of conformer embeddings
- `process_ensemble_batches`: Processes all batches for a single ensemble and averages the results

Usage example:

```python
from conformer_ensemble_embedding_combiner import process_ensemble_batches

ensemble_embeddings = process_ensemble_batches(list_of_batch_embeddings)
```

### contrastive_loss.py

`contrastive_loss.py` implements a contrastive loss function for semi-supervised learning in the context of molecular conformer analysis. The loss encourages embeddings of the same molecule to be close together in the embedding space, while pushing embeddings of different molecules apart.

Key components:
- `contrastive_loss`: Function that computes the contrastive loss for a batch of embeddings

Usage example:

```python
from contrastive_loss import contrastive_loss

loss = contrastive_loss(embeddings, labels)
```

### train.py

`train.py` is responsible for training the EQUICAT model using the contrastive loss function. It sets up the training pipeline, including data loading, model initialization, and the training loop.

Key functionalities:
- Sets up logging for the training process
- Initializes the EQUICAT model and optimizer
- Implements the training loop using contrastive loss
- Saves the trained model

Usage:

```bash
python train.py
```

## Conformer Embedding Creation Process

The EQUICAT model generates conformer embeddings through the following steps:

1. **Initial Embedding**: Each atom is represented by its atomic number and 3D position. The `node_embedding` block converts atomic numbers into initial node features.

2. **Edge Features**: The `radial_embedding` block computes edge features based on inter-atomic distances. Spherical harmonics encode relative atom orientations, producing `edge_attrs`.

3. **Message Passing**: A series of interaction blocks process the data, updating node features, combining them with edge attributes, aggregating messages, and applying a linear operation.

4. **Equivariant Updates**: After each interaction, an equivariant product basis (`products` blocks) maintains equivariance.

5. **Final Embedding**: The output from the last interaction and product block yields the final node-level embeddings, representing each atom within the context of the entire conformer.

6. **Readout**: The `readout` block (linear or non-linear) is applied to the final node embeddings, producing the final conformer-level embedding or prediction.

This process creates embeddings that capture both local atomic environments and global conformer structure, while maintaining equivariance to rotations and translations.

## Data Flow

1. Conformer data is loaded and processed by `data_loader.py`
2. The processed data is fed into the EQUICAT model (`equicat.py` or `equicat_plus_nonlinear.py`)
3. The model generates embeddings for each conformer
4. These embeddings are combined using methods in `conformer_ensemble_embedding_combiner.py`
5. During training (`train.py`), the contrastive loss (`contrastive_loss.py`) is used to optimize the model
6. The main pipeline (`main.py`) orchestrates this entire process

## Contributing

Contributions to EQUICAT are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```