# Conformer Embedding Creation Process in EquiCat

This document outlines the process of generating conformer embeddings using the MACE.

## Process Overview

### 1. Initial Embedding

- Each atom in the conformer is represented by its atomic number and 3D position.
- The `node_embedding` block converts atomic numbers into initial node features.

### 2. Edge Features

- Edge features are computed using the `radial_embedding` block, which considers inter-atomic distances.
- Spherical harmonics encode relative atom orientations, producing `edge_attrs`.

### 3. Message Passing

The model employs a series of interaction blocks (as defined by `num_interactions` in the configuration).

Each interaction block consists of:
- Updating node features via the `linear_up` operation.
- Combining node features with edge attributes using a tensor product operation (`conv_tp`).
- Aggregating messages with `scatter_sum`.
- Applying a `linear` operation to the aggregated messages.

### 4. Equivariant Updates

After each interaction, an equivariant product basis (`products` blocks) is applied to maintain equivariance.

### 5. Final Embedding

- The output from the last interaction and product block yields the final node-level embeddings.
- These embeddings represent each atom within the context of the entire conformer.

### 6. Readout

- The `readout` block (either linear or non-linear) is applied to the final node embeddings.
- This step produces the final conformer-level embedding or prediction.

## Note

<<<<<<< HEAD
This process creates embeddings that capture both local atomic environments and global conformer structure, while maintaining equivariance to rotations and translations.
=======
This process creates embeddings that capture both local atomic environments and global conformer structure, while maintaining equivariance to rotations and translations.
>>>>>>> ee8f69a (conformer ensemble embedding combiner working)
