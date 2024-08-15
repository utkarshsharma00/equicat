"""
EQUICAT Conformer Embedding Combiner Sanity Check

This script performs sanity checks on the conformer ensemble processing of the EQUICAT model:
1. Creates dummy data for water molecule conformers
2. Tests the processing of conformer ensembles, including optional padding
3. Verifies various embedding combination methods
4. Compares results between original and padded data

Author: Utkarsh Sharma
Version: 1.0.0
Date: 08-14-2024 (MM-DD-YYYY)
License: MIT

Usage:
    python conformer_ensemble_sanity_check.py

Dependencies:
    - torch (>=1.9.0)
    - numpy (>=1.20.0)

TODO:
    - Implement additional embedding combination methods
    - Add tests for edge cases (e.g., single conformer, very large ensembles)
    - Integrate with actual EQUICAT data loading pipeline for real-data testing

Change Log:
    - v1.0.0: Initial implementation with dummy data generation and basic processing checks
"""

import random
import torch
from equicat import ConformerEnsembleEmbeddingCombiner

BATCH_SIZE = 6

def create_dummy_water_data(num_conformers: int, num_atoms: int = 3, scalar_dim: int = 8, vector_dim: int = 8, pad_to_batch_size: bool = False):
    total_dim = scalar_dim + vector_dim * 3
    dummy_data = torch.rand(num_conformers, num_atoms, total_dim)
    
    if pad_to_batch_size and num_conformers < BATCH_SIZE:
        pad_size = BATCH_SIZE - num_conformers
        padding = torch.zeros(pad_size, num_atoms, total_dim)
        dummy_data = torch.cat([dummy_data, padding], dim=0)
    
    print("\nDummy Water Molecule Data (including padded conformers if applicable):")
    for i in range(dummy_data.shape[0]):
        print(f"\nConformer {i + 1}:")
        for j in range(num_atoms):
            scalar = dummy_data[i, j, :scalar_dim]
            vector = dummy_data[i, j, scalar_dim:].view(vector_dim, 3)
            print(f"  Atom {j + 1}:")
            print(f"    Scalar: {scalar.tolist()}")
            print(f"    Vector: {vector.tolist()}")
    
    return dummy_data

def process_conformer_ensemble(conformer_embeddings: torch.Tensor, pad_to_batch_size: bool = False, batch_size = BATCH_SIZE) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    print(f"process_conformer_ensemble input shape: {conformer_embeddings.shape}")
    
    num_conformers, num_atoms, total_dim = conformer_embeddings.shape
    scalar_dim = total_dim // 4
    vector_dim = scalar_dim
    
    print(f"Num conformers: {num_conformers}, Num atoms: {num_atoms}, Total dim: {total_dim}")
    print(f"Scalar dim: {scalar_dim}, Vector dim: {vector_dim}")
    
    if pad_to_batch_size and batch_size > num_conformers:
        pad_size = batch_size - num_conformers
        padding_indices = random.choices(range(num_conformers), k=pad_size)
        padding = conformer_embeddings[padding_indices]
        
        print("\nPadded Conformers:")
        for i, idx in enumerate(padding_indices):
            print(f"\nPadded Conformer {num_conformers + i + 1} (copied from Conformer {idx + 1}):")
            for j in range(num_atoms):
                scalar = padding[i, j, :scalar_dim]
                vector = padding[i, j, scalar_dim:].view(vector_dim, 3)
                print(f"  Atom {j + 1}:")
                print(f"    Scalar: {scalar.tolist()}")
                print(f"    Vector: {vector.tolist()}")
        
        conformer_embeddings = torch.cat([conformer_embeddings, padding], dim=0)
        print(f"\nPadded input shape: {conformer_embeddings.shape}")
        print(f"Indices used for padding: {padding_indices}")
    else:
        print("\nNo padding needed. Number of conformers matches or exceeds batch size.")
    
    combiner = ConformerEnsembleEmbeddingCombiner(scalar_dim, vector_dim).to(conformer_embeddings.device)
    results = combiner(conformer_embeddings)

    return results

def print_means(scalar, vector):
    scalar_mean = scalar.mean(dim=1)
    vector_mean = vector.mean(dim=1)
    
    print("\nMean values:")
    for i in range(scalar_mean.shape[0]):
        print(f"Conformer {i + 1}:")
        print(f"  Scalar mean: {scalar_mean[i].tolist()}")
        print(f"  Vector mean: {vector_mean[i].tolist()}")

def run_sanity_checks(conformer_embeddings: torch.Tensor):
    print("Starting sanity checks...")
    
    num_conformers, num_atoms, total_dim = conformer_embeddings.shape
    scalar_dim = total_dim // 4
    vector_dim = scalar_dim

    def print_detailed_embeddings(results, title):
        print(f"\n{title}")
        for method, (scalar, vector) in results.items():
            print(f"\n{method}:")
            for i in range(scalar.shape[0]):
                print(f"  Conformer {i + 1}:")
                print(f"    Scalar: {scalar[i].tolist()}")
                print(f"    Vector: {vector[i].tolist()}")

    def print_means(scalar, vector):
        scalar_mean = scalar.mean(dim=1)
        vector_mean = vector.mean(dim=1)
        
        print("\nMean values:")
        for i in range(scalar_mean.shape[0]):
            print(f"Conformer {i + 1}:")
            print(f"  Scalar mean: {scalar_mean[i].tolist()}")
            print(f"  Vector mean: {vector_mean[i].tolist()}")

    print("\nRunning sanity checks with original data:")
    results_original = process_conformer_ensemble(conformer_embeddings)
    print_detailed_embeddings(results_original, "Original Data Embeddings")

    # print("\nMean calculations for original data:")
    # scalar = conformer_embeddings[:, :, :scalar_dim]
    # vector = conformer_embeddings[:, :, scalar_dim:].view(num_conformers, num_atoms, vector_dim, 3)
    # print_means(scalar, vector)

    print("\nRunning sanity checks with padded data:")
    results_padded = process_conformer_ensemble(conformer_embeddings, pad_to_batch_size=False)
    print_detailed_embeddings(results_padded, "Padded Data Embeddings")

    # print("\nMean calculations for padded data:")
    # padded_embeddings = torch.zeros(BATCH_SIZE, num_atoms, total_dim)
    # padded_embeddings[:num_conformers] = conformer_embeddings
    # scalar_padded = padded_embeddings[:, :, :scalar_dim]
    # vector_padded = padded_embeddings[:, :, scalar_dim:].view(BATCH_SIZE, num_atoms, vector_dim, 3)
    # print_means(scalar_padded, vector_padded)

    print("\nComparing original and padded results:")
    for method in results_original.keys():
        scalar_orig, vector_orig = results_original[method]
        scalar_pad, vector_pad = results_padded[method]
        
        print(f"\n{method}:")
        print(f"  Original scalar shape: {scalar_orig.shape}")
        print(f"  Padded scalar shape: {scalar_pad.shape}")
        print(f"  Original vector shape: {vector_orig.shape}")
        print(f"  Padded vector shape: {vector_pad.shape}")
        
        print("  Scalar difference (first 5 elements):")
        print(f"    {(scalar_pad[:num_conformers] - scalar_orig)[:5]}")
        print("  Vector difference (first 5 elements):")
        print(f"    {(vector_pad[:num_conformers] - vector_orig)[:5]}")

    print("Sanity checks completed.")

def main():
    print("Starting main function...")
    
    # Create dummy data for water molecules
    num_conformers = 4
    num_atoms = 3
    scalar_dim = 2
    vector_dim = 2
    dummy_data = create_dummy_water_data(num_conformers, num_atoms, scalar_dim, vector_dim)

    print(f"\nCreated dummy data for {num_conformers} water molecule conformers")
    print(f"Dummy data shape: {dummy_data.shape}")

    # Run sanity checks
    results = process_conformer_ensemble(dummy_data, pad_to_batch_size=False, batch_size=BATCH_SIZE)
    run_sanity_checks(dummy_data)

    print("\nSanity checks completed.")

if __name__ == "__main__":
    main()
    