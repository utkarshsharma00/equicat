# main.py

import torch
import numpy as np
import molli as ml
import torch.nn.functional as F
from e3nn import o3
from mace import data, modules, tools
from equicat import EQUICAT
from data_loader import process_data, ConformerDataset
from conformer_ensemble_embedding_combiner import process_conformer_ensemble
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout

# Set the cutoff value
cutoff = 5.0

# Define the model configuration
model_config = dict(
    atomic_numbers=None,
    r_max=cutoff,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    num_interactions=2,
    interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    hidden_irreps=o3.Irreps("32x0e + 32x1o"),
    correlation=3,
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,

)

# Create the ConformerDataset
conformer_ensemble = ml.ConformerLibrary("/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib")
conformer_dataset = ConformerDataset(conformer_ensemble, cutoff, num_ensembles=2)

# Process each batch from the DataLoader
for batch_conformers, unique_atomic_numbers, avg_num_neighbors in process_data(conformer_dataset, batch_size=32):
    print(f"Processing batch of {len(batch_conformers)} conformers")
    
    # Create an AtomicNumberTable using the unique atomic numbers in the batch
    z_table = tools.AtomicNumberTable(unique_atomic_numbers)
    print(f"Created AtomicNumberTable: {z_table}")

    # Create atomic energies array with zeros
    atomic_energies = np.zeros(len(z_table), dtype=float)
    print(f"Created atomic energies array: {atomic_energies}")

    # Update the model configuration
    model_config['num_elements'] = len(z_table)
    model_config['atomic_energies'] = atomic_energies
    model_config['atomic_numbers'] = torch.tensor(z_table.zs)
    model_config['avg_num_neighbors'] = avg_num_neighbors
    print(f"Updated model configuration: {model_config}")

    # Create an EQUICAT model using the updated model configuration and z_table
    equicat_model = EQUICATPlusNonLinearReadout(model_config, z_table)
    print(f"Created EQUICAT model: {equicat_model}")

    # Process each conformer in the batch
    conformer_embeddings = []
    for conformer in batch_conformers:
        # Prepare the input dictionary for the EQUICAT model
        print(f"Conformer with {conformer.num_nodes} atoms and {conformer.edge_index.shape[1]} edges")
        print(f"batch_conformers: {batch_conformers}")
        input_dict = {
            'positions': conformer.positions,
            'atomic_numbers': conformer.atomic_numbers,
            'edge_index': conformer.edge_index
        }

        # Pass the conformer data through the EQUICAT model to obtain output
        output = equicat_model(input_dict) # * original
        #output, out_irreps = equicat_model(input_dict) # ! added out_irreps
        #loss = F.mse_loss(output, target)
        #print(f"Loss: {loss.item()}")
        conformer_embeddings.append(output)
        print(f"Conformer embeddings shape: {output.shape}")
        print(f"Conformer embeddings:\n{output}")

    # Stack all conformer embeddings
    conformer_embeddings = torch.stack(conformer_embeddings)
    print(f"Stacked conformer embeddings shape: {conformer_embeddings.shape}")
    print(f"Conformer embeddings:\n{conformer_embeddings}")

    # Process the conformer ensemble
    ensemble_embeddings = process_conformer_ensemble(conformer_embeddings)
    #print(f"Main - output shape: {output.shape}, out_irreps: {out_irreps}") # ! added out_irreps
    #ensemble_embeddings = process_conformer_ensemble(output, str(out_irreps))
    print(f"Ensemble embeddings:\n{ensemble_embeddings}")

    # print("Ensemble embeddings:")
    # for key, value in ensemble_embeddings.items():
    #     print(f"{key} shape: {value.shape}")
    # print("Mean Pooling shape:", ensemble_embeddings['mean_pooling'].shape)
    # print("Deep Sets shape:", ensemble_embeddings['deep_sets'].shape)
    # print("Self-Attention shape:", ensemble_embeddings['self_attention'].shape)

    for method, (scalar, vector) in ensemble_embeddings.items():
        print(f"{method}:")
        print(f"  Scalar shape: {scalar.shape}")
        print(f"  Vector shape: {vector.shape}")
        print(f"  Scalar: {scalar}")
        print(f"  Vector: {vector}")
        print("-" * 50)

    # You can now use these ensemble embeddings for further processing or prediction
    # For example:
    # mean_pooled_prediction = some_prediction_model(ensemble_embeddings['mean_pooling'])
    # deep_sets_prediction = some_prediction_model(ensemble_embeddings['deep_sets'])
    # self_attention_prediction = some_prediction_model(ensemble_embeddings['self_attention'])

    print("=" * 50)  # Separator between batches

print("Finished processing all conformers in all ensembles.")