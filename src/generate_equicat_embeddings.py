# import torch
# import os
# import json
# import logging
# from mace import data, tools, modules
# from e3nn import o3
# import numpy as np
# from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
# from conformer_ensemble_embedding_combiner import process_molecule_conformers, move_to_device
# import molli as ml
# from data_loader import MultiFamilyConformerDataset, compute_avg_num_neighbors, get_unique_atomic_numbers

# torch.set_default_dtype(torch.float64)
# np.set_printoptions(precision=15)
# np.random.seed(42)

# # Constants
# CONFORMER_LIBRARY_PATHS = {

#     # "family1": "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib",
#     # "family2": "/Users/utkarsh/MMLI/molli-data/00-libraries/thiol_confs.clib",
#     # "family3": "/Users/utkarsh/MMLI/molli-data/00-libraries/imine_confs.clib",
#     # "family4": "/Users/utkarsh/MMLI/molli-data/00-libraries/product_confs.clib",

#     "family1": "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib",
#     "family2": "/eagle/FOUND4CHEM/utkarsh/dataset/thiol_confs.clib",
#     "family3": "/eagle/FOUND4CHEM/utkarsh/dataset/imine_confs.clib",
#     "family4": "/eagle/FOUND4CHEM/utkarsh/dataset/product_confs.clib",
# }
# CHECKPOINT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/develop_op/checkpoints/final_equicat_model.pt"
# OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/develop_op/final_embeddings"
# CHECKPOINT_PATH = "/Users/utkarsh/MMLI/equicat/develop_op/checkpoints/best_model.pt"
# OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/develop_op/final_embeddings"
# CUTOFF = 6.0
# EMBEDDING_TYPE = 'improved_self_attention'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EXCLUDED_MOLECULES = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']
# MAX_CONFORMERS = 1000

# def setup_logging(log_file):
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )

# def load_checkpoint(checkpoint_path, model, device):
#     logging.info(f"Loading checkpoint from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     logging.info(f"Checkpoint keys: {checkpoint.keys()}")
    
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     elif 'state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['state_dict'])
#     else:
#         raise ValueError(f"Unexpected checkpoint format. Keys found: {checkpoint.keys()}")
    
#     logging.info("Checkpoint loaded successfully")
#     model.eval()
#     return model

# def main():
#     setup_logging(f"{OUTPUT_PATH}/final_embedding_generation.log")
#     logging.info("Starting final embedding generation process")

#     # Load conformer libraries
#     conformer_libraries = {family: ml.ConformerLibrary(path) for family, path in CONFORMER_LIBRARY_PATHS.items()}
#     logging.info(f"Loaded conformer libraries: {', '.join(conformer_libraries.keys())}")

#     # Verify total number of molecules
#     total_molecules = sum(len(lib) for lib in conformer_libraries.values())
#     logging.info(f"Total molecules across all families: {total_molecules}")

#     # Create dataset
#     dataset = MultiFamilyConformerDataset(
#         conformer_libraries=conformer_libraries,
#         cutoff=CUTOFF,
#         sample_size=1,  # We'll process one molecule at a time
#         max_conformers=MAX_CONFORMERS,
#         exclude_molecules=EXCLUDED_MOLECULES
#     )
#     logging.info(f"Dataset created with {dataset.total_molecules} molecules from {len(dataset.family_keys)} families")

#     # Verify dataset creation
#     for family, keys in dataset.family_keys.items():
#         logging.info(f"Family {family}: {len(keys)} molecules")

#     # Calculate average number of neighbors and unique atomic numbers
#     avg_num_neighbors = 0
#     unique_atomic_numbers = set()
#     total_conformers = 0
#     for sample in dataset:
#         for atomic_data_list, _, _ in sample:
#             total_conformers += len(atomic_data_list)
#             for conformer in atomic_data_list:
#                 avg_num_neighbors += compute_avg_num_neighbors(conformer)
#                 unique_atomic_numbers.update(conformer.atomic_numbers.tolist())
#     avg_num_neighbors /= total_conformers
#     unique_atomic_numbers = list(unique_atomic_numbers)

#     logging.info(f"Average number of neighbors: {avg_num_neighbors}")
#     logging.info(f"Unique atomic numbers: {unique_atomic_numbers}")
#     logging.info(f"Total conformers: {total_conformers}")

#     z_table = tools.AtomicNumberTable(unique_atomic_numbers)

#     # Load model configuration
#     model_config = {
#         "r_max": CUTOFF,
#         "num_bessel": 8,
#         "num_polynomial_cutoff": 6,
#         "max_ell": 2,
#         "num_interactions": 1,
#         "num_elements": len(z_table),
#         "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
#         "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
#         "hidden_irreps": o3.Irreps("256x0e + 256x1o"),
#         "MLP_irreps": o3.Irreps("16x0e"),
#         "atomic_energies": tools.to_numpy(torch.zeros(len(z_table), dtype=torch.float64)),
#         "correlation": 3,
#         "gate": torch.nn.functional.silu,
#         "avg_num_neighbors": avg_num_neighbors,
#     }

#     # Initialize and load model
#     model = EQUICATPlusNonLinearReadout(model_config, z_table).to(DEVICE)
#     model = load_checkpoint(CHECKPOINT_PATH, model, DEVICE)
#     logging.info(f"Loaded model from checkpoint: {CHECKPOINT_PATH}")

#     # Process molecules and generate embeddings
#     final_embeddings = {}
#     dataset.reset()

#     processed_count = 0
#     skipped_count = 0
#     error_count = 0

#     with torch.no_grad():
#         for sample_idx in range(len(dataset)):
#             sample = dataset.get_next_sample()
#             if sample is None:
#                 break

#             for atomic_data_list, key, family in sample:
#                 logging.info(f"Processing molecule {key} from {family} with {len(atomic_data_list)} conformers")

#                 try:
#                     molecule_embeddings = []
#                     for conformer_idx, conformer in enumerate(atomic_data_list):
#                         conformer = move_to_device(conformer, DEVICE)
#                         input_dict = {
#                             'positions': conformer.positions,
#                             'atomic_numbers': conformer.atomic_numbers,
#                             'edge_index': conformer.edge_index
#                         }
#                         output = model(input_dict)

#                         # Check for NaN or inf values
#                         if torch.isnan(output).any() or torch.isinf(output).any():
#                             logging.warning(f"NaN or inf detected in output for molecule {key}, conformer {conformer_idx}")
#                             continue

#                         molecule_embeddings.append(output)

#                     if not molecule_embeddings:
#                         logging.warning(f"No valid embeddings generated for molecule {key}")
#                         skipped_count += 1
#                         continue

#                     molecule_embeddings = torch.stack(molecule_embeddings)
#                     averaged_embeddings = process_molecule_conformers(molecule_embeddings, model.non_linear_readout.irreps_out)

#                     scalar, vector = averaged_embeddings[EMBEDDING_TYPE]
#                     if scalar is not None and vector is not None:
#                         combined = torch.cat([scalar.view(-1), vector.view(-1)])
#                     elif scalar is not None:
#                         combined = scalar.view(-1)
#                     else:
#                         combined = vector.view(-1)

#                     final_embeddings[key] = combined.cpu().numpy().tolist()
#                     processed_count += 1
#                     logging.info(f"Successfully processed molecule {key}")

#                 except Exception as e:
#                     logging.error(f"Error processing molecule {key} from {family}: {str(e)}")
#                     error_count += 1

#             if (sample_idx + 1) % 100 == 0:
#                 logging.info(f"Progress: Processed {processed_count}, Skipped {skipped_count}, Errors {error_count}, Total {sample_idx + 1}"
# )

#     # Save final embeddings
#     os.makedirs(OUTPUT_PATH, exist_ok=True)
#     with open(f"{OUTPUT_PATH}/embeddings.json", 'w') as f:
#         json.dump(final_embeddings, f)

#     logging.info(f"Saved final embeddings for {len(final_embeddings)} molecules")
#     logging.info(f"Total molecules attempted: {dataset.total_molecules}")
#     logging.info(f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
#     logging.info("Final embedding generation process completed")

# if __name__ == "__main__":
#     main()



import torch
import os
import h5py
import logging
from mace import data, tools, modules
from e3nn import o3
import numpy as np
from equicat_plus_nonlinear import EQUICATPlusNonLinearReadout
from conformer_ensemble_embedding_combiner import process_molecule_conformers, move_to_device
import molli as ml
from data_loader import MultiFamilyConformerDataset, compute_avg_num_neighbors, get_unique_atomic_numbers

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)
np.random.seed(42)

# Constants
CONFORMER_LIBRARY_PATHS = {

    "family1": "/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib",
    "family2": "/Users/utkarsh/MMLI/molli-data/00-libraries/thiol_confs.clib",
    "family3": "/Users/utkarsh/MMLI/molli-data/00-libraries/imine_confs.clib",
    "family4": "/Users/utkarsh/MMLI/molli-data/00-libraries/product_confs.clib",
    # "family1": "/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib",
    # "family2": "/eagle/FOUND4CHEM/utkarsh/dataset/thiol_confs.clib",
    # "family3": "/eagle/FOUND4CHEM/utkarsh/dataset/imine_confs.clib",
    # "family4": "/eagle/FOUND4CHEM/utkarsh/dataset/product_confs.clib",
}
# CHECKPOINT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/develop_op/checkpoints/final_equicat_model.pt"
# OUTPUT_PATH = "/eagle/FOUND4CHEM/utkarsh/project/equicat/develop_op/final_embeddings"
CHECKPOINT_PATH = "/Users/utkarsh/MMLI/equicat/develop_op/checkpoints/best_model.pt"
OUTPUT_PATH = "/Users/utkarsh/MMLI/equicat/develop_op/final_embeddings"
CUTOFF = 6.0
EMBEDDING_TYPE = 'improved_self_attention'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDED_MOLECULES = ['179_vi', '181_i', '180_i', '180_vi', '178_i', '178_vi']
MAX_CONFORMERS = 1000

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_checkpoint(checkpoint_path, model, device):
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logging.info(f"Checkpoint keys: {checkpoint.keys()}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f"Unexpected checkpoint format. Keys found: {checkpoint.keys()}")
    
    logging.info("Checkpoint loaded successfully")
    model.eval()
    return model

def save_embedding_hdf5(file_path, family, key, embedding):
    unique_key = f"{family}_{key}"
    with h5py.File(file_path, 'a') as f:
        if unique_key in f:
            del f[unique_key]  # Override if exists
        dataset = f.create_dataset(unique_key, data=np.array(embedding))
        dataset.attrs['family'] = family
        dataset.attrs['molecule_id'] = key

def generate_unique_key(family, key):
    return f"{family}_{key}"

def main():
    setup_logging(f"{OUTPUT_PATH}/final_embedding_generation.log")
    logging.info("Starting final embedding generation process")

    # Load conformer libraries
    conformer_libraries = {family: ml.ConformerLibrary(path) for family, path in CONFORMER_LIBRARY_PATHS.items()}
    logging.info(f"Loaded conformer libraries: {', '.join(conformer_libraries.keys())}")

    # Verify total number of molecules
    total_molecules = sum(len(lib) for lib in conformer_libraries.values())
    logging.info(f"Total molecules across all families: {total_molecules}")

    # Create dataset
    dataset = MultiFamilyConformerDataset(
        conformer_libraries=conformer_libraries,
        cutoff=CUTOFF,
        sample_size=1,  # We'll process one molecule at a time
        max_conformers=MAX_CONFORMERS,
        exclude_molecules=EXCLUDED_MOLECULES
    )
    logging.info(f"Dataset created with {dataset.total_molecules} molecules from {len(dataset.family_keys)} families")

    # Verify dataset creation
    for family, keys in dataset.family_keys.items():
        logging.info(f"Family {family}: {len(keys)} molecules")

    # Calculate average number of neighbors and unique atomic numbers
    avg_num_neighbors = 0
    unique_atomic_numbers = set()
    total_conformers = 0
    for sample in dataset:
        for atomic_data_list, _, _ in sample:
            total_conformers += len(atomic_data_list)
            for conformer in atomic_data_list:
                avg_num_neighbors += compute_avg_num_neighbors(conformer)
                unique_atomic_numbers.update(conformer.atomic_numbers.tolist())
    avg_num_neighbors /= total_conformers
    unique_atomic_numbers = list(unique_atomic_numbers)

    logging.info(f"Average number of neighbors: {avg_num_neighbors}")
    logging.info(f"Unique atomic numbers: {unique_atomic_numbers}")
    logging.info(f"Total conformers: {total_conformers}")

    z_table = tools.AtomicNumberTable(unique_atomic_numbers)

    # Load model configuration
    model_config = {
        "r_max": CUTOFF,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 2,
        "num_interactions": 1,
        "num_elements": len(z_table),
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "hidden_irreps": o3.Irreps("256x0e + 256x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "atomic_energies": tools.to_numpy(torch.zeros(len(z_table), dtype=torch.float64)),
        "correlation": 3,
        "gate": torch.nn.functional.silu,
        "avg_num_neighbors": avg_num_neighbors,
    }

    # Initialize and load model
    model = EQUICATPlusNonLinearReadout(model_config, z_table).to(DEVICE)
    model = load_checkpoint(CHECKPOINT_PATH, model, DEVICE)
    logging.info(f"Loaded model from checkpoint: {CHECKPOINT_PATH}")

    # Process molecules and generate embeddings
    output_file = f"{OUTPUT_PATH}/embeddings.h5"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # Initialize the file
        pass

    dataset.reset()

    processed_count = 0
    skipped_count = 0
    error_count = 0
    duplicate_count = 0

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            sample = dataset.get_next_sample()
            if sample is None:
                break

            for atomic_data_list, key, family in sample:
                unique_key = generate_unique_key(family, key)
                logging.info(f"Processing molecule {unique_key} with {len(atomic_data_list)} conformers")

                try:
                    molecule_embeddings = []
                    for conformer_idx, conformer in enumerate(atomic_data_list):
                        conformer = move_to_device(conformer, DEVICE)
                        input_dict = {
                            'positions': conformer.positions,
                            'atomic_numbers': conformer.atomic_numbers,
                            'edge_index': conformer.edge_index
                        }
                        output = model(input_dict)

                        # Check for NaN or inf values
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            logging.warning(f"NaN or inf detected in output for molecule {unique_key}, conformer {conformer_idx}")
                            continue

                        molecule_embeddings.append(output)

                    if not molecule_embeddings:
                        logging.warning(f"No valid embeddings generated for molecule {unique_key}")
                        skipped_count += 1
                        continue

                    molecule_embeddings = torch.stack(molecule_embeddings)
                    averaged_embeddings = process_molecule_conformers(molecule_embeddings, model.non_linear_readout.irreps_out)

                    scalar, vector = averaged_embeddings[EMBEDDING_TYPE]
                    if scalar is not None and vector is not None:
                        combined = torch.cat([scalar.view(-1), vector.view(-1)])
                    elif scalar is not None:
                        combined = scalar.view(-1)
                    else:
                        combined = vector.view(-1)

                    # Check for duplicate keys
                    with h5py.File(output_file, 'r') as f:
                        if unique_key in f:
                            logging.warning(f"Duplicate key detected: {unique_key}. Overwriting.")
                            duplicate_count += 1

                    # Save embedding immediately
                    save_embedding_hdf5(output_file, family, key, combined.cpu().numpy())
                    
                    processed_count += 1
                    logging.info(f"Successfully processed and saved embedding for molecule {unique_key}")

                except Exception as e:
                    logging.error(f"Error processing molecule {unique_key}: {str(e)}")
                    error_count += 1

            if (sample_idx + 1) % 100 == 0:
                logging.info(f"Progress: Processed {processed_count}, Skipped {skipped_count}, Errors {error_count}, Duplicates {duplicate_count}, Total {sample_idx + 1}")

    # Verify final count
    with h5py.File(output_file, 'r') as f:
        final_count = len(f.keys())

    logging.info(f"Total unique embeddings saved: {final_count}")
    logging.info(f"Total molecules attempted: {dataset.total_molecules}")
    logging.info(f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}, Duplicates: {duplicate_count}")
    
    # Log family distribution
    with h5py.File(output_file, 'r') as f:
        family_counts = {}
        for key in f.keys():
            family = f[key].attrs['family']
            family_counts[family] = family_counts.get(family, 0) + 1
    
    logging.info("Family distribution in saved embeddings:")
    for family, count in family_counts.items():
        logging.info(f"  {family}: {count}")
    
    logging.info("Final embedding generation process completed")

if __name__ == "__main__":
    main()