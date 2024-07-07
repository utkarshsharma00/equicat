# data_loader.py 

import torch
import molli as ml
from mace import data, tools
from torch_geometric.data import Data, Batch

class ConformerDataset:
    def __init__(self, conformer_ensemble, cutoff, num_ensembles=2):
        self.conformer_ensemble = conformer_ensemble
        self.cutoff = cutoff
        with self.conformer_ensemble.reading():
            self.keys = list(self.conformer_ensemble.keys())[:num_ensembles]
        print(f"Initialized ConformerDataset with {len(self.keys)} conformer ensembles")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.conformer_ensemble.reading():
            conformer = self.conformer_ensemble[key]
            coords = torch.tensor(conformer.coords, dtype=torch.float32)
            atomic_numbers = torch.tensor([atom.element for atom in conformer.atoms], dtype=torch.long)
            print(f"Retrieved conformer ensemble {key} with {coords.shape[0]} conformers")
            print(f"Atomic Numbers: {atomic_numbers}")
            
            atomic_data_list = []
            for i in range(coords.shape[0]):
                config = data.Configuration(
                    atomic_numbers=atomic_numbers.numpy(),
                    positions=coords[i].numpy()
                )
                z_table = tools.AtomicNumberTable(torch.unique(atomic_numbers).tolist())
                atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=self.cutoff)
                
                torch_geo_data = Data(
                    x=torch.tensor(atomic_data.node_attrs, dtype=torch.float32),
                    positions=torch.tensor(coords[i], dtype=torch.float32),
                    edge_index=torch.tensor(atomic_data.edge_index, dtype=torch.long),
                    atomic_numbers=atomic_numbers,
                    key=key
                )
                atomic_data_list.append(torch_geo_data)
            
            return atomic_data_list, key

def compute_avg_num_neighbors(batch):
    _, receivers = batch.edge_index
    _, counts = torch.unique(receivers, return_counts=True)
    avg_num_neighbors = torch.mean(counts.float())
    return avg_num_neighbors.item()

def custom_collate(batch):
    # batch is a list of tuples (atomic_data_list, key)
    all_conformers = [item for sublist, _ in batch for item in sublist]
    keys = [key for _, key in batch]
    return Batch.from_data_list(all_conformers), keys

from torch_geometric.data import Batch

def process_data(conformer_dataset, batch_size=32):
    total_batches = 0
    total_conformers = 0

    # Create a DataLoader for the entire dataset
    data_loader = torch.utils.data.DataLoader(
        dataset=conformer_dataset,
        batch_size=1,  # Process one ensemble at a time
        shuffle=False,
        collate_fn=lambda x: x[0]  # Don't batch, just return the first (and only) item
    )

    for atomic_data_list, key in data_loader:
        num_conformers = len(atomic_data_list)
        total_conformers += num_conformers

        print(f"\nProcessing Conformer Ensemble: {key}")
        print(f"Number of conformers in this ensemble: {num_conformers}")

        # Process conformers in batches
        for i in range(0, num_conformers, batch_size):
            batch_conformers = atomic_data_list[i:i+batch_size]
            total_batches += 1

            print(f"\nBatch {total_batches} in Ensemble: {key}")
            print(f"Number of conformers in this batch: {len(batch_conformers)}")

            # Create unique_atomic_numbers list while preserving order
            unique_atomic_numbers = []
            for conformer in batch_conformers:
                for atomic_number in conformer.atomic_numbers:
                    if atomic_number.item() not in unique_atomic_numbers:
                        unique_atomic_numbers.append(atomic_number.item())

            avg_num_neighbors = sum(compute_avg_num_neighbors(conformer) for conformer in batch_conformers) / len(batch_conformers)
        
            print(f"Unique Atomic Numbers: {unique_atomic_numbers}")
            print(f"Average number of neighbors: {avg_num_neighbors:.2f}")

            # Yield the batch of conformers, unique atomic numbers, and average number of neighbors
            yield batch_conformers, unique_atomic_numbers, avg_num_neighbors

        print(f"\nFinished processing Conformer Ensemble: {key}")
        print("=" * 50)

    print(f"\nTotal number of batches processed: {total_batches}")
    print(f"Total number of conformers processed: {total_conformers}")