# conformer_ensemble_embedding_combiner.py

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3

class ConformerEnsembleEmbeddingCombiner(nn.Module):
    def __init__(self, scalar_dim, vector_dim):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

        # Deep Sets
        self.deep_sets_phi_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
        )
        self.deep_sets_phi_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU(),
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU(),
        )
        self.deep_sets_rho_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
        )
        self.deep_sets_rho_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU(),
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU(),
        )

        # Self-Attention
        self.attention_scalar = nn.Linear(scalar_dim, scalar_dim)
        self.attention_vector = nn.Linear(vector_dim * 3, vector_dim * 3)
        self.self_attention_phi_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU()
        )
        self.self_attention_phi_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU(),
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU()
        )
        self.self_attention_rho_scalar = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU()
        )
        self.self_attention_rho_vector = nn.Sequential(
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU(),
            nn.Linear(vector_dim * 3, vector_dim * 3),
            nn.SiLU()
        )

    def mean_pooling(self, scalar, vector, batch):
        scalar_result = scatter(scalar, batch, dim=0, reduce='mean')
        vector_result = scatter(vector, batch, dim=0, reduce='mean')
        return scalar_result, vector_result

    def deep_sets(self, scalar, vector, batch):
        scalar = self.deep_sets_phi_scalar(scalar)
        vector = self.deep_sets_phi_vector(vector.view(vector.shape[0], -1))
        scalar = scatter(scalar, batch, dim=0, reduce='mean')
        vector = scatter(vector, batch, dim=0, reduce='mean')
        scalar = self.deep_sets_rho_scalar(scalar)
        vector = self.deep_sets_rho_vector(vector)
        return scalar, vector.view(vector.shape[0], -1, 3)

    def self_attention(self, scalar, vector, batch):
        scalar = self.self_attention_phi_scalar(scalar)
        vector = self.self_attention_phi_vector(vector.view(vector.shape[0], -1))
        scalar_scores = self.attention_scalar(scalar)
        vector_scores = self.attention_vector(vector)
        
        unique_batches = batch.unique()
        scalar_outputs, vector_outputs = [], []
        
        for b in unique_batches:
            mask = (batch == b)
            scalar_batch, vector_batch = scalar[mask], vector[mask]
            scalar_scores_batch, vector_scores_batch = scalar_scores[mask], vector_scores[mask]
            
            attention_weights = torch.softmax(
                torch.matmul(scalar_scores_batch, scalar_scores_batch.transpose(0, 1)) +
                torch.matmul(vector_scores_batch, vector_scores_batch.transpose(0, 1)),
                dim=1
            )
            
            scalar_weighted = torch.matmul(attention_weights, scalar_batch)
            vector_weighted = torch.matmul(attention_weights, vector_batch)
            
            scalar_aggregated = scalar_weighted.sum(dim=0, keepdim=True)
            vector_aggregated = vector_weighted.sum(dim=0, keepdim=True)
            
            scalar_outputs.append(scalar_aggregated)
            vector_outputs.append(vector_aggregated)
        
        scalar_encoded = torch.cat(scalar_outputs, dim=0)
        vector_encoded = torch.cat(vector_outputs, dim=0)
        
        scalar_encoded = self.self_attention_rho_scalar(scalar_encoded)
        vector_encoded = self.self_attention_rho_vector(vector_encoded)
        return scalar_encoded, vector_encoded.view(vector_encoded.shape[0], -1, 3)

    def forward(self, conformer_embeddings, batch_indices):
        total_dim = conformer_embeddings.shape[1]
        scalar_dim = min(self.scalar_dim, total_dim // 4)
        vector_dim = min(self.vector_dim, (total_dim - scalar_dim) // 3)
        scalar = conformer_embeddings[:, :self.scalar_dim]
        vector = conformer_embeddings[:, self.scalar_dim:].view(-1, self.vector_dim, 3)
        
        mean_pooled_scalar, mean_pooled_vector = self.mean_pooling(scalar, vector, batch_indices)
        deep_sets_scalar, deep_sets_vector = self.deep_sets(scalar, vector, batch_indices)
        self_attention_scalar, self_attention_vector = self.self_attention(scalar, vector, batch_indices)

        return {
            'mean_pooling': (mean_pooled_scalar, mean_pooled_vector),
            'deep_sets': (deep_sets_scalar, deep_sets_vector),
            'self_attention': (self_attention_scalar, self_attention_vector)
        }

def process_conformer_ensemble(conformer_embeddings):
    print(f"process_conformer_ensemble input shape: {conformer_embeddings.shape}")
    
    # Flatten all dimensions except the first one
    num_conformers = conformer_embeddings.shape[0]
    flattened_embeddings = conformer_embeddings.view(num_conformers, -1)
    total_dim = flattened_embeddings.shape[1]
    
    scalar_dim = total_dim // 4  # Assuming 1/4 of the features are scalars
    vector_dim = scalar_dim  # Assuming the remaining 3/4 are vectors (3 components each)
    
    print(f"Num conformers: {num_conformers}, Total dim: {total_dim}")
    print(f"Scalar dim: {scalar_dim}, Vector dim: {vector_dim}")
    
    # Create batch indices
    batch_indices = torch.arange(num_conformers)

    # Initialize the combiner
    combiner = ConformerEnsembleEmbeddingCombiner(scalar_dim, vector_dim)

    # Process the conformer ensemble
    results = combiner(flattened_embeddings, batch_indices)

    return results