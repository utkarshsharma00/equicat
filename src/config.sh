#!/bin/bash

# Detect environment
if [ "$HOSTNAME" = "your_eagle_hostname" ]; then
    # Eagle paths
    export EQUICAT_FAMILY1_PATH="/eagle/FOUND4CHEM/utkarsh/dataset/bpa_aligned.clib"
    export EQUICAT_FAMILY2_PATH="/eagle/FOUND4CHEM/utkarsh/dataset/imine_confs.clib"
    export EQUICAT_FAMILY3_PATH="/eagle/FOUND4CHEM/utkarsh/dataset/thiols.clib"
    export EQUICAT_FAMILY4_PATH="/eagle/FOUND4CHEM/utkarsh/dataset/product_confs.clib"
    
    # Output directories
    export EQUICAT_OUTPUT_PATH="/eagle/FOUND4CHEM/utkarsh/project/equicat/epoch_large"
    export EQUICAT_CLUSTERING_DIR="/eagle/FOUND4CHEM/utkarsh/project/equicat/src/clustering_results"
    export EQUICAT_LOG_FILE="/eagle/FOUND4CHEM/utkarsh/project/equicat/epoch_large/data_loader.log"
    export EQUICAT_PROFILE_OUTPUT="/eagle/FOUND4CHEM/utkarsh/project/equicat/epoch_large/profiler_output.json"
    export EQUICAT_CHECKPOINT_PATH="/eagle/FOUND4CHEM/utkarsh/project/equicat/epoch_large/checkpoints/final_equicat_model.pt"
else
    # Local paths
    export EQUICAT_FAMILY1_PATH="/Users/utkarsh/MMLI/molli-data/00-libraries/bpa_aligned.clib"
    export EQUICAT_FAMILY2_PATH="/Users/utkarsh/MMLI/molli-data/00-libraries/imine_confs.clib"
    export EQUICAT_FAMILY3_PATH="/Users/utkarsh/MMLI/molli-data/00-libraries/thiols.clib"
    export EQUICAT_FAMILY4_PATH="/Users/utkarsh/MMLI/molli-data/00-libraries/product_confs.clib"
    
    # Output directories
    export EQUICAT_OUTPUT_PATH="/Users/utkarsh/MMLI/equicat/epoch_large"
    export EQUICAT_CLUSTERING_DIR="/Users/utkarsh/MMLI/equicat/src/clustering_results"
    export EQUICAT_LOG_FILE="/Users/utkarsh/MMLI/equicat/epoch_large/data_loader.log"
    export EQUICAT_PROFILE_OUTPUT="/Users/utkarsh/MMLI/equicat/epoch_large/profiler_output.json"
    export EQUICAT_CHECKPOINT_PATH="/Users/utkarsh/MMLI/equicat/develop_op/checkpoints/best_model.pt"
fi

# Model parameters
export EQUICAT_CUTOFF=6.0
export EQUICAT_MAX_CONFORMERS=10
export EQUICAT_SAMPLE_SIZE=30
export EQUICAT_LEARNING_RATE=1e-3
export EQUICAT_EPOCHS=500
export EQUICAT_GRADIENT_CLIP=1.0
export EQUICAT_CHECKPOINT_INTERVAL=25

# Model configuration
export EQUICAT_NUM_BESSEL=8
export EQUICAT_NUM_POLYNOMIAL_CUTOFF=6
export EQUICAT_MAX_ELL=2
export EQUICAT_NUM_INTERACTIONS=2
export EQUICAT_HIDDEN_IRREPS="256x0e + 256x1o"
export EQUICAT_MLP_IRREPS="16x0e"
export EQUICAT_CORRELATION=3

# Clustering parameters
export EQUICAT_CLUSTERING_CUTOFF=0.2

# Training parameters
export EQUICAT_EARLY_STOPPING_PATIENCE=200
export EQUICAT_DEFAULT_SCHEDULER="step"
export EQUICAT_DEFAULT_EMBEDDING_TYPE="improved_self_attention"

# Loss function weights
export EQUICAT_SAME_FAMILY_SAME_CLUSTER_WEIGHT=1.0
export EQUICAT_SAME_FAMILY_DIFF_CLUSTER_WEIGHT=0.3
export EQUICAT_DIFF_FAMILY_WEIGHT=-2.0
export EQUICAT_TEMPERATURE=0.25

# Excluded molecules
export EQUICAT_EXCLUDED_MOLECULES="179_vi,181_i,180_i,180_vi,178_i,178_vi"

# Random seeds
export EQUICAT_TORCH_SEED=42
export EQUICAT_NUMPY_SEED=42
export EQUICAT_RANDOM_SEED=42