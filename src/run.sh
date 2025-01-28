#!/bin/bash

# Source the configuration
source ./config.sh

# Default values for optional arguments
embedding_type=${EQUICAT_DEFAULT_EMBEDDING_TYPE}
scheduler=${EQUICAT_DEFAULT_SCHEDULER}
num_families=""
ensembles_per_family=""
resume_checkpoint=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --embedding_type)
            embedding_type="$2"
            shift 2
            ;;
        --scheduler)
            scheduler="$2"
            shift 2
            ;;
        --num_families)
            num_families="--num_families $2"
            shift 2
            ;;
        --ensembles_per_family)
            ensembles_per_family="--ensembles_per_family $2"
            shift 2
            ;;
        --resume_from_checkpoint)
            resume_checkpoint="--resume_from_checkpoint $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate embedding type
valid_embedding_types=("mean_pooling" "deep_sets" "self_attention" "improved_deep_sets" "improved_self_attention" "all")
if [[ ! " ${valid_embedding_types[@]} " =~ " ${embedding_type} " ]]; then
    echo "Invalid embedding type. Must be one of: ${valid_embedding_types[*]}"
    exit 1
fi

# Validate scheduler
valid_schedulers=("plateau" "step" "cosine" "cosine_restart" "onecycle")
if [[ ! " ${valid_schedulers[@]} " =~ " ${scheduler} " ]]; then
    echo "Invalid scheduler. Must be one of: ${valid_schedulers[*]}"
    exit 1
fi

# Run the training script with all arguments
python train.py \
    --embedding_type "${embedding_type}" \
    --scheduler "${scheduler}" \
    ${num_families} \
    ${ensembles_per_family} \
    ${resume_checkpoint}

exit_code=$?

# Log completion
echo "Training completed with exit code: ${exit_code}"
if [ ${exit_code} -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code ${exit_code}"
fi