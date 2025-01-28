import os
import logging

logger = logging.getLogger(__name__)

def get_family_paths():
    """Get family paths from environment variables."""
    return {
        "family1": os.environ["EQUICAT_FAMILY1_PATH"],
        "family2": os.environ["EQUICAT_FAMILY2_PATH"],
        "family3": os.environ["EQUICAT_FAMILY3_PATH"],
        "family4": os.environ["EQUICAT_FAMILY4_PATH"],
    }

def get_model_params():
    """Get model parameters from environment variables."""
    return {
        "cutoff": float(os.environ["EQUICAT_CUTOFF"]),
        "max_conformers": int(os.environ["EQUICAT_MAX_CONFORMERS"]),
        "sample_size": int(os.environ["EQUICAT_SAMPLE_SIZE"]),
        "learning_rate": float(os.environ["EQUICAT_LEARNING_RATE"]),
        "epochs": int(os.environ["EQUICAT_EPOCHS"]),
        "gradient_clip": float(os.environ["EQUICAT_GRADIENT_CLIP"]),
        "checkpoint_interval": int(os.environ["EQUICAT_CHECKPOINT_INTERVAL"]),
        "num_bessel": int(os.environ["EQUICAT_NUM_BESSEL"]),
        "num_polynomial_cutoff": int(os.environ["EQUICAT_NUM_POLYNOMIAL_CUTOFF"]),
        "max_ell": int(os.environ["EQUICAT_MAX_ELL"]),
        "num_interactions": int(os.environ["EQUICAT_NUM_INTERACTIONS"]),
        "hidden_irreps": os.environ["EQUICAT_HIDDEN_IRREPS"],
        "mlp_irreps": os.environ["EQUICAT_MLP_IRREPS"],
        "correlation": int(os.environ["EQUICAT_CORRELATION"]),
    }

def get_training_params():
    """Get training parameters from environment variables."""
    return {
        "early_stopping_patience": int(os.environ["EQUICAT_EARLY_STOPPING_PATIENCE"]),
        "default_scheduler": os.environ["EQUICAT_DEFAULT_SCHEDULER"],
        "default_embedding_type": os.environ["EQUICAT_DEFAULT_EMBEDDING_TYPE"],
    }

def get_loss_weights():
    """Get loss function weights from environment variables."""
    return {
        "same_family_same_cluster": float(os.environ["EQUICAT_SAME_FAMILY_SAME_CLUSTER_WEIGHT"]),
        "same_family_diff_cluster": float(os.environ["EQUICAT_SAME_FAMILY_DIFF_CLUSTER_WEIGHT"]),
        "diff_family": float(os.environ["EQUICAT_DIFF_FAMILY_WEIGHT"]),
        "temperature": float(os.environ["EQUICAT_TEMPERATURE"]),
    }

def get_paths():
    """Get output paths from environment variables."""
    return {
        "output_path": os.environ["EQUICAT_OUTPUT_PATH"],
        "clustering_dir": os.environ["EQUICAT_CLUSTERING_DIR"],
        "log_file": os.environ["EQUICAT_LOG_FILE"],
        "profile_output": os.environ["EQUICAT_PROFILE_OUTPUT"],
        "checkpoint_path": os.environ["EQUICAT_CHECKPOINT_PATH"],
    }

def get_excluded_molecules():
    """Get list of excluded molecules from environment variables."""
    return os.environ["EQUICAT_EXCLUDED_MOLECULES"].split(",")