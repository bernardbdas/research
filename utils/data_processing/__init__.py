"""Data processing sub-package for raw caching, preprocessing, and Parquet export."""

from utils.data_processing.catalog import (
    DATASET_ALIASES,
    DATASET_CATALOG,
    DATASET_DEFAULT_CONFIGS,
    DATASET_TO_MODALITY,
    HF_MODERN_MIRRORS,
    get_dataset_modality,
    list_supported_datasets,
    resolve_dataset_name,
)
from utils.data_processing.cleanup import (
    clean_dataset_data,
    clean_modality_data,
    clean_processed_data,
    clean_raw_data,
)
from utils.data_processing.config import (
    configure_hf_cache_dir,
    get_project_root,
)
from utils.data_processing.loader import (
    get_federated_dataset,
    get_federated_dataset_from_parquet,
    load_processed_dataset,
)
from utils.data_processing.paths import DatasetPaths
from utils.data_processing.pipeline import (
    prepare_and_save_dataset,
    prepare_modality_datasets,
)

__all__ = [
    "DATASET_ALIASES",
    "DATASET_CATALOG",
    "DATASET_DEFAULT_CONFIGS",
    "DATASET_TO_MODALITY",
    "HF_MODERN_MIRRORS",
    "DatasetPaths",
    "clean_dataset_data",
    "clean_modality_data",
    "clean_processed_data",
    "clean_raw_data",
    "configure_hf_cache_dir",
    "get_dataset_modality",
    "get_federated_dataset",
    "get_federated_dataset_from_parquet",
    "get_project_root",
    "list_supported_datasets",
    "load_processed_dataset",
    "prepare_and_save_dataset",
    "prepare_modality_datasets",
    "resolve_dataset_name",
]
