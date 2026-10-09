"""Environment and cache directory configuration for Hugging Face datasets."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def get_project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def configure_hf_cache_dir() -> Path:
    """Ensure local.env is loaded and Hugging Face tokens and cache point strictly to <project_root>/data/raw."""
    # Load credentials from local.env or .env
    env_file = find_dotenv("local.env") or find_dotenv(".env")
    if env_file:
        load_dotenv(env_file)

    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token

    root = get_project_root()
    raw_dir = (root / "data" / "raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    hub_dir = (raw_dir / "hub").resolve()
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch_dir = (raw_dir / "torch").resolve()

    os.environ["HF_HOME"] = str(raw_dir)
    os.environ["HF_DATASETS_CACHE"] = str(raw_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["TORCH_HOME"] = str(torch_dir)
    os.environ["HUGGINGFACE_ASSETS_CACHE"] = str((raw_dir / "assets").resolve())

    # Patch already-imported module configs if any
    try:
        import datasets.config as _ds_cfg

        _ds_cfg.HF_DATASETS_CACHE = raw_dir
    except (ImportError, AttributeError):
        pass

    try:
        import huggingface_hub.constants as _hub_c

        _hub_c.HF_HOME = raw_dir
        _hub_c.HF_HUB_CACHE = hub_dir
        _hub_c.HUGGINGFACE_HUB_CACHE = hub_dir
        _hub_c.HF_ASSETS_CACHE = (raw_dir / "assets").resolve()
    except (ImportError, AttributeError):
        pass

    return raw_dir
