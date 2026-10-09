"""Benchmark dataset catalog, aliases, mirrors, and modality categorization."""


# ==============================================================================
# 1. Dataset Catalog by Modality
# ==============================================================================

DATASET_CATALOG: dict[str, list[str]] = {
    "vision": [
        "ylecun/mnist",
        "uoft-cs/cifar10",
        "uoft-cs/cifar100",
        "zalando-datasets/fashion_mnist",
        "flwrlabs/femnist",
        "zh-plus/tiny-imagenet",
        "flwrlabs/usps",
        "flwrlabs/pacs",
        "flwrlabs/cinic10",
        "flwrlabs/caltech101",
        "flwrlabs/office-home",
        "flwrlabs/fed-isic2019",
        "ufldl-stanford/svhn",
        "sasha/dog-food",
        "Mike0307/MNIST-M",
    ],
    "audio": [
        "google/speech_commands",
        "flwrlabs/ambient-acoustic-context",
        "fixie-ai/common_voice_17_0",
        "fixie-ai/librispeech_asr",
    ],
    "tabular": [
        "scikit-learn/adult-census-income",
        "jlh/uci-mushrooms",
        "scikit-learn/iris",
        "jiahborcn/chembl_aqsol",
        "jiahborcn/chembl_multiassay_activity",
    ],
    "text": [
        "google-research-datasets/mbpp",
        "openai/openai_humaneval",
        "lukaemon/mmlu",
        "takala/financial_phrasebank",
        "pauri32/fiqa-2018",
        "zeroshot/twitter-financial-news-sentiment",
        "bigbio/pubmed_qa",
        "openlifescienceai/medmcqa",
        "bigbio/med_qa",
    ],
    "multimodal": [
        "panitsasi/fedjam",
    ],
}

# Intuitive short aliases mapped to canonical Hugging Face Hub repositories
DATASET_ALIASES: dict[str, str] = {
    # Vision
    "mnist": "ylecun/mnist",
    "cifar10": "uoft-cs/cifar10",
    "cifar100": "uoft-cs/cifar100",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
    "femnist": "flwrlabs/femnist",
    "tiny_imagenet": "zh-plus/tiny-imagenet",
    "tiny-imagenet": "zh-plus/tiny-imagenet",
    "usps": "flwrlabs/usps",
    "pacs": "flwrlabs/pacs",
    "cinic10": "flwrlabs/cinic10",
    "caltech101": "flwrlabs/caltech101",
    "office_home": "flwrlabs/office-home",
    "office-home": "flwrlabs/office-home",
    "fed_isic2019": "flwrlabs/fed-isic2019",
    "svhn": "ufldl-stanford/svhn",
    "dog_food": "sasha/dog-food",
    "mnist_m": "Mike0307/MNIST-M",
    "mnist-m": "Mike0307/MNIST-M",
    # Audio
    "speech_commands": "google/speech_commands",
    "ambient_acoustic": "flwrlabs/ambient-acoustic-context",
    "common_voice": "fixie-ai/common_voice_17_0",
    "librispeech": "fixie-ai/librispeech_asr",
    # Tabular
    "adult_census": "scikit-learn/adult-census-income",
    "mushrooms": "jlh/uci-mushrooms",
    "iris": "scikit-learn/iris",
    "chembl_aqsol": "jiahborcn/chembl_aqsol",
    "chembl_multiassay": "jiahborcn/chembl_multiassay_activity",
    # Text
    "mbpp": "google-research-datasets/mbpp",
    "humaneval": "openai/openai_humaneval",
    "mmlu": "lukaemon/mmlu",
    "financial_phrasebank": "takala/financial_phrasebank",
    "fiqa": "pauri32/fiqa-2018",
    "twitter_financial_news": "zeroshot/twitter-financial-news-sentiment",
    "pubmed_qa": "bigbio/pubmed_qa",
    "medmcqa": "openlifescienceai/medmcqa",
    "med_qa": "bigbio/med_qa",
    # Multimodal
    "fedjam": "panitsasi/fedjam",
}

# Modern Parquet mirrors for legacy Hub repos with deprecated .py loader scripts
HF_MODERN_MIRRORS: dict[str, tuple[str, str | None]] = {
    "google/speech_commands": ("arbml/Speech_Commands_Dataset", None),
    "speech_commands": ("arbml/Speech_Commands_Dataset", None),
    "takala/financial_phrasebank": ("warwickai/financial_phrasebank_mirror", None),
    "financial_phrasebank": ("warwickai/financial_phrasebank_mirror", None),
    "lukaemon/mmlu": ("cais/mmlu", "all"),
    "mmlu": ("cais/mmlu", "all"),
    "bigbio/pubmed_qa": ("qiaojin/PubMedQA", "pqa_labeled"),
    "pubmed_qa": ("qiaojin/PubMedQA", "pqa_labeled"),
}

# Default configuration (subset) required by certain multi-config Hub datasets
DATASET_DEFAULT_CONFIGS: dict[str, str] = {
    "ufldl-stanford/svhn": "cropped_digits",
    "cais/mmlu": "all",
    "lukaemon/mmlu": "all",
    "qiaojin/PubMedQA": "pqa_labeled",
    "bigbio/pubmed_qa": "pqa_labeled",
    "fixie-ai/common_voice_17_0": "en",
    "fixie-ai/librispeech_asr": "clean",
}

# Auto-register repo suffix shortcuts (e.g. 'cifar10' from 'uoft-cs/cifar10')
for _repo_list in DATASET_CATALOG.values():
    for _repo in _repo_list:
        _slug = _repo.split("/")[-1].lower()
        DATASET_ALIASES.setdefault(_slug, _repo)
        DATASET_ALIASES.setdefault(_slug.replace("-", "_"), _repo)

# Map canonical IDs, aliases, and slugs to their respective modality
DATASET_TO_MODALITY: dict[str, str] = {}
for _mod, _repo_list in DATASET_CATALOG.items():
    for _repo in _repo_list:
        DATASET_TO_MODALITY[_repo] = _mod
        DATASET_TO_MODALITY[_repo.replace("/", "_").lower()] = _mod
        _slug = _repo.split("/")[-1].lower()
        DATASET_TO_MODALITY[_slug] = _mod
        DATASET_TO_MODALITY[_slug.replace("-", "_")] = _mod
for _alias, _repo in DATASET_ALIASES.items():
    if _repo in DATASET_TO_MODALITY:
        DATASET_TO_MODALITY[_alias] = DATASET_TO_MODALITY[_repo]


def resolve_dataset_name(dataset_name: str) -> str:
    """Resolve shorthand dataset names to canonical Hugging Face repository IDs."""
    key = dataset_name.strip().lower()
    return DATASET_ALIASES.get(
        key, DATASET_ALIASES.get(key.replace("-", "_"), dataset_name)
    )


def get_dataset_modality(dataset_name: str) -> str:
    """Return the modality category ('vision', 'audio', 'tabular', 'text', 'multimodal') for a dataset."""
    canonical = resolve_dataset_name(dataset_name)
    if canonical in DATASET_TO_MODALITY:
        return DATASET_TO_MODALITY[canonical]
    clean = canonical.replace("/", "_").lower()
    if clean in DATASET_TO_MODALITY:
        return DATASET_TO_MODALITY[clean]
    raw_slug = dataset_name.strip().lower()
    if raw_slug in DATASET_TO_MODALITY:
        return DATASET_TO_MODALITY[raw_slug]
    return "misc"


def list_supported_datasets(
    modality: str | None = None,
) -> list[str] | dict[str, list[str]]:
    """List available benchmark datasets, optionally filtered by modality.

    Parameters
    ----------
    modality : Optional[str]
        Filter by modality ('vision', 'audio', 'tabular', 'text', 'multimodal').
        If None, returns all datasets categorized by modality dictionary.

    Returns
    -------
    list[str] | dict[str, list[str]]
        List of canonical repository IDs or full dictionary.
    """
    if modality:
        key = modality.strip().lower()
        if key not in DATASET_CATALOG:
            valid = list(DATASET_CATALOG.keys())
            raise ValueError(
                f"Unknown modality '{modality}'. Supported modalities: {valid}"
            )
        return DATASET_CATALOG[key]
    return DATASET_CATALOG
