"""Inference, evaluation, and visual classification utilities with confidence scoring."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from utils.augmentation.normalization import CIFAR10_MEAN, CIFAR10_STD

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_class_names(dataset_name: str) -> list[str]:
    """Retrieve human-readable class names for canonical benchmark datasets."""
    name = dataset_name.lower()
    if "cifar" in name:
        return CIFAR10_CLASSES
    if "mnist" in name:
        return MNIST_CLASSES
    return [str(i) for i in range(10)]


def predict_with_confidence(
    model: Any,
    params: Any,
    images: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform real-time model inference and compute softmax confidence scores.

    Parameters
    ----------
    model : Any
        Flax model instance (with .apply method).
    params : Any
        Trained model parameters dictionary.
    images : jnp.ndarray
        Batch of input images (N, H, W, C).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - pred_labels: Array of predicted integer class indices.
        - confidences: Array of highest softmax probabilities (range 0.0 - 1.0).
        - all_probs: Array of softmax probability distributions (N, num_classes).
    """

    @jax.jit
    def _forward(p: Any, x: jnp.ndarray) -> jnp.ndarray:
        logits = model.apply({"params": p}, x, train=False)
        return jax.nn.softmax(logits, axis=-1)

    probs = np.array(_forward(params, images))
    pred_labels = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)
    return pred_labels, confidences, probs


def prepare_display_image(img: np.ndarray, is_cifar: bool = False) -> np.ndarray:
    """Prepare image array for matplotlib display, handling normalization and channel dims."""
    arr = np.array(img, dtype=np.float32)

    # Invert CIFAR-10 standardization if already normalized (negative values present)
    if is_cifar and arr.ndim == 3 and arr.shape[-1] == 3:
        if arr.min() < 0.0:
            arr = arr * CIFAR10_STD + CIFAR10_MEAN
        return np.clip(arr, 0.0, 1.0)

    # Single-channel grayscale (e.g. MNIST)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)

    return np.clip(arr, 0.0, 1.0)


def plot_sample_classifications(
    images: Any,
    true_labels: Any,
    model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    class_names: list[str] | None = None,
    num_samples: int = 8,
    is_cifar: bool = False,
    title: str = "Real-Time Validation Sample Classifications & Confidence Scores",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot real test set images with model predictions, confidence scores, and accuracy checks.

    Parameters
    ----------
    images : Any
        Array or list of test images (N, H, W, C).
    true_labels : Any
        Array or list of ground truth integer labels.
    model_predictions : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping of model label (e.g. "Centralized", "FedAvg") to (predicted_labels, confidences).
    class_names : Optional[list[str]]
        Human-readable class name mapping. Defaults to string integers.
    num_samples : int
        Number of validation images to display. Default is 8.
    is_cifar : bool
        Whether images are RGB CIFAR-10 format. Default is False.
    title : str
        Figure main super title.
    figsize : Optional[tuple[int, int]]
        Custom figure size. If None, dynamically scaled.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    n = min(len(images), num_samples)
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    if figsize is None:
        figsize = (max(16, n * 2.8), 5.5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for i in range(n):
        ax = axes[i]
        disp_img = prepare_display_image(images[i], is_cifar=is_cifar)
        cmap = None if is_cifar else "gray"
        ax.imshow(disp_img, cmap=cmap)
        ax.axis("off")

        true_lbl = int(true_labels[i])
        true_name = (
            class_names[true_lbl] if true_lbl < len(class_names) else str(true_lbl)
        )

        # Build classification telemetry text for each model
        text_lines = [f"True: {true_name}"]
        for model_name, (preds, confs) in model_predictions.items():
            pred_lbl = int(preds[i])
            pred_name = (
                class_names[pred_lbl] if pred_lbl < len(class_names) else str(pred_lbl)
            )
            conf_pct = float(confs[i]) * 100.0
            is_correct = pred_lbl == true_lbl
            mark = "✓" if is_correct else "✗"
            text_lines.append(f"{model_name}: {pred_name} ({conf_pct:4.1f}%) {mark}")

        ax.set_title("\n".join(text_lines), fontsize=9, pad=8)

    plt.tight_layout()
    return fig


def plot_prediction_confidence_bars(
    sample_img: Any,
    true_label: int,
    model_probabilities: dict[str, np.ndarray],
    class_names: list[str] | None = None,
    is_cifar: bool = False,
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot detailed class probability distribution comparison for an individual sample.

    Parameters
    ----------
    sample_img : Any
        Single image array (H, W, C).
    true_label : int
        Ground truth integer label.
    model_probabilities : dict[str, np.ndarray]
        Mapping of model label to softmax probability vector (num_classes,).
    class_names : Optional[list[str]]
        Human-readable class labels.
    is_cifar : bool
        Whether image is CIFAR-10 RGB format.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    num_classes = len(class_names)
    num_models = len(model_probabilities)

    fig, (ax_img, ax_bars) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 2.8]}
    )

    disp_img = prepare_display_image(sample_img, is_cifar=is_cifar)
    cmap = None if is_cifar else "gray"
    ax_img.imshow(disp_img, cmap=cmap)
    ax_img.axis("off")
    true_name = (
        class_names[true_label] if true_label < len(class_names) else str(true_label)
    )
    ax_img.set_title(f"True Class: {true_name}", fontsize=11, fontweight="bold")

    x = np.arange(num_classes)
    bar_width = 0.8 / max(1, num_models)

    colors = ["#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]

    for idx, (model_name, probs) in enumerate(model_probabilities.items()):
        offset = (idx - (num_models - 1) / 2) * bar_width
        c = colors[idx % len(colors)]
        ax_bars.bar(
            x + offset,
            probs * 100.0,
            width=bar_width,
            label=model_name,
            color=c,
            alpha=0.85,
        )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax_bars.set_ylabel("Confidence Score (%)", fontsize=10)
    ax_bars.set_ylim(0, 105)
    ax_bars.grid(axis="y", linestyle="--", alpha=0.5)
    ax_bars.legend(loc="upper right", framealpha=0.9)
    ax_bars.set_title(
        "Softmax Class Probability Distribution Across Models",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


def unpack_history_and_params(entry: Any) -> tuple[dict[str, Any], Any]:
    """Extract history dictionary and model parameters from tuple or dict format.

    Supports (history, params) 2-tuples, {'history': ..., 'params': ...} dictionaries,
    or direct history dictionaries.
    """
    if isinstance(entry, tuple):
        return entry[0], (entry[1] if len(entry) > 1 else None)
    if isinstance(entry, dict) and "history" in entry:
        return entry["history"], entry.get("params")
    return entry, None
