from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# ``np.trapezoid`` was introduced in NumPy 2.0; keep compatibility with 1.x.
_trapezoid = getattr(np, "trapezoid", np.trapz)


def oscr_score(
    true_labels: np.ndarray,
    candidate_labels: np.ndarray,
    unknown_scores: np.ndarray,
) -> float:
    """Area under the Correct Classification Rate vs unknown FPR curve."""
    true_labels = np.asarray(true_labels, dtype=np.int64)
    candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    known = true_labels >= 0
    unknown = ~known
    if not np.any(known) or not np.any(unknown):
        return float("nan")

    order = np.argsort(unknown_scores)
    sorted_scores = unknown_scores[order]
    sorted_unknown = unknown[order]
    sorted_correct = (known & (candidate_labels == true_labels))[order]
    cumulative_unknown = np.cumsum(sorted_unknown)
    cumulative_correct = np.cumsum(sorted_correct)

    change = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    fpr = cumulative_unknown[change] / np.sum(unknown)
    ccr = cumulative_correct[change] / np.sum(known)
    fpr = np.r_[0.0, fpr]
    ccr = np.r_[0.0, ccr]
    return float(_trapezoid(ccr, fpr))


def evaluate_open_set(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    candidate_labels: np.ndarray,
    unknown_scores: np.ndarray,
) -> dict[str, float]:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    known = true_labels >= 0
    unknown = ~known

    correct_known = known & (predicted_labels == true_labels)
    accepted = predicted_labels >= 0
    rejected = ~accepted

    metrics = {
        "known_accuracy": float(correct_known.sum() / max(known.sum(), 1)),
        "true_known_rate": float((known & accepted).sum() / max(known.sum(), 1)),
        "true_unknown_rate": float((unknown & rejected).sum() / max(unknown.sum(), 1)),
        "known_precision": float(correct_known.sum() / max(accepted.sum(), 1)),
        "unknown_precision": float((unknown & rejected).sum() / max(rejected.sum(), 1)),
        "macro_f1_open": float(f1_score(true_labels, predicted_labels, average="macro")),
        "oscr": oscr_score(true_labels, candidate_labels, unknown_scores),
    }
    binary_unknown = unknown.astype(np.int64)
    if np.unique(binary_unknown).size == 2:
        metrics["auroc"] = float(roc_auc_score(binary_unknown, unknown_scores))
        metrics["aupr"] = float(average_precision_score(binary_unknown, unknown_scores))
    else:
        metrics["auroc"] = float("nan")
        metrics["aupr"] = float("nan")
    return metrics


def open_set_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    known_class_names: list[str],
    unknown_name: str = "unknown",
) -> dict[str, object]:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    class_ids = [-1, *range(len(known_class_names))]
    label_names = [unknown_name, *known_class_names]
    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_ids)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        np.maximum(row_sums, 1),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )
    return {
        "labels": label_names,
        "class_ids": class_ids,
        "counts": matrix.astype(int).tolist(),
        "row_normalized": normalized.tolist(),
    }


def format_confusion_matrix(confusion: dict[str, object]) -> str:
    labels = [str(label) for label in confusion["labels"]]
    counts = np.asarray(confusion["counts"], dtype=np.int64)
    normalized = np.asarray(confusion["row_normalized"], dtype=np.float64)
    width = max(10, *(len(label) for label in labels))

    lines = ["Confusion matrix counts (rows=true, cols=predicted)"]
    header = " " * width + "  " + "  ".join(label.rjust(width) for label in labels)
    lines.append(header)
    for label, row in zip(labels, counts):
        values = "  ".join(str(int(value)).rjust(width) for value in row)
        lines.append(label.rjust(width) + "  " + values)

    lines.append("")
    lines.append("Confusion matrix row-normalized (rows sum to 1.0)")
    lines.append(header)
    for label, row in zip(labels, normalized):
        values = "  ".join(f"{value:.4f}".rjust(width) for value in row)
        lines.append(label.rjust(width) + "  " + values)
    return "\n".join(lines) + "\n"


def _draw_confusion_matrix(ax: object, confusion: dict[str, object]) -> object:
    """Draw one confusion matrix heatmap onto ``ax``.

    Cells are colored by the row-normalized fraction (single-hue ``Blues``
    ramp, light -> dark) and annotated with the raw count plus the row
    percentage. Returns the ``imshow`` artist for shared colorbar use. Assumes
    the caller has already selected a non-interactive Matplotlib backend.
    """
    import matplotlib.pyplot as plt

    labels = [str(label) for label in confusion["labels"]]
    counts = np.asarray(confusion["counts"], dtype=np.int64)
    normalized = np.asarray(confusion["row_normalized"], dtype=np.float64)
    n = len(labels)

    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    # Thin white separators between cells.
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)

    for i in range(n):
        for j in range(n):
            value = normalized[i, j]
            rgba = image.cmap(image.norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.5 else "#1a1a1a"
            ax.text(
                j,
                i,
                f"{counts[i, j]:,}\n({value:.1%})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, right=False, bottom=False, left=False)
    return image


def save_confusion_matrix_image(
    confusion: dict[str, object],
    path: str | Path,
    title: str | None = None,
    dpi: int = 200,
) -> None:
    """Render a single open-set confusion matrix as a PNG heatmap.

    Matplotlib is imported lazily so the core training/evaluation paths do not
    require it as a hard dependency.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(confusion["labels"])
    fig, ax = plt.subplots(figsize=(1.7 * n + 2.6, 1.7 * n + 1.8))
    image = _draw_confusion_matrix(ax, confusion)
    if title:
        ax.set_title(title, pad=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
