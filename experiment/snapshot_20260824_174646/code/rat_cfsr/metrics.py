from __future__ import annotations

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
