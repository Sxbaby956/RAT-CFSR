from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

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

