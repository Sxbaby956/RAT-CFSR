from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CalibrationPrediction:
    labels: np.ndarray
    candidate_labels: np.ndarray
    candidate_scores: np.ndarray
    class_scores: np.ndarray


class ClassConditionalCalibrator:
    """Empirical-CDF calibration for per-class reconstruction errors.

    Each class-specific AE has its own error scale. Converting an error to its
    rank within that class's matching calibration distribution makes scores
    comparable across class branches. Lower scores are more class-compatible.
    """

    def __init__(self, threshold_quantile: float = 0.95) -> None:
        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0.5, 1.0)")
        self.threshold_quantile = float(threshold_quantile)
        self.sorted_errors: list[np.ndarray] = []
        self.thresholds: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return bool(self.sorted_errors) and self.thresholds is not None

    def fit(self, errors: np.ndarray, labels: np.ndarray) -> "ClassConditionalCalibrator":
        errors = np.asarray(errors, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if errors.ndim != 2 or labels.shape != (errors.shape[0],):
            raise ValueError("Expected errors [N, K] and labels [N]")

        self.sorted_errors = []
        thresholds = []
        for class_index in range(errors.shape[1]):
            matching = np.sort(errors[labels == class_index, class_index])
            if matching.size < 2:
                raise ValueError(
                    f"Class {class_index} needs at least two calibration samples"
                )
            self.sorted_errors.append(matching)
            matching_scores = self._cdf(matching, matching)
            thresholds.append(
                float(np.quantile(matching_scores, self.threshold_quantile))
            )
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        return self

    @staticmethod
    def _cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
        ranks = np.searchsorted(reference, values, side="right")
        return ranks.astype(np.float64) / float(reference.size)

    def transform(self, errors: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        errors = np.asarray(errors, dtype=np.float64)
        if errors.ndim != 2 or errors.shape[1] != len(self.sorted_errors):
            raise ValueError("Error matrix has the wrong number of classes")
        scores = np.empty_like(errors, dtype=np.float64)
        for class_index, reference in enumerate(self.sorted_errors):
            scores[:, class_index] = self._cdf(reference, errors[:, class_index])
        return scores

    def predict(self, errors: np.ndarray) -> CalibrationPrediction:
        scores = self.transform(errors)
        candidates = np.argmin(scores, axis=1)
        candidate_scores = scores[np.arange(scores.shape[0]), candidates]
        assert self.thresholds is not None
        accepted = candidate_scores <= self.thresholds[candidates]
        labels = np.where(accepted, candidates, -1)
        return CalibrationPrediction(
            labels=labels.astype(np.int64),
            candidate_labels=candidates.astype(np.int64),
            candidate_scores=candidate_scores,
            class_scores=scores,
        )

    def to_dict(self) -> dict[str, object]:
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        assert self.thresholds is not None
        return {
            "threshold_quantile": self.threshold_quantile,
            "thresholds": self.thresholds.tolist(),
            "sorted_errors": [values.tolist() for values in self.sorted_errors],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ClassConditionalCalibrator":
        calibrator = cls(float(payload["threshold_quantile"]))
        calibrator.thresholds = np.asarray(payload["thresholds"], dtype=np.float64)
        calibrator.sorted_errors = [
            np.asarray(values, dtype=np.float64) for values in payload["sorted_errors"]
        ]
        return calibrator

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClassConditionalCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class ClassConditionalScoreCalibrator:
    """Class-conditional thresholding for scalar unknown scores.

    Scores are expected to be oriented so that larger values are more unknown.
    Thresholds are estimated from correctly labeled calibration samples for each
    known class, then applied to the class predicted by the classifier.
    """

    def __init__(self, threshold_quantile: float = 0.95) -> None:
        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0.5, 1.0)")
        self.threshold_quantile = float(threshold_quantile)
        self.sorted_scores: list[np.ndarray] = []
        self.thresholds: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return bool(self.sorted_scores) and self.thresholds is not None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ClassConditionalScoreCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if scores.ndim != 1 or labels.shape != scores.shape:
            raise ValueError("Expected scores [N] and labels [N]")

        self.sorted_scores = []
        thresholds = []
        for class_index in range(int(labels.max()) + 1):
            matching = np.sort(scores[labels == class_index])
            if matching.size < 2:
                raise ValueError(
                    f"Class {class_index} needs at least two calibration samples"
                )
            self.sorted_scores.append(matching)
            thresholds.append(float(np.quantile(matching, self.threshold_quantile)))
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        return self

    def predict(
        self, scores: np.ndarray, candidate_labels: np.ndarray
    ) -> CalibrationPrediction:
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        scores = np.asarray(scores, dtype=np.float64)
        candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
        if scores.ndim != 1 or candidate_labels.shape != scores.shape:
            raise ValueError("Expected scores [N] and candidate_labels [N]")
        assert self.thresholds is not None
        accepted = scores <= self.thresholds[candidate_labels]
        labels = np.where(accepted, candidate_labels, -1)
        class_scores = np.repeat(scores[:, None], len(self.thresholds), axis=1)
        return CalibrationPrediction(
            labels=labels.astype(np.int64),
            candidate_labels=candidate_labels.astype(np.int64),
            candidate_scores=scores,
            class_scores=class_scores,
        )

    def to_dict(self) -> dict[str, object]:
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        assert self.thresholds is not None
        return {
            "threshold_quantile": self.threshold_quantile,
            "thresholds": self.thresholds.tolist(),
            "sorted_scores": [values.tolist() for values in self.sorted_scores],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ClassConditionalScoreCalibrator":
        calibrator = cls(float(payload["threshold_quantile"]))
        calibrator.thresholds = np.asarray(payload["thresholds"], dtype=np.float64)
        calibrator.sorted_scores = [
            np.asarray(values, dtype=np.float64) for values in payload["sorted_scores"]
        ]
        return calibrator

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClassConditionalScoreCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
