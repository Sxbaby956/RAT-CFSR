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
    """CFSR-style class-conditional thresholds for reconstruction errors.

    Reconstruction error is not comparable across SNR levels: at -20 dB every
    class reconstructs poorly, so a single per-class threshold fitted over the
    full SNR sweep is dominated by noisy samples and rejects nothing at high
    SNR. CFSR's protocol reports AUROC/OSCR *per SNR*, so the matching-branch
    thresholds are fitted per (class, SNR) group. See docs section 6.1 plus the
    per-SNR evaluation of section 7.1.
    """

    def __init__(self, threshold_quantile: float = 0.95) -> None:
        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0.5, 1.0)")
        self.threshold_quantile = float(threshold_quantile)
        self.sorted_errors: list[np.ndarray] = []
        self.thresholds: np.ndarray | None = None
        # Per-(class, snr) thresholds used by predict_snr. Keys are int snr.
        self.snr_thresholds: dict[tuple[int, int], float] = {}

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
            thresholds.append(float(np.quantile(matching, self.threshold_quantile)))
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.snr_thresholds = {}
        return self

    def fit_snr(
        self, errors: np.ndarray, labels: np.ndarray, snrs: np.ndarray
    ) -> "ClassConditionalCalibrator":
        """Fit one threshold per (class, SNR) group.

        Also fills the flat ``thresholds`` array from the pooled per-class
        errors as a fallback for the (impossible for known classes) case where a
        test SNR was never observed during calibration.
        """
        self.fit(errors, labels)
        errors = np.asarray(errors, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        snrs = np.asarray(snrs, dtype=np.int64)
        if labels.shape != (errors.shape[0],) or snrs.shape != labels.shape:
            raise ValueError("Expected aligned errors [N,K], labels [N], snrs [N]")

        self.snr_thresholds = {}
        for class_index in range(errors.shape[1]):
            class_mask = labels == class_index
            for snr in np.unique(snrs[class_mask]):
                matching = errors[class_mask & (snrs == snr), class_index]
                if matching.size < 2:
                    continue
                key = (int(class_index), int(snr))
                self.snr_thresholds[key] = float(
                    np.quantile(matching, self.threshold_quantile)
                )
        return self

    def transform(self, errors: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        errors = np.asarray(errors, dtype=np.float64)
        if errors.ndim != 2 or errors.shape[1] != len(self.sorted_errors):
            raise ValueError("Error matrix has the wrong number of classes")
        return errors

    def _predict(
        self, scores: np.ndarray, candidate_thresholds: np.ndarray
    ) -> CalibrationPrediction:
        candidates = np.argmin(scores, axis=1)
        candidate_scores = scores[np.arange(scores.shape[0]), candidates]
        accepted = candidate_scores <= candidate_thresholds
        labels = np.where(accepted, candidates, -1)
        return CalibrationPrediction(
            labels=labels.astype(np.int64),
            candidate_labels=candidates.astype(np.int64),
            candidate_scores=candidate_scores,
            class_scores=scores,
        )

    def predict(self, errors: np.ndarray) -> CalibrationPrediction:
        scores = self.transform(errors)
        assert self.thresholds is not None
        candidates = np.argmin(scores, axis=1)
        return self._predict(scores, self.thresholds[candidates])

    def predict_snr(
        self, errors: np.ndarray, snrs: np.ndarray
    ) -> CalibrationPrediction:
        """Predict using per-(class, SNR) thresholds when available."""
        scores = self.transform(errors)
        snrs = np.asarray(snrs, dtype=np.int64)
        if snrs.shape != (scores.shape[0],):
            raise ValueError("Expected snrs [N] aligned with errors [N]")
        assert self.thresholds is not None
        candidates = np.argmin(scores, axis=1)
        candidate_thresholds = np.empty(scores.shape[0], dtype=np.float64)
        for index, (candidate, snr) in enumerate(zip(candidates, snrs)):
            key = (int(candidate), int(snr))
            candidate_thresholds[index] = self.snr_thresholds.get(
                key, float(self.thresholds[candidate])
            )
        return self._predict(scores, candidate_thresholds)

    def to_dict(self) -> dict[str, object]:
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted")
        assert self.thresholds is not None
        return {
            "threshold_quantile": self.threshold_quantile,
            "thresholds": self.thresholds.tolist(),
            "sorted_errors": [values.tolist() for values in self.sorted_errors],
            "snr_thresholds": {
                f"{cls}:{snr}": value for (cls, snr), value in self.snr_thresholds.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ClassConditionalCalibrator":
        calibrator = cls(float(payload["threshold_quantile"]))
        calibrator.thresholds = np.asarray(payload["thresholds"], dtype=np.float64)
        calibrator.sorted_errors = [
            np.asarray(values, dtype=np.float64) for values in payload["sorted_errors"]
        ]
        calibrator.snr_thresholds = {
            tuple(int(part) for part in key.split(":")): float(value)
            for key, value in payload.get("snr_thresholds", {}).items()
        }
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
