import numpy as np

from rat_cfsr.metrics import format_confusion_matrix, open_set_confusion_matrix
from rat_cfsr.train import (
    correctly_classified_calibration_mask,
    fit_feature_prototypes,
    grouped_open_set_metrics,
    prototype_unknown_scores,
)


def test_open_set_confusion_matrix_includes_unknown_and_known_classes() -> None:
    confusion = open_set_confusion_matrix(
        true_labels=np.array([-1, -1, 0, 0, 1]),
        predicted_labels=np.array([-1, 0, 0, -1, 1]),
        known_class_names=["4G", "WiFi"],
        unknown_name="unknown:5G",
    )

    assert confusion["labels"] == ["unknown:5G", "4G", "WiFi"]
    assert confusion["class_ids"] == [-1, 0, 1]
    assert confusion["counts"] == [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]
    assert "rows=true" in format_confusion_matrix(confusion)


def test_prototype_unknown_scores_use_predicted_class_distance() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [2.0, 0.0],
            [2.2, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    prototypes = fit_feature_prototypes(features, labels)

    scores = prototype_unknown_scores(
        features=np.array([[0.1, 0.0], [4.0, 0.0]]),
        candidate_labels=np.array([0, 1]),
        prototypes=prototypes,
    )

    assert scores[0] < scores[1]


def test_calibration_mask_keeps_only_correctly_classified_examples() -> None:
    logits = np.array(
        [
            [3.0, 0.0],
            [2.5, 0.0],
            [0.0, 3.0],
            [4.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.5],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    mask = correctly_classified_calibration_mask(logits, labels)

    assert mask.tolist() == [True, True, False, False, True, True]


def test_grouped_open_set_metrics_adds_snr_and_modulation_breakdowns() -> None:
    true_labels = np.array([0, 0, 1, 1, -1, -1])
    predicted_labels = np.array([0, -1, 1, 0, -1, 1])
    candidate_labels = np.array([0, 0, 1, 0, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.3, 1.0, 0.8])
    snrs = np.array([-2, -2, 0, 0, 0, 2])
    modulations = np.array(["BPSK", "BPSK", "QPSK", "QPSK", "WBFM", "WBFM"])

    grouped = grouped_open_set_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        candidate_labels=candidate_labels,
        unknown_scores=scores,
        snrs=snrs,
        modulations=modulations,
        known_class_names=["BPSK", "QPSK"],
        unknown_modulations=["WBFM"],
    )

    assert set(grouped["snr_metrics"]) == {"-2", "0", "2"}
    assert grouped["unknown_modulation_metrics"]["WBFM"]["unknown_sample_count"] == 2
    assert grouped["known_modulation_accuracy"]["BPSK"]["sample_count"] == 2
