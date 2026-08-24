import numpy as np

from rat_cfsr.metrics import format_confusion_matrix, open_set_confusion_matrix
from rat_cfsr.train import fit_feature_prototypes, prototype_unknown_scores


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
