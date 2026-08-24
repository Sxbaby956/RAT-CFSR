import numpy as np

from rat_cfsr.metrics import format_confusion_matrix, open_set_confusion_matrix


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
