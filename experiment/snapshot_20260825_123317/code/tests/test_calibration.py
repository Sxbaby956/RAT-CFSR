import numpy as np

from rat_cfsr.calibration import ClassConditionalCalibrator, ClassConditionalScoreCalibrator


def test_class_conditional_calibration_rejects_large_errors() -> None:
    calibration_errors = np.array(
        [
            [0.10, 0.80],
            [0.12, 0.75],
            [0.14, 0.85],
            [0.70, 0.08],
            [0.80, 0.10],
            [0.75, 0.12],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator = ClassConditionalCalibrator(0.95).fit(calibration_errors, labels)

    errors = np.array([[0.11, 0.90], [0.90, 0.09], [1.20, 1.10]])
    prediction = calibrator.predict(errors)
    assert prediction.labels.tolist() == [0, 1, -1]
    assert prediction.candidate_scores[-1] == 1.0


def test_score_calibrator_rejects_high_unknown_scores() -> None:
    scores = np.array([0.10, 0.12, 0.14, 0.20, 0.22, 0.24])
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator = ClassConditionalScoreCalibrator(0.95).fit(scores, labels)

    prediction = calibrator.predict(
        scores=np.array([0.11, 0.21, 0.80]),
        candidate_labels=np.array([0, 1, 1]),
    )

    assert prediction.labels.tolist() == [0, 1, -1]
    assert prediction.candidate_scores.tolist() == [0.11, 0.21, 0.80]
