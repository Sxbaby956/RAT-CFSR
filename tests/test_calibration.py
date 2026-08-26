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
    assert np.isclose(prediction.candidate_scores[-1], 1.10)


def test_per_snr_thresholds_reject_only_within_their_snr_band() -> None:
    # Same class has tight errors at high SNR and huge errors at low SNR. A
    # global threshold would accept everything (v3's true_unknown_rate=0
    # failure); per-(class, SNR) thresholds must reject only in the tight band.
    errors = np.array(
        [
            [0.10, 0.80],  # class 0, snr 10
            [0.12, 0.75],  # class 0, snr 10
            [0.14, 0.85],  # class 0, snr 10
            [5.00, 0.80],  # class 0, snr -20
            [5.20, 0.75],  # class 0, snr -20
            [5.40, 0.85],  # class 0, snr -20
            [0.70, 0.08],  # class 1, snr 10
            [0.80, 0.10],  # class 1, snr 10
            [0.75, 0.12],  # class 1, snr 10
        ]
    )
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    snrs = np.array([10, 10, 10, -20, -20, -20, 10, 10, 10])
    calibrator = ClassConditionalCalibrator(0.95).fit_snr(errors, labels, snrs)

    test_errors = np.array([[0.50, 0.90]])  # class 0 candidate, snr 10
    assert calibrator.predict_snr(test_errors, np.array([10])).labels.tolist() == [-1]
    # The same error at snr -20 is below that band's (huge) threshold -> accepted.
    assert calibrator.predict_snr(test_errors, np.array([-20])).labels.tolist() == [0]


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
