import pickle
from pathlib import Path

import numpy as np

from rat_cfsr.data import (
    MODULATIONS,
    SNRS,
    ModulationDataset,
    Sample,
    filter_samples_by_snr,
    load_samples,
    split_samples,
)


def _make_sample(modulation: str, snr: int, index: int, label: int) -> Sample:
    return Sample(modulation=modulation, snr=snr, index=index, label=label)


def test_split_samples_is_modulation_snr_stratified() -> None:
    samples = []
    for modulation in ("BPSK", "QPSK", "WBFM"):
        for snr in (0, 10):
            for index in range(10):
                label = {"BPSK": 0, "QPSK": 1}.get(modulation, -1)
                samples.append(_make_sample(modulation, snr, index, label))

    splits = split_samples(samples, known_modulations=["BPSK", "QPSK"], seed=7)

    assert len(splits["train"]) == 24
    assert len(splits["calibration"]) == 8
    assert len(splits["test"]) == 12
    assert all(sample.modulation != "WBFM" for sample in splits["train"])
    assert all(sample.modulation != "WBFM" for sample in splits["calibration"])

    for split_name, expected in {
        "train": 6,
        "calibration": 2,
        "test": 2,
    }.items():
        for modulation in ("BPSK", "QPSK"):
            for snr in (0, 10):
                count = sum(
                    sample.modulation == modulation and sample.snr == snr
                    for sample in splits[split_name]
                )
                assert count == expected

    unknown_test_count = sum(
        sample.modulation == "WBFM" for sample in splits["test"]
    )
    assert unknown_test_count == 4


def test_filter_samples_by_snr_uses_inclusive_bounds() -> None:
    samples = [
        _make_sample("BPSK", -2, 0, 0),
        _make_sample("BPSK", 0, 1, 0),
        _make_sample("BPSK", 2, 2, 0),
        _make_sample("BPSK", 4, 3, 0),
    ]

    filtered = filter_samples_by_snr(samples, min_snr=0, max_snr=2)

    assert [sample.snr for sample in filtered] == [0, 2]


def test_modulation_dataset_normalizes_and_returns_2x128() -> None:
    data = np.random.randn(16, 2, 128).astype(np.float32)
    samples = [
        _make_sample("BPSK", 0, index, 0) for index in range(8)
    ] + [
        _make_sample("QPSK", 0, index, 1) for index in range(8)
    ]
    dataset = ModulationDataset(data, samples)
    item = dataset[0]
    assert item["iq"].shape == (2, 128)
    assert item["label"].item() == 0
    assert np.isfinite(item["iq"].numpy()).all()
    assert item["modulation"] == "BPSK"


def test_load_samples_roundtrip(tmp_path: Path) -> None:
    raw = {}
    per_group = 4
    for modulation in MODULATIONS:
        for snr in SNRS:
            raw[(modulation, snr)] = np.random.randn(per_group, 2, 128).astype(np.float32)
    pkl_path = tmp_path / "RML2016.10a_dict.pkl"
    with open(pkl_path, "wb") as handle:
        pickle.dump(raw, handle)

    label_map = {m: index for index, m in enumerate(MODULATIONS)}
    data, samples = load_samples(tmp_path, label_map)

    expected = len(MODULATIONS) * len(SNRS) * per_group
    assert data.shape == (expected, 2, 128)
    assert len(samples) == expected
    assert samples[0].label == 0
    assert samples[0].modulation == MODULATIONS[0]
