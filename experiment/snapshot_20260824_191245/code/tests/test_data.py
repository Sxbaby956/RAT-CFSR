import json
from pathlib import Path

import numpy as np

from rat_cfsr.data import (
    PowderWindowDataset,
    Recording,
    build_windows,
    discover_recordings,
    split_recordings,
)


def test_powder_memmap_and_fixed_duration_resampling(tmp_path: Path) -> None:
    iq = (np.arange(2000) + 1j * np.arange(2000)[::-1]).astype(np.complex64)
    bin_path = tmp_path / "4G_Day_1_bes_s1.bin"
    iq.tofile(bin_path)
    metadata = {
        "global": {"core:datatype": "cf32", "core:sample_rate": "1000"},
        "captures": {
            "core:day": "1",
            "core:set": "1",
            "core:center_frequency": "1000000",
        },
        "annotations": {
            "core:sample_count": "2000",
            "core:protocol": "4G",
            "transmitter": {"core:location": "bes"},
        },
    }
    (tmp_path / "4G_Day_1_bes_s1.json").write_text(json.dumps(metadata))

    recordings = discover_recordings(tmp_path)
    windows = build_windows(
        recordings,
        label_map={"4G": 0},
        window_ms=100.0,
        max_windows_per_recording=2,
    )
    dataset = PowderWindowDataset(windows, target_samples=128)
    item = dataset[0]
    assert item["iq"].shape == (2, 128)
    assert item["label"].item() == 0
    assert np.isfinite(item["iq"].numpy()).all()


def test_split_recordings_is_protocol_day_stratified() -> None:
    recordings = []
    for protocol in ("4G", "5G", "WiFi"):
        for day in (1, 2):
            for index in range(10):
                recordings.append(
                    Recording(
                        recording_id=f"{protocol}_Day_{day}_tx_s{index}",
                        protocol=protocol,
                        day=day,
                        transmitter="tx",
                        set_index=index,
                        sample_rate=1.0,
                        sample_count=100,
                        center_frequency=0.0,
                        bin_path=Path("x.bin"),
                        metadata_path=Path("x.json"),
                    )
                )

    splits = split_recordings(recordings, known_protocols=["4G", "WiFi"], seed=7)

    assert len(splits["train"]) == 24
    assert len(splits["calibration"]) == 8
    assert len(splits["test"]) == 12
    assert all(recording.protocol != "5G" for recording in splits["train"])
    assert all(recording.protocol != "5G" for recording in splits["calibration"])

    for split_name, expected in {
        "train": 6,
        "calibration": 2,
        "test": 2,
    }.items():
        for protocol in ("4G", "WiFi"):
            for day in (1, 2):
                count = sum(
                    recording.protocol == protocol and recording.day == day
                    for recording in splits[split_name]
                )
                assert count == expected

    for day in (1, 2):
        unknown_test_count = sum(
            recording.protocol == "5G" and recording.day == day
            for recording in splits["test"]
        )
        assert unknown_test_count == 2
