import json
from pathlib import Path

import numpy as np

from rat_cfsr.data import PowderWindowDataset, build_windows, discover_recordings


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

