from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


PROTOCOLS = ("4G", "5G", "WiFi")
FILENAME_RE = re.compile(
    r"^(?P<protocol>4G|5G|WiFi)_Day_(?P<day>[12])_"
    r"(?P<transmitter>[A-Za-z0-9-]+)_s(?P<set>[1-5])$"
)


@dataclass(frozen=True)
class Recording:
    recording_id: str
    protocol: str
    day: int
    transmitter: str
    set_index: int
    sample_rate: float
    sample_count: int
    center_frequency: float
    bin_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class WindowRecord:
    recording: Recording
    start: int
    length: int
    label: int


def _as_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _canonical_protocol(stem_protocol: str, metadata_protocol: str | None) -> str:
    if stem_protocol == "WiFi":
        if metadata_protocol not in {None, "802.11a", "WiFi"}:
            raise ValueError(f"Unexpected Wi-Fi metadata protocol: {metadata_protocol}")
        return "WiFi"
    if stem_protocol == "5G":
        if metadata_protocol not in {None, "5G", "5G NR"}:
            raise ValueError(
                f"Filename protocol {stem_protocol} disagrees with metadata {metadata_protocol}"
            )
        return "5G"
    if metadata_protocol not in {None, stem_protocol}:
        raise ValueError(
            f"Filename protocol {stem_protocol} disagrees with metadata {metadata_protocol}"
        )
    return stem_protocol


def discover_recordings(root: str | Path) -> list[Recording]:
    """Discover and validate POWDER recordings without loading their IQ samples."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"POWDER data directory not found: {root}")

    recordings: list[Recording] = []
    for metadata_path in sorted(root.glob("*.json")):
        match = FILENAME_RE.match(metadata_path.stem)
        if match is None:
            continue
        bin_path = metadata_path.with_suffix(".bin")
        if not bin_path.is_file():
            raise FileNotFoundError(f"Missing IQ binary for {metadata_path.name}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        global_meta = _as_mapping(metadata.get("global"))
        capture_meta = _as_mapping(metadata.get("captures"))
        annotation_meta = _as_mapping(metadata.get("annotations"))

        datatype = str(global_meta.get("core:datatype", ""))
        if datatype.lower() != "cf32":
            raise ValueError(f"Unsupported datatype {datatype!r} in {metadata_path.name}")

        protocol = _canonical_protocol(
            match.group("protocol"), annotation_meta.get("core:protocol")
        )
        sample_rate = float(global_meta["core:sample_rate"])
        file_sample_count = bin_path.stat().st_size // np.dtype("<c8").itemsize
        annotated_count = int(annotation_meta.get("core:sample_count", file_sample_count))
        sample_count = min(file_sample_count, annotated_count)
        if sample_count <= 0:
            raise ValueError(f"Empty recording: {bin_path}")

        transmitter_meta = _as_mapping(annotation_meta.get("transmitter"))
        transmitter = str(
            transmitter_meta.get("core:location", match.group("transmitter"))
        )
        expected_transmitter = match.group("transmitter")
        if transmitter.lower() != expected_transmitter.lower():
            raise ValueError(
                f"Transmitter mismatch in {metadata_path.name}: "
                f"{transmitter} != {expected_transmitter}"
            )

        day = int(capture_meta.get("core:day", match.group("day")))
        set_index = int(capture_meta.get("core:set", match.group("set")))
        recordings.append(
            Recording(
                recording_id=metadata_path.stem,
                protocol=protocol,
                day=day,
                transmitter=transmitter.lower(),
                set_index=set_index,
                sample_rate=sample_rate,
                sample_count=sample_count,
                center_frequency=float(capture_meta.get("core:center_frequency", 0.0)),
                bin_path=bin_path.resolve(),
                metadata_path=metadata_path.resolve(),
            )
        )

    if not recordings:
        raise FileNotFoundError(f"No POWDER metadata files found in {root}")
    return recordings


def split_recordings(
    recordings: Sequence[Recording],
    known_protocols: Sequence[str],
    seed: int = 42,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, list[Recording]]:
    """Split each protocol/day group into train/calibration/test recordings.

    Each protocol is split independently inside each collection day, so Day 1
    and Day 2 are represented evenly in every split. Unknown protocols are kept
    out of train/calibration and only their test fraction is used.
    """
    known = set(known_protocols)
    if not known or not known.issubset(PROTOCOLS):
        raise ValueError(f"Invalid known protocol set: {sorted(known)}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train_fraction + calibration_fraction must be < 1")

    splits = {"train": [], "calibration": [], "test": []}
    grouped: dict[tuple[str, int], list[Recording]] = {}
    for recording in recordings:
        grouped.setdefault((recording.protocol, recording.day), []).append(recording)

    rng = np.random.default_rng(seed)
    for (protocol, _day), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: item.recording_id)
        permutation = rng.permutation(len(ordered))
        shuffled = [ordered[index] for index in permutation]
        train_count = int(round(len(shuffled) * train_fraction))
        calibration_count = int(round(len(shuffled) * calibration_fraction))
        train_count = max(1, min(train_count, len(shuffled) - 2))
        calibration_count = max(
            1, min(calibration_count, len(shuffled) - train_count - 1)
        )
        train_values = shuffled[:train_count]
        calibration_values = shuffled[train_count : train_count + calibration_count]
        test_values = shuffled[train_count + calibration_count :]

        if protocol in known:
            splits["train"].extend(train_values)
            splits["calibration"].extend(calibration_values)
        splits["test"].extend(test_values)

    for split_name, values in splits.items():
        if not values:
            raise ValueError(f"Split {split_name!r} is empty")
        splits[split_name] = sorted(values, key=lambda item: item.recording_id)
    return splits


def _window_starts(
    sample_count: int,
    window_length: int,
    stride_length: int,
    max_windows: int | None,
) -> np.ndarray:
    if sample_count < window_length:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, sample_count - window_length + 1, stride_length, dtype=np.int64)
    if max_windows is not None and max_windows > 0 and starts.size > max_windows:
        indices = np.linspace(0, starts.size - 1, max_windows, dtype=np.int64)
        starts = starts[indices]
    return np.unique(starts)


def build_windows(
    recordings: Iterable[Recording],
    label_map: dict[str, int],
    window_ms: float,
    stride_fraction: float = 1.0,
    max_windows_per_recording: int | None = 256,
) -> list[WindowRecord]:
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if not 0 < stride_fraction <= 1:
        raise ValueError("stride_fraction must be in (0, 1]")

    windows: list[WindowRecord] = []
    duration_s = window_ms / 1000.0
    for recording in recordings:
        raw_length = max(2, int(round(recording.sample_rate * duration_s)))
        stride = max(1, int(round(raw_length * stride_fraction)))
        starts = _window_starts(
            recording.sample_count,
            raw_length,
            stride,
            max_windows_per_recording,
        )
        label = label_map.get(recording.protocol, -1)
        windows.extend(
            WindowRecord(recording=recording, start=int(start), length=raw_length, label=label)
            for start in starts
        )
    if not windows:
        raise ValueError("No windows could be generated with the requested duration")
    return windows


def _resample_complex_linear(iq: np.ndarray, target_length: int) -> np.ndarray:
    if iq.size == target_length:
        return np.asarray(iq, dtype=np.complex64)
    source_axis = np.linspace(0.0, 1.0, iq.size, endpoint=False, dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, target_length, endpoint=False, dtype=np.float64)
    real = np.interp(target_axis, source_axis, iq.real)
    imag = np.interp(target_axis, source_axis, iq.imag)
    return (real + 1j * imag).astype(np.complex64, copy=False)


class PowderWindowDataset(Dataset):
    """Memory-mapped, fixed-duration POWDER windows.

    Every source window represents the same physical duration. It is resampled to
    ``target_samples`` so that the network cannot identify Wi-Fi from the original
    5 MS/s versus LTE/NR at 7.69 MS/s tensor length.
    """

    def __init__(
        self,
        windows: Sequence[WindowRecord],
        target_samples: int = 8192,
        augment: bool = False,
        frequency_shift_max: float = 0.02,
        awgn_probability: float = 0.5,
        awgn_snr_db: tuple[float, float] = (15.0, 35.0),
    ) -> None:
        if target_samples < 64:
            raise ValueError("target_samples must be at least 64")
        self.windows = list(windows)
        self.target_samples = int(target_samples)
        self.augment = bool(augment)
        self.frequency_shift_max = float(frequency_shift_max)
        self.awgn_probability = float(awgn_probability)
        self.awgn_snr_db = awgn_snr_db
        self._memmaps: dict[Path, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_memmaps"] = {}
        return state

    def _memmap(self, path: Path) -> np.memmap:
        if path not in self._memmaps:
            self._memmaps[path] = np.memmap(path, dtype="<c8", mode="r")
        return self._memmaps[path]

    @staticmethod
    def _normalize(iq: np.ndarray) -> np.ndarray:
        iq = iq - np.mean(iq)
        rms = math.sqrt(float(np.mean(np.abs(iq) ** 2)))
        return (iq / max(rms, 1e-8)).astype(np.complex64, copy=False)

    def _augment(self, iq: np.ndarray) -> np.ndarray:
        phase = np.random.uniform(-math.pi, math.pi)
        iq = iq * np.exp(1j * phase)

        max_shift = self.frequency_shift_max
        if max_shift > 0:
            normalized_shift = np.random.uniform(-max_shift, max_shift)
            sample_index = np.arange(iq.size, dtype=np.float32)
            iq = iq * np.exp(2j * math.pi * normalized_shift * sample_index)

        if np.random.random() < self.awgn_probability:
            snr_db = np.random.uniform(*self.awgn_snr_db)
            noise_power = 10.0 ** (-snr_db / 10.0)
            noise = (
                np.random.normal(size=iq.size) + 1j * np.random.normal(size=iq.size)
            ) * math.sqrt(noise_power / 2.0)
            iq = iq + noise
        return self._normalize(iq)

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        source = self._memmap(window.recording.bin_path)
        iq = np.asarray(source[window.start : window.start + window.length]).copy()
        iq = self._normalize(iq)
        iq = _resample_complex_linear(iq, self.target_samples)
        iq = self._normalize(iq)
        if self.augment:
            iq = self._augment(iq)

        tensor = np.stack((iq.real, iq.imag), axis=0).astype(np.float32, copy=False)
        return {
            "iq": torch.from_numpy(tensor),
            "label": torch.tensor(window.label, dtype=torch.long),
            "protocol": window.recording.protocol,
            "recording_id": window.recording.recording_id,
        }


def summarize_recordings(recordings: Sequence[Recording]) -> dict[str, object]:
    protocols = sorted({recording.protocol for recording in recordings})
    return {
        "recording_count": len(recordings),
        "protocol_counts": {
            protocol: sum(recording.protocol == protocol for recording in recordings)
            for protocol in protocols
        },
        "days": sorted({recording.day for recording in recordings}),
        "transmitters": sorted({recording.transmitter for recording in recordings}),
        "sample_rates": sorted({recording.sample_rate for recording in recordings}),
        "total_complex_samples": int(sum(recording.sample_count for recording in recordings)),
    }
