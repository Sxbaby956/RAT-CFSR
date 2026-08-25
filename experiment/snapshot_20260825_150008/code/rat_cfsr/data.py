from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# The 11 modulation types present in RML2016.10a.
MODULATIONS = (
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
)

# SNR levels in RML2016.10a, in dB, from -20 to 18 in 2 dB steps.
SNRS = tuple(range(-20, 20, 2))

# Every RML2016.10a sample is fixed-length: 2 channels (I/Q) x 128 samples.
NUM_IQ_SAMPLES = 128


@dataclass(frozen=True)
class Sample:
    """One labelled RML2016.10a IQ example.

    ``index`` is the row index into the in-memory data array. ``label`` is the
    index into the known-class label map, or ``-1`` for a modulation held out
    as unknown.
    """

    modulation: str
    snr: int
    index: int
    label: int


def _find_pkl(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RML2016.10a data directory not found: {root}")
    candidates = sorted(root.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No .pkl dataset found in {root}")
    return candidates[0]


def load_samples(
    root: str | Path,
    label_map: dict[str, int],
) -> tuple[np.ndarray, list[Sample]]:
    """Load the RML2016.10a pickle into a single in-memory array.

    The array is stacked in a deterministic order (modulation, then SNR, then
    the original sample index) so that ``Sample.index`` is monotonic within each
    ``(modulation, snr)`` group.
    """
    pkl_path = _find_pkl(root)
    with open(pkl_path, "rb") as handle:
        raw = pickle.load(handle, encoding="latin1")

    arrays: list[np.ndarray] = []
    samples: list[Sample] = []
    for modulation in MODULATIONS:
        for snr in SNRS:
            key = (modulation, snr)
            if key not in raw:
                raise KeyError(f"Missing modulation/SNR key {key} in {pkl_path.name}")
            block = np.asarray(raw[key], dtype=np.float32)
            if block.ndim != 3 or block.shape[1:] != (2, NUM_IQ_SAMPLES):
                raise ValueError(
                    f"Unexpected sample shape {block.shape} for {key}; "
                    f"expected (N, 2, {NUM_IQ_SAMPLES})"
                )
            label = label_map.get(modulation, -1)
            for local_index in range(block.shape[0]):
                samples.append(
                    Sample(
                        modulation=modulation,
                        snr=snr,
                        index=len(arrays),
                        label=label,
                    )
                )
                arrays.append(block[local_index])

    data = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    return data, samples


def split_samples(
    samples: Sequence[Sample],
    known_modulations: Sequence[str],
    seed: int = 42,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, list[Sample]]:
    """Split each ``(modulation, snr)`` group into train/calibration/test.

    Every group (1000 samples in RML2016.10a) is shuffled and split
    independently, so all SNR levels appear in every split. Known modulations
    contribute to train/calibration/test; an unknown modulation only contributes
    its test fraction, mirroring the open-set protocol-recognition split.
    """
    known = set(known_modulations)
    if not known or not known.issubset(MODULATIONS):
        raise ValueError(f"Invalid known modulation set: {sorted(known)}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train_fraction + calibration_fraction must be < 1")

    splits: dict[str, list[Sample]] = {"train": [], "calibration": [], "test": []}
    grouped: dict[tuple[str, int], list[Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.modulation, sample.snr), []).append(sample)

    rng = np.random.default_rng(seed)
    for (modulation, _snr), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: item.index)
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

        if modulation in known:
            splits["train"].extend(train_values)
            splits["calibration"].extend(calibration_values)
        splits["test"].extend(test_values)

    for split_name, values in splits.items():
        if not values:
            raise ValueError(f"Split {split_name!r} is empty")
        splits[split_name] = sorted(values, key=lambda item: item.index)
    return splits


def filter_samples_by_snr(
    samples: Sequence[Sample],
    min_snr: int | None = None,
    max_snr: int | None = None,
) -> list[Sample]:
    """Keep samples whose SNR falls within the optional inclusive range."""
    if min_snr is not None and min_snr not in SNRS:
        raise ValueError(f"min_snr must be one of {SNRS}; got {min_snr}")
    if max_snr is not None and max_snr not in SNRS:
        raise ValueError(f"max_snr must be one of {SNRS}; got {max_snr}")
    if min_snr is not None and max_snr is not None and min_snr > max_snr:
        raise ValueError("min_snr cannot exceed max_snr")
    return [
        sample
        for sample in samples
        if (min_snr is None or sample.snr >= min_snr)
        and (max_snr is None or sample.snr <= max_snr)
    ]


class ModulationDataset(Dataset):
    """In-memory RML2016.10a windows with optional on-the-fly augmentation."""

    def __init__(
        self,
        data: np.ndarray,
        samples: Sequence[Sample],
        augment: bool = False,
        frequency_shift_max: float = 0.02,
        awgn_probability: float = 0.5,
        awgn_snr_db: tuple[float, float] = (15.0, 35.0),
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.samples = list(samples)
        self.augment = bool(augment)
        self.frequency_shift_max = float(frequency_shift_max)
        self.awgn_probability = float(awgn_probability)
        self.awgn_snr_db = awgn_snr_db

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_complex(iq: np.ndarray) -> np.ndarray:
        return iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)

    @staticmethod
    def _from_complex(complex_iq: np.ndarray) -> np.ndarray:
        return np.stack((complex_iq.real, complex_iq.imag), axis=0).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _normalize(complex_iq: np.ndarray) -> np.ndarray:
        complex_iq = complex_iq - np.mean(complex_iq)
        rms = math.sqrt(float(np.mean(np.abs(complex_iq) ** 2)))
        return complex_iq / max(rms, 1e-8)

    def _augment(self, complex_iq: np.ndarray) -> np.ndarray:
        phase = np.random.uniform(-math.pi, math.pi)
        complex_iq = complex_iq * np.exp(1j * phase)

        max_shift = self.frequency_shift_max
        if max_shift > 0:
            normalized_shift = np.random.uniform(-max_shift, max_shift)
            sample_index = np.arange(complex_iq.size, dtype=np.float32)
            complex_iq = complex_iq * np.exp(2j * math.pi * normalized_shift * sample_index)

        if np.random.random() < self.awgn_probability:
            snr_db = np.random.uniform(*self.awgn_snr_db)
            noise_power = 10.0 ** (-snr_db / 10.0)
            noise = (
                np.random.normal(size=complex_iq.size) + 1j * np.random.normal(size=complex_iq.size)
            ) * math.sqrt(noise_power / 2.0)
            complex_iq = complex_iq + noise
        return complex_iq

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        complex_iq = self._to_complex(self.data[sample.index])
        complex_iq = self._normalize(complex_iq)
        if self.augment:
            complex_iq = self._augment(complex_iq)
            complex_iq = self._normalize(complex_iq)

        tensor = torch.from_numpy(self._from_complex(complex_iq))
        return {
            "iq": tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "modulation": sample.modulation,
            "snr": sample.snr,
        }


def summarize_samples(samples: Sequence[Sample]) -> dict[str, object]:
    modulations = sorted({sample.modulation for sample in samples})
    return {
        "sample_count": len(samples),
        "modulation_counts": {
            modulation: sum(sample.modulation == modulation for sample in samples)
            for modulation in modulations
        },
        "snrs": sorted({sample.snr for sample in samples}),
    }
