from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import MODULATIONS, load_samples, split_samples, summarize_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RML2016.10a data and split sizes")
    parser.add_argument("--data-root", type=Path, default=Path("/home/zjut/public/zjm/RML2016.10a"))
    parser.add_argument(
        "--unknown", nargs="+", choices=MODULATIONS, default=["WBFM"]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    unknown = list(args.unknown)
    known = [m for m in MODULATIONS if m not in set(unknown)]
    label_map = {m: index for index, m in enumerate(known)}
    data, samples = load_samples(args.data_root, label_map)
    splits = split_samples(samples, known, seed=args.seed)
    payload = {
        "dataset": summarize_samples(samples),
        "data_shape": list(data.shape),
        "known_modulations": known,
        "unknown_modulations": unknown,
        "split_strategy": "modulation_snr_stratified_random_60_20_20",
        "split_seed": args.seed,
        "splits": {},
    }
    for name, values in splits.items():
        payload["splits"][name] = {
            "samples": len(values),
            "modulations": {
                modulation: sum(value.modulation == modulation for value in values)
                for modulation in MODULATIONS
            },
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
