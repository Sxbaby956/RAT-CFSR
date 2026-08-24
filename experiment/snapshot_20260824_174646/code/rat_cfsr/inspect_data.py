from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import PROTOCOLS, build_windows, discover_recordings, split_recordings, summarize_recordings


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect POWDER metadata and split sizes")
    parser.add_argument("--data-root", type=Path, default=Path("GlobecomPOWDER"))
    parser.add_argument("--unknown", choices=PROTOCOLS, default="5G")
    parser.add_argument("--window-ms", type=float, default=1.0)
    parser.add_argument("--max-windows-per-recording", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    recordings = discover_recordings(args.data_root)
    known = [protocol for protocol in PROTOCOLS if protocol != args.unknown]
    label_map = {protocol: index for index, protocol in enumerate(known)}
    splits = split_recordings(recordings, known, seed=args.seed)
    payload = {
        "dataset": summarize_recordings(recordings),
        "known_protocols": known,
        "unknown_protocol": args.unknown,
        "split_strategy": "protocol_day_stratified_random_60_20_20",
        "split_seed": args.seed,
        "splits": {},
    }
    for name, values in splits.items():
        windows = build_windows(
            values,
            label_map,
            window_ms=args.window_ms,
            max_windows_per_recording=args.max_windows_per_recording,
        )
        payload["splits"][name] = {
            "recordings": len(values),
            "windows": len(windows),
            "protocols": {
                protocol: sum(value.protocol == protocol for value in values)
                for protocol in PROTOCOLS
            },
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
