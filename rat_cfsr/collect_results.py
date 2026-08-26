"""Summarize the judged metrics across experiment versions.

The acceptance criterion for this reproduction is, on the SNR>=0 subset,
``known_accuracy >= 0.90`` and ``auroc >= 0.90``. Training may span all SNRs,
so the headline numbers always come from ``high_snr_metrics`` when present and
fall back to the top-level block when a run was itself restricted to SNR>=0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("known_accuracy", "auroc", "oscr", "true_known_rate", "true_unknown_rate")


def judged_block(metrics: dict) -> dict:
    """Return the SNR>=0 metric block, whichever shape the run produced."""
    high = metrics.get("high_snr_metrics") or {}
    if high.get("sample_count"):
        return high
    return metrics


def load_run(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    block = judged_block(metrics)
    row = {"run": run_dir.name, **{f: block.get(f) for f in FIELDS}}
    row["folds"] = 1
    return row


def load_matrix(root: Path) -> dict | None:
    """Average per-fold runs (v5 onward writes one subdirectory per fold)."""
    folds = [load_run(d) for d in sorted(root.iterdir()) if d.is_dir()]
    folds = [f for f in folds if f]
    if not folds:
        return None
    row = {"run": root.name, "folds": len(folds)}
    for field in FIELDS:
        values = [f[field] for f in folds if isinstance(f.get(field), (int, float))]
        row[field] = sum(values) / len(values) if values else None
    return row


def summarize(root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        row = load_run(run_dir) or load_matrix(run_dir)
        if row:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    rows = summarize(args.root)
    if not rows:
        print(f"no metrics.json found under {args.root}")
        return

    header = f"{'run':<34} {'folds':>5} " + " ".join(f"{f:>16}" for f in FIELDS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(
            f"{row[f]:>16.4f}" if isinstance(row.get(f), (int, float)) else f"{'-':>16}"
            for f in FIELDS
        )
        target = ""
        ka, au = row.get("known_accuracy"), row.get("auroc")
        if isinstance(ka, float) and isinstance(au, float):
            target = "  <== PASS" if ka >= 0.90 and au >= 0.90 else ""
        print(f"{row['run']:<34} {row['folds']:>5} {cells}{target}")


if __name__ == "__main__":
    main()
