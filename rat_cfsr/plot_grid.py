from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import _draw_confusion_matrix

# Fixed layout for the experiment matrix: rows are the held-out ("unknown")
# protocol, columns are the random seed.
UNKNOWN_ORDER = ("5G", "4G", "WiFi")
SEED_ORDER = ("42", "123", "2026")


def parse_run_name(name: str) -> tuple[str, str] | None:
    """Parse ``unknown_<protocol>_seed<seed>`` into ``(protocol, seed)``."""
    if not name.startswith("unknown_") or "_seed" not in name:
        return None
    protocol, seed = name[len("unknown_") :].split("_seed", 1)
    if not protocol or not seed:
        return None
    return protocol, seed


def discover_runs(output_root: Path) -> dict[tuple[str, str], Path]:
    """Map ``(unknown, seed)`` to each run's ``confusion_matrix.json``.

    The unknown protocol and seed are read from ``config.json`` when present
    (robust to arbitrary output-directory names), falling back to parsing the
    directory name.
    """
    runs: dict[tuple[str, str], Path] = {}
    for conf_path in sorted(output_root.glob("*/confusion_matrix.json")):
        run_dir = conf_path.parent
        unknown, seed = parse_run_name(run_dir.name) or ("", "")
        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            unknown = str(config.get("unknown", unknown))
            seed = str(config.get("seed", seed))
        if unknown and seed:
            runs[(unknown, seed)] = conf_path
    return runs


def load_confusion(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion_matrix_grid(
    runs: dict[tuple[str, str], Path],
    path: str | Path,
    title: str | None = None,
    dpi: int = 200,
) -> None:
    """Render the 3-unknown x 3-seed confusion-matrix grid as one PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(UNKNOWN_ORDER)
    n_cols = len(SEED_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    images = []
    for row, unknown in enumerate(UNKNOWN_ORDER):
        for col, seed in enumerate(SEED_ORDER):
            ax = axes[row][col]
            conf_path = runs.get((unknown, seed))
            if conf_path is not None:
                images.append(_draw_confusion_matrix(ax, load_confusion(conf_path)))
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    color="#9a9a9a",
                    fontsize=12,
                )
        axes[row][0].set_ylabel(
            f"unknown={unknown}", fontsize=13, labelpad=18, rotation=90
        )
    for col, seed in enumerate(SEED_ORDER):
        axes[0][col].set_title(f"seed={seed}", fontsize=13)

    if images:
        fig.colorbar(images[0], ax=axes, fraction=0.03, pad=0.03, label="row fraction")
    if title:
        fig.suptitle(title, fontsize=15)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the 3-unknown x 3-seed open-set confusion-matrix grid."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Directory containing the per-run output directories.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Output PNG path (default: <output-root>/confusion_matrix_grid.png).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runs = discover_runs(args.output_root)
    image_path = args.image or (args.output_root / "confusion_matrix_grid.png")
    plot_confusion_matrix_grid(runs, image_path, dpi=args.dpi)
    print(f"Rendered {len(runs)} run(s) to {image_path}")
    missing = [
        (unknown, seed)
        for unknown in UNKNOWN_ORDER
        for seed in SEED_ORDER
        if (unknown, seed) not in runs
    ]
    if missing:
        print("Missing runs:", ", ".join(f"{u}/seed{s}" for u, s in missing))


if __name__ == "__main__":
    main()
