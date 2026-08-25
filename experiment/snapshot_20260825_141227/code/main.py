from __future__ import annotations

import copy
import sys
from pathlib import Path

from rat_cfsr.data import MODULATIONS
from rat_cfsr.snapshot import copy_current_snapshot
from rat_cfsr.train import build_arg_parser, run, run_test_only

DEFAULT_SEEDS = (42,)
DEFAULT_OUTPUT_DIR = Path("outputs/rml2016_v8/pruned_prototype_snr0_seed42")
DEFAULT_MIN_SNR = 0


def parse_args():
    parser = build_arg_parser()
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR, min_snr=DEFAULT_MIN_SNR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--single",
        action="store_true",
        help="Run one experiment using --unknown/--seed/--output-dir.",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run one train-only experiment using --unknown/--seed/--output-dir.",
    )
    mode.add_argument(
        "--test",
        action="store_true",
        help="Only test an existing checkpoint from --output-dir.",
    )
    mode.add_argument(
        "--matrix",
        action="store_true",
        help="Run the full unknown-modulation experiment matrix.",
    )
    parser.add_argument(
        "--matrix-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seeds used by the default experiment matrix (one per unknown modulation).",
    )
    return parser.parse_args()


def run_matrix(args) -> None:
    output_root = args.output_dir
    Path(output_root).mkdir(parents=True, exist_ok=True)
    print(
        "[matrix] "
        f"running {len(MODULATIONS) * len(args.matrix_seeds)} experiment(s); "
        f"output_root={output_root}"
    )
    for unknown in MODULATIONS:
        for seed in args.matrix_seeds:
            run_args = copy.deepcopy(args)
            run_args.unknown = [unknown]
            run_args.seed = int(seed)
            run_args.output_dir = Path(output_root) / f"unknown_{unknown}_seed{seed}"
            if (run_args.output_dir / "metrics.json").is_file():
                print(
                    "[matrix] "
                    f"skip unknown={unknown} seed={seed}; metrics.json already exists"
                )
                continue
            print(
                "[matrix] "
                f"start unknown={unknown} seed={seed} "
                f"output_dir={run_args.output_dir}"
            )
            run(run_args, evaluate_after_training=True)
            print(f"[matrix] done unknown={unknown} seed={seed}")

    try:
        from rat_cfsr.plot_grid import discover_runs, plot_confusion_matrix_grid

        runs = discover_runs(Path(output_root))
        if not runs:
            print("[matrix] skipped confusion matrix grid; no confusion matrices found")
            return
        image_path = Path(output_root) / "confusion_matrix_grid.png"
        plot_confusion_matrix_grid(runs, image_path)
        print(f"[matrix] rendered confusion matrix grid: {image_path}")
    except ImportError as exc:
        print(f"[matrix] skipped confusion matrix grid; install plot extras: {exc}")

    snapshot_dir = copy_current_snapshot(
        repo_root=Path(__file__).resolve().parent,
        output_root=Path(output_root),
    )
    print(f"[snapshot] saved current code and results: {snapshot_dir}")


def save_run_snapshot(output_dir: Path) -> None:
    snapshot_dir = copy_current_snapshot(
        repo_root=Path(__file__).resolve().parent,
        output_root=Path(output_dir),
    )
    print(f"[snapshot] saved current code and results: {snapshot_dir}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    if args.matrix:
        run_matrix(args)
    elif args.test:
        run_test_only(args)
    else:
        run(args, evaluate_after_training=not args.train)
        if not args.dry_run:
            save_run_snapshot(args.output_dir)


if __name__ == "__main__":
    main()
