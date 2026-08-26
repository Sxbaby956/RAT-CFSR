from __future__ import annotations

import copy
import sys
from pathlib import Path

from rat_cfsr.data import MODULATIONS
from rat_cfsr.snapshot import copy_current_snapshot
from rat_cfsr.train import build_arg_parser, run, run_test_only

DEFAULT_SEEDS = (42,)
DEFAULT_OUTPUT_DIR = Path("outputs/cfsr_v19_direct_ce_q99")
DEFAULT_FOLDS = 1
DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_OPEN_WEIGHT = 0.5
DEFAULT_CLASSIFICATION_WEIGHT = 1.0
DEFAULT_THRESHOLD_QUANTILE = 0.99
DEFAULT_STAGE1_EPOCHS = 10
DEFAULT_STAGE2_EPOCHS = 50
# v10+: align training/calibration with the judged region (SNR>=0). The full-SNR
# AUROC/OSCR-vs-SNR curves remain available as an optional `--min-snr None` run.
DEFAULT_MIN_SNR = 0


def parse_args():
    parser = build_arg_parser()
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR,
        min_snr=DEFAULT_MIN_SNR,
        folds=DEFAULT_FOLDS,
        early_stop_patience=DEFAULT_EARLY_STOP_PATIENCE,
        open_weight=DEFAULT_OPEN_WEIGHT,
        classification_weight=DEFAULT_CLASSIFICATION_WEIGHT,
        threshold_quantile=DEFAULT_THRESHOLD_QUANTILE,
        stage1_epochs=DEFAULT_STAGE1_EPOCHS,
        stage2_epochs=DEFAULT_STAGE2_EPOCHS,
    )
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


def run_folds(args) -> None:
    output_root = args.output_dir
    Path(output_root).mkdir(parents=True, exist_ok=True)
    print(
        "[folds] "
        f"running {args.folds} fold(s) x {len(args.matrix_seeds)} seed(s); "
        f"output_root={output_root}"
    )
    for fold in range(args.folds):
        for seed in args.matrix_seeds:
            run_args = copy.deepcopy(args)
            run_args.fold = fold
            run_args.seed = int(seed)
            run_args.output_dir = Path(output_root) / f"fold{fold}_seed{seed}"
            if (run_args.output_dir / "metrics.json").is_file():
                print(f"[folds] skip fold={fold} seed={seed}; metrics.json exists")
                continue
            print(f"[folds] start fold={fold} seed={seed} -> {run_args.output_dir}")
            run(run_args, evaluate_after_training=True)
            print(f"[folds] done fold={fold} seed={seed}")

    try:
        from rat_cfsr.plot_grid import discover_runs, plot_confusion_matrix_grid

        runs = discover_runs(Path(output_root))
        if runs:
            plot_confusion_matrix_grid(
                runs, Path(output_root) / "confusion_matrix_grid.png"
            )
    except ImportError:
        pass

    snapshot_dir = copy_current_snapshot(
        repo_root=Path(__file__).resolve().parent,
        output_root=Path(output_root),
    )
    print(f"[snapshot] saved current code and results: {snapshot_dir}")


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
    elif args.folds > 1:
        run_folds(args)
    elif args.test:
        run_test_only(args)
    else:
        run(args, evaluate_after_training=not args.train)
        if not args.dry_run:
            save_run_snapshot(args.output_dir)


if __name__ == "__main__":
    main()
