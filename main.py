from __future__ import annotations

import copy
import sys
from pathlib import Path

from rat_cfsr.data import PROTOCOLS
from rat_cfsr.train import build_arg_parser, run, run_test_only

DEFAULT_SEEDS = (42, 123, 2026)


def parse_args():
    parser = build_arg_parser()
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
    parser.add_argument(
        "--matrix-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seeds used by the default 3-unknown experiment matrix.",
    )
    return parser.parse_args()


def run_matrix(args) -> None:
    output_root = args.output_dir
    Path(output_root).mkdir(parents=True, exist_ok=True)
    print(
        "[matrix] "
        f"running {len(PROTOCOLS) * len(args.matrix_seeds)} experiment(s); "
        f"output_root={output_root}"
    )
    for unknown in PROTOCOLS:
        for seed in args.matrix_seeds:
            run_args = copy.deepcopy(args)
            run_args.unknown = unknown
            run_args.seed = int(seed)
            run_args.output_dir = Path(output_root) / f"unknown_{unknown}_seed{seed}"
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    if args.test:
        run_test_only(args)
    elif args.single or args.train:
        run(args, evaluate_after_training=not args.train)
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
