from __future__ import annotations

import sys

from rat_cfsr.train import build_arg_parser, run, run_test_only


def parse_args():
    parser = build_arg_parser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train",
        action="store_true",
        help="Only train and save checkpoint/history; skip test evaluation.",
    )
    mode.add_argument(
        "--test",
        action="store_true",
        help="Only test an existing checkpoint from --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    if args.test:
        run_test_only(args)
    else:
        run(args, evaluate_after_training=not args.train)


if __name__ == "__main__":
    main()
