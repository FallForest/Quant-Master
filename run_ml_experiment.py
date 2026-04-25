from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.backfill import backfill_signal_metrics_from_path, backfill_signal_metrics_group_from_path
from ml.experiments.runner import run_experiment_from_path, run_experiment_group_from_path
from ml.experiments.scheduler import GroupExecutionOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ML experiment manifest or a batch group manifest.")
    parser.add_argument("--experiment", help="Path to one experiment YAML file")
    parser.add_argument("--group", help="Path to one experiment group YAML file")
    parser.add_argument("--continue-on-error", action="store_true", help="Only applies to --group")
    parser.add_argument("--parallel", action="store_true", help="Enable resource-aware parallel execution for --group")
    parser.add_argument("--cpu-workers", type=int, help="Max concurrent CPU experiments for --group")
    parser.add_argument("--gpu-workers", type=int, help="Max concurrent GPU experiments for --group")
    parser.add_argument("--gpu-device", action="append", default=[], help="Visible GPU device ids for scheduling")
    parser.add_argument("--backfill-signal-metrics", action="store_true", help="Recompute and rewrite OOS signal metrics using existing artifacts")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.experiment) == bool(args.group):
        parser.error("Provide exactly one of --experiment or --group.")
        return

    if args.experiment:
        if args.backfill_signal_metrics:
            summary = backfill_signal_metrics_from_path(args.experiment)
        else:
            summary = run_experiment_from_path(args.experiment)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.backfill_signal_metrics:
        summary = backfill_signal_metrics_group_from_path(args.group)
    else:
        summary = run_experiment_group_from_path(
            args.group,
            continue_on_error=args.continue_on_error,
            execution_options=GroupExecutionOptions(
                parallel=bool(args.parallel),
                cpu_workers=args.cpu_workers,
                gpu_workers=args.gpu_workers,
                gpu_devices=[str(item) for item in args.gpu_device],
            ),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
