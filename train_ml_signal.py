from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.factors import get_default_registry
from ml.models import SUPPORTED_MODEL_NAMES
from ml.training import train_ml_signal_model
from ml.tuning import TuningConfig
from ml.validation import WalkForwardConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a machine-learning signal artifact")
    parser.add_argument("--market", default="ashare")
    parser.add_argument("--provider", default="csv", choices=["csv", "parquet", "duckdb"])
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--universe-root", default="data/universe")
    parser.add_argument("--reference-root", default="data/reference")
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--timeframe", default="1d")

    parser.add_argument("--symbol", dest="symbols", action="append", help="Repeat for multiple symbols")
    parser.add_argument("--universe", help="Universe csv name under data/universe/<market>")

    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--train-end-date", help="Optional last date of the training split")
    parser.add_argument("--valid-start-date", help="Optional first date of the validation split")
    parser.add_argument("--valid-end-date", help="Optional last date of the validation split")

    parser.add_argument("--artifact-path")
    parser.add_argument("--model", default="ridge", choices=list(SUPPORTED_MODEL_NAMES))
    parser.add_argument("--model-param", action="append", default=[], help="Model param in key=value form")
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--target-mode", choices=["future_return", "cross_sectional_rank"], default="future_return")

    parser.add_argument("--feature", action="append", default=[], help="Repeat for multiple input features")
    parser.add_argument("--feature-family", action="append", default=[], help="Optional family filter for --list-features")
    parser.add_argument("--list-features", action="store_true")

    parser.add_argument("--validation-mode", choices=["holdout", "walk_forward"], default="holdout")
    parser.add_argument("--wf-train-size", type=int)
    parser.add_argument("--wf-valid-size", type=int)
    parser.add_argument("--wf-step-size", type=int)
    parser.add_argument("--wf-expanding", action="store_true")
    parser.add_argument("--purge-size", type=int, default=0, help="Drop the last N timestamps from each training split")
    parser.add_argument("--embargo-size", type=int, default=0, help="Skip the first N timestamps after each split boundary")

    parser.add_argument("--optuna-trials", type=int)
    parser.add_argument("--optuna-timeout", type=int)
    parser.add_argument("--optuna-jobs", type=int, help="Parallel Optuna trial workers")
    parser.add_argument("--gpu-device", action="append", default=[], help="Optional GPU device ids for parallel trial assignment")
    parser.add_argument("--tune-metric", default="spearman_ic")
    parser.add_argument("--tune-direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--keep-top-trials", type=int, default=5, help="How many tuned trial candidates to keep for second-stage selection")
    return parser


def parse_kv_pairs(items: list[str]) -> dict:
    values: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid key=value argument: {item}")
        key, raw = item.split("=", 1)
        values[key.strip()] = coerce_value(raw.strip())
    return values


def coerce_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if any(token in raw for token in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def format_factor_list(*, family_filters: list[str] | None = None) -> str:
    registry = get_default_registry()
    filters = set(family_filters or [])
    lines: list[str] = []
    for factor in registry.list_factors():
        if filters and factor.family not in filters:
            continue
        lines.append(
            f"{factor.name} | family={factor.family} | stability={factor.stability_level} | {factor.description}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_features:
        print(format_factor_list(family_filters=args.feature_family))
        return

    missing = [
        name
        for name, value in {
            "--start-date": args.start_date,
            "--end-date": args.end_date,
            "--artifact-path": args.artifact_path,
        }.items()
        if not value
    ]
    if missing:
        parser.error(f"Missing required arguments for training mode: {', '.join(missing)}")
    if not args.feature:
        parser.error("Provide at least one --feature.")
    if args.validation_mode == "walk_forward" and (args.wf_train_size is None or args.wf_valid_size is None):
        parser.error("--validation-mode walk_forward requires --wf-train-size and --wf-valid-size.")

    metadata = train_ml_signal_model(
        market=args.market,
        provider_name=args.provider,
        data_root=args.data_root,
        universe_root=args.universe_root,
        reference_root=args.reference_root,
        adjust=args.adjust,
        timeframe=args.timeframe,
        symbols=args.symbols,
        universe=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        artifact_path=args.artifact_path,
        model_name=args.model,
        model_params=parse_kv_pairs(args.model_param),
        feature_columns=list(args.feature),
        label_horizon=args.label_horizon,
        target_mode=args.target_mode,
        train_end_date=args.train_end_date,
        valid_start_date=args.valid_start_date,
        valid_end_date=args.valid_end_date,
        validation_mode=args.validation_mode,
        walk_forward_config=WalkForwardConfig(
            train_size=args.wf_train_size,
            valid_size=args.wf_valid_size,
            step_size=args.wf_step_size,
            expanding=bool(args.wf_expanding),
            purge_size=int(args.purge_size),
            embargo_size=int(args.embargo_size),
        )
        if args.validation_mode == "walk_forward"
        else None,
        tuning_config=TuningConfig(
            trials=int(args.optuna_trials),
            metric=str(args.tune_metric),
            direction=str(args.tune_direction),
            timeout_seconds=args.optuna_timeout,
            keep_top_trials=int(args.keep_top_trials),
            parallel_jobs=int(args.optuna_jobs) if args.optuna_jobs is not None else None,
            gpu_devices=[str(item) for item in args.gpu_device],
        )
        if args.optuna_trials
        else None,
        purge_size=int(args.purge_size),
        embargo_size=int(args.embargo_size),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
