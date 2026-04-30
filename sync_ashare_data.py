from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.sync_ashare_data import resolve_sync_symbols, sync_ashare_daily
from research.profiles import load_research_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync A-share daily data via AKShare")
    parser.add_argument("--research-profile", help="Optional research profile that supplies the default universe and adjust mode")
    parser.add_argument("--symbols", nargs="+", help="A-share symbols, e.g. 000001 600519 000333")
    parser.add_argument("--universe", help="Universe csv name under data/universe/ashare, for example hs300")
    parser.add_argument("--universe-root", default="data/universe", help="Universe file root")
    parser.add_argument("--start", required=True, help="Start date in YYYYMMDD")
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y%m%d"),
        help="End date in YYYYMMDD. Defaults to today.",
    )
    parser.add_argument("--adjust", choices=["", "qfq", "hfq"], help="Adjustment mode. Defaults to the profile setting or qfq")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for normalized CSV files. Defaults to data/raw/ashare/daily/<adjust>",
    )
    parser.add_argument("--max-workers", type=int, help="Optional upper bound for adaptive download concurrency")
    parser.add_argument("--overwrite", action="store_true", help="Force re-download even if the local file is already fresh")
    parser.add_argument(
        "--incremental-lookback-days",
        type=int,
        default=5,
        help="When a local CSV already exists, re-pull this many calendar days before the last local bar and merge/deduplicate.",
    )
    parser.add_argument("--subtask-timeout-seconds", type=float, default=60.0, help="Per-symbol remote request timeout in seconds")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any symbol still fails after retries")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = load_research_profile(args.research_profile) if args.research_profile else None
    adjust = args.adjust if args.adjust is not None else (profile.adjust if profile is not None else "qfq")
    universe = args.universe or (profile.universe if profile is not None else None)
    adjust_dir = adjust if adjust else "raw"
    output_dir = args.output_dir or f"data/raw/ashare/daily/{adjust_dir}"
    symbols = resolve_sync_symbols(
        symbols=args.symbols,
        universe=universe,
        universe_root=args.universe_root,
        adjust=adjust,
    )
    summary = sync_ashare_daily(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        adjust=adjust,
        output_dir=output_dir,
        max_workers=args.max_workers,
        timeout_seconds=args.subtask_timeout_seconds,
        incremental_lookback_days=args.incremental_lookback_days,
        allow_partial=not bool(args.fail_on_error),
        overwrite=bool(args.overwrite),
    )
    for path in summary.successful_paths:
        print(path)


if __name__ == "__main__":
    main()
