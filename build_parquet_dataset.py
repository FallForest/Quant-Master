from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.build_parquet_dataset import build_parquet_dataset
from research.profiles import load_research_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build parquet dataset from local CSV market data")
    parser.add_argument("--research-profile", help="Optional research profile that supplies market, universe, and adjust defaults")
    parser.add_argument("--market", help="Market id, e.g. ashare. Defaults to the profile setting or ashare")
    parser.add_argument("--input-root", default="data/raw", help="CSV data root")
    parser.add_argument("--output-root", default="data/lake", help="Parquet data root")
    parser.add_argument("--universe-root", help="Universe file root. Defaults to the profile setting or data/universe")
    parser.add_argument("--timeframe", help="Data timeframe. Defaults to the profile setting or 1d")
    parser.add_argument("--adjust", help="Adjustment mode directory, e.g. qfq, hfq, raw. Defaults to the profile setting or qfq")
    parser.add_argument("--symbols", nargs="+", help="Optional symbol list to convert")
    parser.add_argument("--universe", help="Optional universe file name or CSV path")
    parser.add_argument("--max-workers", type=int, help="Optional upper bound for adaptive build concurrency")
    parser.add_argument("--overwrite", action="store_true", help="Force rebuild even if parquet is newer than source CSV")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any symbol fails parquet conversion")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = load_research_profile(args.research_profile) if args.research_profile else None
    summary = build_parquet_dataset(
        market=args.market if args.market else (profile.market if profile is not None else "ashare"),
        input_root=args.input_root,
        output_root=args.output_root,
        universe_root=args.universe_root if args.universe_root else (profile.universe_root if profile is not None else "data/universe"),
        timeframe=args.timeframe if args.timeframe else (profile.timeframe if profile is not None else "1d"),
        adjust=args.adjust if args.adjust else (profile.adjust if profile is not None else "qfq"),
        symbols=args.symbols,
        universe=args.universe or (profile.universe if profile is not None else None),
        max_workers=args.max_workers,
        overwrite=bool(args.overwrite),
        allow_partial=not bool(args.fail_on_error),
    )
    for path in summary.successful_paths:
        print(path)


if __name__ == "__main__":
    main()
