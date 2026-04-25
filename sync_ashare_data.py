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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync A-share daily data via AKShare")
    parser.add_argument("--symbols", nargs="+", help="A-share symbols, e.g. 000001 600519 000333")
    parser.add_argument("--universe", help="Universe csv name under data/universe/ashare, for example hs300")
    parser.add_argument("--universe-root", default="data/universe", help="Universe file root")
    parser.add_argument("--start", required=True, help="Start date in YYYYMMDD")
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y%m%d"),
        help="End date in YYYYMMDD. Defaults to today.",
    )
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="Adjustment mode")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for normalized CSV files. Defaults to data/raw/ashare/daily/<adjust>",
    )
    parser.add_argument("--max-workers", type=int, help="Optional upper bound for adaptive download concurrency")
    parser.add_argument("--allow-partial", action="store_true", help="Keep successful files even if some symbols fail")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adjust_dir = args.adjust if args.adjust else "raw"
    output_dir = args.output_dir or f"data/raw/ashare/daily/{adjust_dir}"
    symbols = resolve_sync_symbols(
        symbols=args.symbols,
        universe=args.universe,
        universe_root=args.universe_root,
        adjust=args.adjust,
    )
    summary = sync_ashare_daily(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
        output_dir=output_dir,
        max_workers=args.max_workers,
        allow_partial=bool(args.allow_partial),
    )
    for path in summary.successful_paths:
        print(path)


if __name__ == "__main__":
    main()
