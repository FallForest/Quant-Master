from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.build_parquet_dataset import build_parquet_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build parquet dataset from local CSV market data")
    parser.add_argument("--market", default="ashare", help="Market id, e.g. ashare")
    parser.add_argument("--input-root", default="data/raw", help="CSV data root")
    parser.add_argument("--output-root", default="data/lake", help="Parquet data root")
    parser.add_argument("--universe-root", default="data/universe", help="Universe file root")
    parser.add_argument("--timeframe", default="1d", help="Data timeframe, current default is 1d")
    parser.add_argument("--adjust", default="qfq", help="Adjustment mode directory, e.g. qfq, hfq, raw")
    parser.add_argument("--symbols", nargs="+", help="Optional symbol list to convert")
    parser.add_argument("--universe", help="Optional universe file name or CSV path")
    parser.add_argument("--max-workers", type=int, help="Optional upper bound for adaptive build concurrency")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = build_parquet_dataset(
        market=args.market,
        input_root=args.input_root,
        output_root=args.output_root,
        universe_root=args.universe_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
        symbols=args.symbols,
        universe=args.universe,
        max_workers=args.max_workers,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
