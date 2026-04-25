from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.sync_ashare_data import resolve_sync_symbols
from app.sync_ashare_reference_data import sync_ashare_reference_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync benchmark and fundamental reference data for auxiliary ML factors")
    parser.add_argument("--symbols", nargs="+", help="A-share symbols, e.g. 000001 600519 000333")
    parser.add_argument("--universe", help="Universe csv name under data/universe/ashare, for example hs300")
    parser.add_argument("--universe-root", default="data/universe", help="Universe file root")
    parser.add_argument("--start", required=True, help="Benchmark start date in YYYYMMDD")
    parser.add_argument("--end", required=True, help="Benchmark end date in YYYYMMDD")
    parser.add_argument("--benchmark-symbol", default="sh000300", help="Benchmark symbol, default is HS300")
    parser.add_argument("--reference-root", default="data/reference", help="Reference data root")
    parser.add_argument("--max-workers", type=int, help="Optional upper bound for adaptive sync concurrency")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist")
    parser.add_argument("--allow-partial", action="store_true", help="Keep successful files even if some symbols fail")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    symbols = resolve_sync_symbols(
        symbols=args.symbols,
        universe=args.universe,
        universe_root=args.universe_root,
        adjust="qfq",
    )
    summary = sync_ashare_reference_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        benchmark_symbol=args.benchmark_symbol,
        reference_root=args.reference_root,
        max_workers=args.max_workers,
        overwrite=bool(args.overwrite),
        allow_partial=bool(args.allow_partial),
    )
    print(
        json.dumps(
            {
                "benchmark_path": summary.benchmark_path,
                "fundamental_paths": summary.fundamental_paths,
                "total": summary.total,
                "downloaded": summary.succeeded,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "elapsed_seconds": summary.elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
