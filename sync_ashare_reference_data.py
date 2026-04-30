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
from app.sync_ashare_reference_data import REFERENCE_SYNC_SCOPES, sync_ashare_reference_data
from research.profiles import load_research_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync benchmark and fundamental reference data for auxiliary ML factors")
    parser.add_argument("--research-profile", help="Optional research profile that supplies the default universe and benchmark symbol")
    parser.add_argument("--symbols", nargs="+", help="A-share symbols, e.g. 000001 600519 000333")
    parser.add_argument("--universe", help="Universe csv name under data/universe/ashare, for example hs300")
    parser.add_argument("--universe-root", default="data/universe", help="Universe file root")
    parser.add_argument("--start", required=True, help="Benchmark start date in YYYYMMDD")
    parser.add_argument("--end", required=True, help="Benchmark end date in YYYYMMDD")
    parser.add_argument("--benchmark-symbol", help="Benchmark symbol. Defaults to the profile setting or sh000300")
    parser.add_argument("--reference-root", default="data/reference", help="Reference data root")
    parser.add_argument("--max-workers", type=int, help="Optional upper bound for adaptive sync concurrency")
    parser.add_argument("--bundle-workers", type=int, help="Optional per-symbol subtask concurrency across fundamentals/industry/dividends")
    parser.add_argument("--subtask-timeout-seconds", type=float, default=60.0, help="Per-remote-subtask timeout in seconds")
    parser.add_argument(
        "--scope",
        default="all",
        choices=sorted(REFERENCE_SYNC_SCOPES),
        help="Sync scope. Default syncs benchmark, fundamentals, industry, and dividends together.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any symbol still fails after retries")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = load_research_profile(args.research_profile) if args.research_profile else None
    universe = args.universe or (profile.universe if profile is not None else None)
    benchmark_symbol = args.benchmark_symbol or (profile.benchmark_symbol if profile is not None else "sh000300")
    symbols = resolve_sync_symbols(
        symbols=args.symbols,
        universe=universe,
        universe_root=args.universe_root,
        adjust="qfq",
    )
    summary = sync_ashare_reference_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        benchmark_symbol=benchmark_symbol,
        reference_root=args.reference_root,
        max_workers=args.max_workers,
        bundle_workers=args.bundle_workers,
        timeout_seconds=args.subtask_timeout_seconds,
        overwrite=bool(args.overwrite),
        allow_partial=not bool(args.fail_on_error),
        scope=str(args.scope),
    )
    print(
        json.dumps(
            {
                "scope": args.scope,
                "benchmark_path": summary.benchmark_path,
                "fundamental_paths": summary.fundamental_paths,
                "industry_paths": summary.industry_paths,
                "dividend_paths": summary.dividend_paths,
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
