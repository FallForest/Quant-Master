from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.sync_index_universe import sync_csindex_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync a CSI index constituent universe into a local CSV file")
    parser.add_argument("--index-symbol", default="000300", help="CSI index code, for example 000300 for HS300")
    parser.add_argument("--output-name", default="hs300", help="Universe file name without extension")
    parser.add_argument("--market", default="ashare")
    parser.add_argument("--universe-root", default="data/universe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = sync_csindex_universe(
        index_symbol=args.index_symbol,
        output_name=args.output_name,
        market=args.market,
        universe_root=args.universe_root,
    )
    print(path)


if __name__ == "__main__":
    main()
