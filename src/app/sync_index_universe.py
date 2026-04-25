from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd

from data.providers.symbols import normalize_symbol


def sync_csindex_universe(
    *,
    index_symbol: str,
    output_name: str,
    market: str = "ashare",
    universe_root: str = "data/universe",
) -> str:
    try:
        ak = import_module("akshare")
    except ModuleNotFoundError as exc:
        raise RuntimeError("AKShare is not installed. Run `py -m pip install -r requirements.txt` first.") from exc

    raw = ak.index_stock_cons_csindex(symbol=index_symbol)
    if raw.empty:
        raise ValueError(f"No constituents returned for index {index_symbol}")

    normalized = pd.DataFrame(
        {
            "symbol": raw["成分券代码"].map(normalize_symbol),
            "name": raw["成分券名称"].astype(str),
            "index_code": str(index_symbol),
            "index_name": raw["指数名称"].astype(str),
            "as_of_date": pd.to_datetime(raw["日期"]).dt.date.astype(str),
            "exchange": raw["交易所"].astype(str),
        }
    ).drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    target = Path(universe_root) / market / f"{output_name}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(target, index=False, encoding="utf-8-sig")
    return str(target)
