from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.providers.symbols import normalize_symbol


@dataclass(frozen=True)
class SecurityInfo:
    symbol: str
    asset_type: str
    source: str


class AshareSecurityRouter:
    def __init__(
        self,
        security_master_path: str | Path = "data/universe/ashare/security_master.csv",
    ) -> None:
        self.security_master_path = Path(security_master_path)
        self._security_master = self._load_security_master()

    def resolve(self, symbol: str) -> SecurityInfo:
        normalized = self.normalize_symbol(symbol)
        asset_type = self._lookup_asset_type(normalized) or self._infer_asset_type(normalized)

        if asset_type == "stock":
            return SecurityInfo(symbol=normalized, asset_type=asset_type, source="stock_tx")
        if asset_type in {"etf", "lof"}:
            return SecurityInfo(symbol=normalized, asset_type=asset_type, source="fund_sina")

        raise ValueError(f"Unsupported asset type for symbol {normalized}: {asset_type}")

    def normalize_symbol(self, symbol: object) -> str:
        return normalize_symbol(symbol)

    def _load_security_master(self) -> dict[str, str]:
        if not self.security_master_path.exists():
            return {}

        table = pd.read_csv(self.security_master_path)
        if table.empty:
            return {}
        if "symbol" not in table.columns or "asset_type" not in table.columns:
            raise ValueError("security_master.csv must contain `symbol` and `asset_type` columns")

        normalized: dict[str, str] = {}
        for _, row in table.iterrows():
            symbol = self.normalize_symbol(row["symbol"])
            asset_type = str(row["asset_type"]).strip().lower()
            if not symbol or not asset_type:
                continue
            normalized[symbol] = asset_type
        return normalized

    def _lookup_asset_type(self, symbol: str) -> str | None:
        return self._security_master.get(symbol)

    def _infer_asset_type(self, symbol: str) -> str:
        if symbol.startswith(("15", "16", "50", "51", "52", "56", "58", "159")):
            if symbol.startswith("16"):
                return "lof"
            return "etf"

        if symbol.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")):
            return "stock"

        raise ValueError(
            f"Cannot infer asset type for symbol {symbol}. "
            "Add it to data/universe/ashare/security_master.csv with columns: symbol,asset_type."
        )
