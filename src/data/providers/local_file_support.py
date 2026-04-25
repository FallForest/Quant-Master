from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from data.providers.symbols import normalize_symbol

BAR_COLUMNS = [
    "symbol",
    "market",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "open_interest",
]


class LocalFileProviderSupport:
    file_suffix = ""

    def __init__(
        self,
        base_path: str | Path = "data/raw",
        universe_root: str | Path = "data/universe",
        default_adjust: str = "qfq",
    ) -> None:
        self.base_path = Path(base_path)
        self.universe_root = Path(universe_root)
        self.default_adjust = default_adjust or "raw"

    def load_universe_from_csv(
        self,
        market: str,
        universe: str | None,
    ) -> list[str]:
        if universe:
            universe_path = self.resolve_universe_path(market=market, universe=universe)
            if not universe_path.exists():
                return []

            table = pd.read_csv(universe_path)
            if table.empty:
                return []

            if "symbol" in table.columns:
                symbols = table["symbol"]
            elif "code" in table.columns:
                symbols = table["code"]
            else:
                symbols = table.iloc[:, 0]

            normalized = [self.normalize_symbol(item) for item in symbols.dropna().tolist()]
            return list(dict.fromkeys(normalized))
        return []

    def load_bars(
        self,
        symbol: str,
        market: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        file_path = self.resolve_data_path(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            suffix=self.file_suffix,
        )
        if not file_path.exists():
            return self.empty_bars()

        data = self.read_data(file_path)
        data = self.normalize_bar_columns(data=data, symbol=symbol, market=market)
        return self.filter_bars(data=data, start=start, end=end)

    def load_universe(
        self,
        market: str,
        universe: str | None = None,
        date: datetime | None = None,
    ) -> list[str]:
        symbols = self.load_universe_from_csv(market=market, universe=universe)
        if universe is not None:
            return symbols

        files = list(self.iter_market_data_files(market=market, timeframe="1d", suffix=self.file_suffix))
        return sorted({path.stem for path in files})

    def load_fundamentals(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def resolve_data_path(
        self,
        symbol: str,
        market: str,
        timeframe: str,
        suffix: str,
    ) -> Path:
        normalized_symbol = self.normalize_symbol(symbol)
        timeframe_dir = self.normalize_timeframe_dir(timeframe)
        adjust_dir = self.normalize_adjust_dir(self.default_adjust)
        candidates = [
            self.base_path / market / timeframe_dir / adjust_dir / f"{normalized_symbol}{suffix}",
            self.base_path / market / timeframe_dir / f"{normalized_symbol}{suffix}",
            self.base_path / market / f"{normalized_symbol}{suffix}",
            self.base_path / f"{normalized_symbol}{suffix}",
        ]

        market_timeframe_dir = self.base_path / market / timeframe_dir
        if market_timeframe_dir.exists():
            for path in market_timeframe_dir.glob(f"*/*{suffix}"):
                if path.stem == normalized_symbol:
                    candidates.append(path)

        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def iter_market_data_files(self, market: str, timeframe: str, suffix: str) -> list[Path]:
        timeframe_dir = self.normalize_timeframe_dir(timeframe)
        candidates = [
            self.base_path / market / timeframe_dir / self.normalize_adjust_dir(self.default_adjust),
            self.base_path / market / timeframe_dir,
            self.base_path / market,
        ]
        files: list[Path] = []
        seen: set[Path] = set()
        for directory in candidates:
            if not directory.exists():
                continue
            for path in directory.rglob(f"*{suffix}"):
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                files.append(path)
        return files

    def resolve_universe_path(self, market: str, universe: str) -> Path:
        candidate = Path(universe)
        if candidate.is_absolute() or len(candidate.parts) > 1:
            return candidate

        file_name = universe if universe.endswith(".csv") else f"{universe}.csv"
        return self.universe_root / market / file_name

    def normalize_bar_columns(self, data: pd.DataFrame, symbol: str, market: str) -> pd.DataFrame:
        renamed = data.rename(
            columns={
                "date": "timestamp",
                "trade_date": "timestamp",
                "datetime": "timestamp",
                "vol": "volume",
                "turnover": "amount",
            }
        ).copy()

        if "timestamp" not in renamed.columns:
            raise ValueError("Data file must contain one of: timestamp, date, trade_date, datetime")

        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"])
        if "symbol" in renamed.columns:
            renamed["symbol"] = renamed["symbol"].map(self.normalize_symbol)
        else:
            renamed["symbol"] = self.normalize_symbol(symbol)
        renamed["market"] = market
        for column in ["amount", "open_interest"]:
            if column not in renamed.columns:
                renamed[column] = None

        required = ["open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in renamed.columns]
        if missing:
            raise ValueError(f"Data file missing required columns: {missing}")

        return renamed[BAR_COLUMNS]

    def filter_bars(self, data: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        return data[(data["timestamp"] >= start) & (data["timestamp"] <= end)].sort_values("timestamp").reset_index(
            drop=True
        )

    def empty_bars(self) -> pd.DataFrame:
        return pd.DataFrame(columns=BAR_COLUMNS)

    def read_data(self, file_path: Path) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement read_data()")

    def normalize_symbol(self, symbol: object) -> str:
        return normalize_symbol(symbol)

    def normalize_timeframe_dir(self, timeframe: str) -> str:
        if timeframe == "1d":
            return "daily"
        return timeframe

    def normalize_adjust_dir(self, adjust: str) -> str:
        return adjust if adjust else "raw"
