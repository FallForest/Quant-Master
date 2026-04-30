from __future__ import annotations

from datetime import datetime
from importlib import import_module
from pathlib import Path
from threading import Lock

import pandas as pd

from data.providers.security_router import AshareSecurityRouter, SecurityInfo
from data.providers.symbols import normalize_symbol


def _silent_tqdm(iterable, *args, **kwargs):
    return iterable


class AKShareAshareProvider:
    _internal_progress_patch_lock = Lock()
    _internal_progress_patched = False

    def __init__(self) -> None:
        try:
            self.ak = import_module("akshare")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AKShare is not installed. Run `py -m pip install -r requirements.txt` first."
            ) from exc
        self._disable_internal_progress_bars()
        self.router = AshareSecurityRouter()

    def load_bars(
        self,
        symbol: str,
        market: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        if market != "ashare":
            raise ValueError(f"AKShareAshareProvider only supports market='ashare', got {market}")
        if timeframe != "1d":
            raise ValueError(f"AKShareAshareProvider only supports timeframe='1d', got {timeframe}")

        security = self.router.resolve(symbol)
        raw = self._fetch_hist_data(
            security=security,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        return self._normalize_hist_frame(raw=raw, security=security)

    def load_universe(
        self,
        market: str,
        universe: str | None = None,
        date: datetime | None = None,
    ) -> list[str]:
        if market != "ashare":
            return []

        try:
            raw = self.ak.stock_zh_a_spot_em()
        except Exception:
            return []

        code_column = "\u4ee3\u7801"
        if code_column not in raw.columns:
            return []
        return raw[code_column].astype(str).tolist()

    def load_fundamentals(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def download_daily_to_csv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
        output_path: str | Path,
    ) -> Path:
        normalized = self.download_daily_frame(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(target, index=False)
        return target

    def download_daily_frame(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        security = self.router.resolve(symbol)
        raw = self._fetch_hist_data(
            security=security,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        return self._normalize_hist_frame(raw=raw, security=security)

    def _normalize_hist_frame(self, raw: pd.DataFrame, security: SecurityInfo) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(
                columns=[
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
            )

        renamed = raw.rename(
            columns={
                "\u65e5\u671f": "timestamp",
                "\u5f00\u76d8": "open",
                "\u6700\u9ad8": "high",
                "\u6700\u4f4e": "low",
                "\u6536\u76d8": "close",
                "\u6210\u4ea4\u91cf": "volume",
                "\u6210\u4ea4\u989d": "amount",
                "date": "timestamp",
                "close": "close",
                "open": "open",
                "high": "high",
                "low": "low",
            }
        ).copy()

        if security.source == "stock_tx" and "amount" in raw.columns and "date" in raw.columns:
            # AKShare documents TX `amount` in units of hands, so it belongs in volume, not turnover amount.
            renamed["volume"] = raw["amount"]
            renamed["amount"] = None

        if "amount" not in renamed.columns:
            renamed["amount"] = None

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in renamed.columns]
        if missing:
            raise ValueError(f"AKShare returned unexpected columns, missing: {missing}")

        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"])
        renamed["symbol"] = security.symbol
        renamed["market"] = "ashare"
        renamed["open_interest"] = None

        return renamed[
            [
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
        ].sort_values("timestamp").reset_index(drop=True)

    def _fetch_hist_data(
        self,
        security: SecurityInfo,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        if security.source == "stock_tx":
            return self.ak.stock_zh_a_hist_tx(
                symbol=self._to_tx_symbol(security.symbol),
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

        if security.source == "fund_sina":
            return self._filter_sina_hist(
                self.ak.fund_etf_hist_sina(symbol=self._to_tx_symbol(security.symbol)),
                start_date=start_date,
                end_date=end_date,
            )

        raise ValueError(f"Unsupported data source for symbol {security.symbol}: {security.source}")

    @classmethod
    def _disable_internal_progress_bars(cls) -> None:
        if cls._internal_progress_patched:
            return

        with cls._internal_progress_patch_lock:
            if cls._internal_progress_patched:
                return

            # AKShare's Tencent daily bar endpoint prints its own tqdm per symbol,
            # which becomes unreadable once we add our own batch-level progress bar.
            noisy_module = import_module("akshare.stock_feature.stock_hist_tx")
            setattr(noisy_module, "get_tqdm", lambda *args, **kwargs: _silent_tqdm)
            cls._internal_progress_patched = True

    def _to_tx_symbol(self, symbol: str) -> str:
        code = normalize_symbol(symbol)
        prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
        return f"{prefix}{code}"

    def _filter_sina_hist(
        self,
        raw: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if raw.empty or "date" not in raw.columns:
            return raw

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        normalized = raw.copy()
        normalized["date"] = pd.to_datetime(normalized["date"])
        return normalized[(normalized["date"] >= start) & (normalized["date"] <= end)].reset_index(drop=True)
