from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.build_parquet_dataset import build_parquet_dataset
from app.sync_ashare_data import sync_ashare_daily


def test_sync_ashare_daily_skips_fresh_files(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "daily"
    output_dir.mkdir(parents=True)
    existing = output_dir / "000001.csv"
    existing.write_text("timestamp,close\n2026-04-24,1.0\n", encoding="utf-8")
    (output_dir / "000001.csv.sync.json").write_text('{"synced_end_date": "2026-04-24"}', encoding="utf-8")

    def fail_download(**kwargs):
        raise AssertionError("download should not run for fresh files")

    monkeypatch.setattr("app.sync_ashare_data._download_daily_with_retries", fail_download)

    summary = sync_ashare_daily(
        symbols=["000001"],
        start_date="2026-01-01",
        end_date="2026-04-24",
        adjust="qfq",
        output_dir=str(output_dir),
        show_progress=False,
        print_summary=False,
    )

    assert summary.succeeded == 0
    assert summary.skipped == 1
    assert summary.failed == 0


def test_sync_ashare_daily_allows_partial_success_and_writes_failure_manifest(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "daily"
    output_dir.mkdir(parents=True)

    def fake_sync_daily_symbol(**kwargs):
        symbol = str(kwargs["symbol"])
        if symbol == "000001":
            path = output_dir / f"{symbol}.csv"
            path.write_text("timestamp,close\n2026-04-24,1.0\n", encoding="utf-8")
            return type("Result", (), {"symbol": symbol, "path": str(path), "skipped": False, "attempts": 1, "succeeded": True})()
        raise RuntimeError("upstream failed")

    monkeypatch.setattr("app.sync_ashare_data._sync_daily_symbol", fake_sync_daily_symbol)

    summary = sync_ashare_daily(
        symbols=["000001", "000002"],
        start_date="2026-01-01",
        end_date="2026-04-24",
        adjust="qfq",
        output_dir=str(output_dir),
        show_progress=False,
        print_summary=False,
    )

    assert summary.succeeded == 1
    assert summary.failed == 1
    assert summary.manifest_path is not None
    assert summary.failures_path is not None
    failures = json.loads(Path(summary.failures_path).read_text(encoding="utf-8"))
    assert failures["failure_count"] == 1
    assert failures["failures"][0]["symbol"] == "000002"


def test_sync_ashare_daily_incrementally_merges_missing_tail(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "daily"
    output_dir.mkdir(parents=True)
    target = output_dir / "000001.csv"
    target.write_text(
        "\n".join(
            [
                "symbol,market,timestamp,open,high,low,close,volume,amount,open_interest",
                "000001,ashare,2026-04-21,1,1,1,1,1,1,",
                "000001,ashare,2026-04-22,2,2,2,2,2,2,",
            ]
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    class FakeProvider:
        def download_daily_frame(self, *, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
            calls.append(start_date)
            return pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "market": "ashare",
                        "timestamp": pd.Timestamp("2026-04-22"),
                        "open": 20,
                        "high": 20,
                        "low": 20,
                        "close": 20,
                        "volume": 20,
                        "amount": 20,
                        "open_interest": None,
                    },
                    {
                        "symbol": symbol,
                        "market": "ashare",
                        "timestamp": pd.Timestamp("2026-04-23"),
                        "open": 3,
                        "high": 3,
                        "low": 3,
                        "close": 3,
                        "volume": 3,
                        "amount": 3,
                        "open_interest": None,
                    },
                    {
                        "symbol": symbol,
                        "market": "ashare",
                        "timestamp": pd.Timestamp("2026-04-24"),
                        "open": 4,
                        "high": 4,
                        "low": 4,
                        "close": 4,
                        "volume": 4,
                        "amount": 4,
                        "open_interest": None,
                    },
                ]
            )

    monkeypatch.setattr("app.sync_ashare_data.AKShareAshareProvider", FakeProvider)

    summary = sync_ashare_daily(
        symbols=["000001"],
        start_date="2026-04-01",
        end_date="2026-04-24",
        adjust="qfq",
        output_dir=str(output_dir),
        incremental_lookback_days=2,
        show_progress=False,
        print_summary=False,
    )

    assert summary.succeeded == 1
    assert calls == ["20260420"]
    merged = pd.read_csv(target)
    assert list(merged["timestamp"]) == ["2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24"]
    assert float(merged.loc[merged["timestamp"] == "2026-04-22", "close"].iloc[0]) == 20.0


def test_build_parquet_dataset_skips_up_to_date_targets(tmp_path: Path) -> None:
    source_path = tmp_path / "raw" / "ashare" / "daily" / "qfq"
    target_path = tmp_path / "lake" / "ashare" / "daily" / "qfq"
    source_path.mkdir(parents=True)
    target_path.mkdir(parents=True)
    source_file = source_path / "000001.csv"
    target_file = target_path / "000001.parquet"
    source_file.write_text("timestamp,open,high,low,close,volume\n2026-04-24,1,1,1,1,1\n", encoding="utf-8")
    target_file.write_text("placeholder", encoding="utf-8")
    os.utime(source_file, (1_700_000_000, 1_700_000_000))
    os.utime(target_file, (1_700_000_100, 1_700_000_100))

    summary = build_parquet_dataset(
        market="ashare",
        input_root=str(tmp_path / "raw"),
        output_root=str(tmp_path / "lake"),
        universe_root=str(tmp_path / "universe"),
        timeframe="1d",
        adjust="qfq",
        symbols=["000001"],
        show_progress=False,
        print_summary=False,
    )

    assert summary.built == 0
    assert summary.skipped == 1
    assert summary.failed == 0


def test_build_parquet_dataset_allows_partial_success_and_writes_failure_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "raw" / "ashare" / "daily" / "qfq"
    source_path.mkdir(parents=True)
    (source_path / "000001.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-04-24,1,1,1,1,1\n",
        encoding="utf-8",
    )

    summary = build_parquet_dataset(
        market="ashare",
        input_root=str(tmp_path / "raw"),
        output_root=str(tmp_path / "lake"),
        universe_root=str(tmp_path / "universe"),
        timeframe="1d",
        adjust="qfq",
        symbols=["000001", "000002"],
        show_progress=False,
        print_summary=False,
    )

    assert summary.built == 1
    assert summary.failed == 1
    assert summary.failures_path is not None
    failures = json.loads(Path(summary.failures_path).read_text(encoding="utf-8"))
    assert failures["failure_count"] == 1
    assert failures["failures"][0]["symbol"] == "000002"
