from __future__ import annotations

import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
import pandas as pd

from app.sync_ashare_reference_data import (
    ReferenceSyncTask,
    _build_standardized_dividend_frame,
    _call_with_retries,
    _resolve_reference_task_workers,
    _run_reference_sync_task,
    build_standardized_fundamental_frame,
    resolve_symbols_needing_reference_sync,
)
from data.timeout import SubtaskTimeoutError


def test_resolve_symbols_needing_reference_sync_splits_skipped_and_pending(tmp_path: Path) -> None:
    target_dir = tmp_path / "ashare" / "industry"
    target_dir.mkdir(parents=True)
    existing = target_dir / "000001.csv"
    existing.write_text("change_date\n2026-04-24\n", encoding="utf-8")
    (target_dir / "000001.csv.sync.json").write_text('{"synced_end_date": "2026-04-24"}', encoding="utf-8")

    skipped, pending = resolve_symbols_needing_reference_sync(
        symbols=["000001", "000002"],
        end_date="2026-04-24",
        reference_root=str(tmp_path),
        scope="industry-only",
        overwrite=False,
    )

    assert [item.symbol for item in skipped] == ["000001"]
    assert skipped[0].industry_path is not None
    assert skipped[0].skipped is True
    assert pending == ["000002"]


def test_resolve_symbols_needing_reference_sync_requires_all_requested_scope_files(tmp_path: Path) -> None:
    fundamentals_dir = tmp_path / "ashare" / "fundamentals"
    industry_dir = tmp_path / "ashare" / "industry"
    fundamentals_dir.mkdir(parents=True)
    industry_dir.mkdir(parents=True)
    (fundamentals_dir / "000001.csv").write_text("available_date\n2026-04-24\n", encoding="utf-8")
    (fundamentals_dir / "000001.csv.sync.json").write_text('{"synced_end_date": "2026-04-24"}', encoding="utf-8")
    (industry_dir / "000001.csv").write_text("change_date\n2026-04-24\n", encoding="utf-8")
    (industry_dir / "000001.csv.sync.json").write_text('{"synced_end_date": "2026-04-24"}', encoding="utf-8")
    (fundamentals_dir / "000002.csv").write_text("available_date\n2026-04-24\n", encoding="utf-8")
    (fundamentals_dir / "000002.csv.sync.json").write_text('{"synced_end_date": "2026-04-24"}', encoding="utf-8")

    skipped, pending = resolve_symbols_needing_reference_sync(
        symbols=["000001", "000002"],
        end_date="2026-04-24",
        reference_root=str(tmp_path),
        scope="all",
        overwrite=False,
    )

    assert [item.symbol for item in skipped] == []
    assert pending == ["000001", "000002"]


def test_build_standardized_fundamental_frame_emits_extended_canonical_columns() -> None:
    balance = pd.DataFrame(
        {
            "REPORT_DATE": ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"],
            "NOTICE_DATE": ["2023-04-30", "2023-08-30", "2023-10-31", "2024-03-31", "2024-04-30"],
            "TOTAL_ASSETS": [900.0, 950.0, 980.0, 1000.0, 1200.0],
            "TOTAL_LIABILITIES": [360.0, 380.0, 390.0, 400.0, 500.0],
            "TOTAL_PARENT_EQUITY": [540.0, 570.0, 590.0, 600.0, 700.0],
            "SHARE_CAPITAL": [100.0, 100.0, 100.0, 100.0, 100.0],
            "INVENTORY": [95.0, 98.0, 99.0, 100.0, 130.0],
            "ACCOUNTS_RECE": [45.0, 48.0, 49.0, 50.0, 70.0],
            "NOTE_RECE": [9.0, 9.5, 9.8, 10.0, 12.0],
            "FIXED_ASSET": [190.0, 195.0, 198.0, 200.0, 220.0],
            "INTANGIBLE_ASSET": [28.0, 29.0, 29.5, 30.0, 35.0],
            "MONETARYFUNDS": [140.0, 145.0, 148.0, 150.0, 180.0],
        }
    )
    profit = pd.DataFrame(
        {
            "REPORT_DATE": ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"],
            "NOTICE_DATE": ["2023-04-30", "2023-08-30", "2023-10-31", "2024-03-31", "2024-04-30"],
            "TOTAL_OPERATE_INCOME": [120.0, 260.0, 420.0, 600.0, 150.0],
            "OPERATE_COST": [72.0, 156.0, 252.0, 360.0, 90.0],
            "OPERATE_PROFIT": [12.0, 28.0, 45.0, 72.0, 18.0],
            "PARENT_NETPROFIT": [10.0, 22.0, 35.0, 55.0, 14.0],
        }
    )
    cash = pd.DataFrame(
        {
            "REPORT_DATE": ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"],
            "NOTICE_DATE": ["2023-04-30", "2023-08-30", "2023-10-31", "2024-03-31", "2024-04-30"],
            "NETCASH_OPERATE": [15.0, 35.0, 50.0, 80.0, 20.0],
            "CONSTRUCT_LONG_ASSET": [5.0, 11.0, 18.0, 26.0, 7.0],
        }
    )

    standardized = build_standardized_fundamental_frame(
        balance_frame=balance,
        profit_frame=profit,
        cash_flow_frame=cash,
    )

    latest = standardized.iloc[-1]
    assert "total_liabilities" in standardized.columns
    assert "inventory_growth" in standardized.columns
    assert "receivables_growth" in standardized.columns
    assert "capex_growth" in standardized.columns
    assert latest["total_liabilities"] == 500.0
    assert latest["inventory_growth"] == pytest.approx((130.0 / 95.0) - 1.0)
    assert latest["receivables_growth"] == pytest.approx((70.0 / 45.0) - 1.0)


def test_build_standardized_dividend_frame_normalizes_cash_per_share() -> None:
    raw = pd.DataFrame(
        {
            "实施方案公告日期": ["2024-03-01", "2024-09-01"],
            "分红类型": ["年度分红", "中期分红"],
            "派息比例": [12.0, 8.0],
            "股权登记日": ["2024-03-10", "2024-09-10"],
            "除权日": ["2024-03-11", "2024-09-11"],
            "派息日": ["2024-03-15", "2024-09-15"],
            "报告时间": ["2023年报", "2024半年报"],
        }
    )

    standardized = _build_standardized_dividend_frame(symbol="000001", raw=raw)

    assert list(standardized["cash_dividend_per_share"]) == [1.2, 0.8]
    assert standardized.loc[0, "event_date"] == pd.Timestamp("2024-03-11")


def test_run_reference_sync_task_dispatches_scope(monkeypatch) -> None:
    calls: list[str] = []

    def fake_sync_fundamental_frame(**kwargs):
        calls.append("fundamentals")
        from app.sync_ashare_reference_data import ReferenceSymbolSyncResult

        return ReferenceSymbolSyncResult(
            symbol="000001",
            path="fund.csv",
            industry_path=None,
            dividend_path=None,
            skipped=False,
            attempts=1,
        )

    def fake_sync_industry_frame(**kwargs):
        calls.append("industry")
        return "industry.csv", False

    def fake_sync_dividend_frame(**kwargs):
        calls.append("dividends")
        return "dividend.csv", False

    monkeypatch.setattr("app.sync_ashare_reference_data._sync_fundamental_frame", fake_sync_fundamental_frame)
    monkeypatch.setattr("app.sync_ashare_reference_data._sync_industry_frame", fake_sync_industry_frame)
    monkeypatch.setattr("app.sync_ashare_reference_data._sync_dividend_frame", fake_sync_dividend_frame)

    result = _run_reference_sync_task(
        ak=None,
        task=ReferenceSyncTask(task_id="fundamentals:000001", scope="fundamentals", symbol="000001"),
        end_date="2026-04-24",
        reference_root="data/reference",
        max_attempts=3,
        retry_backoff_seconds=1.0,
        timeout_seconds=60.0,
        overwrite=False,
    )

    assert result is not None
    assert result.path == "fund.csv"
    assert calls == ["fundamentals"]


def test_resolve_reference_task_workers_uses_single_level_multiplier() -> None:
    assert _resolve_reference_task_workers(
        task_count=100,
        max_workers=4,
        bundle_workers=2,
        active_scope_count=3,
    ) == 8

    assert _resolve_reference_task_workers(
        task_count=5,
        max_workers=2,
        bundle_workers=None,
        active_scope_count=3,
    ) == 4


def test_call_with_retries_times_out_remote_subtask() -> None:
    started_at = time.monotonic()

    with pytest.raises(SubtaskTimeoutError):
        _call_with_retries(
            lambda: time.sleep(0.1),
            max_attempts=2,
            retry_backoff_seconds=0.0,
            timeout_seconds=0.01,
            timeout_label="slow-task",
        )

    assert time.monotonic() - started_at < 0.08
