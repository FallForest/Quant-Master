from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.factors.builder import build_factor_frame
from ml.factors.families import FACTOR_FAMILIES
from ml.factors.registry import (
    DEFAULT_FACTOR_REGISTRY,
    estimate_factor_history_lookback,
    get_qlib_alpha158_factor_names,
    get_qlib_alpha360_factor_names,
)


def test_default_registry_exposes_new_canonical_factors() -> None:
    names = set(DEFAULT_FACTOR_REGISTRY.list_factor_names())

    assert "return_120" in names
    assert "close_to_sma_120" in names
    assert "momentum_63_21" in names
    assert "industry_momentum_63_21" in names
    assert "price_to_26w_high" in names
    assert "volume_ratio_60" in names
    assert "volume_trend_10_60" in names
    assert "stochastic_k_14" in names
    assert "stochastic_d_14_3" in names
    assert "williams_r_14" in names
    assert "money_flow_index_14" in names
    assert "upper_shadow_pct" in names
    assert "lower_shadow_pct" in names
    assert "real_body_pct" in names
    assert "beta_20_hs300" in names
    assert "beta_252_hs300" in names
    assert "ivol_120_hs300" in names
    assert "amihud_illiquidity_5" in names
    assert "turnover_5" in names
    assert "abnormal_turnover_5" in names
    assert "dollar_volume_5" in names
    assert "channel_position_60" in names
    assert "distance_to_high_60" in names
    assert "distance_to_low_60" in names
    assert "atr_14_pct" in names
    assert "atr_20_pct" in names
    assert "parkinson_volatility_20" in names
    assert "parkinson_volatility_60" in names
    assert "garman_klass_volatility_20" in names
    assert "garman_klass_volatility_60" in names
    assert "cashflow_to_price" in names
    assert "dividend_yield_ttm" in names
    assert "roa_ttm" in names
    assert "gross_margin" in names
    assert "operating_margin" in names
    assert "cash_profitability" in names
    assert "asset_turnover" in names
    assert "dollar_volume_20" in names
    assert "dollar_volume_60" in names
    assert "downside_volatility_20" in names
    assert "downside_volatility_60" in names
    assert "amihud_illiquidity_60" in names
    assert "turnover_60" in names
    assert "abnormal_turnover_60" in names
    assert "volatility_60" in names
    assert "industry_momentum_126_21" in names
    assert "industry_momentum_252_21" in names
    assert "liabilities_to_assets" in names
    assert "liabilities_to_equity" in names
    assert "cash_to_assets" in names
    assert "inventory_to_assets" in names
    assert "receivables_to_assets" in names
    assert "inventory_growth" in names
    assert "receivables_growth" in names
    assert "capex_growth" in names
    assert "dividend_payout_ratio_ttm" in names
    assert "KMID" in names
    assert "OPEN0" in names
    assert "OPEN59" in names
    assert "VWAP59" in names
    assert "VOLUME59" in names
    assert "BETA60" in names
    assert "RSQR30" in names
    assert "IMXD20" in names
    assert "VSUMD10" in names

    assert "qlib_alpha158" not in FACTOR_FAMILIES
    assert "qlib_alpha360" not in FACTOR_FAMILIES
    assert "rolling_stats" in FACTOR_FAMILIES
    assert "raw_price_history" in FACTOR_FAMILIES
    assert "raw_volume_history" in FACTOR_FAMILIES

    open0 = DEFAULT_FACTOR_REGISTRY.get_factor("OPEN0")
    open59 = DEFAULT_FACTOR_REGISTRY.get_factor("OPEN59")
    kmid = DEFAULT_FACTOR_REGISTRY.get_factor("KMID")
    ma5 = DEFAULT_FACTOR_REGISTRY.get_factor("MA5")
    volume59 = DEFAULT_FACTOR_REGISTRY.get_factor("VOLUME59")

    assert open0.family == "raw_price_history"
    assert "alpha158" in open0.tags
    assert "alpha360" in open0.tags
    assert open59.family == "raw_price_history"
    assert "alpha158" not in open59.tags
    assert "alpha360" in open59.tags
    assert kmid.family == "bar_shape"
    assert "alpha158" in kmid.tags
    assert ma5.family == "rolling_stats"
    assert "alpha158" in ma5.tags
    assert volume59.family == "raw_volume_history"
    assert "alpha360" in volume59.tags


def test_qlib_named_sets_follow_expected_membership() -> None:
    alpha158 = get_qlib_alpha158_factor_names()
    alpha360 = get_qlib_alpha360_factor_names()

    assert "KMID" in alpha158
    assert "OPEN0" in alpha158
    assert "HIGH0" in alpha158
    assert "LOW0" in alpha158
    assert "VWAP0" in alpha158
    assert "VOLUME0" not in alpha158
    assert "CLOSE0" not in alpha158
    assert "OPEN1" not in alpha158
    assert "MA5" in alpha158
    assert "VSUMD60" in alpha158

    assert "OPEN0" in alpha360
    assert "OPEN59" in alpha360
    assert "CLOSE0" in alpha360
    assert "VWAP59" in alpha360
    assert "VOLUME0" in alpha360
    assert "VOLUME59" in alpha360
    assert "KMID" not in alpha360
    assert "MA5" not in alpha360


def test_build_factor_frame_computes_new_fundamental_factors(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference" / "ashare" / "fundamentals"
    reference_root.mkdir(parents=True)
    (reference_root / "000001.csv").write_text(
        (
            "symbol,report_date,available_date,total_assets,total_liabilities,total_parent_equity,share_capital,"
            "inventory,accounts_rece,note_rece,fixed_asset,intangible_asset,monetary_funds,"
            "total_operate_income_ttm,operate_cost_ttm,operate_profit_ttm,parent_netprofit_ttm,"
            "netcash_operate_ttm,capex_ttm,asset_growth,investment_to_assets,accruals,inventory_growth,receivables_growth,capex_growth\n"
            "000001,2023-12-31,2023-12-31,2000,800,1000,100,200,120,0,300,50,400,500,300,50,40,60,80,0.1,0.2,0.01,0.3,0.4,0.5\n"
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "symbol": ["000001"],
            "open": [20.0],
            "high": [20.5],
            "low": [19.8],
            "close": [20.0],
            "volume": [1000.0],
        }
    )

    features = build_factor_frame(
        frame,
        factor_names=[
            "cashflow_to_price",
            "roa_ttm",
            "gross_margin",
            "operating_margin",
            "cash_profitability",
            "asset_turnover",
            "liabilities_to_assets",
            "liabilities_to_equity",
            "cash_to_assets",
            "inventory_to_assets",
            "receivables_to_assets",
            "inventory_growth",
            "receivables_growth",
            "capex_growth",
        ],
        reference_root=str(tmp_path / "reference"),
    )

    row = features.iloc[0]
    assert row["cashflow_to_price"] == pytest.approx(0.03)
    assert row["roa_ttm"] == pytest.approx(0.02)
    assert row["gross_margin"] == pytest.approx(0.4)
    assert row["operating_margin"] == pytest.approx(0.1)
    assert row["cash_profitability"] == pytest.approx(0.03)
    assert row["asset_turnover"] == pytest.approx(0.25)
    assert row["liabilities_to_assets"] == pytest.approx(0.4)
    assert row["liabilities_to_equity"] == pytest.approx(0.8)
    assert row["cash_to_assets"] == pytest.approx(0.2)
    assert row["inventory_to_assets"] == pytest.approx(0.1)
    assert row["receivables_to_assets"] == pytest.approx(0.06)
    assert row["inventory_growth"] == pytest.approx(0.3)
    assert row["receivables_growth"] == pytest.approx(0.4)
    assert row["capex_growth"] == pytest.approx(0.5)


def test_build_factor_frame_computes_dollar_volume_and_downside_volatility() -> None:
    returns = [
        -0.01,
        0.02,
        -0.03,
        0.01,
        -0.02,
        0.015,
        -0.005,
        0.01,
        -0.01,
        0.02,
        -0.015,
        0.01,
        -0.02,
        0.005,
        -0.01,
        0.03,
        -0.01,
        0.02,
        -0.025,
        0.01,
        -0.005,
        0.015,
        -0.01,
        0.02,
    ]
    closes = [100.0]
    for daily_return in returns:
        closes.append(closes[-1] * (1.0 + daily_return))

    timestamps = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    amount = pd.Series(range(1, len(closes) + 1), dtype=float) * 1000.0
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["000001"] * len(closes),
            "open": closes,
            "high": [price * 1.01 for price in closes],
            "low": [price * 0.99 for price in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
            "amount": amount,
        }
    )

    features = build_factor_frame(
        frame,
        factor_names=["dollar_volume_20", "downside_volatility_20"],
    )

    expected_dollar_volume = amount.iloc[-20:].mean()
    return_series = pd.Series(closes, dtype=float).pct_change()
    expected_downside = return_series.clip(upper=0.0).pow(2).iloc[-20:].mean() ** 0.5

    row = features.iloc[-1]
    assert row["dollar_volume_20"] == pytest.approx(expected_dollar_volume)
    assert row["downside_volatility_20"] == pytest.approx(expected_downside)


def test_build_factor_frame_computes_dividend_yield_and_payout_ratio(tmp_path: Path) -> None:
    fundamentals_root = tmp_path / "reference" / "ashare" / "fundamentals"
    dividends_root = tmp_path / "reference" / "ashare" / "dividends"
    fundamentals_root.mkdir(parents=True)
    dividends_root.mkdir(parents=True)
    (fundamentals_root / "000001.csv").write_text(
        (
            "symbol,report_date,available_date,total_assets,total_parent_equity,share_capital,"
            "total_operate_income_ttm,operate_cost_ttm,operate_profit_ttm,parent_netprofit_ttm,"
            "netcash_operate_ttm,asset_growth,investment_to_assets,accruals\n"
            "000001,2023-12-31,2023-12-31,2000,1000,100,500,300,50,40,60,0.1,0.2,0.01\n"
        ),
        encoding="utf-8",
    )
    (dividends_root / "000001.csv").write_text(
        (
            "symbol,announcement_date,record_date,ex_date,pay_date,event_date,cash_dividend_per_share,dividend_type,report_period\n"
            "000001,2024-03-01,2024-03-10,2024-03-11,2024-03-15,2024-03-11,1.2,年度分红,2023年报\n"
            "000001,2024-09-01,2024-09-10,2024-09-11,2024-09-15,2024-09-11,0.8,中期分红,2024半年报\n"
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-12-31"]),
            "symbol": ["000001"],
            "open": [20.0],
            "high": [20.5],
            "low": [19.8],
            "close": [20.0],
            "volume": [1000.0],
        }
    )

    features = build_factor_frame(
        frame,
        factor_names=["dividend_yield_ttm", "dividend_payout_ratio_ttm"],
        reference_root=str(tmp_path / "reference"),
    )

    row = features.iloc[0]
    assert row["dividend_yield_ttm"] == pytest.approx(0.1)
    assert row["dividend_payout_ratio_ttm"] == pytest.approx(5.0)


def test_build_factor_frame_computes_industry_momentum_from_peer_returns(tmp_path: Path) -> None:
    industry_root = tmp_path / "reference" / "ashare" / "industry"
    industry_root.mkdir(parents=True)
    for symbol in ["000001", "000002"]:
        (industry_root / f"{symbol}.csv").write_text(
            (
                "symbol,change_date,standard,sector,industry_level_1,industry_level_2,industry_level_3,industry_code\n"
                f"{symbol},2020-01-01,申银万国行业分类标准,消费,食品饮料,白酒,白酒,{801120}\n"
            ),
            encoding="utf-8",
        )

    timestamps = pd.date_range("2024-01-01", periods=140, freq="D")
    close_a = [100.0]
    close_b = [80.0]
    for _ in range(139):
        close_a.append(close_a[-1] * 1.005)
        close_b.append(close_b[-1] * 1.01)

    frame = pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "symbol": ["000001"] * len(timestamps) + ["000002"] * len(timestamps),
            "open": close_a + close_b,
            "high": [value * 1.01 for value in close_a + close_b],
            "low": [value * 0.99 for value in close_a + close_b],
            "close": close_a + close_b,
            "volume": [1000.0] * (len(timestamps) * 2),
        }
    )

    features = build_factor_frame(
        frame,
        factor_names=["industry_momentum_126_21"],
        reference_root=str(tmp_path / "reference"),
    )
    target = features[features["symbol"] == "000001"].sort_values("timestamp").reset_index(drop=True)
    expected = pd.Series(close_b, dtype=float).shift(21) / pd.Series(close_b, dtype=float).shift(126) - 1.0

    assert target.loc[len(target) - 1, "industry_momentum_126_21"] == pytest.approx(float(expected.iloc[-1]))


def test_build_factor_frame_computes_new_technical_factors() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=25, freq="D"),
            "symbol": ["000001"] * 25,
            "open": [10.0 + index * 0.2 for index in range(25)],
            "high": [10.3 + index * 0.2 for index in range(25)],
            "low": [9.8 + index * 0.2 for index in range(25)],
            "close": [10.1 + index * 0.2 for index in range(25)],
            "volume": [1000.0 + index * 50.0 for index in range(25)],
        }
    )

    features = build_factor_frame(
        frame,
        factor_names=[
            "stochastic_k_14",
            "stochastic_d_14_3",
            "williams_r_14",
            "money_flow_index_14",
            "upper_shadow_pct",
            "lower_shadow_pct",
            "real_body_pct",
            "atr_14_pct",
            "parkinson_volatility_20",
            "garman_klass_volatility_20",
        ],
    )

    row = features.iloc[-1]
    rolling = frame.iloc[-14:]
    rolling_high = rolling["high"].max()
    rolling_low = rolling["low"].min()
    expected_k = (frame.iloc[-1]["close"] - rolling_low) / (rolling_high - rolling_low)
    stochastic_k = features["stochastic_k_14"]
    expected_d = float(stochastic_k.iloc[-3:].mean())
    expected_wr = -100.0 * (rolling_high - frame.iloc[-1]["close"]) / (rolling_high - rolling_low)
    expected_upper_shadow = (frame.iloc[-1]["high"] - max(frame.iloc[-1]["open"], frame.iloc[-1]["close"])) / frame.iloc[-1]["close"]
    expected_lower_shadow = (min(frame.iloc[-1]["open"], frame.iloc[-1]["close"]) - frame.iloc[-1]["low"]) / frame.iloc[-1]["close"]
    expected_real_body = abs(frame.iloc[-1]["close"] - frame.iloc[-1]["open"]) / frame.iloc[-1]["open"]

    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    expected_atr = true_range.iloc[-14:].mean() / frame.iloc[-1]["close"]

    log_hl = np.log(frame["high"] / frame["low"])
    expected_parkinson = float(np.sqrt((log_hl.pow(2) / (4.0 * np.log(2.0))).iloc[-20:].mean()))
    log_co = np.log(frame["close"] / frame["open"])
    gk_estimator = 0.5 * log_hl.pow(2) - ((2.0 * np.log(2.0)) - 1.0) * log_co.pow(2)
    expected_gk = float(np.sqrt(gk_estimator.clip(lower=0.0).iloc[-20:].mean()))

    assert row["stochastic_k_14"] == pytest.approx(expected_k)
    assert row["stochastic_d_14_3"] == pytest.approx(expected_d)
    assert row["williams_r_14"] == pytest.approx(expected_wr)
    assert row["money_flow_index_14"] == pytest.approx(100.0)
    assert row["upper_shadow_pct"] == pytest.approx(expected_upper_shadow)
    assert row["lower_shadow_pct"] == pytest.approx(expected_lower_shadow)
    assert row["real_body_pct"] == pytest.approx(expected_real_body)
    assert row["atr_14_pct"] == pytest.approx(expected_atr)
    assert row["parkinson_volatility_20"] == pytest.approx(expected_parkinson)
    assert row["garman_klass_volatility_20"] == pytest.approx(expected_gk)


def test_build_factor_frame_preserves_requested_factor_order_and_history_count() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
            "symbol": ["000001"] * 6,
            "open": [10.0, 10.2, 10.1, 10.4, 10.5, 10.7],
            "high": [10.2, 10.4, 10.3, 10.6, 10.7, 10.9],
            "low": [9.9, 10.1, 10.0, 10.2, 10.3, 10.5],
            "close": [10.1, 10.3, 10.2, 10.5, 10.6, 10.8],
            "volume": [1000.0, 1010.0, 990.0, 1020.0, 1030.0, 1040.0],
        }
    )

    factor_names = ["return_1", "volatility_5", "rsi_14"]
    features = build_factor_frame(frame, factor_names=factor_names)

    expected_suffix = factor_names + ["history_count"]
    assert list(features.columns[-len(expected_suffix) :]) == expected_suffix
    assert features["history_count"].tolist() == [1, 2, 3, 4, 5, 6]


def test_build_factor_frame_computes_representative_qlib_factors() -> None:
    close = [10.0 + index for index in range(70)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=70, freq="D"),
            "symbol": ["000001"] * 70,
            "open": [value - 0.2 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.6 for value in close],
            "close": close,
            "volume": [1000.0 + index * 10.0 for index in range(70)],
            "amount": [(1000.0 + index * 10.0) * 100.0 * (value + 0.1) for index, value in enumerate(close)],
        }
    )

    features = build_factor_frame(
        frame,
        factor_names=["OPEN0", "OPEN59", "VWAP0", "VOLUME1", "KMID", "MA5", "STD5", "IMAX5", "CORR5"],
    )
    row = features.iloc[-1]
    current_close = frame.iloc[-1]["close"]
    expected_open0 = frame.iloc[-1]["open"] / current_close
    expected_open59 = frame.iloc[-60]["open"] / current_close
    expected_vwap0 = (frame.iloc[-1]["amount"] / (frame.iloc[-1]["volume"] * 100.0)) / current_close
    expected_volume1 = frame.iloc[-2]["volume"] / frame.iloc[-1]["volume"]
    expected_kmid = (frame.iloc[-1]["close"] - frame.iloc[-1]["open"]) / frame.iloc[-1]["open"]
    expected_ma5 = frame["close"].iloc[-5:].mean() / current_close
    expected_std5 = frame["close"].iloc[-5:].std(ddof=0) / current_close
    expected_imax5 = 0.0
    expected_corr5 = frame["close"].iloc[-5:].corr(np.log(frame["volume"].iloc[-5:] + 1.0))

    assert row["OPEN0"] == pytest.approx(expected_open0)
    assert row["OPEN59"] == pytest.approx(expected_open59)
    assert row["VWAP0"] == pytest.approx(expected_vwap0)
    assert row["VOLUME1"] == pytest.approx(expected_volume1)
    assert row["KMID"] == pytest.approx(expected_kmid)
    assert row["MA5"] == pytest.approx(expected_ma5)
    assert row["STD5"] == pytest.approx(expected_std5)
    assert row["IMAX5"] == pytest.approx(expected_imax5)
    assert row["CORR5"] == pytest.approx(expected_corr5)


def test_estimate_factor_history_lookback_handles_qlib_factors() -> None:
    assert estimate_factor_history_lookback(["OPEN59"]) == 59
    assert estimate_factor_history_lookback(["MA60"]) == 60
    assert estimate_factor_history_lookback(["CORD30"]) == 31
