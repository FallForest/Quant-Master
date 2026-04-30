from __future__ import annotations

"""Single-stock ML inference entrypoint.

This script reuses an existing ML artifact and produces:
1. A raw model score for one stock on one trading date.
2. A business interpretation of that score based on the artifact target mode.
3. Optional same-day cross-sectional context by scoring a reference universe.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.loading import load_bars_for_symbols_batch
from data.provider_factory import build_data_provider
from ml.artifacts import load_signal_artifact
from ml.dataset import build_inference_dataset
from ml.experiments.baseline import load_official_baseline_manifest
from ml.factors import estimate_factor_history_lookback
from ml.models import score_frame
from research.profiles import load_research_profile


@dataclass(slots=True)
class RuntimeConfig:
    artifact_path: str
    market: str
    provider_name: str
    data_root: str
    universe_root: str
    reference_root: str
    adjust: str
    timeframe: str
    context_universe: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an existing ML artifact to score a single stock."
    )
    parser.add_argument("--symbol", required=True, help="Stock code, for example 000063")
    parser.add_argument("--prediction-date", help="Target trade date, defaults to the latest local bar on or before today")

    parser.add_argument("--research-profile", help="Optional research profile that supplies the baseline manifest and context universe")
    parser.add_argument("--artifact-path", help="Artifact directory containing model.pkl and metadata.json")
    parser.add_argument("--manifest-path", help="Optional baseline manifest JSON used to infer artifact path and context universe")

    parser.add_argument("--market", help="Override market from artifact metadata")
    parser.add_argument("--provider", help="Override provider from artifact metadata")
    parser.add_argument("--data-root", help="Override data root, for example data/lake")
    parser.add_argument("--universe-root", default="data/universe", help="Universe CSV root")
    parser.add_argument("--reference-root", help="Override reference data root")
    parser.add_argument("--adjust", help="Override adjust mode")
    parser.add_argument("--timeframe", help="Override timeframe")

    parser.add_argument("--context-universe", help="Universe used to provide same-day percentile context")
    parser.add_argument("--no-context", action="store_true", help="Disable same-day universe context scoring")

    parser.add_argument("--history-padding-days", type=int, help="Override automatic factor lookback padding in trading days")
    parser.add_argument("--bar-max-workers", type=int, help="Max workers used when loading context-universe bars")
    parser.add_argument("--factor-max-workers", type=int, help="Max workers used by the factor builder")
    parser.add_argument("--json", action="store_true", help="Print raw JSON only")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    profile = load_research_profile(args.research_profile) if args.research_profile else None
    manifest = _resolve_manifest(args=args, profile=profile)
    artifact_path = _resolve_artifact_path(args.artifact_path, manifest)
    estimator, metadata = load_signal_artifact(artifact_path)
    runtime = _resolve_runtime_config(args=args, metadata=metadata, manifest=manifest, artifact_path=artifact_path, profile=profile)

    feature_columns = [str(item) for item in metadata.get("feature_columns", [])]
    if not feature_columns:
        raise ValueError("Artifact metadata has no feature_columns.")

    provider = build_data_provider(
        provider_name=runtime.provider_name,
        data_root=runtime.data_root,
        universe_root=runtime.universe_root,
        adjust=runtime.adjust,
    )
    symbol = provider.normalize_symbol(args.symbol)

    requested_end = pd.Timestamp(args.prediction_date) if args.prediction_date else pd.Timestamp.now().normalize()
    history_padding_days = _resolve_history_padding_days(
        feature_columns=feature_columns,
        override_days=args.history_padding_days,
    )
    start = requested_end - pd.offsets.BDay(max(0, history_padding_days))

    single_report = _score_single_symbol(
        estimator=estimator,
        metadata=metadata,
        provider=provider,
        runtime=runtime,
        symbol=symbol,
        start=start,
        requested_end=requested_end,
        factor_max_workers=args.factor_max_workers,
    )

    context_report = None
    if runtime.context_universe:
        context_report = _score_context_universe(
            estimator=estimator,
            metadata=metadata,
            provider=provider,
            runtime=runtime,
            symbol=symbol,
            start=start,
            prediction_timestamp=single_report["prediction_timestamp"],
            universe_name=runtime.context_universe,
            bar_max_workers=args.bar_max_workers,
            factor_max_workers=args.factor_max_workers,
        )

    output = _build_output(
        symbol=symbol,
        runtime=runtime,
        metadata=metadata,
        single_report=single_report,
        context_report=context_report,
        history_padding_days=history_padding_days,
    )
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    _print_human_readable(output)


def _resolve_artifact_path(artifact_path: str | None, manifest: dict[str, object] | None) -> str:
    if artifact_path:
        return str(Path(artifact_path))
    if manifest is not None and manifest.get("artifact_path"):
        return str(Path(str(manifest["artifact_path"])))
    raise ValueError("No artifact path provided. Use --artifact-path, or provide a manifest that contains artifact_path.")


def _resolve_manifest(*, args, profile) -> dict[str, object] | None:
    if args.manifest_path:
        return load_official_baseline_manifest(args.manifest_path)
    if profile is not None and profile.official_baseline_manifest:
        return load_official_baseline_manifest(profile.official_baseline_manifest)
    if args.artifact_path:
        return None
    return load_official_baseline_manifest()


def _resolve_runtime_config(
    *,
    args,
    metadata: dict[str, object],
    manifest: dict[str, object] | None,
    artifact_path: str,
    profile,
) -> RuntimeConfig:
    manifest_fixed = dict(manifest.get("fixed_config", {})) if manifest is not None else {}
    return RuntimeConfig(
        artifact_path=str(Path(artifact_path)),
        market=str(args.market or metadata.get("market") or manifest_fixed.get("market") or (profile.market if profile is not None else "ashare")),
        provider_name=str(args.provider or metadata.get("provider") or manifest_fixed.get("provider") or (profile.provider if profile is not None else "parquet")),
        data_root=str(args.data_root or manifest_fixed.get("data_root") or (profile.data_root if profile is not None else "data/lake")),
        universe_root=str(args.universe_root or (profile.universe_root if profile is not None else "data/universe")),
        reference_root=str(args.reference_root or metadata.get("reference_root") or manifest_fixed.get("reference_root") or (profile.reference_root if profile is not None else "data/reference")),
        adjust=str(args.adjust or metadata.get("adjust") or (profile.adjust if profile is not None else "qfq")),
        timeframe=str(args.timeframe or metadata.get("timeframe") or (profile.timeframe if profile is not None else "1d")),
        context_universe=None
        if args.no_context
        else (
            str(args.context_universe)
            if args.context_universe
            else _resolve_context_universe(manifest_fixed, profile)
        ),
    )


def _resolve_context_universe(manifest_fixed: dict[str, object], profile) -> str | None:
    universe = manifest_fixed.get("universe")
    if universe in {None, ""} and profile is not None:
        universe = profile.universe
    if universe in {None, ""}:
        return None
    return str(universe)


def _resolve_history_padding_days(*, feature_columns: list[str], override_days: int | None) -> int:
    if override_days is not None:
        return int(override_days)
    estimated = int(estimate_factor_history_lookback(feature_columns))
    # 因子 registry 给的是理论最小 lookback。
    # 单票推理实际按交易日取样，节假日会吞掉一部分 business-day 预热区间，
    # 所以这里自动加一个安全缓冲，避免 20/60/120 日滚动因子在边界处变成空值。
    return max(estimated + 20, int(estimated * 2), 40)


def _score_single_symbol(
    *,
    estimator,
    metadata: dict[str, object],
    provider,
    runtime: RuntimeConfig,
    symbol: str,
    start: pd.Timestamp,
    requested_end: pd.Timestamp,
    factor_max_workers: int | None,
) -> dict[str, object]:
    bars = provider.load_bars(
        symbol=symbol,
        market=runtime.market,
        start=start.to_pydatetime(),
        end=requested_end.to_pydatetime(),
        timeframe=runtime.timeframe,
    )
    if bars.empty:
        data_path = provider.resolve_data_path(
            symbol=symbol,
            market=runtime.market,
            timeframe=runtime.timeframe,
            suffix=getattr(provider, "file_suffix", ""),
        )
        raise FileNotFoundError(
            f"No local bars found for {symbol} under provider={runtime.provider_name}. "
            f"Expected data near: {data_path}"
        )

    inference = build_inference_dataset(
        data=bars,
        feature_columns=[str(item) for item in metadata["feature_columns"]],
        reference_root=runtime.reference_root,
        market=runtime.market,
        feature_normalization=str(metadata.get("feature_normalization", "none")),
        factor_max_workers=factor_max_workers,
    )
    scored = score_frame(
        estimator=estimator,
        frame=inference,
        feature_columns=[str(item) for item in metadata["feature_columns"]],
        score_column="prediction",
    )
    scored = scored.dropna(subset=["prediction"]).reset_index(drop=True)
    if scored.empty:
        raise ValueError(
            f"{symbol} could not produce a complete feature row before {requested_end.date()}. "
            "Increase history padding or verify required columns and reference data."
        )

    candidate = scored.loc[scored["timestamp"] <= requested_end].copy()
    if candidate.empty:
        raise ValueError(f"No scored row is available for {symbol} on or before {requested_end.date()}.")
    row = candidate.sort_values("timestamp").iloc[-1]

    feature_snapshot = {
        column: _json_ready_number(row[column])
        for column in metadata["feature_columns"]
    }
    return {
        "prediction_timestamp": pd.Timestamp(row["timestamp"]),
        "prediction": float(row["prediction"]),
        "feature_snapshot": feature_snapshot,
        "raw_bar_timestamp_max": pd.to_datetime(bars["timestamp"]).max(),
        "bar_count": int(len(bars)),
    }


def _score_context_universe(
    *,
    estimator,
    metadata: dict[str, object],
    provider,
    runtime: RuntimeConfig,
    symbol: str,
    start: pd.Timestamp,
    prediction_timestamp: pd.Timestamp,
    universe_name: str,
    bar_max_workers: int | None,
    factor_max_workers: int | None,
) -> dict[str, object] | None:
    universe_symbols = provider.load_universe(
        market=runtime.market,
        universe=universe_name,
        date=prediction_timestamp.to_pydatetime(),
    )
    if not universe_symbols:
        return {
            "enabled": False,
            "reason": f"Universe {universe_name} resolved to zero symbols.",
        }

    normalized_symbols = [provider.normalize_symbol(item) for item in universe_symbols]
    symbol_in_universe = symbol in set(normalized_symbols)
    if not symbol_in_universe:
        normalized_symbols.append(symbol)
    normalized_symbols = list(dict.fromkeys(normalized_symbols))

    bars_batch = load_bars_for_symbols_batch(
        provider=provider,
        market=runtime.market,
        timeframe=runtime.timeframe,
        symbols=normalized_symbols,
        start=start.to_pydatetime(),
        end=prediction_timestamp.to_pydatetime(),
        progress_desc="Loading context-universe bars",
        show_progress=False,
        max_workers=bar_max_workers,
        empty_columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    )
    if bars_batch.frame.empty:
        return {
            "enabled": False,
            "reason": f"No bars loaded for context universe {universe_name}.",
        }

    inference = build_inference_dataset(
        data=bars_batch.frame,
        feature_columns=[str(item) for item in metadata["feature_columns"]],
        reference_root=runtime.reference_root,
        market=runtime.market,
        feature_normalization=str(metadata.get("feature_normalization", "none")),
        factor_max_workers=factor_max_workers,
    )
    scored = score_frame(
        estimator=estimator,
        frame=inference,
        feature_columns=[str(item) for item in metadata["feature_columns"]],
        score_column="prediction",
    )
    same_day = scored.loc[
        pd.to_datetime(scored["timestamp"]).eq(prediction_timestamp)
    ].dropna(subset=["prediction"]).copy()
    if same_day.empty:
        return {
            "enabled": False,
            "reason": f"No complete same-day context rows on {prediction_timestamp.date()}.",
        }

    same_day["score_rank"] = same_day["prediction"].rank(method="min", ascending=False)
    same_day["score_percentile"] = same_day["prediction"].rank(method="average", pct=True)
    target = same_day.loc[same_day["symbol"] == symbol]
    if target.empty:
        return {
            "enabled": False,
            "reason": f"{symbol} has no complete same-day feature row inside context universe {universe_name}.",
        }
    item = target.iloc[0]
    return {
        "enabled": True,
        "universe_name": universe_name,
        "symbol_in_universe": symbol_in_universe,
        "symbol_count": int(len(same_day)),
        "score_rank": int(item["score_rank"]),
        "score_percentile": float(item["score_percentile"]),
        "top_score": float(same_day["prediction"].max()),
        "median_score": float(same_day["prediction"].median()),
        "bottom_score": float(same_day["prediction"].min()),
    }


def _build_output(
    *,
    symbol: str,
    runtime: RuntimeConfig,
    metadata: dict[str, object],
    single_report: dict[str, object],
    context_report: dict[str, object] | None,
    history_padding_days: int,
) -> dict[str, object]:
    target_mode = str(metadata.get("target_mode", "future_return"))
    label_horizon = int(metadata.get("label_horizon", 0) or 0)
    raw_prediction = float(single_report["prediction"])
    explanation = _prediction_explanation(
        target_mode=target_mode,
        label_horizon=label_horizon,
        raw_prediction=raw_prediction,
        context_report=context_report,
    )
    return {
        "symbol": symbol,
        "prediction_timestamp": str(pd.Timestamp(single_report["prediction_timestamp"]).date()),
        "artifact": {
            "path": runtime.artifact_path,
            "model_name": metadata.get("model_name"),
            "model_params": metadata.get("model_params"),
            "target_mode": target_mode,
            "label_horizon": label_horizon,
            "label_column": metadata.get("label_column"),
            "feature_normalization": metadata.get("feature_normalization"),
            "feature_count": len(metadata.get("feature_columns", [])),
        },
        "runtime": {
            "market": runtime.market,
            "provider": runtime.provider_name,
            "data_root": runtime.data_root,
            "reference_root": runtime.reference_root,
            "adjust": runtime.adjust,
            "timeframe": runtime.timeframe,
            "history_padding_days": history_padding_days,
            "bar_count_loaded": int(single_report["bar_count"]),
        },
        "prediction": {
            "raw_score": raw_prediction,
            "predicted_future_return": raw_prediction if target_mode == "future_return" else None,
            "predicted_future_return_pct": raw_prediction * 100.0 if target_mode == "future_return" else None,
        },
        "context": context_report,
        "interpretation": explanation,
        "feature_snapshot": single_report["feature_snapshot"],
    }


def _prediction_explanation(
    *,
    target_mode: str,
    label_horizon: int,
    raw_prediction: float,
    context_report: dict[str, object] | None,
) -> dict[str, object]:
    if target_mode == "future_return":
        return {
            "score_meaning": f"模型把这个分数当作未来 {label_horizon} 个交易日收益率的点预测。",
            "higher_is_better": True,
            "direct_return_forecast": True,
            "plain_text": f"预测未来 {label_horizon} 个交易日收益约为 {raw_prediction * 100.0:.2f}%。",
        }

    rank_note = "这是横截面相对强弱分数，不是绝对收益率预测。"
    if context_report and context_report.get("enabled"):
        percentile = float(context_report["score_percentile"]) * 100.0
        rank_note = (
            f"这是横截面相对强弱分数，不是绝对收益率预测。"
            f"按同日股票池打分，它大约位于 {percentile:.1f}% 分位。"
        )
    return {
        "score_meaning": f"模型学习的是未来 {label_horizon} 个交易日的横截面相对强弱。",
        "higher_is_better": True,
        "direct_return_forecast": False,
        "plain_text": rank_note,
    }


def _print_human_readable(payload: dict[str, object]) -> None:
    artifact = dict(payload["artifact"])
    runtime = dict(payload["runtime"])
    prediction = dict(payload["prediction"])
    context = payload.get("context")
    interpretation = dict(payload["interpretation"])

    print(f"symbol: {payload['symbol']}")
    print(f"prediction_date: {payload['prediction_timestamp']}")
    print(
        "artifact: "
        f"{artifact['model_name']} | target_mode={artifact['target_mode']} | "
        f"label_horizon={artifact['label_horizon']}d | features={artifact['feature_count']}"
    )
    print(
        "runtime: "
        f"provider={runtime['provider']} | data_root={runtime['data_root']} | "
        f"history_padding_days={runtime['history_padding_days']} | bars_loaded={runtime['bar_count_loaded']}"
    )
    print(f"raw_score: {prediction['raw_score']:.6f}")
    if prediction["predicted_future_return"] is not None:
        print(f"predicted_future_return_{artifact['label_horizon']}d: {prediction['predicted_future_return_pct']:.2f}%")
    if isinstance(context, dict) and context.get("enabled"):
        print(
            "context: "
            f"{context['universe_name']} | rank={context['score_rank']}/{context['symbol_count']} | "
            f"percentile={float(context['score_percentile']) * 100.0:.2f}% | "
            f"symbol_in_universe={context['symbol_in_universe']}"
        )
    elif isinstance(context, dict):
        print(f"context: disabled | reason={context.get('reason')}")
    print(f"interpretation: {interpretation['plain_text']}")
    print("feature_snapshot:")
    for key, value in dict(payload["feature_snapshot"]).items():
        print(f"  {key}: {value}")


def _json_ready_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


if __name__ == "__main__":
    main()
