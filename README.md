# Quant Master

Quant Master is an A-share ML research project focused on signal training, validation, comparison, and data preparation.

The old generic backtest chain built around `selector + timing` has been removed from the codebase. The active project path is now:

- data sync and dataset preparation
- factor-driven ML training
- leak-aware validation
- signal-window candidate reranking
- final pure signal test
- experiment-group comparison and backfill

## What It Supports

- ML signal artifact training and evaluation
- OOS signal metrics for artifacts: `pearson_ic`, `spearman_ic`, `ic_std`, `ic_ir`, `ndcg_at_10`
- research-profile-driven chaining for `HS300`, `ZZ500`, and `ZZ1000`
- per-profile official baseline manifest plus promotion-gate comparison against that profile baseline
- experiment-group resource-aware scheduling for mixed CPU/GPU workloads
- leak-aware validation with purge / embargo controls
- two-stage model selection: tuning metric first, signal-window rerank second
- in-memory dataset reuse across training, candidate rerank, and final signal test
- resource-aware parallel candidate rerank without temporary candidate artifacts
- scoped reference sync with unified `--scope`, `--overwrite`, and partial-success handling
- long-running data stages now persist per-run manifest / summary / failure files for precise补跑
- remote sync subtasks now use bounded per-call timeouts instead of waiting indefinitely
- single-pool task-level reference sync concurrency across `fundamentals / industry / dividends`
- symbol-level factor construction concurrency plus per-stage data-preparation diagnostics
- batched factor-column assembly to avoid pandas fragmentation on large factor libraries
- resumable experiment-group execution with per-task manifest and failure list
- literature-backed canonical factors across price momentum, industry momentum, reversal, stochastic and money-flow oscillators, candlestick structure, liquidity, valuation, quality, investment, and OHLC volatility-estimator families
- local data providers: `csv`, `parquet`, `duckdb`
- signal experiment report export with validation, final signal-test metrics, IC decay, and slice diagnostics

## Quick Start

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

Research profiles live under `configs/research_profiles/ashare/` and carry the reusable pool-level defaults:

- `universe`
- `benchmark_symbol`
- `official_baseline_manifest`
- `provider / data_root / reference_root / timeframe / adjust`

List available ML features:

```bash
py train_ml_signal.py --list-features
```

Run one reproducible experiment:

```bash
py run_ml_experiment.py --experiment configs/experiments/hs300_xgb_core_v1.yaml
```

Each trained artifact stores signal validation metrics in `metadata.json`, including `oos_pearson_ic`, `oos_spearman_ic`, `oos_ic_std`, `oos_ic_ir`, and `oos_ndcg_at_10`.

The official HS300 baseline is fixed at:

- config: `configs/experiments/official/hs300_official_baseline.yaml`
- manifest: `configs/experiments/official/hs300_official_baseline_manifest.json`

Any challenger experiment must beat all four gate metrics from that manifest at the same time:

- `full_oos_spearman_ic`
- `window_mean_spearman_ic`
- `window_min_spearman_ic`
- `window_mean_ic_ir`

The same pipeline can now be switched to another pool by changing the research profile instead of rewriting the code path. Baseline skeleton configs are ready for:

- `configs/experiments/official/zz500_official_baseline.yaml`
- `configs/experiments/official/zz1000_official_baseline.yaml`

Run the current core model group and compare signal results:

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --parallel --cpu-workers 1 --gpu-workers 1
py compare_ml_experiments.py --group hs300_core_models
```

Resume an interrupted group run from the existing `group_summary.json`:

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --parallel --cpu-workers 1 --gpu-workers 1 --resume --continue-on-error
```

Backfill validation OOS signal metrics into existing experiment artifacts and summaries without retraining:

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --backfill-signal-metrics
```

Resume an interrupted backfill run:

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --backfill-signal-metrics --resume
```

Sync A-share daily data:

```bash
py sync_ashare_data.py --research-profile hs300 --start 20210101 --end 20260424
```

Daily sync now defaults to incremental update:

- if a symbol file is already fresh, it is skipped
- if the local file is stale, the sync resumes from the last local bar with a small overlap window and merges/deduplicates the result
- use `--incremental-lookback-days` to widen that overlap window when you want to re-pull more recent history
- use `--overwrite` when you explicitly want a full per-symbol refresh

Sync auxiliary reference data:

```bash
py sync_ashare_reference_data.py --research-profile hs300 --start 20210101 --end 20260424
```

This reference sync now carries the balance-sheet and dividend fields used by the canonical value, quality, and investment factors, including leverage, cash, inventory, receivables, capex growth, and trailing dividend yield proxies.

The technical factor library now also includes standard proxies such as `momentum_63_21`, `stochastic_k_14`, `stochastic_d_14_3`, `williams_r_14`, `money_flow_index_14`, `atr_14_pct`, `parkinson_volatility_20`, `garman_klass_volatility_20`, `upper_shadow_pct`, and `lower_shadow_pct`.

Sync only one reference scope when you only need to repair that layer:

```bash
py sync_ashare_reference_data.py --research-profile hs300 --start 20210101 --end 20260424 --scope industry-only --max-workers 8
```

If one symbol still needs to pull fundamentals, industry, and dividends together, use `--bundle-workers` to raise the single task-pool concurrency multiplier:

```bash
py sync_ashare_reference_data.py --research-profile hs300 --start 20210101 --end 20260424 --max-workers 8 --bundle-workers 3
```

Build parquet data:

```bash
py build_parquet_dataset.py --research-profile hs300 --input-root data/raw --output-root data/lake
```

Data sync, reference sync, and parquet build now default to partial-success mode:

- successful outputs are kept even if a small number of symbols fail
- each run writes `*_summary.json`, `*_manifest.json`, and `*_failures.json` beside the pipeline output
- reference sync now records task-level progress and failures such as `fundamentals:000001` instead of only symbol-level aggregates
- use `--fail-on-error` to restore strict all-green exit behavior
- use `--subtask-timeout-seconds` on sync commands to bound each remote request

Compare a group against the baseline of one research profile explicitly:

```bash
py compare_ml_experiments.py --group hs300_core_models --research-profile hs300
```

Score one stock against the current profile baseline artifact:

```bash
py predict_single_stock.py --research-profile hs300 --symbol 000063
```

All long-running group runs now persist:

- `group_summary.json`
- `group_run_manifest.json`
- `group_run_failures.json`
- `group_backfill_manifest.json`
- `group_backfill_failures.json`

## Docs

- `run_ml_experiment.py` evaluates pure signal quality plus a final `signal_test` window.
- `run_ml_experiment.py` now also writes `signal_test_ic_decay` and `signal_test_slice_diagnostics`.
- `prepare_signal_dataset()` now records bar-loading workers, factor-building workers, and stage timings in artifact metadata under `data_preparation`.
- [Developer Entry](docs/developer-entry.md)
- [Long-Running Pipeline Governance](docs/long-running-pipeline-governance.md)
- [ML Architecture Refactor Plan](docs/ml-architecture-refactor.md)
