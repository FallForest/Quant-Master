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
- experiment-group resource-aware scheduling for mixed CPU/GPU workloads
- leak-aware validation with purge / embargo controls
- two-stage model selection: tuning metric first, signal-window rerank second
- technical, benchmark-relative, liquidity, valuation, quality, and investment factors
- local data providers: `csv`, `parquet`, `duckdb`
- signal experiment report export with validation and final signal-test metrics

## Quick Start

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

List available ML features:

```bash
py train_ml_signal.py --list-features
```

Run one reproducible experiment:

```bash
py run_ml_experiment.py --experiment configs/experiments/hs300_xgb_core_v1.yaml
```

Each trained artifact stores signal validation metrics in `metadata.json`, including `oos_pearson_ic`, `oos_spearman_ic`, `oos_ic_std`, `oos_ic_ir`, and `oos_ndcg_at_10`.

Run the current core model group and compare signal results:

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --parallel --cpu-workers 1 --gpu-workers 1
py compare_ml_experiments.py --group hs300_core_models
```

Backfill validation OOS signal metrics into existing experiment artifacts and summaries without retraining:

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --backfill-signal-metrics
```

Sync A-share daily data:

```bash
py sync_ashare_data.py --universe hs300 --start 20210101 --end 20260424 --adjust qfq
```

Sync auxiliary reference data:

```bash
py sync_ashare_reference_data.py --universe hs300 --start 20210101 --end 20260424
```

Build parquet data:

```bash
py build_parquet_dataset.py --market ashare --input-root data/raw --output-root data/lake --universe hs300 --adjust qfq
```

## Docs

- `run_ml_experiment.py` evaluates pure signal quality plus a final `signal_test` window.
- [Developer Entry](docs/developer-entry.md)
- [ML Architecture Refactor Plan](docs/ml-architecture-refactor.md)
