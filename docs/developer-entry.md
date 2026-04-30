# Quant Master 开发者入口

这份文档面向新接手项目的开发者，目标是快速说明当前项目保留了什么、主链路怎么跑、核心代码在哪。

## 1. 当前项目定位

Quant Master 当前是一个以 A 股 ML 信号研究为核心的项目，不再保留旧的通用回测链路。

已经移除的旧结构：

- `selector`
- `timing`
- `selector + timing` 组合策略
- 通用回测 runner
- 通用回测相关 broker / portfolio / report / risk / market rule 代码

当前主链路只围绕以下问题展开：

- 数据准备
- 因子构造
- 模型训练
- 泄漏控制验证
- 信号层候选筛选
- 最终 signal test
- 实验组对比与指标回填

当前因子库已经覆盖：

- 趋势 / 动量 / 反转
- 价格位置 / 区间
- 震荡指标
- K 线形态
- 波动率与 OHLC 波动率估计
- 成交量 / 流动性
- 估值 / 质量 / 投资

## 2. 根目录入口

当前根目录里保留的入口脚本，按用途分组如下。

ML 主链路：

- `train_ml_signal.py`
- `run_ml_experiment.py`
- `compare_ml_experiments.py`

数据准备：

- `sync_ashare_data.py`
- `sync_ashare_reference_data.py`
- `sync_index_universe.py`
- `build_parquet_dataset.py`

项目说明：

- `README.md`
- `requirements.txt`

## 2.1 研究 Profile

项目现在新增了一层 research profile，用来承载股票池级别的默认参数，而不是继续把 `hs300_*` 口径硬编码到每条命令里。

当前 profile 配置目录：

- `configs/research_profiles/ashare/hs300.yaml`
- `configs/research_profiles/ashare/zz500.yaml`
- `configs/research_profiles/ashare/zz1000.yaml`

每个 profile 统一定义：

- `universe`
- `benchmark_symbol`
- `official_baseline_manifest`
- `provider`
- `data_root`
- `reference_root`
- `timeframe`
- `adjust`

训练核心链路还是同一套，差异只在 profile 和实验 YAML。

## 3. 最常用命令

列出可用特征：

```bash
py train_ml_signal.py --list-features
```

运行单个实验：

```bash
py run_ml_experiment.py --experiment configs/experiments/hs300_xgb_core_v1.yaml
```

运行官方 baseline：

```bash
py run_ml_experiment.py --experiment configs/experiments/official/hs300_official_baseline.yaml
```

运行实验组并比较：

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --parallel --cpu-workers 1 --gpu-workers 1
py compare_ml_experiments.py --group hs300_core_models
```

恢复一个被中断的实验组：

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --parallel --cpu-workers 1 --gpu-workers 1 --resume --continue-on-error
```

回填已有 artifact 的信号指标：

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --backfill-signal-metrics
```

恢复一个被中断的 backfill：

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --backfill-signal-metrics --resume
```

同步和构建数据：

```bash
py sync_ashare_data.py --research-profile hs300 --start 20210101 --end 20260424
py sync_ashare_reference_data.py --research-profile hs300 --start 20210101 --end 20260424
py sync_ashare_reference_data.py --research-profile hs300 --start 20210101 --end 20260424 --scope industry-only --max-workers 8
py build_parquet_dataset.py --research-profile hs300 --input-root data/raw --output-root data/lake
```

数据链路当前默认治理行为：

- `sync_ashare_data.py`、`sync_ashare_reference_data.py`、`build_parquet_dataset.py` 默认保留部分成功结果，不再因为少量 symbol 失败让整轮进度作废
- `sync_ashare_data.py` 默认做增量补尾：已新鲜文件直接跳过，未新鲜文件从本地最后一根 bar 向前带少量 overlap 续拉，再本地合并去重
- `sync_ashare_data.py` 可用 `--incremental-lookback-days` 调大 overlap 窗口；只有显式传 `--overwrite` 才会整票重刷
- 如需严格模式，显式传 `--fail-on-error`
- 两个 sync 入口支持 `--subtask-timeout-seconds`，为单个远程子任务设置超时上限
- `sync_ashare_reference_data.py` 现在用单层 task 池调度 `fundamentals / industry / dividends`，避免 symbol 外层并发叠加 bundle 内层并发
- `--bundle-workers` 现在作为 task 池并发倍率，而不是再开一层嵌套线程池
- 三条链路都会在输出目录旁写：
  - `*_summary.json`
  - `*_manifest.json`
  - `*_failures.json`
- reference sync 的 manifest / failures 已细化到 task 级，例如 `dividends:000001`

显式指定 profile baseline 做比较：

```bash
py compare_ml_experiments.py --group hs300_core_models --research-profile hs300
```

对单只股票复用当前 profile 的 baseline artifact 打分：

```bash
py predict_single_stock.py --research-profile hs300 --symbol 000063
```

## 4. 当前目录地图

核心目录：

- `src/app`
  - 数据同步和 parquet 构建入口逻辑
- `src/data`
  - 数据 provider、加载逻辑、provider factory
- `src/ml`
  - 因子、数据集构造、训练、验证、调参、实验编排、对比、回填
- `configs/experiments`
  - 可复现实验 YAML
- `data/raw`
  - 原始 CSV 行情
- `data/lake`
  - parquet 数据湖
- `data/universe`
  - 股票池
- `data/reference`
  - 基准、基本面、行业分类等参考数据
- `artifacts`
  - 模型 artifact
- `reports`
  - 实验 summary 和比较输出
- `tests`
  - 自动化测试

## 5. ML 实验主链路

当前 `run_ml_experiment.py` 的流程是：

1. 读取实验 YAML，生成 `ExperimentSpec`
2. 调用 `ml.training.train_ml_signal_model()` 训练 artifact
3. 如果配置了 tuning，则保留 top trials
4. 在独立的 signal window 上做 candidate selection
5. 用选中的参数重训最终 artifact
6. 在 `signal_test` 窗口上做最终纯信号评估
7. 计算 `signal_test_ic_decay`
8. 计算 `signal_test_slice_diagnostics`
9. 写出 `experiment_summary.json`

当前技术因子库除了基础的 `return / sma / rsi / bollinger` 之外，还已经补充了多组标准代理，例如：

- `momentum_63_21`
- `industry_momentum_63_21`
- `stochastic_k_14`
- `stochastic_d_14_3`
- `williams_r_14`
- `money_flow_index_14`
- `atr_14_pct`
- `parkinson_volatility_20`
- `garman_klass_volatility_20`
- `upper_shadow_pct`
- `lower_shadow_pct`
- `real_body_pct`

这个流程不再经过任何 `selector / timing / backtest` 组合逻辑。
当前实现还额外做了两件吞吐优化：

- 同一实验内会复用训练窗口、候选训练窗口、候选评估窗口、`signal_test` 窗口的已准备数据集，避免重复读 bars 和重复算特征
- candidate selection 不再为每个候选落临时 artifact，而是直接基于缓存后的数据帧做资源感知并发评估

对 `--group` 批量实验，当前还会额外写三份可恢复执行文件：

- `group_summary.json`
- `group_run_manifest.json`
- `group_run_failures.json`

对 `--backfill-signal-metrics --group`，会写独立的 backfill manifest：

- `group_backfill_manifest.json`
- `group_backfill_failures.json`

恢复执行时，已完成实验会直接复用，不会重复训练。

## 6. 当前实验配置结构

当前实验 YAML 以这几个块为主：

- 顶层通用字段
  - `name`
  - `market`
  - `provider`
  - `data_root`
  - `reference_root`
  - `symbols` 或 `universe`
  - `features`
  - `model`
  - `model_params`
- `train`
  - 训练区间
  - 验证模式
  - `label_horizon`
  - `target_mode`
  - `purge_size`
  - `embargo_size`
  - `walk_forward`
  - `tuning`
  - `candidate_selection`
- `signal_test`
  - 最终样本外纯信号测试窗口
- `signal_windows`
  - 用于稳定性比较的互斥子窗口
- `report`
  - 输出目录和 artifact 路径

## 7. 官方 Baseline

官方 baseline 已经固定为：

- config: `configs/experiments/official/hs300_official_baseline.yaml`
- manifest: `configs/experiments/official/hs300_official_baseline_manifest.json`

固定配置包括：

- model: `ridge`
- `alpha = 1.0`
- `feature_normalization = none`
- label: `future_return_rank_5d`
- target mode: `cross_sectional_rank`
- signal test: `2025-01-01` 到 `2026-04-24`
- signal windows:
  - `oos_2025_h1`
  - `oos_2025_h2`
  - `oos_2026_ytd`

晋级规则已经固定：

- 新实验必须同时超过 `full_oos_spearman_ic`
- 新实验必须同时超过 `window_mean_spearman_ic`
- 新实验必须同时超过 `window_min_spearman_ic`
- 新实验必须同时超过 `window_mean_ic_ir`

比较输出会自动读取 manifest，并给出：

- `beats_official_baseline`
- `baseline_gate_failures`

## 8. 重点看哪些指标

看 artifact 本身好不好，优先看：

- `oos_spearman_ic`
- `oos_pearson_ic`
- `oos_ic_std`
- `oos_ic_ir`
- `oos_ndcg_at_10`

看最终样本外信号窗口，优先看：

- `signal_test_metrics`
- `full_oos_spearman_ic`
- `window_mean_spearman_ic`
- `window_min_spearman_ic`
- `window_mean_ic_ir`
- `test_ndcg_at_10`

看诊断切片，优先看：

- `signal_test_ic_decay`
- `signal_test_slice_diagnostics.year_windows`
- `signal_test_slice_diagnostics.market_style_regimes`
- `signal_test_slice_diagnostics.industry_buckets`
- `signal_test_slice_diagnostics.market_cap_buckets`

## 9. 关键代码位置

训练与验证：

- `src/ml/training.py`
- `src/ml/validation.py`
- `src/ml/models.py`

实验编排：

- `src/ml/experiments/specs.py`
- `src/ml/experiments/loader.py`
- `src/ml/experiments/runner.py`
- `src/ml/experiments/compare.py`
- `src/ml/experiments/baseline.py`
- `src/ml/experiments/scheduler.py`
- `src/ml/prepared_data.py`
- `src/ml/diagnostics.py`

数据与因子：

- `src/data/provider_factory.py`
- `src/data/loading.py`
- `src/ml/dataset.py`
- `src/ml/factors`

回填：

- `src/ml/backfill.py`

## 10. 推荐阅读顺序

建议按这个顺序看代码：

- 先看 `README.md`
- 再看本文档
- 再看 `docs/long-running-pipeline-governance.md`
- 再看 `docs/ml-architecture-refactor.md`
- 然后按实验主链路阅读 `src/ml`

## 11. 长链路治理

这个项目里凡是会跑很久、会并发、会访问外部数据源、会批量处理 symbol 或 experiment 的任务，都按统一治理规则约束。

优先阅读：

- [通用长链路治理清单](long-running-pipeline-governance.md)

包括但不限于这些链路：

- `sync_ashare_data.py`
- `sync_ashare_reference_data.py`
- `build_parquet_dataset.py`
- `run_ml_experiment.py`
- `backfill`

后续如果这类链路出现“慢、卡、失败、需要重跑”的问题，不要直接临时 patch，先回到治理清单里按分类、可观测性、scope、重试、并发层级逐项定位。

## 11. 一句话结论

当前仓库已经不再是“通用回测 + selector/timing 策略”项目，而是一个聚焦在 ML 信号研究、验证、比较和数据准备的研究型代码库，并且已经固化了一条官方 HS300 baseline 作为后续所有实验的统一参照线。
