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

## 3. 最常用命令

列出可用特征：

```bash
py train_ml_signal.py --list-features
```

运行单个实验：

```bash
py run_ml_experiment.py --experiment configs/experiments/hs300_xgb_core_v1.yaml
```

运行实验组并比较：

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --parallel --cpu-workers 1 --gpu-workers 1
py compare_ml_experiments.py --group hs300_core_models
```

回填已有 artifact 的信号指标：

```bash
py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --backfill-signal-metrics
```

同步和构建数据：

```bash
py sync_ashare_data.py --universe hs300 --start 20210101 --end 20260424 --adjust qfq
py sync_ashare_reference_data.py --universe hs300 --start 20210101 --end 20260424
py build_parquet_dataset.py --market ashare --input-root data/raw --output-root data/lake --universe hs300 --adjust qfq
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
  - 基准和基本面参考数据
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
7. 写出 `experiment_summary.json`

这个流程不再经过任何 `selector / timing / backtest` 组合逻辑。

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
- `report`
  - 输出目录和 artifact 路径

## 7. 重点看哪些指标

看 artifact 本身好不好，优先看：

- `oos_spearman_ic`
- `oos_pearson_ic`
- `oos_ic_std`
- `oos_ic_ir`
- `oos_ndcg_at_10`

看最终样本外信号窗口，优先看：

- `signal_test_metrics`
- `test_spearman_ic`
- `test_ic_ir`
- `test_ndcg_at_10`

## 8. 关键代码位置

训练与验证：

- `src/ml/training.py`
- `src/ml/validation.py`
- `src/ml/models.py`

实验编排：

- `src/ml/experiments/specs.py`
- `src/ml/experiments/loader.py`
- `src/ml/experiments/runner.py`
- `src/ml/experiments/compare.py`
- `src/ml/experiments/scheduler.py`

数据与因子：

- `src/data/provider_factory.py`
- `src/data/loading.py`
- `src/ml/dataset.py`
- `src/ml/factors`

回填：

- `src/ml/backfill.py`

## 9. 推荐阅读顺序

建议按这个顺序看代码：

1. `README.md`
2. 本文档
3. `configs/experiments/hs300_xgb_core_v1.yaml`
4. `run_ml_experiment.py`
5. `src/ml/experiments/runner.py`
6. `src/ml/training.py`
7. `src/ml/models.py`
8. `src/ml/validation.py`
9. `src/ml/factors`

## 10. 一句话结论

当前仓库已经不再是“通用回测 + selector/timing 策略”项目，而是一个聚焦在 ML 信号研究、验证、比较和数据准备的研究型代码库。
