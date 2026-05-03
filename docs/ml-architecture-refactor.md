# ML 架构重构方案

这份文档记录本次 ML 主链路重构的设计结论和当前落地结果。

口径说明，截止 2026-04-25：

- 旧的通用回测链路已经从代码库中移除。
- `selector`、`timing`、组合策略注册表、通用回测 runner，以及相关 broker / portfolio / risk / report / market rule 代码已经删除。
- 当前仓库只保留 ML 信号研究主链路和数据准备能力。

## 1. 为什么要删掉旧链路

旧结构的问题不是参数不对，而是层级混淆：

- 模型分数被过早截断成 `top_n` / `min_score`
- `selector` 和 `timing` 共同决定最终信号
- 回测结果被错误地拿来判断信号本身好坏

这样会导致三个问题：

1. 无法单独判断 artifact 的预测能力。
2. 无法区分是模型差，还是组合映射差。
3. 无法把执行影响和信号影响拆开看。

## 2. 现在保留的主链路

当前代码库只保留这条路径：

`data -> factors -> train -> validation -> candidate selection -> signal_test -> compare/backfill`

对应能力是：

- 数据同步和 parquet 构建
- 因子构造
- 模型训练
- 泄漏控制验证
- 调参与候选信号重排
- 最终纯信号测试
- 实验组比较
- 已有 artifact 指标回填

## 3. 当前架构分层

### 3.1 数据层

职责：

- 同步原始行情和参考数据
- 管理 `csv / parquet / duckdb` provider
- 提供统一的数据读取入口

核心代码：

- `src/app/sync_ashare_data.py`
- `src/app/sync_ashare_reference_data.py`
- `src/app/build_parquet_dataset.py`
- `src/data/provider_factory.py`
- `src/data/loading.py`

### 3.2 信号研究层

职责：

- 构造训练集和推理集
- 训练 artifact
- 计算样本外信号指标
- 做 walk-forward / holdout 验证
- 做 tuning 和 candidate selection

核心代码：

- `src/ml/dataset.py`
- `src/ml/training.py`
- `src/ml/models.py`
- `src/ml/validation.py`
- `src/ml/tuning.py`
- `src/ml/selection.py`

### 3.3 实验编排层

职责：

- 加载实验 spec
- 运行单实验和实验组
- 管理 signal test
- 汇总 summary
- 输出 compare 表

核心代码：

- `src/ml/experiments/specs.py`
- `src/ml/experiments/loader.py`
- `src/ml/experiments/runner.py`
- `src/ml/experiments/compare.py`
- `src/ml/experiments/scheduler.py`

### 3.4 维护工具层

职责：

- 回填已有 artifact 的新指标
- 维护历史实验 summary

核心代码：

- `src/ml/backfill.py`

## 4. 删除了什么

这次重构已经删除：

- `src/strategies/*`
- `src/app/runner.py`
- `src/config/*`
- `src/core/*`
- `src/markets/*`
- `src/risk/*`
- `src/report/*`
- `src/data/feature_pipeline.py`
- `src/ml/inference.py`

这意味着仓库里已经不存在：

- 策略注册表
- selector / timing 组合器
- 通用回测配置对象
- 通用回测引擎
- 交易执行模拟
- 风控规则链
- 通用回测报表导出

## 5. 当前实验 spec

当前实验 YAML 只保留这些核心块：

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
  - 训练窗口
  - 验证模式
  - `label_horizon`
  - `target_mode`
  - `purge_size`
  - `embargo_size`
  - `walk_forward`
  - `tuning`
  - `candidate_selection`
- `signal_test`
  - 最终样本外纯信号窗口
- `report`
  - 输出目录和 artifact 路径

## 6. 当前评估原则

模型先作为纯 scorer 被评估，不再通过交易回测间接评估。

优先关注：

- `rank_ic`
- `ic`
- `oos_ic_std`
- `oos_ic_ir`
- `oos_ndcg_at_10`
- `signal_test_metrics`

## 7. 后续边界

当前仓库已经完成“去掉通用回测链路”的清理，但还保留了一些历史命名：

- `train_ml_signal.py`
- `train_ml_signal_model()`
- `load_signal_artifact()`
- `save_signal_artifact()`

## 8. 一句话结论

Quant Master 现在已经是一个纯 ML 信号研究代码库，不再包含旧的 selector/timing 通用回测链路。

## 9. 2026-04-27 并发与治理落地补充

本轮又补了三件之前缺口比较明显的事：

- `prepare_signal_dataset()` 不再只有行情加载并发。
  现在会把 `bar load -> factor build -> normalization/label -> dropna/trim` 分阶段计时，并把实际 workers 写进 `PreparedSignalDataset.diagnostics` 与 artifact metadata。
- `build_factor_frame()` 现在支持按 `symbol` 并发构造因子。
  新增因子族扩容后，CPU-heavy 的特征构建不再完全串行。
- `sync_ashare_reference_data()` 现在除了 symbol 级并发，还支持单个 symbol 内部的 `fundamentals / industry / dividends` 子任务并发。
  CLI 新增 `--bundle-workers`。

同时，`ml.factors.auxiliary` 里原先三份几乎相同的 symbol 级 reference merge 编排已经收敛成一套通用映射框架，避免继续保留重复实现。
