# Quant Master 通用长链路治理清单

这份清单用于约束项目里的长链路任务，避免每次遇到“很慢、卡住、失败、重跑成本高”时临时打补丁。

适用对象：

- `sync_ashare_data.py`
- `sync_ashare_reference_data.py`
- `build_parquet_dataset.py`
- `run_ml_experiment.py`
- `backfill`
- 任何未来新增的批处理、数据同步、实验编排、批量评估链路

## 1. 先判断是哪类问题

遇到长链路异常时，先分类，不要直接加线程。

- `CPU bound`
  - 现象：CPU 长时间高占用，线程跑满
  - 常见原因：特征计算、模型训练、序列化、压缩
- `I/O bound`
  - 现象：CPU 很低，但总时长很长
  - 常见原因：远程接口慢、磁盘读写慢、网络抖动
- `serialization bottleneck`
  - 现象：外层并发了，但关键步骤仍然串行
  - 常见原因：单个 symbol 内部多个子任务串行执行
- `scope too large`
  - 现象：明明只缺一小部分数据，却全量重跑
  - 常见原因：链路粒度太粗，缺少 `*-only` 和 `skip-existing`
- `retry storm`
  - 现象：失败后大量重试，耗时膨胀
  - 常见原因：没有失败分流，没有失败名单
- `tail latency`
  - 现象：大多数任务很快，少数极慢任务拖垮总时长
  - 常见原因：极慢 symbol、极慢接口、极慢窗口
- `memory / dataframe layout bottleneck`
  - 现象：CPU 不高，但 pandas 处理很慢，并伴随大量复制、拼接、警告
  - 常见原因：逐列插入、反复 `concat`、DataFrame 碎片化
- `nested oversubscription`
  - 现象：开了很多并发，但整体提速很差，CPU 也未必打满
  - 常见原因：外层任务并发和内层线程并发叠加，线程数远超机器有效算力

## 2. 所有长链路必须具备的能力

任何新长链路上线前，至少满足下面这些项。

### 2.1 可重复执行

- `idempotent`
  - 同一输入重复执行，不会产生脏数据或重复副作用
- `deterministic enough`
  - 相同输入和固定版本下，结果应稳定可复现
- `safe overwrite`
  - 显式 `overwrite` 时才允许覆盖已有结果

### 2.2 可增量执行

- 支持 `skip-existing`
- 支持只跑缺失部分
- 支持按 `symbols / universe / date range / group` 缩小范围
- 支持把“全量刷新”和“日常增量”分开

### 2.3 可中断恢复

- 中途被中断后可以继续跑
- 已成功部分可以复用
- 失败项可以单独补跑

### 2.4 允许部分成功

- 允许 `allow_partial`
- 输出成功数、失败数、跳过数
- 输出失败项列表，供二次补跑
- 不允许因为少量失败而让全部进度作废

## 3. 设计原则

### 3.1 先缩 scope，再谈提速

默认顺序：

1. 减少要做的工作量
2. 复用已有结果
3. 做增量
4. 再做并发
5. 最后才是加机器和加线程

### 3.2 任务边界必须清晰

长链路不要把很多独立目标绑在一个入口里。

理想形态：

- `benchmark-only`
- `fundamentals-only`
- `industry-only`
- `dividends-only`
- `backfill-only`
- `diagnostics-only`

原则：

- 缺什么补什么
- 改了哪一层就只重跑哪一层
- 默认不要全量

### 3.3 昂贵步骤必须可缓存

对于高成本步骤：

- 数据下载结果要落盘
- 中间标准化结果要可复用
- 数据集准备结果要可复用
- 候选评估结果要避免重复构造

### 3.4 版本和 schema 要显式管理

当输出结构变化时：

- 要有 schema version 或 manifest
- 能识别“旧文件存在但结果已过期”
- 明确是增量更新还是强制重构

## 4. 可观测性要求

没有可观测性，不允许讨论性能优化结论。

每条长链路至少记录：

- 总耗时
- 每阶段耗时
- 成功 / 失败 / 跳过计数
- 重试次数
- 当前并发数
- CPU 使用率快照
- 待处理数量

最好再记录：

- 最慢 `top N` 任务
- 最慢 `top N` symbol
- 最慢 `top N` 外部接口
- 平均耗时 / P95 / P99
- 单任务输入规模

如果是外部数据源任务，还要记录：

- 接口名
- 单接口耗时
- 单接口失败率
- 超时次数

## 5. 并发治理

### 5.1 不要只问“有没有并发”

要明确并发层级：

- `job-level parallelism`
  - 例如多个 symbol 并发
- `subtask-level parallelism`
  - 例如单个 symbol 内部多个接口并发
- `pipeline-level parallelism`
  - 例如下载、标准化、写盘流水线化

很多“开了并发还是慢”的问题，本质上只是做了第一层。

### 5.2 并发要和瓶颈匹配

- `CPU bound`
  - 关注核数、线程数、进程数、BLAS 线程
- `I/O bound`
  - 关注远程限速、连接数、超时、重试、批量粒度

不要因为 CPU 低就得出“没并发”的结论。I/O 任务在等待远程返回时，CPU 本来就可能很低。

### 5.3 并发必须可控

所有并发链路都应支持：

- `max-workers`
- 自动并发上限
- 显式关闭并发
- 日志中输出实际 workers

### 5.4 防止过度并发

要防止：

- 打爆外部接口
- 本地文件句柄耗尽
- 因重试放大请求洪峰
- 小任务反而被调度开销拖慢

### 5.5 防止嵌套并发失控

必须区分：

- 外层并发
  - 例如 `cpu-workers`，表示同时跑几个实验
- 内层并发
  - 例如 `bar_workers`、`factor_workers`
- 模型线程
  - 例如 `n_jobs`、`thread_count`

治理规则：

- 不允许只限制外层任务数，却放任单任务内部线程无限膨胀
- 当单任务已经很重时，外层并发应保守
- 必须能在 summary / metadata 里看到外层任务数和内层线程数

## 6. 超时、重试与失败处理

### 6.1 每个远程子任务都要有超时

- 不允许无限等待
- 超时要带上下文信息
- 超时后要么重试，要么进入失败列表

### 6.2 重试必须有限且有退避

- 限制 `max_attempts`
- 使用 backoff
- 记录最终失败原因
- 区分可重试错误和不可重试错误

### 6.3 失败要分层

至少区分：

- 单子任务失败
- 单 symbol 失败
- 单阶段失败
- 全链路失败

不要把所有错误都压成一句“任务失败”。

### 6.4 必须输出失败清单

失败清单至少包含：

- task id / symbol / experiment name
- 失败阶段
- 重试次数
- 最终错误

这样才能做精确补跑。

## 7. 性能治理流程

任何性能优化，按下面顺序做。

1. 先测量现状
2. 找最大瓶颈阶段
3. 判断是 `CPU / I/O / 串行 / scope / memory layout / nested oversubscription`
4. 先缩 scope
5. 再做缓存和增量
6. 再做并发层级优化
7. 最后再调资源参数

禁止做法：

- 没有分阶段耗时就盲目加线程
- 没有失败清单就全量重跑
- 没有缓存就重复拉远程数据
- 只看 CPU 总占用率就判断“算力没吃满”

## 8. 长链路排障顺序

以后遇到“慢、卡、失败”，统一按这个顺序排。

1. 是否跑了不必要的 scope
2. 是否有已有结果可复用
3. 是否是远程 I/O 在等待
4. 是否外层并发、内层串行
5. 是否少数慢任务拖尾
6. 是否重试次数过多
7. 是否输出已经足够支持精确补跑
8. 是否存在 DataFrame 碎片化或大量内存复制
9. 是否存在外层任务并发和内层线程并发叠加

## 9. 代码评审检查项

新增或修改长链路代码时，评审至少检查：

- 是否支持缩 scope
- 是否支持 `skip-existing`
- 是否支持 `allow_partial`
- 是否有超时
- 是否有有限重试和退避
- 是否输出失败清单
- 是否记录阶段耗时
- 是否区分 CPU / I/O 型工作
- 是否避免重复计算或重复下载
- 是否能在中途中断后恢复
- 是否有对应测试覆盖

针对 pandas / 特征工程链路，额外检查：

- 是否在循环中反复逐列插入 `DataFrame`
- 是否能先构造列字典，再一次性 `concat`
- 是否有 DataFrame fragmentation 风险

## 10. 当前项目的建议落地方向

结合 Quant Master 现状，后续优先级建议如下。

### 高优先级

- `reference sync` 支持细粒度 scope
- 默认支持 `skip-existing`
- 输出失败 symbol 清单
- 统计每个数据源接口耗时
- 跨实验共享 prepared dataset
- 去掉大因子库实验中的逐列插入式特征构建

### 中优先级

- 单 symbol 内部子任务并发
  - `balance / profit / cash / industry / dividend`
- 统一的阶段耗时与 tail-latency 统计
- schema 版本与自动失效机制
- 对外层任务并发和任务内线程预算做协同约束

### 低优先级

- 更复杂的流水线化
- 更细粒度的任务编排器
- 跨任务共享下载缓存

### 当前落地状态补充，截止 2026-04-27

- `reference sync` 已补齐 symbol 内部子任务并发，入口参数为 `--bundle-workers`
- ML 数据准备链路已补齐阶段耗时与实际 workers 记录
- 因子构建阶段已补齐 symbol 级并发

## 11. `all_factors` 慢链路复盘

这次 `hs300_all_factors_models_v1` 组实验暴露出的，不是单点 bug，而是一组典型长链路性能问题。

### 11.1 现象

- 命令：
  - `py run_ml_experiment.py --group configs/experiments/groups/hs300_all_factors_models_v1.yaml --parallel --cpu-workers 3`
- 运行总时长约 `18` 分钟
- 观察到：
  - CPU 总占用率长期只有约 `30%`
  - 已经开启外层并发，但整体仍然偏慢
  - 日志中出现大量 `PerformanceWarning: DataFrame is highly fragmented`

### 11.2 这类问题不能只看 CPU %

遇到“CPU 只有 30%，但任务还是很慢”时，禁止直接得出“并发度不够”的结论。

要优先判断下面三类瓶颈：

- `serial bottleneck`
  - 外层并发了，但关键阶段仍然有大量串行步骤
- `memory / dataframe layout bottleneck`
  - CPU 没打满，但在频繁做列插入、复制、拼接、排序、类型对齐
- `nested oversubscription`
  - 外层任务并发和内层线程并发叠加，线程数量远高于机器有效算力，导致争抢但不一定表现为高 CPU

### 11.3 这次问题的具体成因

#### A. 嵌套并发

这次不是单层并发，而是多层并发叠加：

- group 层：
  - `cpu-workers = 3`
- 单实验内部：
  - `bar_loader.resolved_workers = 32`
  - `factor_builder.resolved_workers = 16`
- 模型内部：
  - `xgboost n_jobs = 8`
  - `catboost thread_count = 8`

治理原则：

- 外层并发数不是越大越好
- 必须同时统计“外层任务数”和“单任务内部线程数”
- 如果已经有强内层并发，外层并发应保守

#### B. 重复做相同的数据准备

这次三个模型：

- `ridge`
- `xgboost`
- `catboost`

使用的是同一份：

- universe
- date range
- feature list
- normalization
- label horizon
- target mode

但当前仍然为每个模型各自重新做了一遍：

- bars loading
- factor building
- normalization
- labeling
- dropna / trim

治理原则：

- “模型不同，但数据准备口径相同”时，必须优先考虑跨实验共享 prepared dataset
- 长链路里最贵的中间结果，应先判断能否跨实验复用，再决定是否继续加并发

缓存适用边界：

- 可以复用：
  - 模型变化
  - 模型参数变化
- 不可复用：
  - 特征列表变化
  - 标签定义变化
  - normalization 变化
  - 时间区间变化
  - universe 变化
  - 数据源 / adjust / reference 口径变化

#### C. DataFrame 碎片化

日志中的 `DataFrame is highly fragmented` 说明当前因子构建方式在反复做逐列插入。

典型反模式：

- 在循环里反复执行：
  - `frame[new_col] = series`

这会导致：

- 底层 block 被切碎
- 后续排序、切片、标准化、训练前矩阵抽取都变慢
- CPU 使用率可能不高，但内存搬运很多

治理原则：

- 因子列不要一列一列插入到主表
- 应先构造成：
  - `dict[str, Series]`
  - 或单独的 feature frame
- 最后一次性：
  - `pd.concat(axis=1)`
  - 或一次性 `assign`

### 11.4 这类 case 的正确优化顺序

以后遇到类似 `all_factors` 的慢实验，按这个顺序处理：

1. 先确认是不是重复跑了相同 prepared dataset
2. 再确认是否存在 DataFrame 碎片化
3. 再确认是否存在外层并发和内层线程叠加
4. 最后才调大 `cpu-workers`

禁止反过来做：

- 先盲目提高 `cpu-workers`
- 先盲目提高 `factor_workers`
- 只看 CPU 总占用率就判断“算力没吃满”

### 11.5 这类链路必须新增的观测项

针对 ML 大因子库实验，后续 summary / metadata 至少要稳定记录：

- group 层：
  - 同时运行的 experiment 数
- 单实验层：
  - data preparation 总耗时
  - bar loading 耗时
  - factor building 耗时
  - normalization 耗时
  - label building 耗时
- 并发层：
  - `cpu-workers`
  - `bar_loader.resolved_workers`
  - `factor_builder.resolved_workers`
  - `n_jobs / thread_count`
- 复用层：
  - prepared dataset 是否命中缓存
  - 命中的 cache key 是什么
- 数据结构层：
  - 是否出现 DataFrame fragmentation warning

## 12. 结论

对长链路的治理，核心不是“这次卡了怎么修”，而是：

- 让任务少做事
- 让结果可复用
- 让失败可恢复
- 让瓶颈可观测
- 让优化基于测量而不是猜测

后续项目里的 sync、训练、回填、批量比较，都应该按这份清单逐项对照。
