from __future__ import annotations

"""ML 实验命令行入口。

这个文件只负责把命令行参数翻译成明确的执行意图，然后转发到
`src/ml/...` 中的真正实现。训练、评估、并发调度、回填逻辑都不写在这里。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
# 允许直接通过 `py run_ml_experiment.py ...` 运行仓库入口，
# 不要求先把项目安装成可编辑包。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.backfill import backfill_signal_metrics_from_path, backfill_signal_metrics_group_from_path
from ml.experiments.runner import run_experiment_from_path, run_experiment_group_from_path
from ml.experiments.scheduler import GroupExecutionOptions


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="运行单个 ML 实验，或运行/回填一个可恢复的实验组。",
        epilog=(
            "示例:\n"
            "  py run_ml_experiment.py --experiment configs/experiments/demo.yaml\n"
            "  py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --parallel --cpu-workers 8\n"
            "  py run_ml_experiment.py --group configs/experiments/groups/hs300_core_models.yaml --resume\n"
            "  py run_ml_experiment.py --experiment configs/experiments/demo.yaml --backfill-signal-metrics"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--experiment", help="单个实验 YAML 配置路径。与 --group 二选一。")
    parser.add_argument("--group", help="实验组 YAML 配置路径。与 --experiment 二选一。")
    parser.add_argument("--continue-on-error", action="store_true", help="仅对 --group 生效。某个实验失败后继续跑剩余实验。")
    parser.add_argument("--parallel", action="store_true", help="仅对 --group 生效。启用按 CPU/GPU 资源感知的并发调度。")
    parser.add_argument("--cpu-workers", type=int, help="仅对 --group 生效。最多同时跑多少个 CPU 实验。")
    parser.add_argument("--gpu-workers", type=int, help="仅对 --group 生效。最多同时跑多少个 GPU 实验。")
    parser.add_argument("--gpu-device", action="append", default=[], help="仅对 --group 生效。指定调度可见的 GPU 设备编号，可重复传入。")
    parser.add_argument("--resume", action="store_true", help="仅对 --group 生效。从已有 group summary 继续，跳过已完成任务。")
    parser.add_argument("--backfill-signal-metrics", action="store_true", help="不重新训练模型，复用已有 artifact 重算并回写信号评估结果。")
    return parser


def main() -> None:
    """命令行主流程。"""

    parser = build_parser()
    args = parser.parse_args()

    # 强制调用方二选一，避免下游逻辑去猜到底是跑单实验还是实验组。
    if bool(args.experiment) == bool(args.group):
        parser.error("必须且只能提供 --experiment 或 --group 其中一个。")
        return

    if args.experiment:
        # backfill 复用已有 artifact，只重算并回写信号侧评估结果；
        # 普通 run 会重新训练并生成完整 summary。
        if args.backfill_signal_metrics:
            summary = backfill_signal_metrics_from_path(args.experiment)
        else:
            summary = run_experiment_from_path(args.experiment)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.backfill_signal_metrics:
        # 实验组 backfill 与单实验 backfill 语义一致，
        # 但可以结合 resume 从已有进度快照继续。
        summary = backfill_signal_metrics_group_from_path(args.group, resume=bool(args.resume))
    else:
        # 实验组执行支持按资源感知调度，大批量运行时可以分别约束
        # CPU / GPU 并发度，避免任务互相抢资源。
        summary = run_experiment_group_from_path(
            args.group,
            continue_on_error=args.continue_on_error,
            execution_options=GroupExecutionOptions(
                parallel=bool(args.parallel),
                cpu_workers=args.cpu_workers,
                gpu_workers=args.gpu_workers,
                gpu_devices=[str(item) for item in args.gpu_device],
            ),
            resume=bool(args.resume),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
