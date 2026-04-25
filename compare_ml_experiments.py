from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.experiments.compare import build_comparison_frame, load_group_summary, render_comparison_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare completed ML experiments from one group summary.")
    parser.add_argument("--group", required=True, help="Group name, group YAML, or group_summary.json path")
    parser.add_argument("--output-csv", help="Optional CSV export path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    group_summary = load_group_summary(args.group)
    frame = build_comparison_frame(group_summary)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    print(render_comparison_table(frame))


if __name__ == "__main__":
    main()
