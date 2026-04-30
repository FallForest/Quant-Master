from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.experiments.baseline import load_official_baseline_manifest
from ml.experiments.compare import build_comparison_frame, load_group_summary, render_comparison_table
from research.profiles import load_research_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare completed ML experiments from one group summary.")
    parser.add_argument("--group", required=True, help="Group name, group YAML, or group_summary.json path")
    parser.add_argument("--research-profile", help="Optional research profile that supplies the official baseline manifest")
    parser.add_argument("--baseline-manifest", help="Optional explicit baseline manifest JSON path")
    parser.add_argument("--output-csv", help="Optional CSV export path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    group_summary = load_group_summary(args.group)
    baseline_manifest = None
    if args.baseline_manifest:
        baseline_manifest = load_official_baseline_manifest(args.baseline_manifest)
    elif args.research_profile:
        profile = load_research_profile(args.research_profile)
        if profile.official_baseline_manifest:
            baseline_manifest = load_official_baseline_manifest(profile.official_baseline_manifest)
    frame = build_comparison_frame(group_summary, baseline_manifest=baseline_manifest)
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    print(render_comparison_table(frame))


if __name__ == "__main__":
    main()
