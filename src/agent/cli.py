"""Command-line entry point for ctvlm report generation."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from src.agent import pipeline


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate a ctvlm FINDINGS block.")
    ap.add_argument("study_id", help="Study ID used for CT/features/cache lookup")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-threshold", type=float, default=0.20)
    ap.add_argument("--max-findings", type=int, default=12)
    ap.add_argument("--contrast-phase", choices=("ce", "nc"), default="ce")
    ap.add_argument("--probe-path", type=Path, default=None)
    ap.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    ap.add_argument(
        "--with-llm",
        action="store_true",
        help="Enable the research-only LLM prose layer; production should omit this.",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    report = pipeline.generate_report(
        args.study_id,
        threshold=args.threshold,
        min_threshold=args.min_threshold,
        max_findings=args.max_findings,
        probe_path=args.probe_path,
        contrast_phase=args.contrast_phase,
        skip_llm=not args.with_llm,
    )
    if args.json:
        print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    else:
        print(report.summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
