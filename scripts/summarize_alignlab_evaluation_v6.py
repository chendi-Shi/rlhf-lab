"""Summarize ten complete primary evaluation tasks by independent CPU replay."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_lab.evaluation_analysis_v6 import summarize_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--existing-manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(summarize_evaluation(args.request, args.request_sha256,
        output=args.output, expected_manifest_sha256=args.existing_manifest_sha256)))


if __name__ == "__main__":
    main()
