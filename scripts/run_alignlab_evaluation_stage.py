"""Execute one externally pinned full evaluation task. Never resumes or retries."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_lab.evaluation_execution import run_evaluation_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    receipt = run_evaluation_stage(args.request, args.request_sha256)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
