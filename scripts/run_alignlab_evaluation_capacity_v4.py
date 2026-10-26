"""Execute one pinned evaluation capacity child; no retry, resume or quality claim."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_lab.evaluation_capacity_v4 import run_evaluation_capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", required=True, action="store_true")
    args = parser.parse_args()
    result = run_evaluation_capacity(args.request, args.request_sha256)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
