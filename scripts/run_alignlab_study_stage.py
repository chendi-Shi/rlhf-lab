"""Execute one immutable formal SFT/RM parent request, never automatically resume."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.study_stage import _pin, read_json, run_study_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    _pin(args.request, args.request_sha256)
    request = read_json(args.request)
    _pin(args.request, args.request_sha256)
    result = run_study_stage(request, execute=args.execute)
    print(json.dumps({key: result[key] for key in
                      ("status", "stage", "formal_training", "research_complete")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
