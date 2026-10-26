"""Run exactly one frozen capacity stage; plan-only unless --execute is given."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.capacity_stage import read_json, run_capacity_stage, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if sha(args.request) != args.request_sha256:
        raise ValueError("Capacity stage request SHA changed")
    request = read_json(args.request)
    if sha(args.request) != args.request_sha256:
        raise ValueError("Capacity stage request changed while reading")
    result = run_capacity_stage(request, execute=args.execute)
    print(json.dumps({k: result[k] for k in ("status", "stage", "capacity_verified", "formal_training")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
