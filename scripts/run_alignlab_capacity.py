"""Dispatch one immutable serial capacity workload; no implicit retry/resume."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.capacity_supervisor import run_capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_capacity(args.plan, args.plan_sha256, execute=args.execute)
    print(json.dumps({k: result[k] for k in ("status", "capacity_verified", "formal_training")}, indent=2))
    return 0 if result["status"] in {"plan_only", "capacity_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
