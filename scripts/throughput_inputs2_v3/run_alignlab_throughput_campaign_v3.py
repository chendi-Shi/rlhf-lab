"""Run the explicit trajectory-preserving continuation, never an automatic retry."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alignment_lab.throughput_supervisor_v3_inputs2 import run_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_alignment(args.plan, args.plan_sha256, execute=args.execute)
    print(json.dumps({k: result[k] for k in ("status", "primary_alignment_verified", "parents_verified",
        "capacity_verified", "research_complete")}, indent=2))
    return 0 if result["status"] in {"plan_only", "primary_alignment_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
