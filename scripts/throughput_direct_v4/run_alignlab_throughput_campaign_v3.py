"""Run the explicitly authorized receipt-based direct training entry."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alignment_lab.throughput_supervisor_direct_v4 import run_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pin = {"path": str(args.execution_manifest.resolve()), "sha256": args.execution_manifest_sha256}
    result = run_alignment(args.plan, args.plan_sha256, pin, execute=args.execute)
    print(json.dumps({key: result[key] for key in ("status", "primary_alignment_verified",
        "parents_verified", "capacity_verified", "research_complete")}, indent=2))
    return 0 if result["status"] in {"plan_only", "primary_alignment_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
