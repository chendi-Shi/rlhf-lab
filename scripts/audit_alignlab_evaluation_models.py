"""CPU-only independent audit and roster construction; never starts evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.evaluation_acceptance import audit_evaluation_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True)
    parser.add_argument("--parent-plan-sha256", required=True)
    parser.add_argument("--alignment-plan", required=True)
    parser.add_argument("--alignment-plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_evaluation_models(args.parent_plan, args.parent_plan_sha256,
        args.alignment_plan, args.alignment_plan_sha256, output_dir=args.output)
    print(json.dumps({k: result[k] for k in ("status", "models_verified", "blockers", "roster")}, indent=2))
    return 0 if result["models_verified"] else (2 if result["status"] == "incomplete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
