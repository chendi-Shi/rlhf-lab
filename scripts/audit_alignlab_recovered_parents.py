"""CPU-only independent acceptance of explicitly recovered formal parents."""
from pathlib import Path
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True)
    parser.add_argument("--parent-plan-sha256", required=True)
    parser.add_argument("--recovery-plan", required=True)
    parser.add_argument("--recovery-plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from alignment_lab.recovered_parent_acceptance import audit_recovered_parents, ACCEPTED
    report = audit_recovered_parents(args.parent_plan, args.parent_plan_sha256,
        args.recovery_plan, args.recovery_plan_sha256, output=args.output)
    print(json.dumps({"status": report["status"], "all_checks_passed": report["all_checks_passed"],
                      "output": str(Path(args.output).resolve())}))
    return 0 if report["status"] == ACCEPTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
