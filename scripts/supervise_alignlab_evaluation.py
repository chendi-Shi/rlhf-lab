"""CPU coordinator for one externally pinned primary-evaluation allocation."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.evaluation_supervisor import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_evaluation(args.plan, args.plan_sha256, execute=args.execute)
    print(json.dumps({k: result.get(k) for k in (
        "status", "primary_evaluation_complete", "research_complete", "completed_tasks", "error")}), flush=True)
    return 0 if result["status"] in {"plan_only", "primary_evaluation_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
