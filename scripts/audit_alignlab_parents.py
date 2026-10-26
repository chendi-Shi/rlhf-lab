"""Read-only formal-parent audit; writes one fresh report and never starts training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.parent_acceptance import audit_parents, ACCEPTED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_parents(args.plan, args.plan_sha256, output=args.output)
    print(json.dumps({k: report[k] for k in ("status", "parents_verified", "processes_exited", "blockers")}, indent=2))
    return 0 if report["status"] == ACCEPTED else (2 if report["status"] == "incomplete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
