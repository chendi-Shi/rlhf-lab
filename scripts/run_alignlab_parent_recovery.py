"""One separately authorized fresh-directory RM continuation; no automatic retry."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.parent_recovery import run_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_recovery(args.request, args.request_sha256, execute=args.execute)
    print(json.dumps({key: result[key] for key in ("status", "formal_training", "research_complete")}, indent=2))
    return 0 if result["status"] in {"plan_only", "formal_parent_recovery_passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
