"""Plan-only by default; dispatch one pinned initial GPU probe after download."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.probe_dispatch import run_dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_dispatch(args.request, args.request_sha256, execute=args.execute)
    print(json.dumps({key: result[key] for key in
                     ("status", "report_dir", "capacity_verified", "formal_training", "scope")
                     if key in result}, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("plan_only", "initial_probe_completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
