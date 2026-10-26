"""Aggregate a completed, audited three-seed campaign without model imports."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlhf_lab.study_report import summarize_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = summarize_campaign(args.plan, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "failed", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if result["status"] in {"completed", "reused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
