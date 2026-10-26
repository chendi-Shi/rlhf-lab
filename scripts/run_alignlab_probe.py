"""Execute an explicitly frozen initial GPU probe; defaults to plan inspection."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.gpu_probe import run_probe, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    result = run_probe(plan) if args.execute else {"status": "plan_only", "plan": plan}
    print(json.dumps({k: v for k, v in result.items() if k in
                     {"status", "training_started", "formal_training", "capacity_verified", "plan", "resources"}},
                    ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "waiting_resources" else 0


if __name__ == "__main__":
    raise SystemExit(main())
