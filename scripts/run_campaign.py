"""Plan or supervise sequential SFT/reward/DPO/PPO runs without sharing GPU workers."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlhf_lab.campaign import run_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true", help="Without this switch, only print the plan")
    parser.add_argument("--resume", action="store_true", help="Require and validate an existing campaign identity")
    args = parser.parse_args(argv)
    try:
        result = run_campaign(args.plan, execute=args.execute, resume=args.resume)
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    # The full process/attempt history remains in campaign.json.
    print(json.dumps({key: value for key, value in result.items() if key not in ("identity", "attempts", "sessions")},
                     ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if result["status"] in ("plan_only", "completed") else 2 if result["status"] in ("paused", "budget_exhausted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
