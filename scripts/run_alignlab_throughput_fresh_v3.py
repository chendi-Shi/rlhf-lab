"""Run one authorized fresh DPO/PPO44 with safe stopping boundaries."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.throughput_stage_v1 import pin, read
from alignment_lab.throughput_fresh_v3 import run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    record = {"path": str(args.request.resolve()), "sha256": args.request_sha256}
    request = read(pin(record))
    output = Path(request["legacy_request"]["output"]).resolve()
    if args.request.resolve() != output.parent / (output.name + ".request.json"):
        raise ValueError("Exact supervisor stage launch file required")
    pin(record)
    result = run_stage(request, execute=args.execute)
    print(json.dumps({k: result[k] for k in ("status", "stage", "formal_training", "research_complete")}, indent=2))


if __name__ == "__main__":
    main()
