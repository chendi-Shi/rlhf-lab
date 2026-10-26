"""Run one explicitly authorized, pinned PPO checkpoint continuation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.throughput_stage_v2 import pin, read, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = pin({"path": str(args.request.resolve()), "sha256": args.request_sha256})
    request = read(path)
    output = Path(request["output"]).resolve()
    if path.resolve() != output.parent / (output.name + ".request.json"):
        raise ValueError("Request must be the exact supervisor stage launch file")
    pin({"path": str(path.resolve()), "sha256": args.request_sha256})
    result = run_stage(request, execute=args.execute)
    print(json.dumps({k: result[k] for k in ("status", "stage", "formal_training", "research_complete")}, indent=2))


if __name__ == "__main__":
    main()
