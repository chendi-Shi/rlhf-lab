"""Run only the separately authorized exact CUDA comparison; never training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.throughput_validation_v1 import run_throughput_validation
from alignment_lab.throughput_stage_v1 import pin, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = pin({"path": str(args.request.resolve()), "sha256": args.request_sha256})
    request = read(path)
    result = run_throughput_validation(request, execute=args.execute,
        request_path=path, request_sha256=args.request_sha256)
    print(json.dumps({key: result[key] for key in ("status", "real_cuda", "formal_training")}, indent=2))


if __name__ == "__main__":
    main()
