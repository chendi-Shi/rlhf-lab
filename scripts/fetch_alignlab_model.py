"""Plan by default; --execute downloads the fixed public Qwen3 safetensors snapshot."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.model_snapshot import (
    DEFAULT_OUTPUT, ROOT, download_snapshot, plan_snapshot, verify_snapshot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true",
                        help="Explicitly acquire weights; never overwrites an existing directory")
    action.add_argument("--verify-manifest-sha256",
                        help="Offline full read-only verification anchored to this manifest SHA256")
    args = parser.parse_args()
    if not args.output.absolute().resolve().is_relative_to((ROOT / "models").resolve()):
        parser.error("--output must stay inside this project's models directory")
    if args.verify_manifest_sha256:
        result = verify_snapshot(args.output, args.verify_manifest_sha256)
    elif args.execute:
        result = download_snapshot(args.output, timeout=args.timeout)
    else:
        result = plan_snapshot(args.output, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
