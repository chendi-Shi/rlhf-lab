"""Accept current direct-V4 final models on CPU; never starts inference."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.evaluation_acceptance_v2 import audit_evaluation_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "plan-sha256", "campaign", "campaign-sha256", "execution-manifest", "execution-manifest-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = audit_evaluation_models(args.plan, args.plan_sha256, args.campaign, args.campaign_sha256,
        args.execution_manifest, args.execution_manifest_sha256, output_dir=args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("models_verified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
