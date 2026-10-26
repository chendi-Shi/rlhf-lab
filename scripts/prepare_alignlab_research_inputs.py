"""CPU-only fixed development and reserved diagnostic prompt preparation."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.research_inputs import prepare_research_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare_research_inputs(args.audit_dir, args.asset_manifest, output=args.output)
    print(json.dumps({k: result[k] for k in ("status", "development_prompts",
        "reserved_diagnostic_prompts", "diagnostic_set_complete", "research_complete")}))
