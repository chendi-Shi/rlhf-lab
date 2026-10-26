"""Freeze the authorized independent evaluation inputs; never load a model."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_lab.evaluation_protocol_v2 import (
    build_evaluation_protocol, load_evaluation_assets, load_final_holdout, canonical,
)
from alignment_lab.evaluation_supervisor_v2 import (
    NEW_SOURCES, KIND, MODEL_PYTHON, SCORE_PYTHON, DECODER_ENTRYPOINT,
)
from alignment_lab.evaluation_analysis_v2 import REQUIRED_SOURCES
from alignment_lab.evaluation_acceptance_v2 import SOURCES
from alignment_lab.evaluation_resources_v2 import load_profile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fresh(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))
    return pin(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--tokenizers-version", required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".runtime/alignlab-eval-20261020").resolve():
        raise ValueError("Build the protocol in the pinned scoring environment")
    launch = args.launch.resolve()
    if not launch.is_relative_to(ROOT / "reports") or not launch.is_dir():
        raise ValueError("Use the existing authorized phase launch directory")
    if sha(args.roster) != args.roster_sha256:
        raise ValueError("Actual accepted roster SHA differs")
    roster = read(args.roster)
    expected_models = {"base", "sft", "reward", "dpo42", "ppo42", "dpo43", "ppo43", "dpo44", "ppo44"}
    if set(roster["models"]) != expected_models or roster["schema_version"] != 2:
        raise ValueError("Exact nine accepted models required")
    authority = read(launch / "authorization-and-budget.json")
    profile = pin(launch / "resource-profile.json")
    load_profile(profile)
    assets = load_evaluation_assets(ROOT / "data/evaluation-assets-20261020T103505Z/ASSET_MANIFEST.json",
                                   "6950abb56fc6ae6044b0283c53a7b5fc3e64d57a43ea8ba863621e2274505720")
    holdout = load_final_holdout(ROOT / "reports/alignlab-v2-data-audit-v3-p1024-r768-20261020",
                                "84d059ba111c37d8f89d70cec6db6bd6c4fd7aef8d4cf933343d67556a204e0b")
    manifests = {name: (str(Path(row["artifact_dir"]) / "manifest.json"), row["manifest_sha256"])
                 for name, row in roster["models"].items() if name != "base"}
    manifests["base"] = (str(Path(roster["model_snapshot"]) / "snapshot-manifest.json"),
                         roster["model_manifest_sha256"])
    nltk_path = ROOT / "reports/alignlab-v2-evaluator-env-20261020/nltk-resources.json"
    if sha(nltk_path) != "4b3cdaa01441787d113ab92f5237f97477239169056e2deb5dba64cf205f513f":
        raise ValueError("Accepted NLTK resource inventory changed")
    nltk_root = ROOT / ".runtime/alignlab-eval-nltk-data-20261020"
    nltk_files = {f["path"]: (str(nltk_root / f["path"]), f["sha256"])
                  for resource in read(nltk_path)["resources"] for f in resource["files"]}
    protocol = build_evaluation_protocol(assets, holdout, model_manifests=manifests,
        scorer_seed=0, bootstrap_seed=42, nltk_resource_files=nltk_files, nltk_data_dir=str(nltk_root))
    if any(v is None for v in protocol.identity["dependency_versions"].values()):
        raise ValueError("Official scoring dependencies are missing")
    protocol_pin = fresh(launch / "protocol.json", protocol.identity)
    source_names = REQUIRED_SOURCES | set(SOURCES) | NEW_SOURCES | {
        "scripts/verify_alignlab_cpu.py", "scripts/prepare_alignlab_evaluation_v2.py",
        "scripts/verify_alignlab_evaluation_v2.py",
        "tests_v2/test_evaluation_model_v2.py", "tests_v2/test_official_evaluation_v2.py",
        "tests_v2/test_evaluation_scoring_cli_v2.py",
        "tests_v2/test_evaluation_execution.py", "alignment_lab/evaluation_execution.py",
        "alignment_lab/evaluation_protocol.py",
        "tests_v2/test_stage_artifacts.py", "tests_v2/test_shared_qwen.py",
    }
    source_sha = {name: sha(ROOT / name) for name in sorted(source_names)}
    original = read(launch / "source-versioning-provenance.json")
    if not original:
        raise ValueError("Versioning provenance is missing")
    for record in original:
        if sha(ROOT / record["original"]) != record["original_sha256"]:
            raise ValueError("Historical frozen source changed: " + record["original"])
    restoration = Path(authority["restoration_report"])
    training = Path(authority["training_success_evidence"])
    scorer_receipt = ROOT / "reports/alignlab-v2-evaluator-env-20261020/acceptance-20261020T110523775060Z.json"
    output = ROOT / "runs/alignlab-quality-v2-20261025T023011Z"
    if output.exists():
        raise FileExistsError("Never overwrite an earlier evaluation attempt")
    plan = {"schema_version": 1, "kind": KIND, "output": str(output),
        "wall_seconds": authority["wall_seconds"], "analysis_wall_seconds": 21600,
        "started_at_epoch": authority["started_at_epoch"], "deadline_epoch": authority["deadline_epoch"],
        "authorization": pin(launch / "authorization-and-budget.json"), "resource_profile": profile,
        "poll_seconds": 10, "source_sha256": source_sha, "cpu_check_report": pin(args.cpu_report),
        "protocol": protocol_pin, "roster": pin(args.roster),
        "model_python_sha256": sha(ROOT / MODEL_PYTHON), "scoring_python_sha256": sha(ROOT / SCORE_PYTHON),
        "decoder": {"python": str(ROOT / MODEL_PYTHON), "python_sha256": sha(ROOT / MODEL_PYTHON),
            "script": str(ROOT / DECODER_ENTRYPOINT), "script_sha256": sha(ROOT / DECODER_ENTRYPOINT),
            "tokenizer_path": str(Path(roster["tokenizer_dir"]) / "tokenizer.json"),
            "tokenizer_sha256": sha(Path(roster["tokenizer_dir"]) / "tokenizer.json"),
            "tokenizers_version": args.tokenizers_version},
        "prerequisites": [
            {**pin(restoration), "json_equals": {"status": "restored_and_verified", "restoration_complete": True}},
            {**pin(training), "json_equals": {}},
            {**pin(scorer_receipt), "json_equals": {"status": "official_cpu_tests_passed", "tests_run": 48}},
            {**pin(nltk_path), "json_equals": {}},
            {**pin(launch / "source-versioning-provenance.json"), "json_equals": {}},
        ]}
    if "torch" in sys.modules:
        raise RuntimeError("Input preparation unexpectedly imported Torch")
    print(json.dumps({"protocol": protocol_pin, "plan": fresh(launch / "plan.json", plan),
                      "models": sorted(manifests), "nltk_files": len(nltk_files),
                      "cuda_used": False, "model_loaded": False}))


if __name__ == "__main__":
    main()
