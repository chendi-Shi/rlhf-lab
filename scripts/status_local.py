"""Collect observed local run evidence without loading models or initializing CUDA."""
import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from rlhf_lab.local_runtime import load_local_config, local_doctor, STAGES
from rlhf_lab.runtime import sha256_file, utc_now, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/local-smoke.json")
args = parser.parse_args()
config = load_local_config(args.config)
output = Path(config["output_dir"])
report = {"generated_at_utc": utc_now(), "device": config["device"], "config": config,
          "scope": config.get("experiment_scope", "Local diagnostics; no quality improvement claim"), "stages": {},
          "timing_scope": "Stage elapsed_seconds excludes model setup, final validation, export, and artifact hashing",
          "gpu_training_completed": False, "human_judgments": 0, "human_win_rate": None}
for stage in STAGES:
    path = output / stage / "run_manifest.json"
    if not path.exists():
        report["stages"][stage] = {"status": "not_started"}
        continue
    manifest = json.loads(path.read_text(encoding="utf-8"))
    keys = ("status", "completed_optimizer_steps", "completed_rollout_batches", "weights_changed",
            "value_head_weights_changed", "elapsed_seconds", "peak_allocated_bytes", "validation", "dependencies")
    record = {key: manifest[key] for key in keys if key in manifest}
    record["manifest_sha256"] = sha256_file(path)
    record["model_sha256"] = sha256_file(path.parent / "model.safetensors") if (path.parent / "model.safetensors").exists() else None
    record["source_sha256"] = manifest["source_sha256"]
    report["stages"][stage] = record
report["gpu_training_completed"] = config["device"] == "cuda" and all(row.get("status") == "completed" for row in report["stages"].values())
eval_path = output / "evaluation" / "summary.json"
if eval_path.exists():
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    report["evaluation"] = {"status": evaluation["status"], "split": evaluation["split"],
                            "generation_examples_per_model": evaluation["generation_examples"],
                            "reward_model_pairs": evaluation["reward_model"]["pairs"],
                            "reward_model_correct": evaluation["reward_model"]["correct"],
                            "human_judgments": evaluation["human_judgments"],
                            "summary_sha256": sha256_file(eval_path), "path": str(eval_path)}
test_path = root / "reports" / "local-check.json"
if test_path.exists():
    report["software_checks"] = json.loads(test_path.read_text(encoding="utf-8"))
report["gpu_preflight"] = local_doctor(load_local_config("configs/local-gpu.json"))
destination = root / "reports" / "actual-smoke-status.json"
write_json(destination, report)
print(json.dumps({"report": str(destination), "stages": {s: r["status"] for s, r in report["stages"].items()},
                  "evaluation": report.get("evaluation", {}).get("status"),
                  "gpu_blockers": report["gpu_preflight"]["blockers"]}, indent=2))
