"""Freeze a new, same-deadline seed44-only plan after full PPO43 acceptance."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_lab import throughput_supervisor_v3_inputs as campaign
from alignment_lab.training_telemetry import digest, read, sha, write_new

PRIOR = ROOT / "configs/alignlab-v2-throughput-resume609-20261023T080559Z/plan.json"
PRIOR_SHA = "adea484fc1b052d5c19eda58e8b5aba8dd18e4b13ee988e53034181cbea89086"
AUTH = ROOT / "reports/alignlab-ppo43-recovery-authorization-20261024T0859375034524Z.json"
AUTH_SHA = "d66c606d823bf98eab23db4445ccbe3a934b7fbf8efe1a26172dc75e89dc5f70"
BUDGET_AUTH = ROOT / "reports/alignlab-deadline-replacement-authorization-20261024T0916160121615Z.json"
BUDGET_AUTH_SHA = "d387d2fbbb01e862755cd84b1ab40eebaaed2764ad3bfc1a4e120866dd8e069a"

def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", required=True, type=Path)
    parser.add_argument("--supplemental", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if sha(PRIOR) != PRIOR_SHA or sha(AUTH) != AUTH_SHA or sha(BUDGET_AUTH) != BUDGET_AUTH_SHA:
        raise ValueError("Frozen failed plan or concrete user authorization changed")
    if Path(args.run_id).name != args.run_id or not args.run_id.startswith("alignlab-v3-ppo43-recovery-"):
        raise ValueError("Independent direct-child recovery name required")
    old = read(PRIOR)
    cpu_pin, acceptance_pin = pin(args.cpu_report), pin(args.supplemental)
    cpu, receipt = read(args.cpu_report), read(args.supplemental)
    sources = cpu["source_after"]
    campaign._cpu_proof({"cpu_report": cpu_pin, "source_sha256": sources})
    campaign.resume._source_pins(sources, old["source_sha256"])
    if (receipt.get("status") != "ppo43_supplemental_accepted"
            or receipt.get("original_full_acceptance_passed") is not True
            or time.time() >= read(BUDGET_AUTH)["execution_window"]["deadline_epoch"]):
        raise ValueError("Completed supplemental acceptance within the newly authorized window required")
    progress = read(Path(old["output"]) / "ppo43/progress.json")
    config_dir, report_dir, output = (ROOT / name / args.run_id for name in ("configs", "reports", "runs"))
    if any(path.exists() for path in (config_dir, report_dir, output)):
        raise FileExistsError("New plan/report/output paths required")
    config_dir.mkdir()
    report_dir.mkdir()
    report = {"kind": "alignlab-original-continuation-validation-v1", "decision": "fallback_original",
        "reason": "Authorized completed PPO43 supplemental acceptance followed by DPO44/PPO44 only; unchanged original numerical backend.",
        "source_sha256": sources, "authorization_sha256": old["authorization"]["sha256"],
        "original_plan_sha256": old["original_plan"]["sha256"], "original_request": old["original_request"],
        "history_sha256": digest(old["history"]), "cpu_resume_equivalence_passed": True,
        "original_implementation_unchanged": True,
        "original_implementation_scope": "All previously frozen sources and original numerical engines remain byte-identical. New serializer preserves only omitted scalar response work; the stage body and backend are tested identical. Separate supplemental verification and seed44 dispatcher are pinned and tested.",
        "selected_backend": "original", "benchmark_executed": False,
        "benchmark_did_not_commit_training_updates": True, "real_cuda": False,
        "cuda_sampling_equivalence_passed": False, "cuda_resume_equivalence_passed": False,
        "optimized_speedup_proven": False, "ppo43_retraining": False,
        "evidence_pins": {"cpu_report": cpu_pin, "recovery_authorization": pin(AUTH),
            "supplemental_ppo43": acceptance_pin, "prior_validation": old["validation_report"],
            "failed_v2_campaign": pin(Path(old["output"]) / "campaign.json")},
        "created_at_epoch": time.time(), "original_deadline_epoch": campaign.resume.DEADLINE}
    report_pin = write_new(report_dir / "original-backend-validation.json", report)
    plan = copy.deepcopy(old)
    plan.update(kind=campaign.KIND, output=str(output), source_sha256=sources,
        cpu_report=cpu_pin, validation_report=report_pin, failed_v2_plan=pin(PRIOR),
        failed_v2_campaign=pin(Path(old["output"]) / "campaign.json"), recovery_authorization=pin(AUTH),
        supplemental_ppo43=acceptance_pin,
        budget_authorization=pin(BUDGET_AUTH), execution_window=read(BUDGET_AUTH)["execution_window"],
        process_wait_spent_seconds=progress["process_guard"]["total_waited_seconds"])
    plan_pin = write_new(config_dir / "plan.json", plan)
    campaign._checked_prefix(plan)
    campaign.resume._source_pins(sources, old["source_sha256"])
    print(json.dumps({"status": "plan_frozen_metadata_validated_full_validation_pending",
        "plan": plan_pin, "report": report_pin, "output": str(output),
        "cuda_initialized": __import__("torch").cuda.is_initialized(),
        "formal_training_launched": False, "remaining_stages": ["dpo44", "ppo44"]}, indent=2))

if __name__ == "__main__":
    main()
