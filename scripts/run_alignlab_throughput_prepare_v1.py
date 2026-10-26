"""Prepare the explicitly authorized original-backend rollback; never launch CUDA."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_lab.training_telemetry import digest, read, sha, write_new
from alignment_lab import throughput_stage_v1 as stage
from alignment_lab import throughput_supervisor_v1 as campaign
from alignment_lab.throughput_telemetry_v1 import verify_resume_evidence

ORIGINAL_PLAN = ROOT / "configs/alignlab-v2-primary-r3-20261022T064456409883Z/plan.json"
ORIGINAL_SHA = "a98a3f37d81119c4a1658af9d0c5bbe242f5aadf056dd832b4a6b73a190da253"
AUTH = ROOT / "reports/alignlab-throughput-authorized-20261023T0704092135256Z/authorization.json"
AUTH_SHA = "643cfff807174a71ee84d87c46a4d9bc9217369b6ab86202db1da3a9ed37a4db"
INCIDENT = ROOT / "reports/alignlab-r3-throughput-incident-20261023T0724536730981Z"
ROLLBACK_SHA = "3a60e225ebca09ffd334d5fe533647bf82cf4df91a6082ae415114cf69bf5e72"


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", required=True, type=Path)
    parser.add_argument("--cpu-report-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"alignlab-v2-throughput-resume609-\d{8}T\d{6}Z", args.run_id):
        raise ValueError("Unique explicit authorized continuation run ID required")
    cpu_pin = {"path": str(args.cpu_report.resolve()), "sha256": args.cpu_report_sha256}
    stage.pin(cpu_pin)
    cpu = read(args.cpu_report)
    original = campaign.legacy.validate_plan(ORIGINAL_PLAN, ORIGINAL_SHA)
    old_output = Path(original["output"])
    prior = read(old_output / "ppo42/progress.json")
    checkpoint = prior["latest_safe_checkpoint"]
    history = {"prior_progress": pin(old_output / "ppo42/progress.json"),
        "metrics": pin(old_output / "ppo42/metrics.jsonl"),
        "cost_events": pin(old_output / "ppo42/telemetry/cost-events.jsonl"),
        "checkpoint": {"path": checkpoint["checkpoint_path"], "manifest_sha256": checkpoint["manifest_sha256"],
            "sidecar": checkpoint["telemetry_sidecar"], "binding": checkpoint["telemetry_binding"]},
        "rollback_authorization": {"path": str(INCIDENT / "rollback-authorization.json"), "sha256": ROLLBACK_SHA}}
    stage.pin(history["rollback_authorization"])
    stage.pin({"path": str(AUTH), "sha256": AUTH_SHA})
    if (prior["actual_optimizer_updates"] != 630 or checkpoint["optimizer_updates"] != 609
            or time.time() >= stage.DEADLINE):
        raise ValueError("Explicit rollback boundary or original deadline no longer applies")
    source = cpu["source_after"]
    stage._source_pins(source, original["source_sha256"])
    campaign._cpu_proof({"cpu_report": cpu_pin, "source_sha256": source})
    evidence = verify_resume_evidence(history, expected_old_identity_sha256=digest(prior["identity"]))
    if evidence["cursor"]["optimizer_updates"] != 609 or evidence["original_observed_optimizer_updates"] != 630:
        raise ValueError("Independent rollback history verification did not prove 609/630")
    config_dir, reports, output = (ROOT / name / args.run_id for name in ("configs", "reports", "runs"))
    if config_dir.exists() or reports.exists() or output.exists():
        raise FileExistsError("Continuation preparation never overwrites prior output")
    config_dir.mkdir()
    reports.mkdir()
    request_pin = pin(old_output / "ppo42.request.json")
    authority = {"path": str(AUTH), "sha256": AUTH_SHA}
    report = {"kind": "alignlab-original-continuation-validation-v1", "decision": "fallback_original",
        "reason": "User prioritizes resuming effective training; optimized CUDA benchmark has not been run.",
        "source_sha256": source, "authorization_sha256": AUTH_SHA, "original_plan_sha256": ORIGINAL_SHA,
        "original_request": request_pin, "history_sha256": digest(history),
        "cpu_resume_equivalence_passed": True, "original_implementation_unchanged": True,
        "original_implementation_scope": "All original 179 frozen files, sampling backend and numerical PPO engine remain byte-identical; new explicit rollback and stop-save control are separately pinned and tested.",
        "selected_backend": "original", "benchmark_executed": False,
        "benchmark_did_not_commit_training_updates": True, "real_cuda": False,
        "cuda_sampling_equivalence_passed": False, "cuda_resume_equivalence_passed": False,
        "optimized_speedup_proven": False,
        "restore_optimizer_updates": 609, "original_observed_optimizer_updates": 630, "authorized_replay_updates": 21,
        "missing_630_parameter_state_independently_verified": False,
        "evidence_pins": {"cpu_report": cpu_pin, "incident": pin(INCIDENT / "incident.json"),
            "rollback_authorization": history["rollback_authorization"],
            "original_checkpoint_manifest": pin(Path(checkpoint["checkpoint_path"]) / "manifest.json")},
        "created_at_epoch": time.time(), "original_deadline_epoch": stage.DEADLINE}
    report_pin = write_new(reports / "original-continuation-validation.json", report)
    dpo = read(old_output / "dpo42/progress.json")
    accepted_dpo = {"request": pin(old_output / "dpo42.request.json"), "progress": pin(old_output / "dpo42/progress.json"),
        "artifact_manifest": pin(dpo["artifact"]["manifest_path"]), "campaign": pin(old_output / "campaign.json")}
    plan = {"schema_version": 1, "kind": campaign.KIND, "output": str(output),
        "original_plan": {"path": str(ORIGINAL_PLAN), "sha256": ORIGINAL_SHA},
        "prior_campaign": accepted_dpo["campaign"], "original_started_at_epoch": stage.STARTED,
        "original_deadline_epoch": stage.DEADLINE, "original_request": request_pin, "history": history,
        "source_sha256": source, "authorization": authority, "validation_report": report_pin,
        "cpu_report": cpu_pin, "accepted_dpo42": accepted_dpo, "selected_backend": "original",
        "process_wait_spent_seconds": prior["process_guard"]["total_waited_seconds"], "future_ppo_backend": "original"}
    plan_pin = write_new(config_dir / "plan.json", plan)
    campaign.validate_plan(plan_pin["path"], plan_pin["sha256"])
    # Rehash after validation; any source drift invalidates the plan instead of
    # blessing a new inventory after tests.
    stage._source_pins(source, original["source_sha256"])
    print(json.dumps({"status": "validated_plan_only", "plan": plan_pin, "report": report_pin,
        "output": str(output), "cuda_initialized": __import__("torch").cuda.is_initialized(),
        "formal_training_launched": False, "restored_from": 609, "replay_updates": 21}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
