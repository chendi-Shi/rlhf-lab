"""Independent, repeatable, read-only acceptance of completed formal parents.

Only a new report is written. Model weights are hashed, never loaded into a
model; checkpoint tensors are decoded with weights_only=True on CPU. A running
campaign is diagnostic evidence, never an accepted parent for alignment.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import random
import time

from . import study_supervisor as parents
from .alignment_supervisor import _cpu_payload, _checkpoint_content
from .capacity_supervisor import _bundle, _pin, sha
from .model_snapshot import verify_snapshot
from .probe_dispatch import _System, _same_process, _stamp, _parse

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-independent-formal-parents-acceptance-v1"
ACCEPTED = "accepted_full_real_formal_parents"
AUDITOR_SOURCES = (
    "alignment_lab/parent_acceptance.py", "scripts/audit_alignlab_parents.py",
    "tests_v2/test_parent_acceptance.py", "alignment_lab/alignment_supervisor.py",
    "alignment_lab/study_supervisor.py", "alignment_lab/capacity_supervisor.py",
    "alignment_lab/training_data.py", "alignment_lab/model_snapshot.py",
    "alignment_lab/probe_dispatch.py",
)


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _processes(campaign, plan_path, plan_sha, current):
    """PID reuse is an exited historical process; same creation drift is not."""
    own = campaign.get("supervisor_processes", [])
    if not own:
        raise ValueError("Recorded supervisor PID/create-time/command identities missing")
    expected = []
    for item in own:
        if (not parents._command(item, "run_alignlab_parents.py", plan_path)
                or plan_sha not in item["command_line"]):
            raise ValueError("Recorded supervisor command is not the frozen parent command")
        expected.append(item)
    for record in campaign.get("stages", []):
        request_path = Path(record["request"]["output"]).parent / (record["stage"] + ".request.json")
        known = record.get("known_processes", [])
        if record.get("status") == "accepted" and not known:
            raise ValueError("Accepted child lacks recorded real process identities")
        by_pid = {p["pid"]: p for p in known}
        for item in known:
            if (not parents._command(item, "run_alignlab_study_stage.py", request_path)
                    or record["request_sha256"] not in item["command_line"]
                    or _stamp(item["created_at_utc"]).timestamp() < record["launch_epoch"]):
                raise ValueError("Child creation/command does not match its frozen request")
            ancestor, visited = item, set()
            while ancestor["pid"] != record["wrapper_pid"]:
                if ancestor["pid"] in visited or ancestor.get("parent_pid") not in by_pid:
                    raise ValueError("Child is not a recorded descendant of the spawned wrapper")
                visited.add(ancestor["pid"])
                ancestor = by_pid[ancestor["parent_pid"]]
            if ancestor.get("parent_pid") not in {p["pid"] for p in own}:
                raise ValueError("Child wrapper does not descend from the recorded supervisor")
            expected.append(item)
    observed = {p["pid"]: p for p in current}
    result = []
    for item in expected:
        _stamp(item["created_at_utc"])
        actual = observed.get(item["pid"])
        if actual is None:
            state = "exited"
        elif _same_process(actual, item):
            state = "still_alive"
        elif _stamp(actual["created_at_utc"]) == _stamp(item["created_at_utc"]):
            raise ValueError("Same PID/creation has a different command; cannot attest exit")
        else:
            state = "pid_reused_original_exited"
        result.append({"expected": item, "observed": actual, "state": state})
    return result


def _verify_data(plan):
    from .training_data import _verify
    manifest, identity, _ = _verify(Path(plan["audit_dir"]), plan["audit_manifest_sha256"])
    if Path(identity["tokenizer"]["path"]).resolve() != Path(plan["tokenizer_dir"]).resolve():
        raise ValueError("Accepted data tokenizer differs from parent tokenizer")
    snapshot = _parse(Path(plan["model_snapshot"]) / "snapshot-manifest.json")
    for name, digest in identity["tokenizer"]["files"].items():
        if name in snapshot["files"] and snapshot["files"][name]["sha256"] != digest:
            raise ValueError("Base-model and accepted-data tokenizer files disagree")
    return {"manifest": _file(Path(plan["audit_dir"]) / "manifest.json"),
            "identity_sha256": manifest["identity_sha256"],
            "artifact_sha256": manifest["artifacts"],
            "raw_input_sha256": identity["inputs"], "tokenizer": identity["tokenizer"],
            "holdout_content_parsed": False, "retokenized": False}


def _final_weights(checkpoint, exported, role, heads):
    import torch
    normalized = {}
    for name, value in checkpoint["parameters"].items():
        marker = next((f".{kind}.{role}." for kind in ("lora_A", "lora_B")
                       if f".{kind}.{role}." in name), None)
        if marker:
            key = "adapter/" + name.replace(marker, marker.replace(f".{role}.", "."))
        elif name.startswith("scalar_heads."):
            _, head, local = name.split(".", 2)
            if head not in heads:
                raise ValueError("Checkpoint contains an unowned scalar head")
            key = f"heads/{head}/{local}"
        else:
            raise ValueError("Checkpoint contains parameters outside its adapter/head")
        if key in normalized:
            raise ValueError("Parent parameter normalization collision")
        normalized[key] = value
    if (not isinstance(exported, dict) or set(exported) != set(normalized)
            or any(not isinstance(exported[k], torch.Tensor) or exported[k].device.type != "cpu"
                   or exported[k].shape != value.shape or exported[k].dtype != value.dtype
                   or not torch.equal(exported[k], value) for k, value in normalized.items())):
        raise ValueError("Final artifact weights differ from actual final parent checkpoint")
    return len(normalized)


def _payloads(output, stage, progress):
    """Reconstruct expected saved boundaries from real metrics and train order."""
    count, updates = parents.COUNTS[stage], parents.UPDATES[stage]
    required = {item["optimizer_updates"] for item in progress["checkpoints"]}
    boundary_cursors, seen = {}, []
    metrics_path = Path(output) / "metrics.jsonl"
    # Only retain checkpoint boundaries, avoiding hundreds of duplicate shuffle
    # arrays in memory for the long reward-model metrics file.
    import json
    with metrics_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("event") == "optimizer_boundary":
                cursor = item["cursor"]
                update = cursor["optimizer_updates"]
                seen.append(update)
                if update in required:
                    boundary_cursors[update] = cursor
    if seen != list(range(1, updates + 1)) or set(boundary_cursors) != required:
        raise ValueError("Incomplete/discontinuous optimizer boundaries")
    selection = _parse(progress["selection"]["selection_manifest_path"])["records"]
    order = list(range(count))
    random.Random(42).shuffle(order)
    prefix = [0]
    for index in order:
        row = selection[index]
        prefix.append(prefix[-1] + row["chosen_tokens"] + (row["rejected_tokens"] or 0))
    details = []
    for receipt in progress["checkpoints"]:
        directory = Path(receipt["checkpoint_path"])
        update = receipt["optimizer_updates"]
        cursor = boundary_cursors[update]
        offset = min(update * 16, count)
        if (cursor.get("order") != order or cursor.get("offset") != offset
                or cursor.get("response_tokens") != prefix[offset]
                or cursor.get("active_rollout") is not False or cursor.get("completed_rollouts") != 0
                or any(cursor.get(k) != progress["cursor"].get(k) for k in ("config_sha256", "data_sha256", "schema"))):
            raise ValueError("Saved boundary differs from full real shuffled data/token counts")
        manifest = _bundle(directory, receipt["manifest_sha256"], {"state.pt"},
                           "alignlab-optimizer-checkpoint-v1")
        _checkpoint_content(directory, manifest, {**receipt, "cursor": cursor})
        # All parent states are supervised and must contain no active rollout.
        payload = _cpu_payload(directory / "state.pt", manifest["artifacts"]["state.pt"]["sha256"])
        if payload["extra_state"] is not None:
            raise ValueError("Supervised checkpoint contains unexpected active rollout")
        detail = {"checkpoint_path": str(directory), "manifest_sha256": receipt["manifest_sha256"],
                  "payload_sha256": manifest["artifacts"]["state.pt"]["sha256"],
                  "optimizer_updates": update, "rows_processed": offset,
                  "response_tokens": cursor["response_tokens"], "parameters_finite": True,
                  "optimizer_steps_match": True, "payload_cursor_matches_metric": True,
                  "extra_state_none": True, "parameter_tensors": len(payload["parameters"])}
        if receipt == progress["latest_safe_checkpoint"]:
            if update != updates or payload["cursor"] != progress["cursor"]:
                raise ValueError("Latest checkpoint is not the completed actual final cursor")
            final = progress["artifact"]
            artifact = _bundle(final["artifact_dir"], final["manifest_sha256"], {"weights.pt"},
                               "alignlab-completed-stage-v1")
            expected_role = "sft" if stage == "sft" else "reward"
            expected_heads = [] if stage == "sft" else ["reward"]
            if artifact.get("source_role") != expected_role or artifact.get("heads") != expected_heads:
                raise ValueError("Final parent artifact owns the wrong adapter/head")
            weights = _cpu_payload(Path(final["artifact_dir"]) / "weights.pt",
                                   artifact["artifacts"]["weights.pt"]["sha256"])
            detail["final_exact_tensor_count"] = _final_weights(payload, weights, expected_role, expected_heads)
            detail["final_artifact_exact"] = True
            detail["final_artifact_weights_sha256"] = artifact["artifacts"]["weights.pt"]["sha256"]
        details.append(detail)
    return {"metrics": _file(metrics_path), "checkpoints": details,
            "latest_safe_checkpoint": progress["latest_safe_checkpoint"], "continuous_updates": updates}


def audit_parents(plan_path, expected_plan_sha256, *, output, system=None):
    """Write one fresh accepted/incomplete/failed report; never modify a run."""
    import json
    plan_path, output = Path(plan_path).resolve(), Path(output).absolute()
    if (output.is_symlink() or output.exists() or output.suffix != ".json"
            or not output.resolve().is_relative_to((ROOT / "reports").resolve())):
        raise ValueError("Audit output must be a new .json inside this project's reports")
    if not output.parent.is_dir():
        raise ValueError("Report parent directory must already exist")
    started = time.time()
    report = {"schema_version": 1, "kind": KIND, "status": "incomplete",
              "all_checks_passed": False, "parents_verified": False, "formal_training": False,
              "research_complete": False, "processes_exited": False,
              "parent_plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
              "stages": {}, "checks": [], "blockers": [], "auditor_pid": os.getpid(),
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "device": "cpu", "model_loaded": False, "gpu_calls": False}
    system = system or _System()
    try:
        sources = {name: sha(ROOT / name) for name in AUDITOR_SOURCES}
        report["auditor_sources"] = sources
        plan = parents.validate_plan(plan_path, expected_plan_sha256)
        report["frozen_parent_source_sha256"] = plan["source_sha256"]
        report["checks"].append("frozen_plan_sources_prerequisites_capacity_and_cpu_proof")
        campaign_path = Path(plan["output"]) / "campaign.json"
        campaign_record = _file(campaign_path)
        campaign = _parse(campaign_path)
        _pin(campaign_path, campaign_record["sha256"])
        report["campaign"] = {"path": str(campaign_path), "sha256": campaign_record["sha256"],
                              "status": campaign.get("status")}
        report["formal_training"] = campaign.get("formal_training") is True
        process_records = _processes(campaign, plan_path, expected_plan_sha256, system.processes())
        report["processes"] = process_records
        report["processes_exited"] = all(p["state"] != "still_alive" for p in process_records)
        if (campaign.get("status") != "parents_passed" or campaign.get("parents_verified") is not True
                or not report["processes_exited"]):
            report["blockers"] = (["parent_campaign_not_complete"] if campaign.get("status") != "parents_passed" else [])
            if not report["processes_exited"]:
                report["blockers"].append("recorded_parent_process_still_alive")
            report["observed_stages"] = [{"stage": s["stage"], "status": s["status"], "exit_code": s.get("exit_code")}
                                          for s in campaign.get("stages", [])]
        else:
            if (campaign.get("kind") != parents.KIND or campaign.get("formal_training") is not True
                    or campaign.get("research_complete") is not False
                    or campaign.get("plan_sha256") != expected_plan_sha256
                    or Path(campaign.get("plan_path", "")).resolve() != plan_path
                    or [s.get("stage") for s in campaign.get("stages", [])] != ["sft", "reward"]
                    or campaign.get("deadline_epoch") != campaign["started_at_epoch"] + 43200
                    or not campaign["started_at_epoch"] <= campaign["finished_at_epoch"] <= campaign["deadline_epoch"]):
                raise ValueError("Parent campaign has incomplete lineage/stages/budget evidence")
            if (ROOT / "reports/.alignlab-parents.lock").exists():
                raise ValueError("Parent lock remains after claimed process exit")
            report["model"] = {"manifest": _file(Path(plan["model_snapshot"]) / "snapshot-manifest.json")}
            snapshot = verify_snapshot(plan["model_snapshot"], plan["model_manifest_sha256"])
            report["model"].update(repo_id=snapshot["repo_id"], revision=snapshot["revision"],
                                    actual_files_verified=True)
            report["data"] = _verify_data(plan)
            for stage_record in campaign["stages"]:
                stage = stage_record["stage"]
                directory = Path(plan["output"]) / stage
                request_path = Path(plan["output"]) / (stage + ".request.json")
                _pin(request_path, stage_record["request_sha256"])
                request = _parse(request_path)
                expected_parents = {} if stage == "sft" else {
                    "sft": {k: report["stages"]["sft"][k] for k in ("artifact_dir", "manifest_sha256")}}
                if (request != stage_record["request"] or request["parent_plan_sha256"] != expected_plan_sha256
                        or request["source_sha256"] != plan["source_sha256"]
                        or Path(request["parent_plan_path"]).resolve() != plan_path
                        or Path(request["output"]).resolve() != directory.resolve()
                        or request.get("kind") != parents.STAGE_KIND or request.get("schema_version") != 1
                        or request.get("stage") != stage or request.get("seed") != 42
                        or request.get("expected_rows") != parents.COUNTS[stage]
                        or request.get("expected_updates") != parents.UPDATES[stage]
                        or request.get("deadline_epoch") != campaign["deadline_epoch"]
                        or request.get("parents") != expected_parents
                        or stage_record.get("status") != "accepted" or stage_record.get("exit_code") != 0
                        or stage_record.get("active_processes") != [] or stage_record.get("monitoring_error")
                        or stage_record.get("stop_reason") is not None):
                    raise ValueError("Stage record is not a clean accepted frozen child")
                for name in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
                             "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256"):
                    if request[name] != plan[name]:
                        raise ValueError("Parent child input differs from the frozen campaign")
                _pin(directory / "progress.json", stage_record["progress_sha256"])
                progress = parents._verify_stage(directory, request,
                    {p["pid"]: p for p in stage_record["known_processes"]}, stage_record["exit_code"])
                if progress["artifact"] != stage_record.get("artifact"):
                    raise ValueError("Campaign artifact differs from its pinned child progress")
                details = _payloads(directory, stage, progress)
                artifact = progress["artifact"]
                report["stages"][stage] = {
                    "artifact_dir": artifact["artifact_dir"], "manifest_sha256": artifact["manifest_sha256"],
                    "progress_path": str(directory / "progress.json"), "progress_sha256": stage_record["progress_sha256"],
                    "rows": parents.COUNTS[stage], "optimizer_updates": parents.UPDATES[stage],
                    "payload_audit": details, "gpu_samples": progress["gpu_samples"]}
                if stage == "reward" and progress["identity"]["parents"]["sft"]["sha256"] != report["stages"]["sft"]["manifest_sha256"]:
                    raise ValueError("Reward model does not descend from the accepted common SFT")
                report["checks"].append(stage + "_actual_full_work_cuda_metrics_payload_and_final_weights")
            _pin(campaign_path, campaign_record["sha256"])
            parents.validate_plan(plan_path, expected_plan_sha256)
            final_processes = _processes(campaign, plan_path, expected_plan_sha256, system.processes())
            if any(p["state"] == "still_alive" for p in final_processes):
                raise ValueError("A recorded parent process appeared during audit")
            report["processes_after"] = final_processes
            if sources != {name: sha(ROOT / name) for name in AUDITOR_SOURCES}:
                raise ValueError("Auditor source changed during verification")
            report.update(status=ACCEPTED, all_checks_passed=True, parents_verified=True,
                          processes_exited=True, formal_training=True, blockers=[])
    except Exception as exc:
        report.update(status="failed", all_checks_passed=False, parents_verified=False,
                      error=f"{type(exc).__name__}: {exc}")
        report["blockers"].append("verification_failed")
    report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started)
    # Inspect initialization state only. This does not initialize a CUDA context.
    import sys
    report["torch_imported"] = "torch" in sys.modules
    report["cuda_initialized"] = sys.modules["torch"].cuda.is_initialized() if "torch" in sys.modules else False
    if report["cuda_initialized"]:
        report.update(status="failed", all_checks_passed=False, parents_verified=False,
                      error="Auditor process unexpectedly initialized CUDA")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report
