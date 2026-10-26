"""Explicit PPO continuation over a separately authorized committed R3 boundary.

The original request and checkpoint validators remain authoritative.  The new
identity describes the execution change; it is never substituted for the old
identity when loading old weights, Adam state, RNG state, or active rollouts.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

from . import alignment_stage_r3 as original
from .capacity_stage import boundary, _head_anchor
from .training_telemetry import digest, read, sha, write_new

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-ppo-throughput-resume-v2"
STARTED = 1792634845.7777236
DEADLINE = 1792894045.7777236
BACKENDS = {"original", "deferred_scalar", "cached_topology"}
NEW_SOURCES = {
    "alignment_lab/throughput_stage_v2.py", "scripts/run_alignlab_throughput_stage_v2.py",
    "alignment_lab/throughput_sampling_v1.py", "alignment_lab/throughput_telemetry_v1.py",
    "tests_throughput_v2/test_stage_v2.py", "tests_throughput_v2/__init__.py",
}


def validate_model_metadata(actual, expected):
    """Compare every metadata field using its strict persisted JSON meaning.

    JSON persists tuple/list sequences identically. No other type conversion,
    omitted field, non-finite value, custom object or key coercion is allowed.
    """
    def normalized(value):
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) in (tuple, list):
            return [normalized(item) for item in value]
        if type(value) is dict and all(type(key) is str for key in value):
            return {key: normalized(item) for key, item in value.items()}
        raise ValueError("Model metadata must contain finite strict JSON values")

    if type(actual) is not dict or type(expected) is not dict:
        raise ValueError("Full model metadata dictionaries required")
    before, after = (json.dumps(normalized(item), sort_keys=True, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False).encode("utf-8") for item in (actual, expected))
    if before != after:
        raise ValueError("Actual loaded CUDA model/runtime metadata differs from original PPO")
    return True


def pin(record):
    if (type(record) is not dict or set(record) != {"path", "sha256"}
            or type(record["path"]) is not str or not Path(record["path"]).is_absolute()
            or any(p.is_symlink() for p in (Path(record["path"]), *Path(record["path"]).parents))
            or sha(record["path"]) != record["sha256"]):
        raise ValueError("Exact absolute ordinary-file SHA pin required")
    return Path(record["path"])


def _source_pins(sources, old_sources):
    if (type(sources) is not dict or not NEW_SOURCES <= sources.keys()
            or not old_sources.keys() <= sources.keys()
            or any(sources.get(k) != v for k, v in old_sources.items())):
        raise ValueError("Complete unchanged original source pins plus new implementation required")
    for name, expected in sources.items():
        path = ROOT / name
        if (Path(name).is_absolute() or not path.resolve().is_relative_to(ROOT)
                or any(p.is_symlink() for p in (path, *path.parents)) or sha(path) != expected):
            raise ValueError("Execution source inventory changed")


def _accepted_dpo42(record, old):
    if type(record) is not dict or set(record) != {"request", "progress", "artifact_manifest", "campaign"}:
        raise ValueError("Pinned accepted DPO42 predecessor required")
    request, progress, manifest = (read(pin(record[k])) for k in ("request", "progress", "artifact_manifest"))
    campaign = read(pin(record["campaign"]))
    accepted = [stage for stage in campaign.get("stages", []) if stage.get("name") == "dpo42"]
    if (campaign.get("plan_sha256") != old["alignment_plan_sha256"] or len(accepted) != 1
            or accepted[0].get("status") != "accepted" or accepted[0].get("exit_code") != 0
            or accepted[0].get("request_sha256") != record["request"]["sha256"]
            or accepted[0].get("progress_sha256") != record["progress"]["sha256"]
            or accepted[0].get("artifact", {}).get("manifest_sha256") != record["artifact_manifest"]["sha256"]):
        raise ValueError("Original campaign has not accepted the exact DPO42 artifact")
    original.validate_request(request)
    if (request["stage"] != "dpo" or request["seed"] != 42
            or request["alignment_plan_sha256"] != old["alignment_plan_sha256"]
            or progress.get("status") != "formal_alignment_stage_passed"
            or progress.get("actual_optimizer_updates") != 822
            or progress.get("completion", {}).get("rows_processed") != 13139
            or progress.get("completion", {}).get("protocol_complete") is not True
            or manifest.get("stage") != "dpo" or manifest.get("status") != "complete"
            or manifest.get("completion") != progress.get("completion")
            or progress.get("artifact", {}).get("manifest_sha256") != record["artifact_manifest"]["sha256"]
            or Path(progress["artifact"]["artifact_dir"]).resolve() / "manifest.json"
               != Path(record["artifact_manifest"]["path"]).resolve()):
        raise ValueError("DPO42 accepted predecessor evidence differs")
    for name, artifact in manifest["artifacts"].items():
        path = Path(record["artifact_manifest"]["path"]).parent / name
        if sha(path) != artifact["sha256"] or path.stat().st_size != artifact["size_bytes"]:
            raise ValueError("Accepted DPO42 artifact changed")
    return progress


def validate_execution_authority(request, old):
    """Pins bind the concrete authorization, benchmark and resumed request.

    Both successful optimization and measured rejection may resume; rejection
    must explicitly select the unchanged original backend.
    """
    authorization = read(pin(request["authorization"]))
    report = read(pin(request["validation_report"]))
    if (authorization.get("kind") != "alignlab_throughput_preserving_trajectory_authorization_v1"
            or authorization.get("status") != "authorized_preparation_validation_and_conditional_safe_switch"
            or authorization.get("user_selected_answer") != "保留原采样轨迹：验证等价且确实更快后再安全切换（推荐）"
            or authorization.get("started_at_epoch") != STARTED
            or authorization.get("deadline_epoch") != DEADLINE or authorization.get("wall_seconds") != 259200
            or authorization.get("original_plan_sha256") != old["alignment_plan_sha256"]):
        raise ValueError("Explicit trajectory-preserving throughput authorization required")
    pin({"path": str((ROOT / authorization["proposal"]).resolve()), "sha256": authorization["proposal_sha256"]})
    if (report.get("source_sha256") != request["source_sha256"]
            or report.get("authorization_sha256") != request["authorization"]["sha256"]
            or report.get("original_plan_sha256") != old["alignment_plan_sha256"]
            or report.get("original_request") != request["original_request"]
            or report.get("history_sha256") != digest(request["history"])
            or report.get("cpu_resume_equivalence_passed") is not True
            or report.get("selected_backend") != request["selected_backend"]
            or report.get("benchmark_did_not_commit_training_updates") is not True):
        raise ValueError("Validation report must bind this exact restoration and source inventory")
    evidence = report.get("evidence_pins")
    if type(evidence) is not dict or not evidence:
        raise ValueError("Independent validation evidence inventory required")
    for item in evidence.values():
        pin(item)
    if request["selected_backend"] != "original":
        baseline, candidate = report.get("baseline_wall_seconds"), report.get("candidate_wall_seconds")
        if (report.get("kind") != "alignlab-throughput-validation-v1"
                or report.get("cuda_sampling_equivalence_passed") is not True
                or report.get("cuda_resume_equivalence_passed") is not True or report.get("real_cuda") is not True
                or type(baseline) not in (int, float) or type(candidate) not in (int, float)
                or not math.isfinite(baseline) or not math.isfinite(candidate) or not 0 < candidate < baseline
                or report.get("speedup_ratio") != baseline / candidate):
            raise ValueError("Optimized backend requires exact CUDA equivalence and measured speedup")
    elif (report.get("kind") != "alignlab-original-continuation-validation-v1"
            or report.get("decision") != "fallback_original" or report.get("original_implementation_unchanged") is not True):
        raise ValueError("Original-backend continuation requires an explicit audited fallback decision")
    return authorization, report


def validate_request(request):
    keys = {"schema_version", "kind", "stage", "output", "original_started_at_epoch",
        "original_deadline_epoch", "original_request", "history", "source_sha256",
        "authorization", "validation_report", "accepted_dpo42", "plan",
        "supervisor_processes", "selected_backend", "process_wait_spent_seconds"}
    if (type(request) is not dict or set(request) != keys or type(request["schema_version"]) is not int
            or request["schema_version"] != 1 or request["kind"] != KIND or request["stage"] != "ppo"
            or request["original_started_at_epoch"] != STARTED
            or request["original_deadline_epoch"] != DEADLINE or request["selected_backend"] not in BACKENDS):
        raise ValueError("Exact PPO continuation schema and original budget required")
    old = read(pin(request["original_request"]))
    # This is the *unmodified* original validator, including every frozen source,
    # real data selection, original plan/repair authorization and parent chain.
    original.validate_request(old)
    if old["stage"] != "ppo" or old["seed"] != 42 or old["deadline_epoch"] != DEADLINE:
        raise ValueError("Only the explicitly authorized PPO42 R3 boundary may resume")
    output = Path(request["output"]).resolve()
    if (not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs"
            or output.is_relative_to(Path(old["output"]).resolve())
            or Path(old["output"]).resolve().is_relative_to(output)
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("Continuation requires a separate ordinary output under runs")
    _source_pins(request["source_sha256"], old["source_sha256"])
    from .alignment_process_guard_r3 import validate_background
    validate_background(request["supervisor_processes"])
    if (not request["supervisor_processes"] or type(request["process_wait_spent_seconds"]) not in (float, int)
            or not math.isfinite(request["process_wait_spent_seconds"])
            or not old["process_wait_spent_seconds"] <= request["process_wait_spent_seconds"] < 259200):
        raise ValueError("Exact active supervisor and inherited process wait ledger required")
    plan = read(pin(request["plan"]))
    if (plan.get("original_started_at_epoch") != STARTED or plan.get("original_deadline_epoch") != DEADLINE
            or plan.get("source_sha256") != request["source_sha256"]
            or plan.get("authorization") != request["authorization"]):
        raise ValueError("Resumed stage differs from its pinned supervisor plan")
    _accepted_dpo42(request["accepted_dpo42"], old)
    validate_execution_authority(request, old)
    history = request["history"]
    history_keys = {"prior_progress", "metrics", "cost_events", "checkpoint"}
    if type(history) is not dict or set(history) not in (history_keys, history_keys | {"rollback_authorization"}):
        raise ValueError("Explicit complete history required")
    prior = read(pin(history["prior_progress"]))
    if (prior.get("request") != old or prior.get("status") != "paused"
            or prior.get("error", "").startswith("InterruptedError:") is not True
            or prior.get("outcome_stop_reason") not in (None, "paused") or not prior.get("safe_resume_available")
            or prior.get("deadline_epoch") != DEADLINE or prior.get("stage") != "ppo"
            or not 0 < prior.get("actual_optimizer_updates", 0) < old["workload"]["optimizer_updates"]):
        raise ValueError("Only a stopped original PPO42 with intact history may resume")
    from .throughput_telemetry_v1 import validate_rollback_authorization
    rollback = validate_rollback_authorization(history, prior)
    checkpoint = history["checkpoint"]
    latest = prior["latest_safe_checkpoint"]
    if (set(checkpoint) != {"path", "manifest_sha256", "sidecar", "binding"}
            or Path(checkpoint["path"]).resolve() != Path(latest["checkpoint_path"]).resolve()
            or checkpoint["manifest_sha256"] != latest["manifest_sha256"]
            or checkpoint["sidecar"] != latest["telemetry_sidecar"]
            or checkpoint["binding"] != latest["telemetry_binding"]
            or (rollback is None and (latest["optimizer_updates"] != prior["actual_optimizer_updates"]
                                      or latest["cursor"] != prior["cursor"]))):
        raise ValueError("Resume must consume the final committed boundary, never an earlier update")
    for key in ("metrics", "cost_events"):
        pin(history[key])
    for key in ("sidecar", "binding"):
        pin(checkpoint[key])
    manifest_path = Path(checkpoint["path"]) / "manifest.json"
    pin({"path": str(manifest_path), "sha256": checkpoint["manifest_sha256"]})
    manifest = read(manifest_path)
    if manifest.get("identity") != prior.get("identity") or manifest.get("identity_sha256") != digest(prior["identity"]):
        raise ValueError("Original checkpoint identity differs from the original progress")
    return output, old, prior


def update_accounting(committed, original_observed, current):
    """Logical work excludes the explicitly abandoned physical suffix."""
    if (any(type(v) is not int for v in (committed, original_observed, current))
            or not 0 < committed <= original_observed or current < committed):
        raise ValueError("Invalid restored, observed or current optimizer counters")
    fresh = current - committed
    return {"actual_optimizer_updates": current, "logical_optimizer_updates": current,
        "inherited_optimizer_updates": committed, "new_optimizer_updates": fresh,
        "original_observed_optimizer_updates": original_observed,
        "physical_optimizer_updates": original_observed + fresh,
        "replayed_optimizer_updates": min(fresh, original_observed - committed),
        "abandoned_optimizer_updates": original_observed - committed}


def require_saved_guard_boundary(guard, name, update, save, *, on_stop=None):
    """Preserve the current numerical boundary before every refused admission.

    The frozen guard remains the sole admission authority. Its wait callback
    and its immediate stop path share one save attempt; failed writes propagate
    without retrying or being replaced by the less precise guard exception.
    """
    attempted, saved, failure = False, None, None

    def save_once():
        nonlocal attempted, saved, failure
        if attempted:
            if failure is not None:
                raise failure
            return saved
        attempted = True
        try:
            saved = save()
            if (type(saved) is not dict or saved.get("complete") is not True
                    or saved.get("optimizer_updates") != update):
                raise RuntimeError("Stopping boundary lacks its exact complete save receipt")
        except BaseException as exc:
            failure = exc
            raise
        return saved

    result = guard.check(boundary=name, optimizer_updates=update, save_boundary=save_once)
    if failure is not None:
        raise failure
    if not result["admission_allowed"]:
        receipt = save_once()
        if on_stop is not None:
            on_stop(result, receipt)
        if result["reason"] == "deadline_exhausted":
            raise TimeoutError("Original absolute deadline reached; completed boundary saved")
        raise InterruptedError("Strict process boundary blocked after saving current state: " + result["reason"])
    return result


def replay_comparison(reference, values, event, *, sampling_rng_hex, expected_sampling_rng_hex):
    """Compare only historical observations that actually survived the incident."""
    numeric = ("update", "responses", "action_tokens", "objective", "gradient_norm")
    details = ("target_update", "rollout_index", "record_ids", "response_indices", "epoch", "rollout_offset")
    if (any(event.get(k) != values.get(k) for k in numeric)
            or any(k not in reference or event.get(k) != reference[k] for k in details)
            or reference.get("cursor_sha256") != digest(event["cursor"])
            or reference.get("order_sha256") != digest(event["cursor"]["order"])
            or not expected_sampling_rng_hex or sampling_rng_hex != expected_sampling_rng_hex):
        raise ValueError("Replayed optimizer step differs from preserved original observations")
    return {"event": "replay_equivalence", "update": event["update"],
        "numeric_fields": {k: event[k] for k in numeric},
        "minibatch_fields": {k: event[k] for k in details},
        "cursor_sha256": digest(event["cursor"]), "order_sha256": digest(event["cursor"]["order"]),
        "sampling_rng_sha256": hashlib.sha256(bytes.fromhex(sampling_rng_hex)).hexdigest(),
        "preserved_observations_equal": True, "prior_final_state_bitwise_verified": False,
        "rng_comparison_scope": "sampling_generator_from_saved_active_rollout_only"}


def joined_metrics(previous, current, committed, final):
    """Aggregate disjoint evidence; historical rows never enter the new log."""
    previous, current = tuple(previous), tuple(current)
    if ([m.get("update") for m in previous] != list(range(1, committed + 1))
            or [m.get("update") for m in current] != list(range(committed + 1, final + 1))):
        raise ValueError("Optimizer history has duplicated, skipped or replayed updates")
    return previous + current


def restore_boundary(checkpoint, old_identity, *, parameters, optimizer, generator):
    """Unmodified transactional restore validates the original identity first."""
    from .checkpointing import load_checkpoint
    result = load_checkpoint(checkpoint["path"], expected_identity=old_identity,
        parameters=parameters, optimizer=optimizer, generators={"sampling": generator},
        expected_manifest_sha256=checkpoint["manifest_sha256"])
    cursor = copy.deepcopy(result["cursor"])
    sidecar = cursor.pop("telemetry", None)
    if sidecar != checkpoint["sidecar"]:
        raise ValueError("Restored checkpoint lacks its exact original sidecar")
    result["cursor"] = cursor
    return result


def _equal_state(left, right):
    """Bitwise equality for CPU checkpoint trees, including every saved RNG."""
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return list(left) == list(right) and all(_equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal_state(a, b) for a, b in zip(left, right))
    return left == right


def verify_migration_payload(old_payload, new_payload):
    before, after = copy.deepcopy(old_payload), copy.deepcopy(new_payload)
    before["cursor"].pop("telemetry", None)
    after["cursor"].pop("telemetry", None)
    if not _equal_state(before, after):
        raise ValueError("Migration changed weights, Adam, RNG, cursor/order or active rollout")
    return True


def _owned(snapshot, request, request_path, request_sha256, *, pid=None):
    from .process_guard_v2 import ProcessIdentity
    from .study_supervisor import _command
    pid = os.getpid() if pid is None else pid
    supervisors = {p["pid"]: ProcessIdentity.from_record(p) for p in request["supervisor_processes"]}
    by_pid = {p["pid"]: p for p in snapshot}
    current, result, seen = by_pid.get(pid), [], set()
    while current is not None:
        identity = ProcessIdentity.from_record(current)
        if identity.pid in seen:
            raise RuntimeError("Cyclic child ancestry")
        seen.add(identity.pid)
        if identity.pid in supervisors:
            if identity != supervisors[identity.pid] or not result:
                raise RuntimeError("Supervisor identity mismatch")
            return result
        command = current["command_line"]
        if (not _command(current, "run_alignlab_throughput_stage_v2.py", request_path)
                or request_sha256 not in command or "--request-sha256" not in command):
            raise RuntimeError("Unverified continuation child/wrapper command")
        result.append(current)
        current = by_pid.get(identity.parent_pid)
    raise RuntimeError("Continuation must descend from its exact active supervisor")


def _new_identity(request, old_identity, selection_path, request_path):
    from .checkpointing import build_identity
    files = {name: Path(record["path"]) for name, record in old_identity["inputs"].items()}
    files.update({"source/" + name: ROOT / name for name in request["source_sha256"]})
    files.update({"throughput/request": request_path, "throughput/selection": selection_path,
        "throughput/original_request": pin(request["original_request"]),
        "throughput/authorization": pin(request["authorization"]),
        "throughput/validation": pin(request["validation_report"]), "throughput/plan": pin(request["plan"]),
        "throughput/prior_progress": pin(request["history"]["prior_progress"]),
        "throughput/prior_metrics": pin(request["history"]["metrics"]),
        "throughput/prior_cost_events": pin(request["history"]["cost_events"]),
        "throughput/prior_checkpoint": Path(request["history"]["checkpoint"]["path"]) / "manifest.json",
        "throughput/prior_sidecar": pin(request["history"]["checkpoint"]["sidecar"]),
        "throughput/prior_binding": pin(request["history"]["checkpoint"]["binding"])})
    files.update({"throughput/accepted_dpo42/" + key: pin(record) for key, record in request["accepted_dpo42"].items()})
    if "rollback_authorization" in request["history"]:
        files["throughput/rollback_authorization"] = pin(request["history"]["rollback_authorization"])
    config = copy.deepcopy(old_identity["config"])
    config.update(purpose=KIND, throughput_revision="v2", selected_backend=request["selected_backend"],
        prior_identity_sha256=digest(old_identity), history_sha256=digest(request["history"]),
        original_started_at_epoch=STARTED, original_deadline_epoch=DEADLINE)
    return build_identity(files, config, {name: Path(record["path"]) for name, record in old_identity["parents"].items()})


def run_stage(request, *, execute=False):
    output, old, prior = validate_request(request)
    if not execute:
        return {"status": "plan_only", "stage": "ppo", "formal_training": False, "research_complete": False}
    if output.exists():
        raise FileExistsError("Fresh continuation output required; existing evidence must never be overwritten")
    request_path = output.parent / (output.name + ".request.json")
    request_sha256 = sha(request_path)
    if read(request_path) != request:
        raise ValueError("CLI request differs from the exact stage launch record")
    stop = lambda: ((output / "STOP").exists() or (output.parent / "STOP").exists()
                    or Path(request["plan"]["path"] + ".stop").exists())
    from .alignment_supervisor_r3 import JsonWriter
    from .alignment_process_guard_r3 import GuardSession, strict_admission
    from .probe_dispatch import _System
    from .gpu_probe import resources
    boundary(DEADLINE, stop)
    system = _System()
    owned = _owned(system.processes(), request, request_path, request_sha256)
    allowed = old["allowed_background_processes"] + request["supervisor_processes"]
    admission_guard = strict_admission(system.processes, owned_identities=owned,
        allowed_identities=allowed, deadline_epoch=DEADLINE)
    if not admission_guard["caller_admission_allowed"]:
        from .process_guard_v4 import WAITABLE
        codes = {item["code"] for item in admission_guard["blockers"]}
        fresh = bool(admission_guard["observations"] and admission_guard["observations"][-1]["snapshot_valid"])
        if (not codes or not codes <= WAITABLE or not admission_guard["allowed_ancestry_verified"]
                or fresh and not admission_guard["owned_identity_verified"]):
            raise RuntimeError("Hard process conflict or owned identity blocks continuation")
    write = JsonWriter(DEADLINE, stop)
    owner = {"pid": os.getpid(), "output": str(output), "token": uuid.uuid4().hex, "kind": KIND}
    lock = ROOT / "reports/.alignlab-gpu.lock"
    write(lock, owner, exclusive=True)
    committed = prior["latest_safe_checkpoint"]["optimizer_updates"]
    original_observed = prior["actual_optimizer_updates"]
    status = {"kind": KIND, "stage": "ppo", "status": "starting", "request": request,
        "pid": os.getpid(), "started_at_epoch": prior["started_at_epoch"], "attempt_started_at_epoch": time.time(),
        "deadline_epoch": DEADLINE, "original_started_at_epoch": STARTED,
        "formal_training": True, "research_complete": False, "synthetic_rows": 0,
        **update_accounting(committed, original_observed, committed),
        "training_started": False, "resume_training_started": False,
        "checkpoints": copy.deepcopy(prior["checkpoints"]), "latest_safe_checkpoint": None,
        "gpu_samples": [], "objective_logging": "historical_prefix_plus_disjoint_live_suffix",
        "process_guard": {"owned_identities": owned, "initial_admission": admission_guard,
            "inherited_waited_seconds": request["process_wait_spent_seconds"],
            "total_waited_seconds": request["process_wait_spent_seconds"], "last": None}}

    def progress(phase):
        status.update(status=phase, updated_at_epoch=time.time(), active_elapsed_seconds=time.time() - status["started_at_epoch"])
        write(output / "progress.json", status)
        print(json.dumps({"phase": phase, "stage": "ppo", "optimizer_updates": status["actual_optimizer_updates"],
                          "new_optimizer_updates": status["new_optimizer_updates"]}), flush=True)

    def metric(value):
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_epoch": time.time(), **value}, allow_nan=False) + "\n")
            stream.flush()

    def unchanged():
        try:
            _source_pins(request["source_sha256"], old["source_sha256"])
            return (sha(request_path) == request_sha256 and read(output / "request.json") == request
                    and sha(request["plan"]["path"]) == request["plan"]["sha256"])
        except (OSError, ValueError):
            return False

    def guard_event(event):
        with (output / "process-guard.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
        if event["event"] == "wait_start":
            progress("waiting_process_boundary")
        elif event["event"] == "result":
            status["process_guard"].update(last=event, total_waited_seconds=event["total_waited_seconds"])

    guard = GuardSession(system.processes, owned_identities=owned, allowed_identities=allowed,
        deadline_epoch=DEADLINE, policy=old["process_wait_policy"], spent_seconds=request["process_wait_spent_seconds"],
        stop_requested=stop, identity_unchanged=unchanged, on_event=guard_event)

    def receipt(path, kind, update):
        return {"complete": True, "optimizer_updates": update, "kind": kind,
                "path": str(Path(path).resolve()), "sha256": sha(path)}

    def require_guard(name, update, save):
        def stopped(result, saved):
            status["process_guard"]["blocked_reason"] = result["reason"]
            status["process_guard"]["stop_boundary_receipt"] = saved
            metric({"event": "guard_stop_boundary", "boundary": name, "optimizer_updates": update,
                "reason": result["reason"], "saved_boundary": saved})
        return require_saved_guard_boundary(guard, name, update, save, on_stop=stopped)

    try:
        output.mkdir(parents=True, exist_ok=False)
        write(output / "request.json", request, exclusive=True)
        require_guard("before_cuda", committed, lambda: receipt(output / "request.json", "before_cuda", committed))
        status["resources_before"] = resources()
        if status["resources_before"]["blockers"]:
            raise RuntimeError("CUDA resources are not clear for continuation")
        from .throughput_telemetry_v1 import ResumedTrainingTelemetry, verify_resume_evidence
        evidence = verify_resume_evidence(request["history"], expected_old_identity_sha256=digest(prior["identity"]))
        if (evidence["cursor"] != prior["latest_safe_checkpoint"]["cursor"]
                or evidence["cursor"]["optimizer_updates"] != committed
                or evidence["original_observed_optimizer_updates"] != original_observed):
            raise ValueError("Verified telemetry history differs from the exact restoration boundary")
        replay_references = {p["update"]: p for p in evidence["all_points"]
                             if p.get("event") == "optimizer_end" and committed < p["update"] <= original_observed}
        replay_values = {p["update"]: p for p in evidence["replay_reference_metrics"]}
        if set(replay_references) != set(range(committed + 1, original_observed + 1)) or set(replay_values) != set(replay_references):
            raise ValueError("Explicit replay is missing original observed numerical evidence")
        status["rollback"] = {"authorization": request["history"].get("rollback_authorization"),
            "restored_update": committed, "original_observed_update": original_observed,
            "first_replay_update": committed + 1 if replay_values else None,
            "last_replay_update": original_observed if replay_values else None,
            "required_replay_updates": len(replay_values), "verified_replay_updates": 0,
            "prior_final_state_bitwise_verified": False,
            "preserved_history": True}
        from .training_data import load_training_partition
        partition = load_training_partition(old["audit_dir"], "ppo_prompts", old["audit_manifest_sha256"])
        rows, selection, records = original.select_real_rows(partition, old)
        original_selection = read(Path(old["output"]) / "selection.json")
        if original_selection != {"summary": selection, "records": records}:
            raise ValueError("Recomputed exact real PPO selection differs from original order")
        selection_path = output / "selection.json"
        write(selection_path, original_selection, exclusive=True)
        selection.update(selection_manifest_path=str(selection_path), selection_manifest_sha256=sha(selection_path))
        status["selection"] = selection
        identity = _new_identity(request, prior["identity"], selection_path, request_path)
        status["identity"] = identity
        write(output / "identity.json", identity, exclusive=True)
        migration = {"kind": "alignlab-throughput-telemetry-migration-v1",
            "history_sha256": digest(request["history"]), "old_identity_sha256": digest(prior["identity"]),
            "new_identity_sha256": digest(identity), "original_deadline_epoch": DEADLINE}
        migration_pin = write_new(output / "telemetry-migration.json", migration)
        status["telemetry_migration"] = migration_pin
        telemetry = ResumedTrainingTelemetry(output / "telemetry", history=request["history"],
            migration_pin=migration_pin, new_identity=identity)
        boundary(DEADLINE, stop)
        validate_request(request)
        if resources()["blockers"]:
            raise RuntimeError("Resources changed before CUDA load")
        import torch
        from .model_loader import ModelLoadConfig, load_quantized_stage, QUANTIZATION
        from .stage_artifacts import import_stage, export_stage
        from .stage_engine import parameter_mapping, StageOutcome
        from .stage_engine_observed import run_ppo, event_to_plain
        from .checkpointing import save_checkpoint
        from .gpu_reference import capture_frozen_reference
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        progress("loading_nf4")
        loaded = load_quantized_stage(old["model_snapshot"], old["model_manifest_sha256"], old["tokenizer_dir"],
                                      ModelLoadConfig("ppo", seed=old["seed"]))
        validate_model_metadata(loaded.metadata, prior["model"])
        if request["selected_backend"] != "original":
            from .throughput_sampling_v1 import make_sampling_backend
            mode = "cached_topology" if request["selected_backend"] == "cached_topology" else "original"
            loaded.backend = make_sampling_backend(loaded.backend, guard_mode=mode)
        backend = loaded.backend
        status["model"] = loaded.metadata
        status["sampling_backend"] = request["selected_backend"]
        files = {name: Path(record["path"]) for name, record in identity["inputs"].items()}
        status["imports"] = {}
        for parent, role, heads in (("sft", loaded.role, ()), ("sft", "reference", ()), ("reward", "reward", ("reward",))):
            p = old["parents"][parent]
            status["imports"][role] = import_stage(p["artifact_dir"], backend, role,
                expected_manifest_sha256=p["manifest_sha256"], expected_identity=identity, heads=heads)
        receipts = {}
        for role, parent in (("reference", "sft"), ("reward", "reward")):
            progress("capturing_" + role + "_receipt")
            artifact = Path(old["parents"][parent]["artifact_dir"])
            receipts[role] = capture_frozen_reference(backend, role, artifact_files={
                "base_files": {k: v for k, v in files.items() if k.startswith("base/")},
                "tokenizer_files": {k: v for k, v in files.items() if k.startswith("tokenizer/")},
                "source_files": {k: v for k, v in files.items() if k.startswith("source/")},
                "adapter_files": {"manifest": artifact / "manifest.json", "weights": artifact / "weights.pt"},
                "dataset_manifest": Path(old["audit_dir"]) / "manifest.json"},
                accepted_data_manifest_sha256=old["audit_manifest_sha256"], precision=original.PRECISION,
                quantization=QUANTIZATION)
            status[role + "_anchor"] = receipts[role].anchor
            if receipts[role].anchor != prior[role + "_anchor"]:
                raise ValueError("Loaded frozen reference/reward tensors differ from original PPO")
        head_anchor = _head_anchor(backend._heads["reward"])
        if head_anchor != prior["reward_head_anchor"]:
            raise ValueError("Frozen reward head differs from original PPO")
        status["reward_head_anchor"] = head_anchor

        def audit_frozen(*, full, phase=None):
            for reference in receipts.values():
                reference.verify(full=full)
            if _head_anchor(backend._heads["reward"]) != head_anchor:
                raise RuntimeError("Frozen reward head changed")
            if phase is not None:
                status.setdefault("frozen_full_audits", []).append({"time_epoch": time.time(), "phase": phase,
                    "roles": ["reference", "reward"], "reward_head_content_verified": True,
                    "cache_manifest_sha256": None})

        mapping = parameter_mapping(backend, loaded.role, heads=loaded.heads)
        optimizer = torch.optim.AdamW(mapping.values(), lr=1e-6)
        generator = torch.Generator(device=loaded.device).manual_seed(old["seed"])
        progress("restoring_committed_boundary")
        restored = restore_boundary(request["history"]["checkpoint"], prior["identity"],
            parameters=mapping, optimizer=optimizer, generator=generator)
        if restored["cursor"] != evidence["cursor"]:
            raise ValueError("Transactional restore and verified telemetry cursor differ")
        # The checkpoint loader already validated and restored every RNG and
        # active-rollout tensor.  Do not initialize RNG again after this point.
        state, extra = restored["cursor"], restored["extra_state"]
        replay_rng_hex = None
        if replay_values:
            replay_rng_hex = extra["observation_origin"]["rng_after_hex"]
            if (len({p["rollout_index"] for p in replay_references.values()}) != 1
                    or next(iter(replay_references.values()))["rollout_index"] != extra["observation_origin"]["rollout_index"]):
                raise ValueError("This explicit rollback covers only the preserved active rollout")
        status["cursor"] = copy.deepcopy(state)
        status["restore_receipt"] = {k: restored[k] for k in ("checkpoint_path", "manifest_sha256", "identity_sha256")}
        status["restore_receipt"].update(committed_optimizer_updates=committed, next_optimizer_update=committed + 1,
            active_rollout_restored=extra is not None, all_rng_and_adam_restored=True)
        initial = {name: p.detach().cpu().clone() for name, p in mapping.items()}
        schedule = original.BoundaryCheckpoints(old["workload"]["optimizer_updates"])
        for record in prior["checkpoints"]:
            schedule.schedule._costs(record["reasons"])
            schedule.warm_costs.update(hour for hour in schedule.schedule.hours if f"warm_cost_{hour}h" in record["reasons"])

        def gpu_sample():
            from rlhf_lab.runtime import gpu_inventory
            torch.cuda.synchronize(0)
            sample = {"time_epoch": time.time(), **gpu_inventory(),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(0), "max_reserved_bytes": torch.cuda.max_memory_reserved(0)}
            status["gpu_samples"].append(sample)
            return sample

        def write_checkpoint(name, cursor, active):
            before = time.time()
            sidecar = telemetry.checkpoint_sidecar(name, cursor, active)
            saved = save_checkpoint(output / name, identity=identity, parameters=mapping, optimizer=optimizer,
                cursor={**cursor, "telemetry": sidecar}, generators={"sampling": generator}, extra_state=active)
            bound = telemetry.bind_checkpoint(saved, sidecar, started_at_epoch=before, finished_at_epoch=time.time())
            return {**saved, "telemetry_sidecar": sidecar, "telemetry_binding": bound}

        def save_safe(cursor, active, *, safe_return=False):
            record = schedule.save(cursor, time.time() - status["started_at_epoch"], safe_return=safe_return,
                warm_elapsed=telemetry.warm_elapsed(), writer=lambda name: write_checkpoint(name, cursor, active))
            if record is not None:
                status["checkpoints"] = copy.deepcopy(prior["checkpoints"]) + list(schedule.records.values())
                status["latest_safe_checkpoint"] = record

        def optimizer_boundary(cursor, active, name):
            def save_wait():
                audit_frozen(full=True)
                save_safe(cursor, active, safe_return=True)
                saved = status["latest_safe_checkpoint"]
                if saved is None or saved["cursor"] != cursor:
                    raise RuntimeError("Current continuation boundary was not durably saved")
                return receipt(Path(saved["checkpoint_path"]) / "manifest.json", "optimizer_checkpoint", cursor["optimizer_updates"])
            return require_guard(name, cursor["optimizer_updates"], save_wait)

        replay_failure = None

        def checkpoint(cursor, active):
            current = cursor["optimizer_updates"]
            status.update(cursor=cursor, **update_accounting(committed, original_observed, current),
                training_started=current > committed, resume_training_started=current > committed)
            if replay_failure is not None:
                audit_frozen(full=True)
                save_safe(cursor, active, safe_return=True)
                raise ValueError(replay_failure)
            optimizer_boundary(cursor, active, "optimizer_complete")
            audit_frozen(full=False)
            save_safe(cursor, active)
            metric({"event": "engine_boundary", "cursor": cursor, "active_rollout_saved": active is not None,
                "active_elapsed_seconds": time.time() - status["started_at_epoch"], "gpu": gpu_sample()})
            progress("training")

        origin_checked = False
        next_update = committed + 1
        def observe(event):
            nonlocal origin_checked, next_update, replay_failure
            plain_event = event_to_plain(event)
            if not origin_checked:
                if plain_event["cursor"] != state:
                    raise RuntimeError("Resumed engine origin differs from committed cursor/order")
                optimizer_boundary(state, extra, "engine_origin")
                origin_checked = True
            if plain_event["event"] == "optimizer_end":
                if plain_event["update"] != next_update:
                    raise RuntimeError("A previously committed optimizer update was repeated or skipped")
                next_update += 1
                metric({"event": "live_optimizer_end", **{k: plain_event[k] for k in
                    ("update", "responses", "action_tokens", "objective", "gradient_norm")}})
            before = time.time()
            torch.cuda.synchronize(loaded.device)
            finished = time.time()
            telemetry.cost["observer_cuda_sync_seconds"] = telemetry.cost.get("observer_cuda_sync_seconds", 0.0) + finished - before
            telemetry.point("cuda_synchronization", started_at_epoch=before, finished_at_epoch=finished, seconds=finished - before)
            telemetry.observe(event, backend=backend, role=loaded.role)
            if plain_event["event"] == "optimizer_end" and plain_event["update"] in replay_values:
                try:
                    comparison = replay_comparison(replay_references[plain_event["update"]],
                        replay_values[plain_event["update"]], plain_event,
                        sampling_rng_hex=bytes(generator.get_state().tolist()).hex(),
                        expected_sampling_rng_hex=replay_rng_hex)
                except ValueError as exc:
                    replay_failure = str(exc)
                    metric({"event": "replay_mismatch", "update": plain_event["update"], "error": replay_failure})
                else:
                    metric(comparison)
                    status["rollback"]["verified_replay_updates"] += 1

        audit_frozen(full=True, phase="before_training")
        # New identity checkpoint proves the migration itself, with zero new
        # updates.  It is never represented as an optimizer step or new rollout.
        save_safe(state, extra, safe_return=True)
        gpu_sample()
        progress("training")
        telemetry.warm_start()
        with loaded.autocast():
            outcome = run_ppo(backend, loaded.role, optimizer, rows, original.stage_config(old), device=loaded.device,
                generator=generator, reference_role="reference", reward_role="reward", state=state, extra_state=extra,
                checkpoint_callback=checkpoint, stop_requested=stop, event_callback=observe)
        telemetry.safe_engine_return(outcome.cursor, outcome.extra_state, outcome.stop_reason)
        status.update(cursor=outcome.cursor,
            **update_accounting(committed, original_observed, outcome.cursor["optimizer_updates"]),
            outcome_stop_reason=outcome.stop_reason)
        for item in outcome.metrics:
            metric({"event": "objective_after_engine_return", **item})
        accumulated = joined_metrics(evidence["metric_values"], outcome.metrics, committed, outcome.cursor["optimizer_updates"])
        status.update(objective_records=len(accumulated), inherited_objective_records=committed,
                      new_objective_records=len(outcome.metrics))
        audit_frozen(full=True, phase="after_training")
        save_safe(outcome.cursor, outcome.extra_state, safe_return=True)
        if outcome.stop_reason == "paused":
            raise InterruptedError("Stopped at a committed continuation boundary")
        if outcome.stop_reason == "budget_exhausted":
            raise TimeoutError("Original absolute budget exhausted")
        optimizer_boundary(outcome.cursor, outcome.extra_state, "engine_final")
        if status["rollback"]["verified_replay_updates"] != len(replay_values):
            raise RuntimeError("Authorized rollback replay was not completely verified")
        status["completion"] = original.verify_completion(StageOutcome(outcome.cursor, outcome.extra_state,
            outcome.stop_reason, accumulated), rows, old)
        changed = [name for name, p in mapping.items() if not torch.equal(initial[name], p.detach().cpu())]
        status["adapter_weights_changed"] = any(".lora_" in name for name in changed)
        status["scalar_heads_changed"] = {head: any(name.startswith(f"scalar_heads.{head}.") for name in changed) for head in loaded.heads}
        if not status["adapter_weights_changed"] or not all(status["scalar_heads_changed"].values()):
            raise RuntimeError("Continuation must perform real adapter and value-head changes")
        validate_request(request)
        boundary(DEADLINE, stop)
        artifact = export_stage(output / "final-artifact", backend, loaded.role, stage="ppo", identity=identity,
                                heads=loaded.heads, completion=status["completion"])
        status["artifact"] = {k: artifact[k] for k in ("artifact_dir", "manifest_path", "manifest_sha256")}
        gpu_sample()
        audit_frozen(full=True, phase="after_export")
        status["telemetry"] = telemetry.finish(cursor=outcome.cursor, completion=status["completion"], checkpoints=status["checkpoints"])
        from .throughput_telemetry_v1 import verify_resumed_training_telemetry
        verify_resumed_training_telemetry(output / "telemetry", expected_manifest_sha256=status["telemetry"]["sha256"],
            expected_identity_sha256=digest(identity), expected_completion=status["completion"], checkpoints=status["checkpoints"])
        boundary(DEADLINE, stop)
        progress("formal_alignment_stage_passed")
        return status
    except BaseException as exc:
        status.update(status="budget_exhausted" if isinstance(exc, TimeoutError) else "paused" if isinstance(exc, InterruptedError) else "failed",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            safe_resume_available=status["latest_safe_checkpoint"] is not None,
            emergency_checkpoint_attempted=False, recovery_policy="No automatic retry")
        raise
    finally:
        status.update(finished_at_epoch=time.time(), elapsed_seconds=time.time() - status["started_at_epoch"])
        try:
            if output.exists():
                write(output / "progress.json", status, terminal=True)
        finally:
            if lock.exists() and read(lock) == owner:
                lock.unlink()


def verify_resumed_guard_log(path, *, request, progress, old, committed):
    """Replay actual strict snapshots and waits without renumbering old steps."""
    from . import alignment_process_guard_r3 as guard_module
    from .process_guard_v2 import ProcessIdentity
    from .recovered_parent_acceptance_r5 import _guard_record
    guard_module.validate_policy(old["process_wait_policy"])
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    owned = progress["process_guard"]["owned_identities"]
    guard_module.validate_background(owned)
    expected_owned = tuple(ProcessIdentity.from_record(p) for p in owned)
    allowed = owned + request["supervisor_processes"] + old["allowed_background_processes"]
    inherited = request["process_wait_spent_seconds"]
    previous_total, previous_at = inherited, progress["attempt_started_at_epoch"]
    results, observations, saved, call = [], [], None, 1
    group = None
    for row in rows:
        if (row.get("call") != call or row.get("deadline_epoch") != DEADLINE
                or type(row.get("optimizer_updates")) is not int or row["optimizer_updates"] < committed
                or type(row.get("at_epoch")) not in (int, float) or not previous_at <= row["at_epoch"] < DEADLINE):
            raise ValueError("Continuation guard call/time/update sequence changed")
        previous_at = row["at_epoch"]
        current_group = (row.get("boundary"), row["optimizer_updates"])
        if group is None:
            group = current_group
        elif group != current_group:
            raise ValueError("Guard call changed its phase or committed update")
        event = row.get("event")
        if event == "observation":
            evidence = row["guard"]
            _guard_record(evidence, owned, allowed,
                {"started_at_epoch": progress["attempt_started_at_epoch"], "deadline_epoch": DEADLINE})
            if any(not guard_module._allowed_ancestry_unchanged(allowed,
                    tuple(ProcessIdentity.from_record(p) for p in observation["snapshot"]))
                    for observation in evidence["observations"] if observation["snapshot_valid"]):
                raise ValueError("Allowed process ancestry changed")
            if (row.get("cycle") != len(observations) + 1
                    or observations and observations[-1]["guard"]["admission_allowed"]
                    or evidence["observations"][-1]["finished_at_epoch"] > row["at_epoch"]):
                raise ValueError("Reordered or post-clear guard observation")
            observations.append({key: row[key] for key in ("cycle", "at_epoch", "guard", "waited_seconds", "total_waited_seconds")})
        elif event == "wait_start":
            if saved is not None or not observations or observations[-1]["guard"]["admission_allowed"]:
                raise ValueError("Wait must follow a classified unknown boundary")
            saved = guard_module.verify_boundary_receipt(row["saved_boundary"], row["optimizer_updates"])
            if saved["kind"] == "before_cuda":
                if group[0] != "before_cuda" or Path(saved["path"]).resolve() != Path(request["output"]).resolve() / "request.json":
                    raise ValueError("Initial wait lacks its exact new request")
            elif saved["kind"] == "optimizer_checkpoint":
                candidates = [c for c in progress["checkpoints"] if c["optimizer_updates"] == row["optimizer_updates"]
                    and str((Path(c["checkpoint_path"]) / "manifest.json").resolve()) == saved["path"]
                    and c["manifest_sha256"] == saved["sha256"]]
                if not candidates or read(saved["path"])["identity"] != progress["identity"]:
                    raise ValueError("Wait was not committed under the new identity at current cursor")
            else:
                raise ValueError("PPO continuation cannot use cache boundary receipts")
        elif event == "result":
            if (row.get("schema_version") != 1 or row.get("kind") != "alignlab-process-guard-v4-deadline-wait"
                    or row.get("admission_allowed") is not True or row.get("reason") != "cleared"
                    or row.get("unknown_processes_allowed") is not False or row.get("training_during_wait") is not False
                    or row.get("inherited_waited_seconds") != inherited or row.get("saved_boundary") != saved
                    or row.get("observations") != observations or not observations
                    or row.get("last_guard") != observations[-1]["guard"]):
                raise ValueError("Continuation guard release is not a proven strict-clear result")
            last = row["last_guard"]
            if (last.get("blockers") or not last["observations"][-1]["snapshot_valid"]
                    or not guard_module._owned_present(expected_owned, tuple(ProcessIdentity.from_record(p) for p in last["final_snapshot"]))):
                raise ValueError("Only fresh clear snapshots release continuation")
            wait, total = row["waited_seconds"], row["total_waited_seconds"]
            if (type(wait) not in (int, float) or type(total) not in (int, float)
                    or not math.isfinite(wait) or not math.isfinite(total) or not 0 <= wait < 259200
                    or not previous_total <= total < 259200
                    or not math.isclose(total - previous_total, wait, abs_tol=1e-6, rel_tol=1e-12)):
                raise ValueError("Inherited wait ledger was reset or changed")
            waiting = observations[0]["guard"]["admission_allowed"] is False
            if (waiting and (saved is None or len(observations) < 2 or wait <= 0)
                    or not waiting and (saved is not None or len(observations) != 1 or wait != 0)):
                raise ValueError("Unknown wait needs a real saved boundary and later clearance")
            last_waited = 0.0
            for i, item in enumerate(observations):
                waited, cumulative = item["waited_seconds"], item["total_waited_seconds"]
                if (type(waited) not in (int, float) or type(cumulative) not in (int, float)
                        or not math.isfinite(waited) or not math.isfinite(cumulative)
                        or i == 0 and (waited != 0 or cumulative != previous_total)
                        or waited < last_waited - 1e-6 or waited > wait + 1e-6
                        or not math.isclose(cumulative, previous_total + waited, rel_tol=1e-12, abs_tol=1e-6)
                        or item["at_epoch"] - observations[0]["at_epoch"] > waited + 1e-6):
                    raise ValueError("Guard observation waiting time was undercounted")
                last_waited = waited
            previous_total = total
            results.append(row)
            observations, saved, group, call = [], None, None, call + 1
        else:
            raise ValueError("Unknown guard evidence event")
    expected = [("before_cuda", committed), ("engine_origin", committed)]
    expected += [("optimizer_complete", i) for i in range(committed + 1, old["workload"]["optimizer_updates"] + 1)]
    expected += [("engine_final", old["workload"]["optimizer_updates"])]
    if (observations or saved is not None or group is not None
            or [(row["boundary"], row["optimizer_updates"]) for row in results] != expected
            or not results or progress["process_guard"]["last"] != results[-1]
            or progress["process_guard"]["total_waited_seconds"] != previous_total):
        raise ValueError("Continuation guard omits or duplicates actual optimizer boundaries")
    return {"verified": True, "inherited_waited_seconds": inherited, "total_waited_seconds": previous_total}


def verify_stage(output, request, known_processes, exit_code):
    """Independent full state/artifact/history audit after the child has exited."""
    validated_output, old, prior = validate_request(request)
    output = Path(output).resolve()
    if exit_code != 0 or output != validated_output:
        raise RuntimeError("Continuation child did not exit successfully at its exact output")
    from .alignment_supervisor_r3 import _bundle, _cpu_payload, _checkpoint_content, NF4
    from .checkpointing import _validate_identity
    from .throughput_telemetry_v1 import verify_resume_evidence, verify_resumed_training_telemetry
    result = read(output / "progress.json")
    target = old["workload"]["optimizer_updates"]
    committed, original_observed = prior["latest_safe_checkpoint"]["optimizer_updates"], prior["actual_optimizer_updates"]
    if isinstance(known_processes, list):
        known = {p["pid"]: p for p in known_processes}
    else:
        known = known_processes
    if (result.get("status") != "formal_alignment_stage_passed" or result.get("request") != request
            or read(output / "request.json") != request or result.get("kind") != KIND or result.get("stage") != "ppo"
            or result.get("formal_training") is not True or result.get("research_complete") is not False
            or result.get("pid") not in known or result.get("actual_optimizer_updates") != target
            or result.get("inherited_optimizer_updates") != committed or result.get("new_optimizer_updates") != target - committed
            or result.get("deadline_epoch") != DEADLINE or result.get("original_started_at_epoch") != STARTED
            or result.get("started_at_epoch") != prior["started_at_epoch"]
            or not prior["finished_at_epoch"] <= result["attempt_started_at_epoch"] <= result["finished_at_epoch"] <= DEADLINE):
        raise ValueError("Missing exact continuation process, budget or completion evidence")
    if any(result.get(k) != v for k, v in update_accounting(committed, original_observed, target).items()):
        raise ValueError("Logical, physical and explicit replay update ledgers disagree")
    owned = result["process_guard"]["owned_identities"]
    if {p["pid"] for p in owned} != set(known) or any(p != known[p["pid"]] for p in owned):
        raise ValueError("Guard child chain differs from externally observed exact identities")
    model = result["model"]
    validate_model_metadata(model, prior["model"])
    if (model.get("device") != "cuda:0" or model.get("quantization") != NF4
            or model.get("quantized_linear_count") != 196 or model.get("parameters_selected", 0) <= 0
            or result.get("sampling_backend") != request["selected_backend"]):
        raise ValueError("Continuation lacks actual same-model CUDA evidence")
    if not result.get("gpu_samples") or any(not any(g.get("index") == 0 for g in sample.get("gpus", []))
            or not all(type(sample.get(k)) in (float, int) and math.isfinite(sample[k]) and sample[k] > 0
                       for k in ("max_allocated_bytes", "max_reserved_bytes")) for sample in result["gpu_samples"]):
        raise ValueError("Missing positive actual GPU memory measurements")
    from .training_data import load_training_partition
    rows, selection, records = original.select_real_rows(load_training_partition(old["audit_dir"], "ppo_prompts", old["audit_manifest_sha256"]), old)
    selection_path = output / "selection.json"
    if read(selection_path) != {"summary": selection, "records": records}:
        raise ValueError("Continuation selection differs from real accepted IDs/tokens")
    selection.update(selection_manifest_path=str(selection_path), selection_manifest_sha256=sha(selection_path))
    if result["selection"] != selection:
        raise ValueError("Continuation progress selection disagrees with actual selected rows")
    request_path = output.parent / (output.name + ".request.json")
    expected_identity = _new_identity(request, prior["identity"], selection_path, request_path)
    identity = result["identity"]
    if identity != expected_identity or read(output / "identity.json") != identity:
        raise ValueError("Execution bridge identity differs from independently rebuilt original lineage")
    _validate_identity(identity)
    expected_imports = {"policy": ("sft", []), "reference": ("sft", []), "reward": ("reward", ["reward"])}
    imports = result.get("imports", {})
    if set(imports) != set(expected_imports):
        raise ValueError("Missing immutable SFT/reference/reward initialization receipts")
    for role, (parent, heads) in expected_imports.items():
        imported = imports[role]
        if (imported.get("stage") != parent or imported.get("target_stage") != "ppo"
                or imported.get("target_role") != role or imported.get("heads") != heads
                or imported.get("target_identity_sha256") != digest(identity)
                or imported.get("manifest_sha256") != old["parents"][parent]["manifest_sha256"]):
            raise ValueError("Imported frozen parent differs from the new execution identity")
    evidence = verify_resume_evidence(request["history"], expected_old_identity_sha256=digest(prior["identity"]))
    if (evidence["cursor"]["optimizer_updates"] != committed
            or evidence["original_observed_optimizer_updates"] != original_observed):
        raise ValueError("Independent restoration and observed physical history disagree")
    restore = result["restore_receipt"]
    if (restore.get("checkpoint_path") != str(Path(request["history"]["checkpoint"]["path"]).resolve())
            or restore.get("manifest_sha256") != request["history"]["checkpoint"]["manifest_sha256"]
            or restore.get("identity_sha256") != digest(prior["identity"])
            or restore.get("committed_optimizer_updates") != committed or restore.get("next_optimizer_update") != committed + 1
            or restore.get("active_rollout_restored") is not (evidence["extra_state"] is not None)
            or restore.get("all_rng_and_adam_restored") is not True):
        raise ValueError("Checkpoint restoration provenance differs")
    log = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    fresh = [row for row in log if row.get("event") == "objective_after_engine_return"]
    live = [row for row in log if row.get("event") == "live_optimizer_end"]
    boundaries = [row for row in log if row.get("event") == "engine_boundary"]
    replay_rows = [row for row in log if row.get("event") == "replay_equivalence"]
    replay_updates = list(range(committed + 1, original_observed + 1))
    expected_rollback = {"authorization": request["history"].get("rollback_authorization"),
        "restored_update": committed, "original_observed_update": original_observed,
        "first_replay_update": committed + 1 if replay_updates else None,
        "last_replay_update": original_observed if replay_updates else None,
        "required_replay_updates": len(replay_updates), "verified_replay_updates": len(replay_updates),
        "prior_final_state_bitwise_verified": False, "preserved_history": True}
    if (result.get("rollback") != expected_rollback
            or [row.get("update") for row in replay_rows] != replay_updates
            or any(row.get("event") == "replay_mismatch" for row in log)):
        raise ValueError("Missing explicit and truthful rollback/replay evidence")
    if replay_updates:
        references = {p["update"]: p for p in evidence["all_points"] if p.get("event") == "optimizer_end"}
        values = {p["update"]: p for p in evidence["replay_reference_metrics"]}
        current_points = [json.loads(line) for line in (output / "telemetry/cost-events.jsonl").read_text(encoding="utf-8").splitlines()]
        current_ends = {p["update"]: p for p in current_points if p.get("event") == "optimizer_end"}
        rng_hex = evidence["extra_state"]["observation_origin"]["rng_after_hex"]
        for comparison in replay_rows:
            update = comparison["update"]
            point = copy.deepcopy(current_ends[update])
            point["responses"] = len(point["response_indices"])
            point["cursor"]["order"] = evidence["cursor"]["order"]
            expected_comparison = replay_comparison(references[update], values[update], point,
                sampling_rng_hex=rng_hex, expected_sampling_rng_hex=rng_hex)
            if {k: v for k, v in comparison.items() if k != "time_epoch"} != expected_comparison:
                raise ValueError("Replayed metrics differ from preserved independently observed data")
    objectives = joined_metrics(evidence["metric_values"], fresh, committed, target)
    numeric = ("update", "responses", "action_tokens", "objective", "gradient_norm")
    if ([{k: row[k] for k in numeric} for row in live] != [{k: row[k] for k in numeric} for row in fresh]
            or [row["cursor"]["optimizer_updates"] for row in boundaries] != list(range(committed + 1, target + 1))
            or result["objective_records"] != target or result["new_objective_records"] != target - committed
            or result["inherited_objective_records"] != committed
            or any(row.get("responses") != 1 or type(row.get("action_tokens")) is not int or not 1 <= row["action_tokens"] <= 256
                   or not all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ("objective", "gradient_norm")) for row in objectives)):
        raise ValueError("Actual new live objectives are not the exact disjoint finite suffix")
    from .stage_engine import StageOutcome
    completion = original.verify_completion(StageOutcome(result["cursor"], None, result["outcome_stop_reason"], objectives), rows, old)
    if (result["completion"] != completion or sum(row["action_tokens"] for row in objectives) != 2 * result["cursor"]["response_tokens"]
            or result.get("adapter_weights_changed") is not True or result.get("scalar_heads_changed") != {"value": True}):
        raise ValueError("Incomplete true optimizer, prompt, token or parameter progress")
    artifact = result["artifact"]
    if Path(artifact["artifact_dir"]).resolve() != output / "final-artifact" or Path(artifact["manifest_path"]).resolve() != output / "final-artifact/manifest.json":
        raise ValueError("Final artifact escaped continuation output")
    saved = _bundle(output / "final-artifact", artifact["manifest_sha256"], {"weights.pt"}, "alignlab-completed-stage-v1")
    if saved["identity"] != identity or saved["completion"] != completion or saved["stage"] != "ppo":
        raise ValueError("Final artifact identity or complete workload differs")
    exported = _cpu_payload(output / "final-artifact/weights.pt", saved["artifacts"]["weights.pt"]["sha256"])
    checkpoints = result["checkpoints"]
    if checkpoints[:len(prior["checkpoints"])] != prior["checkpoints"]:
        raise ValueError("Inherited checkpoint receipts were rewritten")
    new_checkpoints = checkpoints[len(prior["checkpoints"]):]
    expected_updates = {committed, target} | {i for i in range(committed + 1, target + 1) if i % 50 == 0}
    if (not expected_updates <= {r["optimizer_updates"] for r in new_checkpoints}
            or result["latest_safe_checkpoint"] not in new_checkpoints
            or result["latest_safe_checkpoint"]["cursor"] != result["cursor"]):
        raise ValueError("Missing migration, periodic or final complete checkpoint")
    if new_checkpoints[0]["cursor"] != evidence["cursor"]:
        raise ValueError("The first new checkpoint must represent zero additional numerical work")
    original_directory = Path(request["history"]["checkpoint"]["path"])
    original_manifest = read(original_directory / "manifest.json")
    original_payload = _cpu_payload(original_directory / "state.pt", original_manifest["artifacts"]["state.pt"]["sha256"])
    migration_directory = Path(new_checkpoints[0]["checkpoint_path"])
    migration_manifest = read(migration_directory / "manifest.json")
    migration_payload = _cpu_payload(migration_directory / "state.pt", migration_manifest["artifacts"]["state.pt"]["sha256"])
    verify_migration_payload(original_payload, migration_payload)
    keys = set()
    for record in checkpoints:
        directory = Path(record["checkpoint_path"]).resolve()
        update = record["optimizer_updates"]
        prior_record = record in prior["checkpoints"]
        wanted_identity = prior["identity"] if prior_record else identity
        wanted_parent = Path(old["output"]).resolve() if prior_record else output
        key = (str(directory), record["manifest_sha256"])
        if (key in keys or directory.parent != wanted_parent or record["cursor_sha256"] != digest(record["cursor"])
                or directory.name != f"update-{update:06d}-{record['cursor_sha256'][:16]}"
                or record["cursor"]["optimizer_updates"] != update):
            raise ValueError("Invalid or duplicate checkpoint boundary receipt")
        keys.add(key)
        bundle = _bundle(directory, record["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
        if bundle["identity"] != wanted_identity:
            raise ValueError("Checkpoint is bound to the wrong side of the identity bridge")
        _checkpoint_content(directory, bundle, record, final=record == result["latest_safe_checkpoint"], artifact=exported, require_observation=True)
    final_directory = Path(result["latest_safe_checkpoint"]["checkpoint_path"])
    final_manifest = read(final_directory / "manifest.json")
    final_payload = _cpu_payload(final_directory / "state.pt", final_manifest["artifacts"]["state.pt"]["sha256"])
    changed = [name for name, value in final_payload["parameters"].items()
               if not _equal_state(value, original_payload["parameters"][name])]
    if not any(".lora_" in name for name in changed) or not any(name.startswith("scalar_heads.value.") for name in changed):
        raise ValueError("Final actual adapter/value weights did not change from inherited checkpoint")
    for role in ("reference", "reward"):
        if result[role + "_anchor"] != prior[role + "_anchor"]:
            raise ValueError("Frozen CUDA content anchor changed during continuation")
    if result["reward_head_anchor"] != prior["reward_head_anchor"] or [a.get("phase") for a in result["frozen_full_audits"]] != ["before_training", "after_training", "after_export"]:
        raise ValueError("Frozen reference/reward content audit is incomplete")
    verify_resumed_training_telemetry(output / "telemetry", expected_manifest_sha256=result["telemetry"]["sha256"],
        expected_identity_sha256=digest(identity), expected_completion=completion, checkpoints=checkpoints)
    verify_resumed_guard_log(output / "process-guard.jsonl", request=request, progress=result, old=old, committed=committed)
    return result
