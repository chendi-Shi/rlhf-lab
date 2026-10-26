"""Pinned, disposable CUDA replay for the trajectory-preserving PPO switch.

No trial updates are written to the formal stage or accepted as training work.
The same loaded NF4 model is restored from the original complete checkpoint
before every trial and again before returning. Timing is measured, never inferred
from GPU utilization. Importing this module does not load a model or initialize
CUDA. The executing caller must already have stopped the original campaign.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time
import uuid

from .training_telemetry import digest, read, sha, write_new
from .throughput_stage_v1 import ROOT, STARTED, DEADLINE, pin, restore_boundary

KIND = "alignlab-throughput-validation-request-v1"
BACKENDS = {"deferred_scalar": "original", "cached_topology": "cached_topology"}
OWN_SOURCES = {
    "alignment_lab/throughput_validation_v1.py",
    "scripts/run_alignlab_throughput_validation_v1.py",
    "tests_throughput_v1/test_validation_v1.py",
    "alignment_lab/throughput_sampling_v1.py",
    "tests_throughput_v1/test_sampling_v1.py",
}
# Runtime pointers/version counters are observational identity, not numeric state.
# The frozen engine validates their schema and preserves inherited origins. Only
# newly generated origins differ after a real checkpoint restore increments the
# target tensors' version counters; their exact original records remain evidence.
EPHEMERAL_ORIGIN_FIELDS = {"generation_model_versions", "model_versions", "origin_sha256"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_cpu_report(report, sources):
    """Accept only the source-bound CPU runner's actual complete suite result."""
    modules = report.get("modules", [])
    names = {str(name).rsplit(".", 1)[-1] for name in modules}
    required = {Path(name).stem for name in sources
                if name.startswith("tests_throughput_v1/test_") and name.endswith(".py")}
    required.update({"test_stage_v1", "test_telemetry_v1", "test_sampling_v1", "test_validation_v1"})
    _require(report.get("kind") == "alignlab-throughput-cpu-validation-v1"
        and report.get("status") == "passed" and report.get("device") == "cpu"
        and report.get("cuda_initialized") is False
        and type(report.get("tests_run")) is int and report["tests_run"] > 0
        and all(type(report.get(key)) is int and report[key] == 0 for key in ("failures", "errors", "skipped"))
        and report.get("source_before") == sources == report.get("source_after")
        and required <= names,
        "Current full throughput CPU checks, including actual resume tests, must pass before CUDA")
    return True


def validate_rollback_authorization(history, prior, old, throughput_authorization):
    """Authorize only the explicitly accepted 609 -> 630 replay, never invent 630 state."""
    auth = read(pin(history["rollback_authorization"]))
    _require(auth.get("kind") == "alignlab-explicit-rollback-replay-authorization-v1"
        and auth.get("source") == "direct_user_reply_in_this_chat"
        and auth.get("user_selected_answer") == "允许从609恢复并重算21步，验证后继续（推荐）"
        and auth.get("original_plan") == {"path": old["alignment_plan_path"], "sha256": old["alignment_plan_sha256"]}
        and auth.get("throughput_authorization") == throughput_authorization
        and auth.get("original_started_at_epoch") == STARTED and auth.get("original_deadline_epoch") == DEADLINE
        and auth.get("wall_seconds") == 259200
        and [auth.get(key) for key in ("restore_optimizer_updates", "prior_observed_optimizer_updates",
            "replay_first_update", "replay_last_update", "replay_updates")] == [609, 630, 610, 630, 21]
        and all(auth.get(key) is True for key in ("permit_original_backend_continuation",
            "permit_optimized_backend_only_after_actual_cuda_equivalence_and_measured_speedup",
            "preserve_failed_history", "replayed_updates_must_be_separately_accounted",
            "require_new_entry_safe_boundary_save_before_guard_stop", "do_not_change_original_179_sources",
            "no_new_seed_or_data_or_protocol", "no_parallel_gpu_training", "no_deadline_extension")),
        "Explicit exact 609 rollback/replay authority required")
    pin(auth["original_plan"])
    incident = read(pin(auth["incident"]))
    latest = prior["latest_safe_checkpoint"]
    _require(prior["actual_optimizer_updates"] == prior["cursor"]["optimizer_updates"] == 630
        and latest["optimizer_updates"] == latest["cursor"]["optimizer_updates"] == 609
        and incident.get("observed_optimizer_updates") == 630
        and incident.get("last_saved_optimizer_updates") == 609
        and incident.get("campaign_status") == "failed"
        and incident.get("updates_after_last_saved_checkpoint") == 21
        and incident["pins"]["ppo42/progress.json"] == history["prior_progress"]
        and incident["pins"]["ppo42/metrics.jsonl"] == history["metrics"]
        and incident["pins"]["ppo42/telemetry/cost-events.jsonl"] == history["cost_events"]
        and all(incident["checkpoint"].get(key) == history["checkpoint"][key]
                for key in ("path", "manifest_sha256", "sidecar", "binding")),
        "Rollback authorization must bind the actual incident and surviving checkpoint")
    for record in incident["pins"].values():
        pin(record)
    return auth


def source_pins(sources, old_sources):
    from .throughput_stage_v1 import _source_pins
    _source_pins(sources, old_sources)
    _require(OWN_SOURCES <= sources.keys(), "Validation implementation and tests must be frozen too")


def validate_request(request):
    keys = {"schema_version", "kind", "output", "original_started_at_epoch", "original_deadline_epoch",
        "original_request", "history", "source_sha256", "authorization", "cpu_report", "accepted_dpo42",
        "candidate_backends", "paired_rounds", "minimum_speedup_ratio", "max_validation_seconds",
        "process_wait_spent_seconds", "executor_entry"}
    _require(type(request) is dict and set(request) == keys and request["schema_version"] == 1
        and request["kind"] == KIND and request["original_started_at_epoch"] == STARTED
        and request["original_deadline_epoch"] == DEADLINE, "Exact validation request and original budget required")
    _require(type(request["candidate_backends"]) is list and 1 <= len(request["candidate_backends"]) <= 2
        and len(set(request["candidate_backends"])) == len(request["candidate_backends"])
        and set(request["candidate_backends"]) <= BACKENDS.keys(), "Explicit candidate set required")
    _require(type(request["paired_rounds"]) is int and request["paired_rounds"] == 2,
        "Two counterbalanced replay rounds are required")
    _require(type(request["minimum_speedup_ratio"]) in (int, float)
        and math.isfinite(request["minimum_speedup_ratio"]) and request["minimum_speedup_ratio"] >= 1,
        "Finite explicit speed acceptance margin required")
    _require(type(request["max_validation_seconds"]) in (int, float)
        and 0 < request["max_validation_seconds"] <= 7200, "Bounded validation cost required")
    from . import alignment_stage_r3 as original
    from .throughput_stage_v1 import _accepted_dpo42
    old = read(pin(request["original_request"]))
    original.validate_request(old)
    _require(old["stage"] == "ppo" and old["seed"] == 42 and old["deadline_epoch"] == DEADLINE,
        "Only the original R3 PPO42 trajectory is authorized")
    source_pins(request["source_sha256"], old["source_sha256"])
    entry = pin(request["executor_entry"])
    _require(entry.resolve().is_relative_to(ROOT) and request["source_sha256"].get(
        entry.resolve().relative_to(ROOT).as_posix()) == request["executor_entry"]["sha256"],
        "The actual executing entry must be in the frozen source inventory")
    auth = read(pin(request["authorization"]))
    _require(auth.get("kind") == "alignlab_throughput_preserving_trajectory_authorization_v1"
        and auth.get("status") == "authorized_preparation_validation_and_conditional_safe_switch"
        and auth.get("user_selected_answer") == "保留原采样轨迹：验证等价且确实更快后再安全切换（推荐）"
        and auth.get("started_at_epoch") == STARTED and auth.get("deadline_epoch") == DEADLINE
        and auth.get("wall_seconds") == 259200 and auth.get("original_plan_sha256") == old["alignment_plan_sha256"],
        "Concrete trajectory-preserving validation authority required")
    pin({"path": str((ROOT / auth["proposal"]).resolve()), "sha256": auth["proposal_sha256"]})
    validate_cpu_report(read(pin(request["cpu_report"])), request["source_sha256"])
    _accepted_dpo42(request["accepted_dpo42"], old)
    history = request["history"]
    normal_history = {"prior_progress", "metrics", "cost_events", "checkpoint"}
    _require(type(history) is dict and set(history) in (normal_history, normal_history | {"rollback_authorization"}),
        "Complete original checkpoint and history pins required")
    prior = read(pin(history["prior_progress"]))
    _require(prior.get("request") == old and prior.get("status") == "paused"
        and prior.get("safe_resume_available") is True and prior.get("deadline_epoch") == DEADLINE,
        "Validation requires an actually stopped, complete resumable original stage")
    for key in ("metrics", "cost_events"):
        pin(history[key])
    checkpoint = history["checkpoint"]
    _require(type(checkpoint) is dict and set(checkpoint) == {"path", "manifest_sha256", "sidecar", "binding"},
        "Exact final checkpoint schema required")
    latest = prior["latest_safe_checkpoint"]
    _require(checkpoint == {"path": latest["checkpoint_path"], "manifest_sha256": latest["manifest_sha256"],
        "sidecar": latest["telemetry_sidecar"], "binding": latest["telemetry_binding"]},
        "Validation must start at the actual last saved checkpoint")
    if "rollback_authorization" in history:
        validate_rollback_authorization(history, prior, old, request["authorization"])
    else:
        _require(latest["optimizer_updates"] == prior["actual_optimizer_updates"] and latest["cursor"] == prior["cursor"],
            "An unsaved update gap needs separate direct rollback authorization")
    manifest = read(pin({"path": str(Path(checkpoint["path"]) / "manifest.json"),
        "sha256": checkpoint["manifest_sha256"]}))
    _require(manifest.get("identity") == prior["identity"]
        and manifest.get("identity_sha256") == digest(prior["identity"]), "Original checkpoint identity changed")
    for key in ("sidecar", "binding"):
        pin(checkpoint[key])
    _require(type(request["process_wait_spent_seconds"]) in (int, float)
        and math.isfinite(request["process_wait_spent_seconds"])
        and old["process_wait_spent_seconds"] <= request["process_wait_spent_seconds"] < 259200,
        "Inherited wait time cannot be reset")
    output = Path(request["output"])
    _require(output.is_absolute() and output.resolve().is_relative_to(ROOT / "reports")
        and output.resolve() != ROOT / "reports"
        and not any(p.is_symlink() for p in (output, *output.parents)), "Fresh ordinary report directory required")
    return output, old, prior


def state_digest(value):
    """Bitwise tree hash, retaining signed zero, tensor dtype/shape and key type."""
    import torch
    result = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            add(list(tensor.shape))
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif type(item) is dict:
            result.update(b"dict\0")
            # Preserve ordered optimizer and named-parameter bindings.
            for key, child in item.items():
                add(key)
                add(child)
            result.update(b"end\0")
        elif type(item) in (list, tuple):
            result.update(type(item).__name__.encode() + b"\0")
            for child in item:
                add(child)
            result.update(b"end\0")
        elif type(item) is float:
            _require(math.isfinite(item), "Nonfinite validation state")
            result.update(b"float\0" + struct.pack("!d", item))
        elif item is None or type(item) in (str, int, bool):
            result.update(type(item).__name__.encode() + b"\0" + json.dumps(item, ensure_ascii=False).encode() + b"\0")
        else:
            raise TypeError("Unsupported exact-state type: " + type(item).__name__)
    add(value)
    return result.hexdigest()


def numeric_active(active):
    if active is None:
        return None
    value = copy.deepcopy(active)
    if "observation_origin" in value:
        value["observation_origin"] = {key: item for key, item in value["observation_origin"].items()
                                        if key not in EPHEMERAL_ORIGIN_FIELDS}
    return value


def replay_target(cursor, config, rows):
    """The next not-yet-generated full rollout, following any inherited active one."""
    _require(config.rollout_batch == 16 and config.optimizer_batch == 1 and config.epochs == 2,
        "Validation is for the frozen 16-response, two-epoch PPO protocol")
    offset = cursor["offset"]
    _require(offset + 16 <= min(len(rows), config.max_responses),
        "There must be a complete next 16-prompt rollout; do not substitute final/held-out prompts")
    indices = cursor["order"][offset:offset + 16]
    return {"rollout_index": cursor["completed_rollouts"] + int(cursor["active_rollout"]),
            "completed_rollouts": cursor["completed_rollouts"] + int(cursor["active_rollout"]) + 1,
            "record_ids": [rows[index]["record_id"] for index in indices],
            "prompt_ids_sha256": digest([rows[index]["prompt_ids"] for index in indices])}


def replay_segment(backend, role, optimizer, rows, config, *, device, generator, state, extra_state,
                   target, checkpoint_callback=lambda cursor, active: None, stop_requested=lambda: False,
                   boundary_callback=lambda cursor, active, name: None, event_sink=lambda event: None,
                   synchronize=lambda: None, clock=time.perf_counter):
    """Run the unchanged observed PPO engine only to a full next-rollout boundary.

    This reusable low-level primitive is also exercised with a real tiny Qwen on
    CPU. Its returned state digests include every optimizer boundary. The caller
    must restore before each replay, enforce admission and discard trial work.
    """
    from .checkpointing import _capture_rng
    from .stage_engine import parameter_mapping
    from .stage_engine_observed import run_ppo, event_to_plain
    mapping = parameter_mapping(backend, role, heads=("value",))
    completed, boundaries, collected = False, [], []
    sampling_started, sampling_seconds = None, None
    latest = {"cursor": copy.deepcopy(state), "active": copy.deepcopy(extra_state)}
    boundary_callback(latest["cursor"], latest["active"], "validation_engine_origin")
    synchronize()
    started = clock()
    def checkpoint(cursor, active):
        nonlocal completed
        latest.update(cursor=copy.deepcopy(cursor), active=copy.deepcopy(active))
        synchronize()
        boundaries.append({"update": cursor["optimizer_updates"], "cursor_sha256": digest(cursor),
            "numeric_active_sha256": state_digest(numeric_active(active)),
            "parameters_sha256": state_digest(mapping), "adam_sha256": state_digest(optimizer.state_dict()),
            "all_rng_sha256": state_digest(_capture_rng({"sampling": generator}))})
        checkpoint_callback(cursor, active)
        completed = not cursor["active_rollout"] and cursor["completed_rollouts"] == target["completed_rollouts"]
        boundary_callback(cursor, active, "validation_optimizer_complete")
    def observe(event):
        nonlocal sampling_started, sampling_seconds
        plain = event_to_plain(event)
        name = plain["event"]
        if name == "sampling_start" and plain["rollout_index"] == target["rollout_index"]:
            # A new sample is admitted only at an already complete optimizer boundary.
            boundary_callback(latest["cursor"], latest["active"], "validation_before_new_sampling")
            synchronize()
            sampling_started = clock()
        if name == "sampling_end" and plain["rollout_index"] == target["rollout_index"]:
            synchronize()
            sampling_seconds = clock() - sampling_started
        if name == "rollout_collected":
            _require(plain["rollout_index"] == target["rollout_index"] and plain["record_ids"] == target["record_ids"],
                "Replay sampled an unexpected rollout or changed prompt order")
            evidence = {key: plain[key] for key in ("responses", "record_ids", "advantages", "returns",
                "reference_logprobs", "reward_scores", "old_logprobs", "old_values", "action_mask", "rollout_metadata")}
            evidence["sampling_rng_before"] = plain["observation_origin"]["rng_before_hex"]
            evidence["sampling_rng_after"] = plain["observation_origin"]["rng_after_hex"]
            collected.append({"rollout_index": plain["rollout_index"], "sha256": state_digest(evidence), "evidence": evidence})
        event_sink(plain)
    outcome = run_ppo(backend, role, optimizer, rows, config, device=device, generator=generator,
        reference_role="reference", reward_role="reward", state=state, extra_state=extra_state,
        checkpoint_callback=checkpoint, stop_requested=lambda: completed or stop_requested(), event_callback=observe)
    synchronize()
    elapsed = clock() - started
    _require(completed and outcome.stop_reason == "paused" and outcome.extra_state is None
        and len(collected) == 1 and len(outcome.metrics) == len(boundaries) > 0,
        "Validation did not complete inherited active work plus the next complete rollout")
    _require(type(sampling_seconds) in (float, int) and 0 < sampling_seconds <= elapsed,
        "Actual synchronized next-rollout sampling time missing")
    return {"wall_seconds": elapsed, "sampling_seconds": sampling_seconds, "boundaries": boundaries,
        "rollouts": collected, "metrics": list(outcome.metrics), "final_cursor": outcome.cursor,
        "final_parameters_sha256": state_digest(mapping), "final_adam_sha256": state_digest(optimizer.state_dict()),
        "final_all_rng_sha256": state_digest(_capture_rng({"sampling": generator})),
        "trial_optimizer_updates": len(outcome.metrics), "formal_optimizer_updates_committed": 0}


def equivalence_projection(result):
    return {key: result[key] for key in ("boundaries", "rollouts", "metrics", "final_cursor",
        "final_parameters_sha256", "final_adam_sha256", "final_all_rng_sha256", "trial_optimizer_updates")}


def decide(trials, candidates, minimum_speedup_ratio):
    baseline = [trial for trial in trials if trial["backend"] == "original"]
    _require(len(baseline) == 2, "Both counterbalanced original trials required")
    anchor = state_digest(equivalence_projection(baseline[0]["result"]))
    baseline_equal = all(state_digest(equivalence_projection(t["result"])) == anchor for t in baseline)
    base_wall = statistics.median(t["result"]["wall_seconds"] for t in baseline)
    scores = {}
    for name in candidates:
        rows = [trial for trial in trials if trial["backend"] == name]
        _require(len(rows) == 2, "Both counterbalanced candidate trials required")
        wall = statistics.median(t["result"]["wall_seconds"] for t in rows)
        equal = baseline_equal and all(state_digest(equivalence_projection(t["result"])) == anchor for t in rows)
        uncontaminated = all(t["result"].get("process_wait_seconds", 0) == 0 for t in [*baseline, *rows])
        ratio = base_wall / wall
        scores[name] = {"exactly_equivalent": equal, "baseline_wall_seconds": base_wall,
            "candidate_wall_seconds": wall, "speedup_ratio": ratio,
            "passes": equal and uncontaminated and wall < base_wall and ratio >= minimum_speedup_ratio,
            "timings_without_process_wait": uncontaminated,
            "sampling_seconds": statistics.median(t["result"]["sampling_seconds"] for t in rows)}
    passed = [name for name in candidates if scores[name]["passes"]]
    winner = min(passed, key=lambda name: scores[name]["candidate_wall_seconds"]) if passed else "original"
    return {"selected_backend": winner, "scores": scores, "baseline_repeatability_passed": baseline_equal}


def _owned(snapshot, request, request_path, request_sha256):
    from .process_guard_v2 import ProcessIdentity
    from .study_supervisor import _command
    by_pid = {item["pid"]: item for item in snapshot}
    current, result, seen = by_pid.get(os.getpid()), [], set()
    entry = Path(request["executor_entry"]["path"]).name
    while current is not None:
        identity = ProcessIdentity.from_record(current)
        _require(identity.pid not in seen, "Cyclic validation process ancestry")
        seen.add(identity.pid)
        if (not _command(current, entry, request_path) or request_sha256 not in identity.command_line
                or "--request-sha256" not in identity.command_line or "--execute" not in identity.command_line):
            break
        result.append(current)
        current = by_pid.get(identity.parent_pid)
    _require(result and result[0]["pid"] == os.getpid(), "Validator must execute the exact pinned entry/request/SHA")
    return result


def checked_guard_boundary(guard, *, name, update, save):
    """Even a hard guard rejection must preserve the completed trial boundary.

    This is disposable validation state, never a formal training commit. A
    source-drift checkpoint may fail its own frozen-identity validation; that
    failure is reported and no checkpoint identity is invented to work around it.
    """
    saved = None
    def commit_once():
        nonlocal saved
        if saved is None:
            saved = save()
        return saved
    try:
        answer = guard.check(boundary=name, optimizer_updates=update, save_boundary=commit_once)
    except BaseException:
        commit_once()
        raise
    if not answer["admission_allowed"]:
        commit_once()
        raise InterruptedError("Validation strict boundary denied after saving: " + answer["reason"])
    return answer


def run_throughput_validation(request, *, execute=False, request_path=None, request_sha256=None):
    """Public same-process API. The entry command must bind its exact request SHA."""
    output, old, prior = validate_request(request)
    if not execute:
        return {"status": "plan_only", "real_cuda": False, "formal_training": False}
    _require(request_path is not None and request_sha256 is not None, "Executing request path/SHA required")
    request_path = pin({"path": str(Path(request_path).resolve()), "sha256": request_sha256})
    _require(read(request_path) == request and not output.exists(), "Fresh exact request/output required")
    from .capacity_stage import boundary, _head_anchor
    from .alignment_process_guard_r3 import GuardSession, strict_admission
    from .alignment_supervisor_r3 import JsonWriter
    from .probe_dispatch import _System
    from .gpu_probe import resources
    system = _System()
    snapshot = system.processes()
    owned = _owned(snapshot, request, request_path, request_sha256)
    original_identities = prior["process_guard"]["owned_identities"] + old["supervisor_processes"]
    _require(not any(item in snapshot for item in original_identities), "Original training processes must all exit before validation")
    started = time.time()
    validation_deadline = min(DEADLINE, started + request["max_validation_seconds"])
    stop = lambda: (output / "STOP").exists() or Path(str(request_path) + ".stop").exists() or time.time() >= validation_deadline
    boundary(DEADLINE, stop)
    allowed = old["allowed_background_processes"]
    admission = strict_admission(system.processes, owned_identities=owned, allowed_identities=allowed, deadline_epoch=DEADLINE)
    _require(admission["caller_admission_allowed"], "Initial strict resource/process admission must pass before any CUDA load")
    _require(not resources()["blockers"], "CUDA resources are not available for validation")
    writer = JsonWriter(DEADLINE, stop)
    owner = {"pid": os.getpid(), "output": str(output), "token": uuid.uuid4().hex, "kind": KIND}
    lock = ROOT / "reports/.alignlab-gpu.lock"
    writer(lock, owner, exclusive=True)
    guard_events, evidence_pins = [], {"cpu_report": request["cpu_report"]}
    baseline = candidate = loaded = optimizer = generator = mapping = None
    restored_origin = None
    validation_boundary = None
    def unchanged():
        try:
            source_pins(request["source_sha256"], old["source_sha256"])
            return sha(request_path) == request_sha256 and sha(pin(request["authorization"])) == request["authorization"]["sha256"]
        except (OSError, ValueError):
            return False
    def guard_event(event):
        guard_events.append(event)
        with (output / "process-guard.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
    guard = GuardSession(system.processes, owned_identities=owned, allowed_identities=allowed,
        deadline_epoch=DEADLINE, policy=old["process_wait_policy"], spent_seconds=request["process_wait_spent_seconds"],
        stop_requested=stop, identity_unchanged=unchanged, on_event=guard_event)
    def require_guard(name, cursor=None, active=None):
        if cursor is None and validation_boundary is not None:
            cursor, active = validation_boundary
        update = prior["latest_safe_checkpoint"]["optimizer_updates"] if cursor is None else cursor["optimizer_updates"]
        def save():
            if cursor is None:
                path = output / "request.json"
                kind = "before_cuda"
            else:
                from .checkpointing import save_checkpoint
                path = output / ("disposable-wait-" + uuid.uuid4().hex)
                record = save_checkpoint(path, identity=prior["identity"], parameters=mapping, optimizer=optimizer,
                    cursor=cursor, generators={"sampling": generator}, extra_state=active)
                path = Path(record["checkpoint_path"]) / "manifest.json"
                kind = "optimizer_checkpoint"
            return {"complete": True, "optimizer_updates": update, "kind": kind,
                    "path": str(path.resolve()), "sha256": sha(path)}
        checked_guard_boundary(guard, name=name, update=update, save=save)
        boundary(DEADLINE, stop)
    try:
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "request.json", request)
        require_guard("validation_before_cpu_history_audit")
        from .throughput_telemetry_v1 import verify_resume_evidence
        evidence = verify_resume_evidence(request["history"], expected_old_identity_sha256=digest(prior["identity"]))
        _require(evidence["cursor"] == prior["latest_safe_checkpoint"]["cursor"],
            "Independent history/cursor validation failed")
        from .training_data import load_training_partition
        from . import alignment_stage_r3 as original
        partition = load_training_partition(old["audit_dir"], "ppo_prompts", old["audit_manifest_sha256"])
        rows, selection, records = original.select_real_rows(partition, old)
        _require(read(Path(old["output"]) / "selection.json") == {"summary": selection, "records": records},
            "Actual original real prompt selection/order changed")
        config = original.stage_config(old)
        target = replay_target(evidence["cursor"], config, rows)
        evidence_pins["selection"] = write_new(output / "validation-selection.json", target)
        validate_request(request)
        require_guard("validation_before_cuda")
        _require(not resources()["blockers"], "Resources changed before CUDA")
        import torch
        from .model_loader import load_quantized_stage, ModelLoadConfig, QUANTIZATION
        from .stage_artifacts import import_stage
        from .stage_engine import parameter_mapping
        from .gpu_reference import capture_frozen_reference
        from .checkpointing import _capture_rng
        from .throughput_sampling_v1 import make_sampling_backend
        torch.cuda.set_device(0)
        loaded = load_quantized_stage(old["model_snapshot"], old["model_manifest_sha256"], old["tokenizer_dir"],
            ModelLoadConfig("ppo", seed=old["seed"]))
        _require(loaded.device == torch.device("cuda:0") and digest(loaded.metadata) == digest(prior["model"]),
            "Actual CUDA model/metadata differs from original stage")
        baseline = loaded.backend
        for parent, role, heads in (("sft", loaded.role, ()), ("sft", "reference", ()), ("reward", "reward", ("reward",))):
            record = old["parents"][parent]
            import_stage(record["artifact_dir"], baseline, role, expected_manifest_sha256=record["manifest_sha256"],
                expected_identity=prior["identity"], heads=heads)
        files = {name: Path(record["path"]) for name, record in prior["identity"]["inputs"].items()}
        receipts = {}
        for role, parent in (("reference", "sft"), ("reward", "reward")):
            artifact = Path(old["parents"][parent]["artifact_dir"])
            receipts[role] = capture_frozen_reference(baseline, role, artifact_files={
                "base_files": {k: v for k, v in files.items() if k.startswith("base/")},
                "tokenizer_files": {k: v for k, v in files.items() if k.startswith("tokenizer/")},
                "source_files": {k: v for k, v in files.items() if k.startswith("source/")},
                "adapter_files": {"manifest": artifact / "manifest.json", "weights": artifact / "weights.pt"},
                "dataset_manifest": Path(old["audit_dir"]) / "manifest.json"},
                accepted_data_manifest_sha256=old["audit_manifest_sha256"], precision=original.PRECISION,
                quantization=QUANTIZATION)
            _require(receipts[role].anchor == prior[role + "_anchor"], "Frozen loaded tensors differ from original")
        head_anchor = _head_anchor(baseline._heads["reward"])
        _require(head_anchor == prior["reward_head_anchor"], "Frozen reward head changed")
        def audit_frozen(full):
            for receipt in receipts.values():
                receipt.verify(full=full)
            _require(_head_anchor(baseline._heads["reward"]) == head_anchor, "Reward head changed during replay")
        owners = {"original": baseline}
        owners.update({name: make_sampling_backend(baseline, guard_mode=BACKENDS[name]) for name in request["candidate_backends"]})
        mapping = parameter_mapping(baseline, loaded.role, heads=loaded.heads)
        optimizer = torch.optim.AdamW(mapping.values(), lr=1e-6)
        generator = torch.Generator(device=loaded.device).manual_seed(old["seed"])
        checkpoint = request["history"]["checkpoint"]
        def restore():
            nonlocal validation_boundary
            restored = restore_boundary(checkpoint, prior["identity"], parameters=mapping, optimizer=optimizer, generator=generator)
            _require(restored["cursor"] == evidence["cursor"]
                and state_digest(restored["extra_state"]) == state_digest(evidence["extra_state"]),
                "Transactional restoration changed original cursor/active rollout")
            validation_boundary = (copy.deepcopy(restored["cursor"]), copy.deepcopy(restored["extra_state"]))
            return restored
        restored_origin = restore()
        origin_state = {"parameters": state_digest(mapping), "adam": state_digest(optimizer.state_dict()),
                        "rng": state_digest(_capture_rng({"sampling": generator}))}
        evidence_pins["restoration"] = write_new(output / "restoration.json", {**origin_state,
            "checkpoint": checkpoint, "all_rng_and_adam_restored": True,
            "restored_optimizer_updates": evidence["cursor"]["optimizer_updates"],
            "prior_observed_optimizer_updates": prior["actual_optimizer_updates"],
            "missing_later_checkpoint_state_reconstructed_or_proven": False,
            "active_rollout_sha256": state_digest(restored_origin["extra_state"])})
        trials = []
        order = ["original", *request["candidate_backends"]]
        for round_index in range(2):
            for name in (order if round_index == 0 else list(reversed(order))):
                require_guard("validation_before_restore")
                restored = restore()
                audit_frozen(True)
                _require(origin_state == {"parameters": state_digest(mapping), "adam": state_digest(optimizer.state_dict()),
                    "rng": state_digest(_capture_rng({"sampling": generator}))}, "Trial did not start from exact same full state")
                torch.cuda.synchronize(loaded.device)
                torch.cuda.reset_peak_memory_stats(loaded.device)
                trial_id = f"round-{round_index}-{name}"
                def boundary_callback(cursor, active, phase):
                    nonlocal validation_boundary
                    validation_boundary = (copy.deepcopy(cursor), copy.deepcopy(active))
                    audit_frozen(False)
                    require_guard(phase, cursor, active)
                wait_before = guard.budget.spent_seconds
                with loaded.autocast():
                    result = replay_segment(owners[name], loaded.role, optimizer, rows, config,
                        device=loaded.device, generator=generator, state=restored["cursor"], extra_state=restored["extra_state"],
                        target=target, stop_requested=stop, boundary_callback=boundary_callback,
                        synchronize=lambda: torch.cuda.synchronize(loaded.device))
                audit_frozen(True)
                result["process_wait_seconds"] = guard.budget.spent_seconds - wait_before
                if "rollback_authorization" in request["history"]:
                    replay_reference = evidence.get("replay_reference_metrics")
                    _require(type(replay_reference) is list and [item["update"] for item in replay_reference]
                        == list(range(610, 631)), "Pinned historical replay objectives 610..630 missing")
                    actual_replay = [item for item in result["metrics"] if 610 <= item["update"] <= 630]
                    result["historical_replay_metrics_equal"] = state_digest(actual_replay) == state_digest(replay_reference)
                    _require(result["historical_replay_metrics_equal"], "Recomputed 610..630 objectives/gradients differ from historical events")
                result["max_allocated_bytes"] = torch.cuda.max_memory_allocated(loaded.device)
                result["max_reserved_bytes"] = torch.cuda.max_memory_reserved(loaded.device)
                trial = {"backend": name, "round": round_index, "result": result}
                evidence_pins[trial_id] = write_new(output / (trial_id + ".json"), trial)
                trials.append(trial)
        decision = decide(trials, request["candidate_backends"], request["minimum_speedup_ratio"])
        restore()
        audit_frozen(True)
        _require(origin_state == {"parameters": state_digest(mapping), "adam": state_digest(optimizer.state_dict()),
            "rng": state_digest(_capture_rng({"sampling": generator}))}, "Final rollback did not restore full original state")
        require_guard("validation_final")
        validate_request(request)
        evidence_pins["decision"] = write_new(output / "decision.json", decision)
        guard_path = output / "process-guard.jsonl"
        evidence_pins["guard"] = {"path": str(guard_path), "sha256": sha(guard_path)}
        winner = decision["selected_backend"]
        report = {"kind": "alignlab-throughput-validation-v1" if winner != "original" else "alignlab-original-continuation-validation-v1",
            "status": "validated", "decision": "optimized_backend_accepted" if winner != "original" else "fallback_original",
            "source_sha256": request["source_sha256"], "authorization_sha256": request["authorization"]["sha256"],
            "original_plan_sha256": old["alignment_plan_sha256"], "original_request": request["original_request"],
            "history_sha256": digest(request["history"]), "validation_request": {"path": str(request_path), "sha256": request_sha256},
            "cpu_resume_equivalence_passed": True, "selected_backend": winner,
            "restored_optimizer_updates": evidence["cursor"]["optimizer_updates"],
            "prior_observed_optimizer_updates": prior["actual_optimizer_updates"],
            "rollback_authorization": request["history"].get("rollback_authorization"),
            "historical_unsaved_updates_recomputed_in_validation":
                prior["actual_optimizer_updates"] - evidence["cursor"]["optimizer_updates"],
            "missing_later_checkpoint_state_reconstructed_or_proven": False,
            "benchmark_did_not_commit_training_updates": True, "formal_training": False,
            "formal_optimizer_updates_committed": 0, "trial_updates_discarded_and_original_state_restored": True,
            "real_cuda": True, "device": str(loaded.device), "original_implementation_unchanged": True,
            "cuda_sampling_equivalence_passed": winner != "original", "cuda_resume_equivalence_passed": winner != "original",
            "evidence_pins": evidence_pins, "next_rollout_prompt_count": 16, "counterbalanced_rounds": 2,
            "comparison_scope": "all sampling stats, frozen scores, per-update weights/Adam/all RNG/cursor, inherited active plus next full rollout",
            "ephemeral_origin_fields_excluded_from_numeric_comparison": sorted(EPHEMERAL_ORIGIN_FIELDS),
            "original_deadline_epoch": DEADLINE, "started_at_epoch": started, "finished_at_epoch": time.time(),
            "validation_wall_seconds": time.time() - started, "process_wait_spent_seconds": guard.budget.spent_seconds,
            "speed_acceptance_margin": request["minimum_speedup_ratio"], "scores": decision["scores"],
            "timing_scope": "synchronized replay engine wall including identical validation checks, excluding restore/load/full final audits",
            "switch_cost_payback_gate": "not_proved_here; caller must include validation/load/switch cost before adopting"}
        if winner != "original":
            score = decision["scores"][winner]
            report.update({key: score[key] for key in ("baseline_wall_seconds", "candidate_wall_seconds", "speedup_ratio")})
        boundary(DEADLINE, stop)
        write_new(output / "report.json", report)
        return report
    except BaseException as error:
        # Cleanup restores only the already loaded private validation model. It
        # never resumes training or creates a replacement formal checkpoint.
        rollback = None
        if restored_origin is not None:
            try:
                restore_boundary(request["history"]["checkpoint"], prior["identity"],
                    parameters=mapping, optimizer=optimizer, generator=generator)
                rollback = True
            except BaseException as rollback_error:
                rollback = type(rollback_error).__name__ + ": " + str(rollback_error)
        if output.is_dir() and not (output / "failure.json").exists():
            write_new(output / "failure.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                "formal_training": False, "formal_optimizer_updates_committed": 0,
                "private_validation_state_rollback": rollback, "original_deadline_epoch": DEADLINE,
                "source_sha256": request["source_sha256"], "history_sha256": digest(request["history"]),
                "finished_at_epoch": time.time()})
        raise
    finally:
        if lock.is_file() and read(lock) == owner:
            lock.unlink()

