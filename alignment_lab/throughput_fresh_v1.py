"""Fresh DPO/PPO43/44 with explicit implementation and safe-stop bindings.

The numerical stage body below is an auditable copy of R3's run_alignment_stage;
all original validators/helpers remain unchanged. No module-global replacement,
implicit resume, historical artifact edits, or unvalidated backend selection.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
import uuid

from . import alignment_stage_r3 as original
from . import alignment_supervisor_r3 as legacy_supervisor
from . import throughput_stage_v1 as resume
from .alignment_stage_r3 import (
    ROOT, KIND, PAD, PRECISION, EXECUTION_REVISION, TELEMETRY_PROTOCOL,
    JsonWriter, TrainingTelemetry, BoundaryCheckpoints, parent_stage, boundary,
    sha, read_json, digest, select_real_rows, stage_config, scoring_config,
    bind_complete_cache, verify_completion, _head_anchor,
)

WRAPPER_KIND = "alignlab-throughput-fresh-v1"
REVISION = "throughput-fresh-v1"
NEW_SOURCES = {"alignment_lab/throughput_fresh_v1.py", "scripts/run_alignlab_throughput_fresh_v1.py",
               "tests_throughput_v1/test_fresh_v1.py"}
WRAPPER_KEYS = {"schema_version", "kind", "legacy_request", "source_sha256", "plan",
    "authorization", "validation_report", "original_request", "history", "selected_backend",
    "original_started_at_epoch", "original_deadline_epoch"}


def make_request(inner, plan_authorized, plan_pin):
    """Build, without executing, the exact wrapper consumed by the new CLI."""
    if inner.get("stage") not in ("dpo", "ppo") or inner.get("seed") not in (43, 44):
        raise ValueError("Fresh throughput execution is limited to DPO/PPO43/44")
    inner = copy.deepcopy(inner)
    wrapper = {"schema_version": 1, "kind": WRAPPER_KIND, "legacy_request": inner,
        "source_sha256": copy.deepcopy(plan_authorized["source_sha256"]), "plan": copy.deepcopy(plan_pin),
        **{key: copy.deepcopy(plan_authorized[key]) for key in (
            "authorization", "validation_report", "original_request", "history", "selected_backend",
            "original_started_at_epoch", "original_deadline_epoch")}}
    wrapper["selected_backend"] = "original" if inner["stage"] == "dpo" else plan_authorized["future_ppo_backend"]
    return wrapper


def validate_request(wrapper):
    if (type(wrapper) is not dict or set(wrapper) != WRAPPER_KEYS
            or type(wrapper["schema_version"]) is not int or wrapper["schema_version"] != 1
            or wrapper["kind"] != WRAPPER_KIND or wrapper["selected_backend"] not in resume.BACKENDS
            or wrapper["original_started_at_epoch"] != resume.STARTED
            or wrapper["original_deadline_epoch"] != resume.DEADLINE):
        raise ValueError("Exact fresh-throughput wrapper and original budget required")
    inner = wrapper["legacy_request"]
    if type(inner) is not dict or inner.get("stage") not in ("dpo", "ppo") or inner.get("seed") not in (43, 44):
        raise ValueError("Only fresh DPO/PPO43/44 may use this wrapper")
    output = original.validate_request(inner)
    old = read_json(resume.pin(wrapper["original_request"]))
    original.validate_request(old)
    if old["stage"] != "ppo" or old["seed"] != 42:
        raise ValueError("Throughput validation must belong to the original PPO42 request")
    if inner["source_sha256"] != old["source_sha256"] or not NEW_SOURCES <= wrapper["source_sha256"].keys():
        raise ValueError("Fresh runner, CLI and tests must belong to the new execution inventory")
    resume._source_pins(wrapper["source_sha256"], old["source_sha256"])
    plan = read_json(resume.pin(wrapper["plan"]))
    resume.validate_execution_authority({**wrapper, "selected_backend": plan["selected_backend"]}, old)
    for key in ("source_sha256", "authorization", "validation_report", "original_request", "history",
                "original_started_at_epoch", "original_deadline_epoch"):
        if plan.get(key) != wrapper[key]:
            raise ValueError("Fresh execution differs from the authorized new plan: " + key)
    expected_backend = "original" if inner["stage"] == "dpo" else plan.get("future_ppo_backend")
    if (expected_backend != wrapper["selected_backend"]
            or plan.get("original_plan") != {"path": inner["alignment_plan_path"], "sha256": inner["alignment_plan_sha256"]}
            or output != Path(plan["output"]).resolve() / f"{inner['stage']}{inner['seed']}"
            or output.is_relative_to(Path(old["output"]).resolve())
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("Fresh output, original plan or future backend differs")
    from .throughput_supervisor_v1 import _cpu_proof
    _cpu_proof(plan)
    for key in ("deadline_epoch", "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
                "audit_manifest_sha256", "parent_plan_path", "parent_plan_sha256", "parent_acceptance_path",
                "parent_acceptance_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256",
                "process_wait_policy", "allowed_background_processes",
                "continuation", "repair_attempt", "alignment_plan_path", "alignment_plan_sha256"):
        if inner[key] != old[key]:
            raise ValueError("Fresh branch changed registered numerical work or lineage: " + key)
    from .alignment_supervisor import workload
    original_plan = read_json(resume.pin(plan["original_plan"]))
    expected_parents = {key: old["parents"][key] for key in (("sft",) if inner["stage"] == "dpo" else ("sft", "reward"))}
    if inner["workload"] != workload(inner["stage"], original_plan["ppo_rows"]) or inner["parents"] != expected_parents:
        raise ValueError("Fresh stage changed its registered workload or parent lineage")
    if inner["stage"] == "dpo":
        accepted = read_json(resume.pin(plan["accepted_dpo42"]["progress"]))
        expected_cache = {"mode": "reuse", **{k: accepted["reference_cache"][k]
                          for k in ("path", "manifest_sha256", "identity_sha256")}}
        if inner["reference_cache"] != expected_cache:
            raise ValueError("Fresh DPO must reuse the exact original accepted DPO42 cache")
    if (inner["deadline_epoch"] != resume.DEADLINE
            or inner["process_wait_spent_seconds"] < plan["process_wait_spent_seconds"]):
        raise ValueError("Fresh branch reset the shared deadline or wait ledger")
    return output, inner


def _wrapper_path(wrapper):
    output = Path(wrapper["legacy_request"]["output"])
    return output.parent / (output.name + ".request.json")


def _wrapper_inputs(wrapper, output):
    """Extra implementation/proof inputs, in addition to every R3 input."""
    request_path = _wrapper_path(wrapper)
    if read_json(request_path) != wrapper:
        raise ValueError("Frozen supervisor wrapper differs from the executing request")
    plan = read_json(resume.pin(wrapper["plan"]))
    pins = {"throughput/plan": wrapper["plan"], "throughput/authorization": wrapper["authorization"],
            "throughput/validation_report": wrapper["validation_report"], "throughput/cpu_report": plan["cpu_report"],
            "throughput/original_request": wrapper["original_request"]}
    for name in ("prior_progress", "metrics", "cost_events"):
        pins["throughput/history/" + name] = wrapper["history"][name]
    checkpoint = wrapper["history"]["checkpoint"]
    pins["throughput/history/checkpoint_manifest"] = {"path": str(Path(checkpoint["path"]) / "manifest.json"),
                                                     "sha256": checkpoint["manifest_sha256"]}
    pins["throughput/history/sidecar"] = checkpoint["sidecar"]
    pins["throughput/history/binding"] = checkpoint["binding"]
    if "rollback_authorization" in wrapper["history"]:
        pins["throughput/rollback_authorization"] = wrapper["history"]["rollback_authorization"]
    files = {name: resume.pin(pin) for name, pin in pins.items()}
    files.update({"throughput/source/" + name: ROOT / name for name in wrapper["source_sha256"]})
    files["throughput/wrapper_request"] = request_path
    files["throughput/wrapper_snapshot"] = Path(output) / "wrapper.request.json"
    return files


def _marker(wrapper):
    return {"revision": REVISION, "selected_backend": wrapper["selected_backend"],
            "wrapper_sha256": sha(_wrapper_path(wrapper)), "plan": wrapper["plan"],
            "authorization": wrapper["authorization"], "validation_report": wrapper["validation_report"]}


def _adopt_backend(loaded, wrapper):
    if wrapper["legacy_request"]["stage"] == "dpo" or wrapper["selected_backend"] == "original":
        if wrapper["legacy_request"]["stage"] == "dpo" and wrapper["selected_backend"] != "original":
            raise ValueError("DPO has no changed sampling backend")
        return loaded.backend
    from .throughput_sampling_v1 import make_sampling_backend
    mode = "original" if wrapper["selected_backend"] == "deferred_scalar" else wrapper["selected_backend"]
    loaded.backend = make_sampling_backend(loaded.backend, guard_mode=mode)
    return loaded.backend


def _cache_source_files(files):
    """Frozen R3 cache identity excludes explicitly separate execution inputs."""
    return {key: value for key, value in files.items() if key.startswith("source/")}


def _owned(snapshot, request, request_path, request_sha256, *, pid=None):
    """Exact new-CLI ancestry, including the pinned wrapper hash."""
    from .process_guard_v2 import ProcessIdentity
    from .study_supervisor import _command
    pid = os.getpid() if pid is None else pid
    supervisors = {p["pid"]: ProcessIdentity.from_record(p) for p in request["supervisor_processes"]}
    if not supervisors:
        raise ValueError("Pinned active supervisor identities required")
    by_pid = {p["pid"]: p for p in snapshot}
    current, result, seen = by_pid.get(pid), [], set()
    while current is not None:
        identity = ProcessIdentity.from_record(current)
        if identity.pid in seen:
            raise RuntimeError("Cyclic child ancestry")
        seen.add(identity.pid)
        if identity.pid in supervisors:
            if identity != supervisors[identity.pid] or not result:
                raise RuntimeError("Fresh stage supervisor identity changed")
            return result
        command = current["command_line"]
        if (not _command(current, "run_alignlab_throughput_fresh_v1.py", request_path)
                or request_sha256 not in command or "--request-sha256" not in command):
            raise RuntimeError("Unverified fresh-throughput child/wrapper command")
        result.append(current)
        current = by_pid.get(identity.parent_pid)
    raise RuntimeError("Fresh stage must descend from its exact active supervisor")


def verify_stage(output, request, known, code):
    """Run unchanged complete R3 acceptance, then check the wrapper proof chain."""
    expected_output, inner = validate_request(request)
    output = Path(output).resolve()
    if output != expected_output:
        raise ValueError("Fresh stage output differs from wrapper")
    result = legacy_supervisor._verify_stage(output, inner, known, code)
    if read_json(output / "wrapper.request.json") != request or read_json(_wrapper_path(request)) != request:
        raise ValueError("Fresh wrapper evidence changed")
    identity = result["identity"]
    marker = _marker(request)
    if result.get("throughput") != marker or identity["config"].get("throughput") != marker:
        raise ValueError("Fresh stage identity lost its explicit implementation/proof binding")
    for name, path in _wrapper_inputs(request, output).items():
        record = identity["inputs"].get(name, {})
        if (record.get("path") != str(path.resolve()) or record.get("sha256") != sha(path)
                or record.get("size_bytes") != path.stat().st_size):
            raise ValueError("Fresh stage omitted or changed a wrapper dependency: " + name)
    for observed in result["process_guard"]["owned_identities"]:
        if observed["pid"] not in known or known[observed["pid"]] != observed:
            raise ValueError("Fresh stage owned identity differs from supervisor observation")
    if _owned([*result["process_guard"]["owned_identities"], *inner["supervisor_processes"]], inner,
              _wrapper_path(request), sha(_wrapper_path(request)), pid=result["pid"]) != result["process_guard"]["owned_identities"]:
        raise ValueError("Fresh stage ancestry proof changed")
    return result


# The following stage body is copied verbatim from frozen R3 except the explicit
# wrapper validation/identity/STOP/new-CLI/backend adoption points. Its original
# numerical workload, objective, optimizer, RNG, observations and acceptance stay.
def run_stage(wrapper, *, execute=False):
    output, request = validate_request(wrapper)
    if not execute:
        return {"status": "plan_only", "stage": request["stage"], "formal_training": False, "research_complete": False}
    if output.exists():
        raise FileExistsError("Fresh alignment output required; no automatic resume/retry")
    from .gpu_probe import resources
    stop = lambda: ((output / "STOP").exists() or (output.parent / "STOP").exists()
                    or Path(request["alignment_plan_path"] + ".stop").exists() or Path(wrapper["plan"]["path"] + ".stop").exists())
    stage, deadline = request["stage"], request["deadline_epoch"]
    _write = JsonWriter(deadline, stop)
    boundary(deadline, stop)
    from .alignment_process_guard_r3 import strict_admission, GuardSession
    from .probe_dispatch import _System
    system = _System()
    owned = _owned(system.processes(), request, _wrapper_path(wrapper), sha(_wrapper_path(wrapper)))
    allowed = [*request["allowed_background_processes"], *request["supervisor_processes"]]
    admission_guard = strict_admission(system.processes, owned_identities=owned,
        allowed_identities=allowed, deadline_epoch=deadline)
    if not admission_guard["caller_admission_allowed"]:
        from .process_guard_v4 import WAITABLE
        codes = {item["code"] for item in admission_guard["blockers"]}
        fresh = bool(admission_guard["observations"] and admission_guard["observations"][-1]["snapshot_valid"])
        if (not codes or not codes <= WAITABLE or not admission_guard["allowed_ancestry_verified"]
                or fresh and not admission_guard["owned_identity_verified"]):
            raise RuntimeError("Hard process conflict or owned identity blocks initial child admission")
    # The supervisor has already admitted the campaign. A new transient unknown
    # between spawn and this first instruction waits before resources/CUDA work.
    admission = {"blockers": [], "process_clearance_pending": True}
    lock = ROOT / "reports/.alignlab-gpu.lock"
    owner = {"pid": os.getpid(), "output": str(output), "token": uuid.uuid4().hex}
    _write(lock, owner, exclusive=True)
    started = time.time()
    status = {"kind": KIND, "stage": stage, "status": "starting", "request": request, "pid": os.getpid(),
        "started_at_epoch": started, "deadline_epoch": deadline, "formal_training": True, "research_complete": False,
        "training_started": False, "actual_optimizer_updates": 0, "scope": request["workload"]["scope"],
        "synthetic_rows": 0, "resources_before": admission, "checkpoints": [], "latest_safe_checkpoint": None,
        "gpu_samples": [], "objective_logging": "live_optimizer_end_and_after_engine_returns",
        "execution_revision": EXECUTION_REVISION, "telemetry_protocol": TELEMETRY_PROTOCOL,
        "process_guard": {"owned_identities": owned, "initial_admission": admission_guard,
            "inherited_waited_seconds": request["process_wait_spent_seconds"],
            "total_waited_seconds": request["process_wait_spent_seconds"], "last": None}}

    status["throughput"] = _marker(wrapper)
    proof_files = {}

    def progress(phase, *, terminal=False):
        status.update(status=phase, updated_at_epoch=time.time(), active_elapsed_seconds=time.time() - started)
        _write(output / "progress.json", status, terminal=terminal)
        print(json.dumps({"phase": phase, "stage": stage, "optimizer_updates": status["actual_optimizer_updates"]}), flush=True)

    def metric(value):
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_epoch": time.time(), **value}, allow_nan=False) + "\n")
            stream.flush()

    def gpu_sample():
        import torch
        from rlhf_lab.runtime import gpu_inventory
        torch.cuda.synchronize(0)
        sample = {"time_epoch": time.time(), **gpu_inventory(),
                  "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "max_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        status["gpu_samples"].append(sample)
        return sample

    def unchanged():
        try:
            for path, expected in proof_files.items():
                parent_stage._pin(path, expected)
            for name, expected in wrapper["source_sha256"].items():
                parent_stage._pin(ROOT / name, expected)
            parent_stage._pin(request["alignment_plan_path"], request["alignment_plan_sha256"])
            if read_json(output / "request.json") != request:
                return False
            return True
        except (OSError, ValueError):
            return False

    def guard_event(event):
        with (output / "process-guard.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
        if event["event"] == "wait_start":
            progress("waiting_process_boundary", terminal=True)
        elif event["event"] == "result":
            status["process_guard"].update(last=event, total_waited_seconds=event["total_waited_seconds"])

    guard = GuardSession(system.processes, owned_identities=owned, allowed_identities=allowed,
        deadline_epoch=deadline, policy=request["process_wait_policy"],
        spent_seconds=request["process_wait_spent_seconds"], stop_requested=stop,
        identity_unchanged=unchanged, on_event=guard_event)

    def require_guard(name, update, save):
        def stopped(result, receipt):
            status["process_guard"].update(blocked_reason=result["reason"], stopping_boundary=receipt)
            metric({"event": "guard_stop_boundary_saved", "boundary": name,
                    "optimizer_updates": update, "reason": result["reason"], "receipt": receipt})
            progress("guard_stop_boundary_saved", terminal=True)
        return resume.require_saved_guard_boundary(guard, name, update, save, on_stop=stopped)

    def file_receipt(path, kind, update):
        return {"complete": True, "optimizer_updates": update, "kind": kind,
                "path": str(Path(path).resolve()), "sha256": sha(path)}

    try:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "request.json", request, exclusive=True)
        _write(output / "wrapper.request.json", wrapper, exclusive=True)
        proof_files.update({path: sha(path) for path in _wrapper_inputs(wrapper, output).values()})
        require_guard("before_cuda", 0, lambda: file_receipt(output / "request.json", "before_cuda", 0))
        admission = resources()
        status["resources_before"] = admission
        if admission["blockers"]:
            raise RuntimeError(f"CUDA admission blocked after process clearance: {admission['blockers']}")
        telemetry = TrainingTelemetry(output / "telemetry", stage=stage, started_at_epoch=started)
        progress("loading_real_selection")
        from .training_data import load_training_partition
        part = "preference_train" if stage == "dpo" else "ppo_prompts"
        partition = load_training_partition(request["audit_dir"], part, request["audit_manifest_sha256"])
        rows, selection, records = select_real_rows(partition, request)
        selection_path = output / "selection.json"
        _write(selection_path, {"summary": selection, "records": records}, exclusive=True)
        selection.update(selection_manifest_path=str(selection_path), selection_manifest_sha256=sha(selection_path))
        status["selection"] = selection
        boundary(deadline, stop)
        validate_request(wrapper)
        immediate = resources()
        if immediate["blockers"]:
            raise RuntimeError(f"Resources changed before CUDA loading: {immediate['blockers']}")
        import torch
        from .model_loader import ModelLoadConfig, load_quantized_stage, QUANTIZATION
        from .stage_engine import parameter_mapping, _batch
        from .stage_engine_observed import run_supervised, run_ppo
        from .checkpointing import build_identity, save_checkpoint
        from .stage_artifacts import import_stage, export_stage
        from .gpu_reference import capture_frozen_reference, bind_reference_cache
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        progress("loading_nf4")
        loaded = load_quantized_stage(request["model_snapshot"], request["model_manifest_sha256"],
            request["tokenizer_dir"], ModelLoadConfig(stage, seed=request["seed"]))
        backend = _adopt_backend(loaded, wrapper)
        status["model"] = loaded.metadata
        gpu_sample()
        files, config = original._identity_files(request, selection_path), stage_config(request)
        files.update(_wrapper_inputs(wrapper, output))
        lr = 5e-6 if stage == "dpo" else 1e-6
        identity_config = {"stage": stage, "purpose": KIND, "formal_training": True, "research_complete": False,
            "execution_revision": EXECUTION_REVISION, "telemetry_protocol": TELEMETRY_PROTOCOL,
            "scope": request["workload"]["scope"], "synthetic_rows": 0, "work": asdict(config),
            "registered_workload": request["workload"], "model": loaded.metadata, "selection": selection,
            "continuation_revision": "alignment-r3", "continuation": request["continuation"],
            "repair_attempt": request["repair_attempt"],
            "optimizer": {"name": "AdamW", "lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            "process_guard_cache_record_interval": 16,
            "checkpoint_policy": {"first": 1, "interval_updates": 50, "cost_hours": [1, 2, 4, 8],
                "warm_training_cost_hours": [1, 2, 4, 8], "save_clean_return": True,
                "cost_axes": "stage_wall_since_admission_and_warm_wall_since_engine_entry",
                "same_cursor_labels": "merged_with_actual_observation_times_and_original_save_times"}}
        identity_config["throughput"] = status["throughput"]
        parent_files = {n: Path(p["artifact_dir"]) / "manifest.json" for n, p in request["parents"].items()}
        # Imports precede optimizer construction. The final identity additionally
        # pins a created/reused cache once its manifest actually exists.
        import_identity = build_identity(files, identity_config, parent_files)
        status["imports"] = {}
        for parent, role, heads in [("sft", loaded.role, ()), ("sft", "reference", ())] + (
                [("reward", "reward", ("reward",))] if stage == "ppo" else []):
            p = request["parents"][parent]
            status["imports"][role] = import_stage(p["artifact_dir"], backend, role,
                expected_manifest_sha256=p["manifest_sha256"], expected_identity=import_identity, heads=heads)
        boundary(deadline, stop)
        def capture_role(role, parent):
            artifact = Path(request["parents"][parent]["artifact_dir"])
            return capture_frozen_reference(backend, role, artifact_files={
                "base_files": {k: v for k, v in files.items() if k.startswith("base/")},
                "tokenizer_files": {k: v for k, v in files.items() if k.startswith("tokenizer/")},
                "source_files": _cache_source_files(files),
                "adapter_files": {"manifest": artifact / "manifest.json", "weights": artifact / "weights.pt"},
                "dataset_manifest": Path(request["audit_dir"]) / "manifest.json"},
                accepted_data_manifest_sha256=request["audit_manifest_sha256"],
                precision=PRECISION, quantization=QUANTIZATION)
        receipts = {}
        for role, parent in [("reference", "sft")] + ([("reward", "reward")] if stage == "ppo" else []):
            progress(f"capturing_{role}_receipt")
            receipts[role] = capture_role(role, parent)
            status[f"{role}_anchor"] = receipts[role].anchor
            boundary(deadline, stop)
        head_anchor = _head_anchor(backend._heads["reward"]) if stage == "ppo" else None
        status["reward_head_anchor"] = head_anchor
        binding = None
        if stage == "dpo":
            def pairs():
                for row in rows:
                    boundary(deadline, stop)
                    values = asdict(row)
                    yield _batch([values], "chosen_ids", PAD, loaded.device), _batch([values], "rejected_ids", PAD, loaded.device)

            reference, cache = receipts["reference"], request["reference_cache"]
            score_config = scoring_config(reference, partition.serialization_version)
            status["cache_progress_path"] = str(output / "cache-progress.json")
            progress("building_reference_cache" if cache["mode"] == "create" else "verifying_reference_cache")
            cache_marks = set()
            def cache_progress(**values):
                if values["scored_sequences"] > imported_count and "remaining" not in cache_marks:
                    progress("building_reference_cache_remaining")
                    cache_marks.add("remaining")
                if values["scored_sequences"] == 0 and "prepared" not in cache_marks:
                    telemetry.point("cold_cache_prepared", expected_unique_sequences=values["expected_unique_sequences"])
                    cache_marks.add("prepared")
                if (values["scored_sequences"] == values["expected_unique_sequences"]
                        and not values["complete"] and "scored" not in cache_marks):
                    telemetry.point("cold_cache_scoring_complete", scored_sequences=values["scored_sequences"])
                    cache_marks.add("scored")
                status["cache_scoring_progress"] = dict(values, time_epoch=time.time(), elapsed_seconds=time.time() - started,
                    imported_scored_sequences=min(imported_count, values["scored_sequences"]),
                    newly_scored_sequences=max(0, values["scored_sequences"] - imported_count))
                _write(output / "cache-progress.json", status["cache_scoring_progress"])
            cache_journal = output / "reference-cache-records.jsonl"
            cache_chain = "0" * 64
            importing_prefix = cache["mode"] == "create" and request["stage"] == "dpo" and request["seed"] == 42
            imported_count = request["continuation"]["cache_prefix"]["scored_sequences"] if importing_prefix else 0
            def cache_boundary(**values):
                nonlocal cache_chain
                payload = {"schema_version": 1, "previous_sha256": cache_chain, **values}
                cache_chain = digest(payload)
                row = {**payload, "record_sha256": cache_chain}
                with cache_journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                # This is the committed journal boundary, independently of
                # throttled display JSON. Failed append never advances this count.
                status["cache_scoring_progress"] = dict(
                    scored_sequences=values["scored_sequences"], expected_unique_sequences=values["expected_unique_sequences"],
                    complete=False, time_epoch=time.time(), elapsed_seconds=time.time() - started,
                    imported_scored_sequences=min(imported_count, values["scored_sequences"]),
                    newly_scored_sequences=max(0, values["scored_sequences"] - imported_count))
                def save_cache():
                    path = output / f"cache-safe-boundary-{values['scored_sequences']:06d}-{guard.calls:06d}.json"
                    _write(path, {"kind": "alignlab-safe-cache-record-boundary-v1", "complete": True,
                        "optimizer_updates": 0, "cache_journal": str(cache_journal),
                        "committed_record": row, "identity": import_identity}, exclusive=True, terminal=True)
                    return file_receipt(path, "cache_record_boundary", 0)
                # Every record is committed; expensive Windows snapshots run
                # only at complete groups of 16 and at both endpoint records.
                count, total = values["scored_sequences"], values["expected_unique_sequences"]
                if count == 0 or count == total or count == imported_count or (count > imported_count and count % 16 == 0):
                    require_guard(f"cache_record:{count}", 0, save_cache)
            telemetry.point("cold_cache_start" if cache["mode"] == "create" else "warm_cache_bind_start")
            if cache["mode"] == "reuse":
                cache_boundary(scored_sequences=0, expected_unique_sequences=0, record=None)
            with loaded.autocast():
                if cache["mode"] == "create":
                    if not importing_prefix:
                        raise ValueError("Only the authorized first DPO42 may create the migrated cache")
                    from .reference_cache_recovery_r3 import create_bounded_cache_from_prefix
                    def migration_report(report):
                        path = output / "cache-prefix-migration.json"
                        _write(path, report, exclusive=True)
                        status["cache_prefix_recovery_report"] = {"path": str(path), "sha256": sha(path)}
                        progress("importing_verified_reference_cache_prefix")
                    binding = create_bounded_cache_from_prefix(reference, cache["path"], pairs(),
                        prefix_spec=request["continuation"]["cache_prefix"], config=score_config,
                        deadline_epoch=deadline, stop_requested=stop, progress=cache_progress,
                        record_boundary=cache_boundary, migration_callback=migration_report)
                else:
                    binding = bind_complete_cache(reference, cache, pairs(), config=score_config,
                        deadline_epoch=deadline, stop_requested=stop)
            if cache["mode"] == "reuse":
                cache_boundary(scored_sequences=0, expected_unique_sequences=0, record=None)
            telemetry.point("cold_cache_bind_complete" if cache["mode"] == "create" else "warm_cache_bind_complete")
            status["reference_cache"] = {"mode": cache["mode"], "path": str(binding.path),
                "manifest_sha256": binding.manifest_sha256, "identity_sha256": sha(binding.path / "identity.json"),
                "unique_sequences": len(binding._cache.entries),
                "expected_unique_sequences": len(binding._cache.entries), "exact_key_coverage_verified": True}
            if importing_prefix:
                report_pin = status["cache_prefix_recovery_report"]
                status["reference_cache"].update(migration_report_path=report_pin["path"],
                    migration_report_sha256=report_pin["sha256"], imported_sequences=imported_count,
                    newly_scored_sequences=len(binding._cache.entries) - imported_count)
                files["reference_cache/prefix_migration"] = Path(report_pin["path"])
            files.update({"reference_cache/manifest": binding.path / "manifest.json",
                          "reference_cache/identity": binding.path / "identity.json"})
            identity_config["reference_cache"] = status["reference_cache"]
            boundary(deadline, stop)

        def audit_frozen(*, full, phase=None):
            for receipt in receipts.values():
                receipt.verify(full=full)
            if binding is not None:
                binding.verify(full=full)
            if head_anchor is not None and _head_anchor(backend._heads["reward"]) != head_anchor:
                raise RuntimeError("Frozen reward head changed")
            if full and phase is not None:
                status.setdefault("frozen_full_audits", []).append({"time_epoch": time.time(),
                    "phase": phase,
                    "roles": sorted(receipts), "reward_head_content_verified": head_anchor is not None,
                    "cache_manifest_sha256": None if binding is None else binding.manifest_sha256})

        validate_request(wrapper)
        boundary(deadline, stop)
        identity = build_identity(files, identity_config, parent_files)
        if stage == "dpo":
            # Completed artifacts bind import receipts to the exact destination
            # identity. A newly computed cache SHA cannot be known beforehand.
            # Re-import unchanged SFT bytes before creating an optimizer, then
            # recapture version guards and bind the existing actual-score cache.
            # Never rewrite import receipts or weaken the artifact verifier.
            progress("binding_final_cache_identity")
            for role in (loaded.role, "reference"):
                parent = request["parents"]["sft"]
                status["imports"][role] = import_stage(parent["artifact_dir"], backend, role,
                    expected_manifest_sha256=parent["manifest_sha256"], expected_identity=identity)
            previous = receipts["reference"]
            receipts["reference"] = capture_role("reference", "sft")
            if receipts["reference"].anchor != previous.anchor:
                raise RuntimeError("Reimported frozen SFT contents differ from the actually scored reference")
            binding = bind_reference_cache(receipts["reference"], binding.path,
                expected_identity=binding.identity, expected_manifest_sha256=binding.manifest_sha256)
        audit_frozen(full=True, phase="before_training")
        boundary(deadline, stop)
        _write(output / "identity.json", identity, exclusive=True)
        status["identity"] = identity
        telemetry.set_identity(identity)
        mapping = parameter_mapping(backend, loaded.role, heads=loaded.heads)
        optimizer = torch.optim.AdamW(mapping.values(), lr=lr)
        generator = torch.Generator(device=loaded.device).manual_seed(request["seed"])
        initial = {n: p.detach().cpu().clone() for n, p in mapping.items()}
        schedule = BoundaryCheckpoints(request["workload"]["optimizer_updates"])

        def write_checkpoint(name, cursor, extra):
            before = time.time()
            sidecar = telemetry.checkpoint_sidecar(name, cursor, extra)
            receipt = save_checkpoint(output / name, identity=identity, parameters=mapping,
                optimizer=optimizer, cursor={**cursor, "telemetry": sidecar},
                generators={"sampling": generator}, extra_state=extra)
            bound = telemetry.bind_checkpoint(receipt, sidecar, started_at_epoch=before, finished_at_epoch=time.time())
            return {**receipt, "telemetry_sidecar": sidecar, "telemetry_binding": bound}

        def save_safe(cursor, extra, *, safe_return=False):
            record = schedule.save(cursor, time.time() - started, safe_return=safe_return,
                warm_elapsed=telemetry.warm_elapsed(),
                writer=lambda name: write_checkpoint(name, cursor, extra))
            if record is not None:
                status["checkpoints"] = list(schedule.records.values())
                status["latest_safe_checkpoint"] = record

        def optimizer_boundary(cursor, extra, name):
            def save_wait_boundary():
                audit_frozen(full=True, phase=None)
                save_safe(cursor, extra, safe_return=True)
                record = status["latest_safe_checkpoint"]
                if record is None or record["cursor"] != cursor:
                    raise RuntimeError("Current optimizer/rollout cursor was not committed before waiting")
                manifest = Path(record["checkpoint_path"]) / "manifest.json"
                parent_stage._pin(manifest, record["manifest_sha256"])
                return file_receipt(manifest, "optimizer_checkpoint", cursor["optimizer_updates"])
            return require_guard(name, cursor["optimizer_updates"], save_wait_boundary)

        def checkpoint(cursor, extra):
            status.update(cursor=cursor, actual_optimizer_updates=cursor["optimizer_updates"],
                          training_started=cursor["optimizer_updates"] > 0)
            if stop() or time.time() >= deadline:
                # The unchanged engine can return a newly completed PPO rollout
                # before its first optimizer step. Preserve that complete state
                # on an explicit STOP/deadline, without resuming any computation.
                if not unchanged():
                    raise InterruptedError("Source identity changed before requested safe stop")
                audit_frozen(full=True, phase=None)
                save_safe(cursor, extra, safe_return=True)
                metric({"event": "requested_stop_boundary", "cursor": cursor,
                        "active_rollout_saved": extra is not None})
                progress("requested_stop_boundary_saved", terminal=True)
                return
            optimizer_boundary(cursor, extra, "optimizer_complete")
            audit_frozen(full=False)
            save_safe(cursor, extra)
            metric({"event": "engine_boundary", "cursor": cursor, "active_rollout_saved": extra is not None,
                    "active_elapsed_seconds": time.time() - started, "gpu": gpu_sample()})
            progress("training")

        engine_origin_checked = False
        def observe(event):
            nonlocal engine_origin_checked
            if not engine_origin_checked:
                from .stage_engine_observed import event_to_plain
                cursor = event_to_plain(event["cursor"])
                if cursor["optimizer_updates"] != 0 or cursor["active_rollout"]:
                    raise RuntimeError("Initial engine guard did not observe a clean fresh cursor")
                optimizer_boundary(cursor, None, "engine_origin")
                engine_origin_checked = True
            # Include all queued CUDA work in the phase interval; never call this
            # result kernel utilization or subtract it from the shared deadline.
            before = time.time()
            if loaded.device.type == "cuda":
                torch.cuda.synchronize(loaded.device)
            finished = time.time()
            sync = finished - before
            telemetry.cost["observer_cuda_sync_seconds"] = telemetry.cost.get("observer_cuda_sync_seconds", 0.0) + sync
            telemetry.point("cuda_synchronization", started_at_epoch=before, finished_at_epoch=finished, seconds=sync)
            telemetry.observe(event, backend=backend, role=loaded.role)

        progress("training")
        telemetry.warm_start()
        with loaded.autocast():
            if stage == "dpo":
                outcome = run_supervised(backend, loaded.role, optimizer, rows, config, device=loaded.device,
                    reference_provider=binding.pair_scores, checkpoint_callback=checkpoint, stop_requested=stop,
                    event_callback=observe)
            else:
                outcome = run_ppo(backend, loaded.role, optimizer, rows, config, device=loaded.device,
                    generator=generator, reference_role="reference", reward_role="reward",
                    checkpoint_callback=checkpoint, stop_requested=stop, event_callback=observe)
        telemetry.safe_engine_return(outcome.cursor, outcome.extra_state, outcome.stop_reason)
        status.update(cursor=outcome.cursor, actual_optimizer_updates=outcome.cursor["optimizer_updates"],
                      outcome_stop_reason=outcome.stop_reason)
        for item in outcome.metrics:
            metric({"event": "objective_after_engine_return", **item})
        status["objective_records"] = len(outcome.metrics)
        audit_frozen(full=True, phase="after_training")
        save_safe(outcome.cursor, outcome.extra_state, safe_return=True)
        if outcome.stop_reason == "paused":
            raise InterruptedError("Stopped at a saved clean engine boundary")
        if outcome.stop_reason == "budget_exhausted":
            raise TimeoutError("Absolute alignment deadline exhausted; latest clean boundary saved")
        optimizer_boundary(outcome.cursor, outcome.extra_state, "engine_final")
        status["completion"] = verify_completion(outcome, rows, request)
        boundary(deadline, stop)
        changed = [n for n, p in mapping.items() if not torch.equal(initial[n], p.detach().cpu())]
        status["adapter_weights_changed"] = any(".lora_" in n for n in changed)
        status["scalar_heads_changed"] = {h: any(n.startswith(f"scalar_heads.{h}.") for n in changed) for h in loaded.heads}
        if not status["adapter_weights_changed"] or not all(status["scalar_heads_changed"].values()):
            raise RuntimeError("Every owned adapter/head must actually change")
        validate_request(wrapper)
        boundary(deadline, stop)
        artifact = export_stage(output / "final-artifact", backend, loaded.role, stage=stage,
            identity=identity, heads=loaded.heads, completion=status["completion"])
        status["artifact"] = {k: artifact[k] for k in ("artifact_dir", "manifest_path", "manifest_sha256")}
        gpu_sample()
        audit_frozen(full=True, phase="after_export")
        validate_request(wrapper)
        boundary(deadline, stop)
        status["telemetry"] = telemetry.finish(cursor=outcome.cursor, completion=status["completion"],
                                               checkpoints=status["checkpoints"])
        from .training_telemetry import verify_training_telemetry
        verify_training_telemetry(output / "telemetry", expected_manifest_sha256=status["telemetry"]["sha256"],
            expected_identity_sha256=digest(identity), expected_completion=status["completion"],
            checkpoints=status["checkpoints"])
        boundary(deadline, stop)
        progress("formal_alignment_stage_passed")
        return status
    except BaseException as exc:
        status.update(status="budget_exhausted" if isinstance(exc, TimeoutError) else
            "paused" if isinstance(exc, InterruptedError) else "failed",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            safe_resume_available=status["latest_safe_checkpoint"] is not None,
            emergency_checkpoint_attempted=False,
            recovery_policy="Only a separately authorized pinned committed checkpoint; no automatic resume/retry")
        raise
    finally:
        status.update(finished_at_epoch=time.time(), elapsed_seconds=time.time() - started)
        try:
            if output.exists():
                _write(output / "progress.json", status, terminal=True)
        finally:
            if lock.exists() and read_json(lock) == owner:
                lock.unlink()
