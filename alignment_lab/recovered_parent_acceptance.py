"""Independent acceptance of original SFT plus an explicitly recovered RM.

The original failed campaign remains failed. A new acceptance binds both
segments, their immutable boundaries and the separately authorized deadline.
Initial audits decode tensors on CPU; historical verification is stdlib-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
import time

from .capacity_supervisor import _bundle
from .probe_dispatch import _System, _same_process, _stamp

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-independent-recovered-parents-acceptance-v1"
ACCEPTED = "accepted_full_real_recovered_parents"
SOURCES = (
    "alignment_lab/recovered_parent_acceptance.py",
    "scripts/audit_alignlab_recovered_parents.py",
    "tests_v2/test_recovered_parent_acceptance.py",
    "alignment_lab/parent_recovery.py", "scripts/run_alignlab_parent_recovery.py",
    "tests_v2/test_parent_recovery.py", "alignment_lab/process_guard_v2.py",
    "alignment_lab/parent_acceptance.py", "alignment_lab/alignment_supervisor.py",
    "alignment_lab/alignment_stage.py", "alignment_lab/evaluation_acceptance.py",
    "alignment_lab/study_supervisor.py", "alignment_lab/study_stage.py",
    "alignment_lab/checkpointing.py", "alignment_lab/stage_artifacts.py",
    "alignment_lab/model_snapshot.py", "alignment_lab/training_data.py",
    "alignment_lab/capacity_supervisor.py", "alignment_lab/probe_dispatch.py",
)


def _file(path, expected=None):
    # The same race-aware file reader is used by the independent evaluation
    # auditor. Import lazily so its recovery dispatch cannot form an import cycle.
    from .evaluation_acceptance import _file as read_file
    return read_file(path, expected)


def _read(path, expected=None):
    from .evaluation_acceptance import _read as read_json
    return read_json(path, expected)


def _record(evidence, path, expected=None):
    item = _file(path, expected)
    if evidence.setdefault(item["path"], item) != item:
        raise ValueError("Conflicting recovery evidence identities")
    return item


def _pin(evidence, pin):
    if type(pin) is not dict or set(pin) != {"path", "sha256"}:
        raise ValueError("An exact external path/SHA pin is required")
    return _record(evidence, pin["path"], pin["sha256"])


def _identity_files(evidence, identity):
    for item in [*identity["inputs"].values(), *identity["parents"].values()]:
        actual = _record(evidence, item["path"], item["sha256"])
        if "size_bytes" in item and actual["size_bytes"] != item["size_bytes"]:
            raise ValueError("Recovery identity file size differs")
    if "implementation" in identity:
        item = identity["implementation"]
        _record(evidence, item["path"], item["sha256"])


def _bundle_files(evidence, directory, manifest_sha, *, artifact=False):
    files = {"weights.pt"} if artifact else {"state.pt"}
    format_name = "alignlab-completed-stage-v1" if artifact else "alignlab-optimizer-checkpoint-v1"
    manifest = _bundle(directory, manifest_sha, files, format_name)
    _record(evidence, Path(directory) / "manifest.json", manifest_sha)
    for name, item in manifest["artifacts"].items():
        actual = _record(evidence, Path(directory) / name, item["sha256"])
        if actual["size_bytes"] != item["size_bytes"]:
            raise ValueError("Recovery bundle payload size differs")
    _identity_files(evidence, manifest["identity"])
    return manifest


def _historical_admission(original_plan, recovery_plan, evidence):
    """Verify original admission bytes, never reinterpret them as live priority."""
    prerequisites = original_plan["prerequisites"]
    required = {str(Path(item["path"]).resolve()): item for item in prerequisites}
    snapshots = recovery_plan.get("historical_prerequisites")
    if (len(required) != len(prerequisites) or type(snapshots) is not dict
            or set(snapshots) != set(required)):
        raise ValueError("Every original admission prerequisite needs its exact historical snapshot")
    for original_path, prerequisite in required.items():
        item = snapshots[original_path]
        if (type(item) is not dict
                or set(item) != {"original_path", "original_sha256", "snapshot", "semantics"}
                or item["original_path"] != original_path
                or item["original_sha256"] != prerequisite["sha256"]
                or item["semantics"] != "historical_admission_fact_not_current_resource_state"
                or type(item["snapshot"]) is not dict
                or set(item["snapshot"]) != {"path", "sha256", "size_bytes"}
                or item["snapshot"]["sha256"] != prerequisite["sha256"]):
            raise ValueError("Historical admission snapshot identity differs")
        actual = _record(evidence, item["snapshot"]["path"], prerequisite["sha256"])
        if (actual["path"] == original_path or type(item["snapshot"]["size_bytes"]) is not int
                or actual["size_bytes"] != item["snapshot"]["size_bytes"]):
            raise ValueError("Historical admission needs an independent exact byte snapshot")
        equalities = prerequisite.get("json_equals", {})
        if equalities:
            value = _read(actual["path"], actual["sha256"])[0]
            for dotted, expected in equalities.items():
                observed = value
                for part in dotted.split("."):
                    if type(observed) is not dict or part not in observed:
                        raise ValueError("Historical admission JSON field missing")
                    observed = observed[part]
                if type(observed) is not type(expected) or observed != expected:
                    raise ValueError("Historical admission JSON field differs")


def _sft_context(origin, evidence, *, tensors):
    """Audit the real completed SFT without claiming the old RM campaign passed."""
    from . import parent_acceptance as parents
    from . import study_supervisor as original
    plan, _ = _read(origin["plan"]["path"], origin["plan"]["sha256"])
    campaign, _ = _read(origin["campaign"]["path"], origin["campaign"]["sha256"])
    if (plan.get("kind") != original.KIND or len(plan.get("source_sha256", {})) != 59
            or campaign.get("kind") != original.KIND or campaign.get("status") != "failed"
            or campaign.get("parents_verified") is not False
            or campaign.get("plan_sha256") != origin["plan"]["sha256"]
            or plan.get("wall_seconds") != 43200
            or campaign.get("deadline_epoch") != campaign.get("started_at_epoch", 0) + 43200
            or not campaign["started_at_epoch"] <= campaign.get("finished_at_epoch", math.inf) <= campaign["deadline_epoch"]
            or [s.get("stage") for s in campaign.get("stages", [])] != ["sft", "reward"]):
        raise ValueError("Original failed two-stage parent campaign identity differs")
    for name, expected in plan["source_sha256"].items():
        _record(evidence, ROOT / name, expected)
    _record(evidence, plan["cpu_check_report"], plan["cpu_check_report_sha256"])
    cpu = _read(plan["cpu_check_report"], plan["cpu_check_report_sha256"])[0]
    checked = {k.replace("\\", "/"): v for k, v in
               cpu.get("source_before", cpu.get("source_sha256", {})).items()}
    required = {k for k in plan["source_sha256"] if k.startswith(("alignment_lab/", "tests_v2/"))}
    required.add("scripts/verify_alignlab_cpu.py")
    if (cpu.get("status") != "cpu_software_checks_passed" or cpu.get("source_unchanged") is not True
            or cpu.get("cuda_initialized") is not False or cpu.get("device") != "cpu"
            or type(cpu.get("tests_run")) is not int or cpu["tests_run"] < 253
            or any(type(cpu.get(k)) is not int or cpu[k] != 0 for k in ("errors", "failures", "skipped"))
            or any(checked.get(name) != plan["source_sha256"][name] for name in required)):
        raise ValueError("Original CPU software proof differs from the frozen parent sources")
    original._capacity_acceptance(plan)
    capacity = _read(plan["capacity_acceptance_path"], plan["capacity_acceptance_sha256"])[0]
    _record(evidence, plan["capacity_acceptance_path"], plan["capacity_acceptance_sha256"])
    _record(evidence, capacity["source_snapshot"]["path"], capacity["source_snapshot"]["sha256"])
    archived = _read(capacity["source_snapshot"]["path"], capacity["source_snapshot"]["sha256"])[0]
    for item in archived["files"].values():
        actual = _record(evidence, item["snapshot_path"], item["sha256"])
        if actual["size_bytes"] != item["size_bytes"]:
            raise ValueError("Original frozen capacity source snapshot size differs")
    _pin(evidence, origin["plan"])
    _pin(evidence, origin["campaign"])
    original_snapshot, _ = _read(Path(plan["output"]) / "plan.snapshot.json")
    if original_snapshot != plan:
        raise ValueError("Original parent plan snapshot changed")
    _record(evidence, Path(plan["output"]) / "plan.snapshot.json")
    record = campaign["stages"][0]
    output = Path(plan["output"]) / "sft"
    request_path = Path(plan["output"]) / "sft.request.json"
    request, _ = _read(request_path, record["request_sha256"])
    if (request != record["request"] or record.get("status") != "accepted"
            or record.get("exit_code") != 0 or record.get("active_processes") != []
            or record.get("monitoring_error") or record.get("stop_reason") is not None
            or request.get("parent_plan_sha256") != origin["plan"]["sha256"]
            or request.get("deadline_epoch") != campaign.get("deadline_epoch")
            or request.get("source_sha256") != plan["source_sha256"]
            or request.get("parents") != {}):
        raise ValueError("Original SFT was not a clean completed original child")
    for name in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
                 "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256"):
        if request.get(name) != plan.get(name):
            raise ValueError("Original SFT model/data/capacity differs from its plan")
    _record(evidence, request_path, record["request_sha256"])
    _record(evidence, output / "progress.json", record["progress_sha256"])
    progress = original._verify_stage(output, request,
        {p["pid"]: p for p in record["known_processes"]}, record["exit_code"])
    if progress["artifact"] != record.get("artifact"):
        raise ValueError("Original SFT campaign and output artifacts differ")
    payload = parents._payloads(output, "sft", progress) if tensors else None
    for name in ("request.json", "identity.json", "selection.json", "metrics.jsonl"):
        _record(evidence, output / name)
    for checkpoint in progress["checkpoints"]:
        _bundle_files(evidence, checkpoint["checkpoint_path"], checkpoint["manifest_sha256"])
    artifact = progress["artifact"]
    _bundle_files(evidence, artifact["artifact_dir"], artifact["manifest_sha256"], artifact=True)
    row = {"artifact_dir": artifact["artifact_dir"], "manifest_sha256": artifact["manifest_sha256"],
           "progress_path": str(output / "progress.json"), "progress_sha256": record["progress_sha256"],
           "rows": 4000, "optimizer_updates": 250}
    if payload is not None:
        row["payload_audit"] = payload
    return plan, campaign, row


def _same_tree(left, right):
    """Exact CPU payload equality, including all optimizer and RNG tensors."""
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return list(left) == list(right) and all(_same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right))
    return left == right


def _payload(directory, manifest, cursor):
    from .alignment_supervisor import _cpu_payload
    import torch
    state = _cpu_payload(Path(directory) / "state.pt", manifest["artifacts"]["state.pt"]["sha256"])
    if (type(state) is not dict or set(state) != {"parameters", "optimizer", "cursor", "rng", "extra_state"}
            or state["cursor"] != cursor or cursor.get("active_rollout") is not False
            or state["extra_state"] is not None or not isinstance(state["rng"], dict)
            or not state["rng"] or not state["parameters"]):
        raise ValueError("Actual recovered checkpoint cursor/state differs")
    optimizer = state["optimizer"]
    if type(optimizer) is not dict or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("Complete optimizer state is required")
    indices = [key for group in optimizer["param_groups"] for key in group["params"]]
    if (len(indices) != len(state["parameters"]) or len(set(indices)) != len(indices)
            or set(indices) != set(optimizer["state"])):
        raise ValueError("Optimizer must own every saved RM parameter exactly once")
    for (name, value), index in zip(state["parameters"].items(), indices):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())
                or not (".lora_A.reward." in name or ".lora_B.reward." in name
                        or name.startswith("scalar_heads.reward."))):
            raise ValueError("Checkpoint contains nonfinite or unowned RM parameters")
        saved = optimizer["state"][index]
        step = saved.get("step")
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or step.item() != cursor["optimizer_updates"]):
            raise ValueError("Actual Adam step differs from the recovered boundary")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = saved.get(key)
            if (not isinstance(moment, torch.Tensor) or moment.device.type != "cpu"
                    or moment.shape != value.shape or not bool(torch.isfinite(moment).all())):
                raise ValueError("Invalid recovered Adam moment")
    return state


def _migration_payloads(origin, migration, transition):
    """The only permitted discontinuity is the separately approved config hash."""
    old_cursor, new_cursor = transition["old_cursor"], transition["new_cursor"]
    if (transition.get("kind") != "alignlab-explicit-deadline-transition-v1"
            or transition.get("changed_cursor_fields") != ["config_sha256"]
            or transition.get("changed_work_fields") != ["deadline_epoch"]
            or transition.get("parameter_optimizer_rng_change_permitted") is not False
            or {k: v for k, v in old_cursor.items() if k != "config_sha256"}
                != {k: v for k, v in new_cursor.items() if k != "config_sha256"}
            or old_cursor.get("optimizer_updates") != 256 or old_cursor.get("offset") != 4096):
        raise ValueError("Migration changed numerical state or its approved boundary")
    before = _bundle(origin["path"], origin["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
    after = _bundle(migration["checkpoint_path"], migration["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
    old_state = _payload(origin["path"], before, old_cursor)
    new_state = _payload(migration["checkpoint_path"], after, new_cursor)
    for name in ("parameters", "optimizer", "rng", "extra_state"):
        if not _same_tree(old_state[name], new_state[name]):
            raise ValueError("Migration altered original " + name)
    return {"actual_cpu_payloads_compared": True,
            "parameters_optimizer_rng_exact": True, "only_cursor_config_sha256_changed": True}


def _combined_work(completion, transition, budget, finished):
    expected = {"rows_processed": 13139, "optimizer_updates": 822,
                "historical_optimizer_updates": 256, "segment_optimizer_updates": 566,
                "historical_rows": 4096, "segment_rows": 9043, "synthetic_rows": 0}
    if (any(type(completion.get(k)) is not int or completion[k] != value for k, value in expected.items())
            or any(completion.get(k) is not True for k in
                   ("data_pass_complete", "protocol_complete", "full_accepted_dataset_pass"))
            or completion.get("stop_reason") != "data_exhausted"
            or completion.get("scope") != "full_accepted_real_training_partition"):
        raise ValueError("Only exact 256 + 566 updates/full 13139 real pairs can be accepted")
    numbers = (budget.get("started_at_epoch"), budget.get("deadline_epoch"), finished)
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
            or type(budget.get("wall_seconds")) is not int or budget["wall_seconds"] != 28800
            or budget["deadline_epoch"] != budget["started_at_epoch"] + 28800
            or not 0 < budget["started_at_epoch"] <= finished <= budget["deadline_epoch"]):
        raise ValueError("Recovered completion must fit the separately fixed 8-hour deadline")
    old_work, new_work = transition["old_work"], transition["new_work"]
    if (new_work.get("deadline_epoch") != budget["deadline_epoch"]
            or new_work["deadline_epoch"] <= old_work.get("deadline_epoch", math.inf)
            or {k: v for k, v in old_work.items() if k != "deadline_epoch"}
                != {k: v for k, v in new_work.items() if k != "deadline_epoch"}):
        raise ValueError("Recovery changed training protocol instead of only the authorized deadline")
    old_cursor = transition["old_cursor"]
    if (type(completion.get("response_tokens")) is not int
            or type(completion.get("segment_response_tokens")) is not int
            or completion.get("historical_response_tokens") != old_cursor["response_tokens"]
            or completion["response_tokens"] != completion["historical_response_tokens"] + completion["segment_response_tokens"]
            or completion["segment_response_tokens"] <= 0):
        raise ValueError("Historical and recovered response-token denominators differ")


def _process_history(records, current, protected=()):
    """Readonly exit attestation; unrelated projects may legitimately run later."""
    by_pid = {p["pid"]: p for p in current}
    result = []
    for expected in records:
        if (type(expected.get("pid")) is not int or type(expected.get("command_line")) is not str
                or not expected["command_line"]):
            raise ValueError("Recorded recovery process identity is incomplete")
        created = _stamp(expected["created_at_utc"])
        actual = by_pid.get(expected["pid"])
        if actual is None:
            state = "exited"
        elif _same_process(actual, expected):
            raise ValueError("A recorded recovery process is still alive")
        elif _stamp(actual["created_at_utc"]) == created:
            raise ValueError("Same PID/creation has unknown or changed command")
        else:
            state = "pid_reused_original_exited"
        result.append({"expected": expected, "observed": actual, "state": state})
    for process in current:
        command = (process.get("command_line") or "").replace("\\", "/").lower()
        if process["pid"] != os.getpid() and any(
                str(Path(p).resolve()).replace("\\", "/").lower() in command for p in protected):
            raise ValueError("A potential writer targets the accepted recovery artifacts")
    return result


def _context(parent_pin, recovery_pin, *, tensors, system, manifest_pin=None):
    """Reconstruct required evidence independently; never trust status alone."""
    from . import parent_recovery as recovery
    from . import parent_acceptance as parents
    from .evaluation_acceptance import _model_data_files
    evidence = {}
    _pin(evidence, parent_pin)
    _pin(evidence, recovery_pin)
    plan, _ = _read(recovery_pin["path"], recovery_pin["sha256"])
    output = Path(plan["output"]).resolve()
    manifest_path = output / "manifest.json"
    if manifest_pin is not None and Path(manifest_pin["path"]).resolve() != manifest_path:
        raise ValueError("Recovery completion manifest escaped the fixed output")
    manifest, manifest_file = _read(manifest_path, manifest_pin["sha256"] if manifest_pin else None)
    _record(evidence, manifest_path, manifest_file["sha256"])
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "alignlab-parent-recovery-result-v1"
            or manifest.get("status") != "formal_parent_recovery_passed"
            or manifest.get("formal_training") is not True or manifest.get("research_complete") is not False):
        raise ValueError("Recovery has no complete separately recorded result")
    _pin(evidence, manifest["request"])
    request, _ = _read(manifest["request"]["path"], manifest["request"]["sha256"])
    validated = recovery.validate_request(request)  # Hash/JSON checks only; no model or tensor loading.
    if tensors and recovery.verify_recovery_output(output,
            expected_manifest_sha256=manifest_file["sha256"], expected_request=request) != manifest:
        raise ValueError("Recovery CPU verifier returned a different completion manifest")
    if (request["recovery_plan_path"] != recovery_pin["path"]
            or request["recovery_plan_sha256"] != recovery_pin["sha256"]
            or request["origin"]["plan"] != parent_pin or request["origin"] != manifest["origin"]
            or request["budget"] != manifest["budget"] or request["output"] != str(output)):
        raise ValueError("Recovery result differs from the authorized original/new plan chain")
    files = manifest.get("files")
    if type(files) is not dict or not files:
        raise ValueError("Complete immutable recovery file inventory required")
    actual_names = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual_names != set(files) | {"manifest.json"}:
        raise ValueError("Recovery output has missing or unregistered files")
    for name, item in files.items():
        local = PurePosixPath(name)
        if local.is_absolute() or ".." in local.parts or "\\" in name or str(local) != name:
            raise ValueError("Recovery file inventory escaped its output")
        actual = _record(evidence, output / name, item["sha256"])
        if actual["size_bytes"] != item["size_bytes"]:
            raise ValueError("Recovery result file size differs")
    progress, _ = _read(output / "progress.json", files["progress.json"]["sha256"])
    identity, _ = _read(output / "identity.json", files["identity.json"]["sha256"])
    if (progress.get("request") != request or progress.get("status") != manifest["status"]
            or progress.get("identity") != identity or progress.get("request_file") != manifest["request"]
            or recovery.digest(identity) != manifest["identity_sha256"]
            or any(progress.get(key) != manifest[key] for key in
                   ("completion", "checkpoints", "artifact", "origin_checkpoint", "migration_manifest",
                    "migration_state_audit", "latest_safe_checkpoint", "process_identity",
                    "started_at_epoch", "finished_at_epoch"))
            or progress.get("actual_optimizer_updates") != 822
            or progress.get("segment_optimizer_updates") != 566):
        raise ValueError("Recovery result/progress/identity disagree")
    if _read(output / "request.json")[0] != request:
        raise ValueError("Recovery saved request differs from its external pin")
    for label, filename in (("migration_manifest", "migration-manifest.json"),
                            ("migration_state_audit", "migration-state-audit.json")):
        item = manifest[label]
        if (type(item) is not dict or set(item) != {"path", "sha256"}
                or Path(item["path"]).resolve() != output / filename
                or item["sha256"] != files[filename]["sha256"]):
            raise ValueError("Recovery migration proof escaped its complete output inventory")
    for name in SOURCES:
        _record(evidence, ROOT / name)
    for name, expected in request["source_sha256"].items():
        _record(evidence, ROOT / name, expected)
    _pin(evidence, plan["budget_authorization"])
    for name, item in request["origin"].items():
        if name != "checkpoint":
            _pin(evidence, item)
    original_plan, original_campaign, sft = _sft_context(request["origin"], evidence, tensors=tensors)
    _historical_admission(original_plan, plan, evidence)
    if tensors:
        from .model_snapshot import verify_snapshot
        verify_snapshot(original_plan["model_snapshot"], original_plan["model_manifest_sha256"])
        parents._verify_data(original_plan)
    _model_data_files(evidence, original_plan)
    _identity_files(evidence, identity)
    config = identity["config"]
    if (config.get("purpose") != "alignlab-formal-parent-stage-v1" or config.get("stage") != "reward"
            or config.get("execution_revision") != "parent-recovery-v1"
            or config.get("expected_rows") != 13139 or config.get("expected_updates") != 822
            or config.get("formal_training") is not True or config.get("synthetic_rows") != 0
            or identity["inputs"]["parent_plan"]["sha256"] != parent_pin["sha256"]
            or identity["parents"].get("sft", {}).get("sha256") != sft["manifest_sha256"]
            or Path(identity["parents"]["sft"]["path"]).resolve() != Path(sft["artifact_dir"]).resolve() / "manifest.json"):
        raise ValueError("Recovered RM is not the same full-real SFT/data/protocol lineage")
    allowed_config_changes = {"work", "selection", "execution_revision", "recovery"}
    old_config = validated["old_identity"]["config"]
    if ({k: v for k, v in config.items() if k not in allowed_config_changes}
            != {k: v for k, v in old_config.items() if k not in allowed_config_changes}):
        raise ValueError("Recovery changed frozen model/optimizer/protocol configuration")
    if _read(output / "selection.json")[0] != _read(request["origin"]["selection"]["path"])[0]:
        raise ValueError("Recovered real selection differs from the entire original partition")
    transition, _ = _read(manifest["migration_manifest"]["path"], manifest["migration_manifest"]["sha256"])
    if (transition.get("recovery_request") != manifest["request"]
            or transition.get("recovery_plan") != recovery_pin
            or transition.get("origin_checkpoint") != request["origin"]["checkpoint"]
            or transition.get("budget_authorization") != plan["budget_authorization"]
            or config.get("work") != transition["new_work"]
            or validated["old_identity"]["config"]["work"] != transition["old_work"]
            or transition.get("old_cursor") != validated["old_progress"]["cursor"]
            or transition.get("old_cursor_sha256") != recovery.digest(transition["old_cursor"])
            or transition.get("new_cursor_sha256") != recovery.digest(transition["new_cursor"])
            or transition["old_cursor"]["config_sha256"] != recovery.digest({"config": transition["old_work"], "role": "reward"})
            or transition["new_cursor"]["config_sha256"] != recovery.digest({"config": transition["new_work"], "role": "reward"})
            or manifest["started_at_epoch"] != request["budget"]["started_at_epoch"]):
        raise ValueError("Deadline transition lacks its exact original/new authorization")
    _combined_work(manifest["completion"], transition, request["budget"], manifest["finished_at_epoch"])
    before = _bundle_files(evidence, request["origin"]["checkpoint"]["path"],
                           request["origin"]["checkpoint"]["manifest_sha256"])
    migration = manifest["origin_checkpoint"]
    if Path(migration["checkpoint_path"]).resolve() != output / "origin-000256":
        raise ValueError("Migration checkpoint escaped the new recovery output")
    migrated = _bundle_files(evidence, migration["checkpoint_path"], migration["manifest_sha256"])
    if migrated["identity"] != identity or before["identity"] != validated["old_identity"]:
        raise ValueError("Migrated checkpoint changed the bound old/new identities")
    contract_keys = ("parameter_specs", "optimizer", "generator_specs", "software")
    if any(key not in before or migrated.get(key) != before[key] for key in contract_keys):
        raise ValueError("Migration changed parameter/optimizer/generator/software structure")
    migration_audit = _migration_payloads(request["origin"]["checkpoint"], migration, transition) if tensors else None
    metrics = recovery._objectives(output / "metrics.jsonl", 257, 822)
    if (sum(item["rows"] for item in metrics) != 9043
            or any(item["rows"] != min(16, 13139 - (item["update"] - 1) * 16) for item in metrics)):
        raise ValueError("Recovered objective denominators do not cover exactly the remaining rows")
    boundaries = [row for row in recovery._jsonl(output / "metrics.jsonl") if row.get("event") == "optimizer_boundary"]
    if [row["cursor"]["optimizer_updates"] for row in boundaries] != list(range(257, 823)):
        raise ValueError("Recovered optimizer boundaries are missing, duplicated or reordered")
    selection = _read(output / "selection.json")[0]["records"]
    order = transition["new_cursor"]["order"]
    expected_order = list(range(13139))
    random.Random(42).shuffle(expected_order)
    accepted_rows = recovery._jsonl(Path(original_plan["audit_dir"]) / "preference_train.jsonl")
    if (order != expected_order or transition["old_cursor"]["order"] != expected_order
            or [r["record_id"] for r in selection] != [r["record_id"] for r in accepted_rows]
            or len(selection) != 13139 or len({r["record_id"] for r in selection}) != 13139
            or any(row["rows"] != 16 for row in validated["historical_metrics"])):
        raise ValueError("Original/continued order or full accepted partition membership differs")
    prefix = [0]
    for index in order:
        row = selection[index]
        prefix.append(prefix[-1] + row["chosen_tokens"] + row["rejected_tokens"])
    if (transition["old_cursor"]["response_tokens"] != prefix[4096]
            or manifest["completion"]["response_tokens"] != prefix[-1]):
        raise ValueError("Combined token totals differ from the exact original/recovered selection")
    by_update = {}
    for row in boundaries:
        cursor = row["cursor"]
        update, offset = cursor["optimizer_updates"], min(13139, cursor["optimizer_updates"] * 16)
        if (cursor.get("order") != order or cursor.get("offset") != offset
                or cursor.get("response_tokens") != prefix[offset] or cursor.get("active_rollout") is not False
                or cursor.get("completed_rollouts") != 0
                or any(cursor.get(k) != transition["new_cursor"][k] for k in ("schema", "data_sha256", "config_sha256"))):
            raise ValueError("Recovered cursor differs from the original shuffled pass/token prefix")
        by_update[update] = cursor
    checkpoints = manifest["checkpoints"]
    updates = [row["optimizer_updates"] for row in checkpoints]
    if (len(updates) != len(set(updates)) or not ({257, 822} | set(range(300, 823, 50))) <= set(updates)
            or manifest["latest_safe_checkpoint"] not in checkpoints
            or manifest["latest_safe_checkpoint"]["optimizer_updates"] != 822
            or progress["cursor"] != by_update[822]):
        raise ValueError("First/periodic/final recovered optimizer checkpoints are incomplete")
    final_state = None
    for item in checkpoints:
        update = item["optimizer_updates"]
        if update not in by_update or Path(item["checkpoint_path"]).resolve() != output / f"update-{update:06d}":
            raise ValueError("Recovery checkpoint belongs to a different boundary/output")
        saved = _bundle_files(evidence, item["checkpoint_path"], item["manifest_sha256"])
        if (saved["identity"] != identity
                or any(saved.get(key) != before[key] for key in contract_keys)):
            raise ValueError("Recovery checkpoint identity or numerical structure changed")
        if tensors:
            state = _payload(item["checkpoint_path"], saved, by_update[update])
            from . import checkpointing as ck
            ck._state_tree(state, integer_keys=True)
            specs = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                     for name, value in state["parameters"].items()]
            indices = [i for group in state["optimizer"]["param_groups"] for i in group["params"]]
            names_by_index = dict(zip(indices, state["parameters"]))
            groups = [{"names": [names_by_index[i] for i in group["params"]],
                       "options": ck._json({k: v for k, v in group.items() if k != "params"})}
                      for group in state["optimizer"]["param_groups"]]
            if (ck._tensor_specs(state) != saved.get("tensor_specs") or specs != saved["parameter_specs"]
                    or saved["optimizer"] != {"class": "torch.optim.adamw.AdamW", "groups": groups}):
                raise ValueError("Actual recovered parameter/optimizer state differs from its immutable contract")
            if update == 822:
                final_state = state
    artifact = manifest["artifact"]
    if (Path(artifact["artifact_dir"]).resolve() != output / "final-artifact"
            or Path(artifact["manifest_path"]).resolve() != output / "final-artifact/manifest.json"):
        raise ValueError("Recovered artifact escaped its authorized output")
    exported = _bundle_files(evidence, artifact["artifact_dir"], artifact["manifest_sha256"], artifact=True)
    if (exported.get("stage") != "reward" or exported.get("source_role") != "reward"
            or exported.get("heads") != ["reward"] or exported["identity"] != identity
            or exported.get("completion") != manifest["completion"]):
        raise ValueError("Recovered artifact does not match its actual completed RM")
    if tensors:
        from .alignment_supervisor import _cpu_payload
        weights = _cpu_payload(Path(artifact["artifact_dir"]) / "weights.pt", exported["artifacts"]["weights.pt"]["sha256"])
        parents._final_weights(final_state, weights, "reward", ["reward"])
    model = progress.get("model", {})
    from .capacity_supervisor import NF4
    samples = progress.get("gpu_samples", [])
    if (model != config.get("model") or model.get("device") != "cuda:0" or model.get("quantization") != NF4
            or model.get("quantized_linear_count") != 196 or model.get("parameters_selected", 0) <= 0
            or progress.get("adapter_weights_changed") is not True
            or progress.get("scalar_heads_changed", {}).get("reward") is not True
            or not samples or any(not sample.get("gpus") or any(
                type(sample.get(k)) not in (int, float) or not math.isfinite(sample[k]) or sample[k] <= 0
                for k in ("max_allocated_bytes", "max_reserved_bytes")) for sample in samples)):
        raise ValueError("Actual CUDA RM update/memory evidence is incomplete")
    current = system.processes()
    old_history = parents._processes(original_campaign, Path(parent_pin["path"]), parent_pin["sha256"], current)
    records = [row["expected"] for row in old_history] + manifest["process_identity"]
    request_command = manifest["request"]["path"]
    if (not manifest["process_identity"] or any(not parents.parents._command(
            record, "run_alignlab_parent_recovery.py", request_command)
            or manifest["request"]["sha256"] not in record["command_line"] for record in manifest["process_identity"])):
        raise ValueError("Recovery process identities do not name the exact executed request")
    histories = _process_history(records, current,
        (parent_pin["path"], original_plan["output"], recovery_pin["path"], output))
    reward = {"artifact_dir": artifact["artifact_dir"], "manifest_sha256": artifact["manifest_sha256"],
              "progress_path": str(output / "progress.json"), "progress_sha256": files["progress.json"]["sha256"],
              "rows": 13139, "optimizer_updates": 822, "historical_optimizer_updates": 256,
              "segment_optimizer_updates": 566, "migration_payload_audit": migration_audit}
    return {"parent_plan": parent_pin, "recovery_plan": recovery_pin,
            "original_campaign": {**request["origin"]["campaign"], "status": "failed"},
            "recovery_manifest": {"path": str(manifest_path), "sha256": manifest_file["sha256"]},
            "stages": {"sft": sft, "reward": reward}, "processes": histories,
            "evidence_files": evidence, "budget": request["budget"], "completion": manifest["completion"]}


def audit_recovered_parents(parent_plan_path, parent_plan_sha256, recovery_plan_path,
                            recovery_plan_sha256, *, output, system=None):
    """Publish one new CPU audit; incomplete or invalid recoveries never pass."""
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Use a new ordinary audit path; existing evidence is never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    system = system or _System()
    started = time.time()
    report = {"schema_version": 1, "kind": KIND, "status": "failed", "all_checks_passed": False,
              "parents_verified": False, "formal_training": False, "processes_exited": False,
              "research_complete": False, "device": "cpu", "model_loaded": False,
              "gpu_operations_performed": False, "started_at_epoch": started}
    try:
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise ValueError("Independent acceptance must not initialize CUDA")
        source_before = {name: _file(ROOT / name)["sha256"] for name in SOURCES}
        context = _context({"path": str(Path(parent_plan_path).resolve()), "sha256": parent_plan_sha256},
            {"path": str(Path(recovery_plan_path).resolve()), "sha256": recovery_plan_sha256},
            tensors=True, system=system)
        if source_before != {name: _file(ROOT / name)["sha256"] for name in SOURCES}:
            raise ValueError("Independent auditor source changed during verification")
        for item in context["evidence_files"].values():
            if _file(item["path"], item["sha256"]) != item:
                raise ValueError("Recovery evidence changed before publication")
        report.update(context, status=ACCEPTED, all_checks_passed=True, parents_verified=True,
                      formal_training=True, processes_exited=True, auditor_source_sha256=source_before,
                      actual_cpu_payloads_verified=True,
                      interpretation="Original failed campaign retained; completed SFT plus separately authorized RM continuation.")
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}")
    report.update(finished_at_epoch=time.time(), elapsed_seconds=time.time() - started,
        cuda_initialized=("torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()))
    if report["cuda_initialized"]:
        report.update(status="failed", all_checks_passed=False, parents_verified=False,
                      error="Unexpected CUDA initialization during CPU acceptance")
    from .evaluation_acceptance import _new
    _new(output, report)
    return report


def verify_recovered_parents(path, expected_sha256, *, system=None):
    """Rehash the completed audit and all reconstructed evidence without Torch."""
    report, item = _read(path, expected_sha256)
    if (report.get("schema_version") != 1 or report.get("kind") != KIND or report.get("status") != ACCEPTED
            or any(report.get(k) is not True for k in ("all_checks_passed", "parents_verified",
                "formal_training", "processes_exited", "actual_cpu_payloads_verified"))
            or report.get("research_complete") is not False or report.get("cuda_initialized") is not False
            or report.get("device") != "cpu" or set(report.get("auditor_source_sha256", {})) != set(SOURCES)):
        raise ValueError("Independent complete recovered-parent CPU acceptance required")
    for name, digest in report["auditor_source_sha256"].items():
        _file(ROOT / name, digest)
    rebuilt = _context(report["parent_plan"], report["recovery_plan"], tensors=False,
                       system=system or _System(), manifest_pin=report["recovery_manifest"])
    for key in ("parent_plan", "recovery_plan", "original_campaign", "recovery_manifest", "budget", "completion"):
        if report.get(key) != rebuilt[key]:
            raise ValueError("Accepted recovery provenance/completion changed")
    for stage in ("sft", "reward"):
        keys = ("artifact_dir", "manifest_sha256", "progress_path", "progress_sha256", "rows", "optimizer_updates")
        if any(report["stages"][stage].get(k) != rebuilt["stages"][stage][k] for k in keys):
            raise ValueError("Accepted parent artifact differs from independently reconstructed evidence")
    if report.get("evidence_files") != rebuilt["evidence_files"]:
        raise ValueError("Required original/recovered evidence was removed, added or changed")
    _file(item["path"], item["sha256"])
    result = copy.deepcopy(report)
    result["processes"] = rebuilt["processes"]
    return result


def validate_recovered_for_request(request, *, system=None):
    """Narrow alignment dispatch; ordinary completed-parent path stays separate."""
    report = verify_recovered_parents(request["parent_acceptance_path"],
                                     request["parent_acceptance_sha256"], system=system)
    if report["parent_plan"] != {"path": request["parent_plan_path"], "sha256": request["parent_plan_sha256"]}:
        raise ValueError("Alignment request changed the original parent plan")
    original, _ = _read(report["parent_plan"]["path"], report["parent_plan"]["sha256"])
    for key in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
                "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256"):
        if request[key] != original[key]:
            raise ValueError("Alignment model/data/capacity differs from recovered parents")
    if (not {"sft"} <= set(request["parents"]) <= {"sft", "reward"}
            or "ppo_rows" in request and request["ppo_rows"] != 1024
            or request.get("stage") == "ppo" and request.get("workload", {}).get("rows") != 1024):
        raise ValueError("Exact accepted parent set and currently chosen PPO scope required")
    for name, pin in request["parents"].items():
        accepted = report["stages"][name]
        if set(pin) != {"artifact_dir", "manifest_sha256"} or any(pin[k] != accepted[k] for k in pin):
            raise ValueError("Alignment requested a different recovered parent artifact")
    recovery_plan, _ = _read(report["recovery_plan"]["path"], report["recovery_plan"]["sha256"])
    required = {**original["source_sha256"], **recovery_plan["source_sha256"],
                **report["auditor_source_sha256"]}
    if any(request["source_sha256"].get(name) != digest for name, digest in required.items()):
        raise ValueError("Alignment source inventory omits the exact recovered-parent verification dependencies")
    return report
