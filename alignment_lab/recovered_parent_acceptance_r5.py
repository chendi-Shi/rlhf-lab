"""Independent CPU acceptance of completed SFT and the complete RM R2-R5 chain.

This module is deliberately outside the running R5 source/import graph. It does
not dispatch, resume, change STOP, or rewrite any historical campaign. Initial
acceptance requires actual CPU tensor audits and all recorded processes exited.
Later verification rehashes that pinned audit and every required input/artifact.
It never interprets a paused checkpoint, training reward or CPU test as quality.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time

from . import recovered_parent_acceptance as base
from .probe_dispatch import _System

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-independent-recovered-parents-r5-acceptance-v1"
ACCEPTED = "accepted_full_real_recovered_parents_r5"
ORIGIN_UPDATE, TOTAL_UPDATES, TOTAL_ROWS, BATCH = 684, 822, 13139, 16
CHAIN = (("r2", 256, 671), ("r3", 671, 677), ("r4", 677, 684), ("r5", 684, 822))
SOURCES = tuple(dict.fromkeys((*base.SOURCES,
    "alignment_lab/recovered_parent_acceptance_r5.py", "tests_v2/test_recovered_parent_acceptance_r5.py",
    "alignment_lab/parent_recovery_r2.py", "alignment_lab/parent_recovery_r3.py",
    "alignment_lab/parent_recovery_r4.py", "alignment_lab/parent_recovery_r5.py",
    "alignment_lab/process_guard_v3.py", "tests_v2/test_process_guard_v3.py",
    "scripts/run_alignlab_parent_recovery_r2.py", "scripts/run_alignlab_parent_recovery_r3.py",
    "scripts/run_alignlab_parent_recovery_r4.py", "scripts/run_alignlab_parent_recovery_r5.py",
    "tests_v2/test_parent_recovery_r2.py", "tests_v2/test_parent_recovery_r3.py",
    "tests_v2/test_parent_recovery_r4.py", "tests_v2/test_parent_recovery_r5.py")))
WAIT_POLICY = {"no_unknown_process_whitelist": True, "training_stops_computing_while_unknown": True,
    "safe_boundary_checkpoint_before_wait": True, "max_wait_seconds": 300,
    "max_total_wait_seconds": 900, "poll_interval_seconds": 2,
    "stop_deadline_and_identity_failures_not_ignored": True}


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _close(left, right):
    # JSON preserves double precision, but subtracting an accumulated sum can
    # lose an ulp. This tolerance is one microsecond, not extra wait budget.
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-6)


def _jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Complete nonempty JSONL evidence required")
    return [json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            for line in lines]


def _combined_work(completion, transition, budget, finished):
    expected = {"optimizer_updates": TOTAL_UPDATES, "rows_processed": TOTAL_ROWS,
        "historical_optimizer_updates": ORIGIN_UPDATE, "segment_optimizer_updates": TOTAL_UPDATES - ORIGIN_UPDATE,
        "historical_rows": ORIGIN_UPDATE * BATCH, "segment_rows": TOTAL_ROWS - ORIGIN_UPDATE * BATCH,
        "synthetic_rows": 0}
    if (any(type(completion.get(k)) is not int or completion[k] != v for k, v in expected.items())
            or any(completion.get(k) is not True for k in
                ("data_pass_complete", "protocol_complete", "full_accepted_dataset_pass"))
            or completion.get("scope") != "full_accepted_real_training_partition"
            or completion.get("stop_reason") != "data_exhausted"):
        raise ValueError("R5 requires exactly 684 historical + 138 new updates and all 13139 real pairs")
    start = _number(budget.get("started_at_epoch"), "budget start")
    deadline = _number(budget.get("deadline_epoch"), "budget deadline")
    if (type(budget.get("wall_seconds")) is not int or budget["wall_seconds"] != 28800
            or deadline != start + 28800 or not 0 < start <= _number(finished, "finish") <= deadline):
        raise ValueError("Completion exceeded or reset the same absolute 8-hour budget")
    if (transition.get("kind") != "alignlab-same-budget-continuation-v1"
            or transition.get("changed_cursor_fields") != [] or transition.get("changed_work_fields") != []
            or transition.get("parameter_optimizer_rng_change_permitted") is not False
            or transition["old_work"] != transition["new_work"]
            or transition["new_work"].get("deadline_epoch") != deadline
            or transition["old_cursor"] != transition["new_cursor"]
            or transition["old_cursor"].get("optimizer_updates") != ORIGIN_UPDATE
            or transition["old_cursor"].get("offset") != ORIGIN_UPDATE * BATCH):
        raise ValueError("R5 migration changed the original complete cursor, numerical work or deadline")
    if (type(completion.get("response_tokens")) is not int
            or type(completion.get("segment_response_tokens")) is not int
            or completion.get("historical_response_tokens") != transition["old_cursor"]["response_tokens"]
            or completion["segment_response_tokens"] <= 0
            or completion["response_tokens"] != completion["historical_response_tokens"] + completion["segment_response_tokens"]):
        raise ValueError("R5 complete response-token denominator differs")


def _continuation_payloads(source, migrated, transition):
    """Independently compare every CPU state tree; this is a genuine no-op."""
    if (transition["old_cursor"] != transition["new_cursor"] or transition["old_work"] != transition["new_work"]
            or transition.get("changed_cursor_fields") != [] or transition.get("changed_work_fields") != []):
        raise ValueError("Same-budget continuation must leave every cursor/work field unchanged")
    before = base._bundle(source["path"], source["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
    after = base._bundle(migrated["checkpoint_path"], migrated["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
    for key in ("parameter_specs", "optimizer", "generator_specs", "software"):
        if key not in before or after.get(key) != before[key]:
            raise ValueError("Continuation changed checkpoint numerical structure")
    left = base._payload(source["path"], before, transition["old_cursor"])
    right = base._payload(migrated["checkpoint_path"], after, transition["new_cursor"])
    if not base._same_tree(left, right):
        raise ValueError("Continuation changed parameters, Adam, RNG, cursor or extra state")
    return {"actual_cpu_payloads_compared": True, "parameters_optimizer_rng_cursor_exact": True,
            "changed_cursor_fields": [], "changed_work_fields": []}


def _classify_snapshot(snapshot, allowed):
    """Recompute v2 classification independently from the recorded decision."""
    from .process_guard_v2 import ProcessIdentity, DEFAULT_BUSY_MARKERS
    actual = [ProcessIdentity.from_record(p) for p in snapshot]
    if len({p.pid for p in actual}) != len(actual):
        raise ValueError("Ambiguous historical process snapshot")
    allow = {p.instance: p for p in map(ProcessIdentity.from_record, allowed)}
    codes = []
    for process in actual:
        known = allow.get(process.instance)
        if not process.command_line or not process.command_line.strip():
            code = "unknown_command"
        elif known is not None:
            code = None if process.command_line == known.command_line else "allowed_command_changed"
        elif any(marker in process.command_line.lower() for marker in DEFAULT_BUSY_MARKERS):
            code = "competing_process"
        else:
            code = "unclassified_process"
        if code:
            codes.append((code, process))
    return actual, codes


def _guard_record(record, owned, allowed, budget):
    from .process_guard_v2 import ProcessIdentity
    if (record.get("schema_version") != 1 or record.get("kind") != "alignlab-process-guard-v2"
            or record.get("gpu_resource_admission_verified") is not False):
        raise ValueError("Frozen strict process classifier evidence required")
    observations = record.get("observations", [])
    if not 1 <= len(observations) <= 2 or record.get("attempt_count") != len(observations):
        raise ValueError("R5 boundary guard must have one or two bounded observations")
    last_codes, last_actual, last_valid = [], [], False
    previous_finish = budget["started_at_epoch"]
    for index, observation in enumerate(observations, 1):
        start = _number(observation.get("started_at_epoch"), "snapshot start")
        finish = _number(observation.get("finished_at_epoch"), "snapshot finish")
        if not previous_finish <= start <= finish < budget["deadline_epoch"] or observation.get("attempt") != index:
            raise ValueError("Snapshot is outside the unchanged absolute deadline or reordered")
        previous_finish = finish
        last_valid = observation.get("snapshot_valid")
        if last_valid is True:
            last_actual, last_codes = _classify_snapshot(observation["snapshot"], allowed)
            expected_owned = {p.pid: p for p in map(ProcessIdentity.from_record, owned)}
            actual_owned = {p.pid: p for p in last_actual}
            if any(actual_owned.get(pid) != value for pid, value in expected_owned.items()):
                raise ValueError("Historical owned process identity was missing or changed")
            observed = [(b["code"], ProcessIdentity.from_record(b["process"])) for b in observation["blockers"]]
            if observed != last_codes:
                raise ValueError("Recorded process blockers differ from independently classified snapshot")
        elif last_valid is False:
            last_actual, last_codes = [], [("snapshot_unavailable", None)]
            if [b.get("code") for b in observation.get("blockers", [])] != ["snapshot_unavailable"]:
                raise ValueError("Unavailable snapshot cannot establish a clear boundary")
        else:
            raise ValueError("Explicit snapshot validity required")
    actual_codes = [(b["code"], ProcessIdentity.from_record(b["process"]) if b.get("process") else None)
                    for b in record.get("blockers", [])]
    clear = last_valid and not last_codes
    if (actual_codes != last_codes or record.get("admission_allowed") is not clear
            or record.get("requires_safe_stop") is not (not clear)
            or record.get("decision") != ("allow" if clear else "block_new_dispatch_and_request_safe_stop")):
        raise ValueError("Historical guard decision would allow an unresolved or conflicting process")
    if clear and [ProcessIdentity.from_record(p) for p in record["final_snapshot"]] != last_actual:
        raise ValueError("Clearance must use the fresh final snapshot")
    if not clear and any(code not in {"unknown_command", "unclassified_process", "snapshot_unavailable"}
                         for code, _ in last_codes):
        raise ValueError("A completed run cannot continue through a hard process conflict")
    return clear


def _validate_wait_log(rows, progress, plan, budget, expected_updates, checkpoint_updates):
    """Reconstruct every bounded hold and clearance; never trust summary flags."""
    if plan.get("process_wait_policy") != WAIT_POLICY or not rows:
        raise ValueError("Exact frozen process-wait policy and complete boundary observations required")
    owned = progress.get("process_identity", [])
    if not owned:
        raise ValueError("Owned processes required for wait audit")
    allowed = [*plan["allowed_background_processes"], *owned]
    groups, previous_time = [], budget["started_at_epoch"]
    for row in rows:
        update, cycle = row.get("optimizer_updates"), row.get("cycle")
        at = _number(row.get("at_epoch"), "wait observation")
        if type(update) is not int or type(cycle) is not int or not previous_time <= at < budget["deadline_epoch"]:
            raise ValueError("Wait evidence moved backwards or escaped the same deadline")
        previous_time = at
        if cycle == 1:
            groups.append([])
        if not groups or cycle != len(groups[-1]) + 1 or groups[-1] and groups[-1][0]["optimizer_updates"] != update:
            raise ValueError("Wait observations are missing, duplicated or assigned to another update")
        clear = _guard_record(row["guard"], owned, allowed, budget)
        if row["guard"]["observations"][-1]["finished_at_epoch"] > at:
            raise ValueError("Wait observation predates its process snapshot")
        if groups[-1] and groups[-1][-1]["guard"]["admission_allowed"]:
            raise ValueError("Unexpected observations after a clear boundary")
        groups[-1].append(row)
        if clear is not row["guard"]["admission_allowed"]:
            raise ValueError("Inconsistent wait clearance")
    if [g[0]["optimizer_updates"] for g in groups] != list(expected_updates):
        raise ValueError("Every origin, completed update and final publication needs its exact guard boundary")
    total = _number(progress.get("total_process_wait_seconds"), "total wait")
    if not 0 <= total < WAIT_POLICY["max_total_wait_seconds"]:
        raise ValueError("Successful recovery exceeded cumulative waiting limit")
    charges, waiting_updates = [], []
    for index, group in enumerate(groups):
        first, last = group[0], group[-1]
        initial_total = _number(first.get("total_waited_seconds"), "initial cumulative wait")
        next_total = groups[index + 1][0]["total_waited_seconds"] if index + 1 < len(groups) else total
        charge = _number(next_total, "next cumulative wait") - initial_total
        if (index == 0 and initial_total != 0 or first.get("waited_seconds") != 0
                or not 0 <= initial_total <= next_total <= total or not 0 <= charge < 300
                or last["guard"]["admission_allowed"] is not True):
            raise ValueError("Waiting was reset, exceeded its bound or resumed without clearance")
        waiting = first["guard"]["admission_allowed"] is False
        if waiting:
            if len(group) < 2 or charge <= 0 or first["optimizer_updates"] not in checkpoint_updates:
                raise ValueError("Unknown process waiting requires a current committed checkpoint and later clearance")
            waiting_updates.append(first["optimizer_updates"])
        elif len(group) != 1 or charge != 0:
            raise ValueError("A clean boundary cannot invent waiting time")
        last_wait = 0.0
        for row in group:
            waited = _number(row.get("waited_seconds"), "waited seconds")
            cumulative = _number(row.get("total_waited_seconds"), "cumulative wait")
            if (waited < last_wait - 1e-6 or waited > charge + 1e-6
                    or not _close(cumulative, initial_total + waited)
                    or row["at_epoch"] - first["at_epoch"] > waited + 1e-6
                    or row["at_epoch"] - first["at_epoch"] > charge + 1e-6):
                raise ValueError("Wait elapsed-time ledger is inconsistent or undercounts wall time")
            last_wait = waited
        charges.append(charge)
    last_result = progress.get("last_process_wait_result", {})
    last_rows = [{k: v for k, v in row.items() if k != "optimizer_updates"} for row in groups[-1]]
    if (last_result.get("kind") != "alignlab-process-guard-v3-boundary-wait"
            or last_result.get("schema_version") != 1 or last_result.get("admission_allowed") is not True
            or last_result.get("reason") != "cleared" or last_result.get("unknown_processes_allowed") is not False
            or last_result.get("training_during_wait") is not False
            or last_result.get("total_waited_seconds") != total
            or not _close(_number(last_result.get("waited_seconds"), "last wait charge"), charges[-1])
            or last_result.get("last_guard") != groups[-1][-1]["guard"]
            or last_result.get("observations") != last_rows
            or progress.get("hard_guard_stop") or progress.get("identity_guard_error")
            or progress.get("process_guard_blocked") is not False):
        raise ValueError("Final process clearance receipt differs from the independently reconstructed waits")
    return {"all_boundaries_have_fresh_clearance": True, "unknown_processes_allowed": False,
            "hard_conflicts_ignored": False, "boundaries": len(groups), "waiting_boundaries": len(waiting_updates),
            "waiting_updates": waiting_updates, "total_waited_seconds": total,
            "per_boundary_charges": charges, "policy": copy.deepcopy(WAIT_POLICY)}


def _record_pins(value, evidence):
    """Retain explicit nested authorization/CPU/history pins, not mutable status."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            item = base._record(evidence, value["path"], value["sha256"])
            if "size_bytes" in value and item["size_bytes"] != value["size_bytes"]:
                raise ValueError("Nested evidence size changed")
        for item in value.values():
            _record_pins(item, evidence)
    elif isinstance(value, list):
        for item in value:
            _record_pins(item, evidence)


def _segments(request, final_manifest, evidence):
    """Return chronological immutable segments; old failures remain failures."""
    segments = []
    current, manifest = request, final_manifest
    for revision, begin, end in reversed(CHAIN):
        output = Path(current["output"]).resolve()
        plan, _ = base._read(current["recovery_plan_path"], current["recovery_plan_sha256"])
        progress, item = base._read(output / "progress.json")
        if (plan.get("execution_revision") != "parent-recovery-" + revision
                or current["budget"] != request["budget"] or current["origin"] != request["origin"]
                or progress.get("request") != current or progress.get("actual_optimizer_updates") != end
                or progress.get("segment_optimizer_updates") != end - begin
                or progress.get("status") != ("formal_parent_recovery_passed" if revision == "r5" else "paused")
                or revision != "r5" and (output / "manifest.json").exists()):
            raise ValueError("Recovery segment changed its scope, original budget or historical status")
        _record_pins(current, evidence); _record_pins(plan, evidence); _record_pins(progress, evidence)
        base._record(evidence, output / "progress.json", item["sha256"])
        base._record(evidence, output / "request.json")
        for filename in ("identity.json", "selection.json", "metrics.jsonl", "process-guard.jsonl"):
            base._record(evidence, output / filename)
        for source, expected in current["source_sha256"].items():
            base._record(evidence, ROOT / source, expected)
        for pin in [progress["origin_checkpoint"], *progress["checkpoints"]]:
            base._bundle_files(evidence, pin["checkpoint_path"], pin["manifest_sha256"])
        for stop in (output / "STOP", Path(current["recovery_plan_path"] + ".stop")):
            if stop.exists():
                base._record(evidence, stop)
        segments.append({"revision": revision, "from_update": begin, "to_update": end,
            "new_updates": end - begin, "new_rows": min(TOTAL_ROWS, end * BATCH) - begin * BATCH,
            "status": progress["status"], "output": str(output), "request": current,
            "progress": progress, "progress_pin": item, "plan": plan})
        if revision != "r2":
            current = base._read(current["prior_segment"]["request"]["path"],
                                 current["prior_segment"]["request"]["sha256"])[0]
        else:
            failed = plan["prior_failed_dispatch"]
            failed_request = base._read(failed["request"]["path"], failed["request"]["sha256"])[0]
            failed_progress = base._read(failed["progress"]["path"], failed["progress"]["sha256"])[0]
            if (failed_progress.get("status") != "failed" or failed_progress.get("actual_optimizer_updates") != 256
                    or failed_progress.get("latest_safe_checkpoint") is not None
                    or failed_request["budget"] != request["budget"]):
                raise ValueError("R1 failure must remain a zero-new-update historical failure")
            _record_pins(failed, evidence); _record_pins(failed_request, evidence)
            _record_pins(failed_progress, evidence)
            failed_plan = base._read(failed_request["recovery_plan_path"], failed_request["recovery_plan_sha256"])[0]
            base._record(evidence, failed_request["recovery_plan_path"], failed_request["recovery_plan_sha256"])
            _record_pins(failed_plan, evidence)
            segments[-1]["prior_failed_progress"] = failed_progress
    return list(reversed(segments))


def _inventory(output, manifest, evidence):
    files = manifest.get("files")
    if (type(files) is not dict or not files or
            {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != set(files) | {"manifest.json"}):
        raise ValueError("Complete R5 output inventory required")
    for name, item in files.items():
        local = PurePosixPath(name)
        if local.is_absolute() or ".." in local.parts or "\\" in name or str(local) != name:
            raise ValueError("Unsafe R5 artifact inventory")
        actual = base._record(evidence, output / name, item["sha256"])
        if actual["size_bytes"] != item["size_bytes"]:
            raise ValueError("R5 artifact inventory size differs")
    return files


def _context(parent_pin, recovery_pin, system):
    from . import parent_recovery_r5 as recovery
    from . import parent_acceptance as parents
    from .evaluation_acceptance import _model_data_files
    evidence = {}
    base._pin(evidence, parent_pin); base._pin(evidence, recovery_pin)
    plan = base._read(recovery_pin["path"], recovery_pin["sha256"])[0]
    output = Path(plan["output"]).resolve()
    manifest, manifest_pin = base._read(output / "manifest.json")
    if (manifest.get("kind") != "alignlab-parent-recovery-r5-result-v1"
            or manifest.get("status") != "formal_parent_recovery_passed"
            or manifest.get("formal_training") is not True or manifest.get("research_complete") is not False):
        raise ValueError("Only complete R5 formal RM output can be accepted")
    base._record(evidence, manifest_pin["path"], manifest_pin["sha256"])
    request = base._read(manifest["request"]["path"], manifest["request"]["sha256"])[0]
    # Check the active final writer before importing any tensor-verifier path.
    # The caller must also wait for the process to exit before starting Python.
    base._process_history(manifest.get("process_identity", []), system.processes(),
                          (manifest["request"]["path"], str(output)))
    if (request["origin"]["plan"] != parent_pin or request["recovery_plan_path"] != recovery_pin["path"]
            or request["recovery_plan_sha256"] != recovery_pin["sha256"]
            or manifest["budget"] != request["budget"] or manifest["prior_segment"] != request["prior_segment"]):
        raise ValueError("R5 acceptance must bind exact original/recovery plans")
    validated = recovery.validate_request(request, execute=True)
    if recovery.verify_recovery_output(output, expected_manifest_sha256=manifest_pin["sha256"], expected_request=request) != manifest:
        raise ValueError("Complete independent R5 output verification differs")
    files = _inventory(output, manifest, evidence)
    segments = _segments(request, manifest, evidence)
    original_plan, original_campaign, sft = base._sft_context(request["origin"], evidence, tensors=True)
    base._historical_admission(original_plan, plan, evidence)
    from .model_snapshot import verify_snapshot
    verify_snapshot(original_plan["model_snapshot"], original_plan["model_manifest_sha256"])
    parents._verify_data(original_plan); _model_data_files(evidence, original_plan)
    progress = segments[-1]["progress"]
    identity = progress["identity"]
    if (identity["parents"].get("sft", {}).get("sha256") != sft["manifest_sha256"]
            or Path(identity["parents"]["sft"]["path"]).resolve() != Path(sft["artifact_dir"]) / "manifest.json"):
        raise ValueError("Recovered RM does not descend from the accepted real SFT artifact")
    transition = base._read(manifest["migration_manifest"]["path"], manifest["migration_manifest"]["sha256"])[0]
    _combined_work(manifest["completion"], transition, request["budget"], manifest["finished_at_epoch"])
    migration = _continuation_payloads(request["prior_segment"]["checkpoint"], manifest["origin_checkpoint"], transition)
    boundaries = [row for row in _jsonl(output / "metrics.jsonl") if row.get("event") == "optimizer_boundary"]
    by_update = {row["cursor"]["optimizer_updates"]: row["cursor"] for row in boundaries}
    final_state = None
    for checkpoint in progress["checkpoints"]:
        update = checkpoint["optimizer_updates"]
        if update not in by_update or Path(checkpoint["checkpoint_path"]).resolve() != output / f"update-{update:06d}":
            raise ValueError("R5 checkpoint is outside a complete recorded optimizer boundary")
        bundle = base._bundle_files(evidence, checkpoint["checkpoint_path"], checkpoint["manifest_sha256"])
        state = base._payload(checkpoint["checkpoint_path"], bundle, by_update[update])
        from . import checkpointing as ck
        ck._state_tree(state, integer_keys=True)
        if ck._tensor_specs(state) != bundle["tensor_specs"]:
            raise ValueError("Actual checkpoint tensors differ from immutable tensor descriptors")
        if update == TOTAL_UPDATES:
            final_state = state
    if final_state is None:
        raise ValueError("A complete final RM optimizer checkpoint is required")
    wait_rows = _jsonl(output / "process-guard-wait.jsonl")
    waits = _validate_wait_log(wait_rows, progress, plan, request["budget"],
        [ORIGIN_UPDATE, *range(ORIGIN_UPDATE + 1, TOTAL_UPDATES + 1), TOTAL_UPDATES],
        {ORIGIN_UPDATE, *(p["optimizer_updates"] for p in progress["checkpoints"])})
    artifact = manifest["artifact"]
    exported = base._bundle_files(evidence, artifact["artifact_dir"], artifact["manifest_sha256"], artifact=True)
    from .alignment_supervisor import _cpu_payload
    weights = _cpu_payload(Path(artifact["artifact_dir"]) / "weights.pt", exported["artifacts"]["weights.pt"]["sha256"])
    parents._final_weights(final_state, weights, "reward", ["reward"])
    from .capacity_supervisor import NF4
    for segment in segments:
        recorded = segment["progress"]
        model, samples = recorded.get("model", {}), recorded.get("gpu_samples", [])
        if (model.get("device") != "cuda:0" or model.get("quantization") != NF4
                or model.get("quantized_linear_count") != 196 or model.get("parameters_selected", 0) <= 0
                or not samples or any(not sample.get("gpus") or any(
                    _number(sample.get(key), "CUDA memory") <= 0 for key in ("max_allocated_bytes", "max_reserved_bytes"))
                    for sample in samples)):
            raise ValueError("Each real recovery segment needs actual CUDA model/memory evidence")
    current = system.processes()
    old_history = parents._processes(original_campaign, Path(parent_pin["path"]), parent_pin["sha256"], current)
    records = [row["expected"] for row in old_history]
    protected = [parent_pin["path"], original_plan["output"]]
    failed_progress = segments[0]["prior_failed_progress"]
    records.extend(failed_progress["process_identity"])
    protected.append(failed_progress["request"]["output"])
    for segment in segments:
        request_file = segment["progress"]["request_file"]
        processes = segment["progress"]["process_identity"]
        if not processes or any(not parents.parents._command(p, "run_alignlab_parent_recovery_" + segment["revision"] + ".py",
                request_file["path"]) or request_file["sha256"] not in p["command_line"] for p in processes):
            raise ValueError("A historical recovery process does not name its exact immutable request")
        records.extend(processes)
        protected.extend((request_file["path"], segment["output"], segment["request"]["recovery_plan_path"]))
    histories = base._process_history(records, current, protected)
    summary = [{k: segment[k] for k in ("revision", "from_update", "to_update", "new_updates", "new_rows", "status", "output")}
               for segment in segments]
    if sum(row["new_updates"] for row in summary) + 256 != TOTAL_UPDATES or sum(row["new_rows"] for row in summary) + 4096 != TOTAL_ROWS:
        raise ValueError("Recovered segment chain has missing or duplicate work")
    for name in SOURCES:
        base._record(evidence, ROOT / name)
    return {"parent_plan": parent_pin, "recovery_plan": recovery_pin,
        "original_campaign": {**request["origin"]["campaign"], "status": "failed"},
        "recovery_manifest": manifest_pin, "budget": request["budget"], "completion": manifest["completion"],
        "segments": summary, "process_wait_audit": waits, "processes": histories, "protected_paths": protected,
        "evidence_files": evidence, "stages": {"sft": sft, "reward": {
            "artifact_dir": artifact["artifact_dir"], "manifest_sha256": artifact["manifest_sha256"],
            "progress_path": str(output / "progress.json"), "progress_sha256": files["progress.json"]["sha256"],
            "rows": TOTAL_ROWS, "optimizer_updates": TOTAL_UPDATES, "historical_optimizer_updates": ORIGIN_UPDATE,
            "segment_optimizer_updates": TOTAL_UPDATES - ORIGIN_UPDATE, "migration_payload_audit": migration}}}


def audit_recovered_parents(parent_plan_path, parent_plan_sha256, recovery_plan_path,
                            recovery_plan_sha256, *, output, system=None):
    """New-path CPU audit only after all recorded training processes exit."""
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Use a new ordinary acceptance path; never overwrite evidence")
    started = time.time()
    report = {"schema_version": 1, "kind": KIND, "status": "failed", "all_checks_passed": False,
        "parents_verified": False, "formal_training": False, "processes_exited": False,
        "research_complete": False, "device": "cpu", "model_loaded": False, "gpu_operations_performed": False,
        "human_labels": 0, "human_win_rate": None, "started_at_epoch": started}
    try:
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise ValueError("Independent parent acceptance cannot run in an initialized CUDA process")
        sources = {name: base._file(ROOT / name)["sha256"] for name in SOURCES}
        context = _context({"path": str(Path(parent_plan_path).resolve()), "sha256": parent_plan_sha256},
            {"path": str(Path(recovery_plan_path).resolve()), "sha256": recovery_plan_sha256}, system or _System())
        if sources != {name: base._file(ROOT / name)["sha256"] for name in SOURCES}:
            raise ValueError("Independent acceptance source changed during verification")
        for item in context["evidence_files"].values():
            if base._file(item["path"], item["sha256"]) != item:
                raise ValueError("Training evidence changed before acceptance publication")
        report.update(context, status=ACCEPTED, all_checks_passed=True, parents_verified=True,
            formal_training=True, processes_exited=True, actual_cpu_payloads_verified=True,
            auditor_source_sha256=sources, interpretation="Original campaign failed; R1 failed; R2/R3/R4 paused. Only the combined SFT and fully completed R5-chain RM are accepted. Quality evaluation remains separate.")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report.update(finished_at_epoch=time.time(), elapsed_seconds=time.time() - started,
        cuda_initialized=("torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()))
    if report["cuda_initialized"]:
        report.update(status="failed", all_checks_passed=False, parents_verified=False,
            error="Unexpected CUDA initialization in independent CPU acceptance")
    from .evaluation_acceptance import _new
    output.parent.mkdir(parents=True, exist_ok=True)
    _new(output, report)
    return report


def verify_recovered_parents(path, expected_sha256, *, system=None):
    """Stdlib-only rehash of the independently accepted immutable tensor proof."""
    report, item = base._read(path, expected_sha256)
    if (report.get("schema_version") != 1 or report.get("kind") != KIND or report.get("status") != ACCEPTED
            or any(report.get(k) is not True for k in ("all_checks_passed", "parents_verified", "formal_training",
                "processes_exited", "actual_cpu_payloads_verified"))
            or report.get("research_complete") is not False or report.get("cuda_initialized") is not False
            or report.get("device") != "cpu" or report.get("human_labels") != 0 or report.get("human_win_rate") is not None
            or set(report.get("auditor_source_sha256", {})) != set(SOURCES)):
        raise ValueError("Complete independent R5 parent CPU acceptance required")
    for name, expected in report["auditor_source_sha256"].items():
        base._file(ROOT / name, expected)
    evidence = report.get("evidence_files")
    if type(evidence) is not dict or not evidence:
        raise ValueError("Complete accepted evidence inventory required")
    for key, pin in evidence.items():
        if key != pin["path"] or base._file(pin["path"], pin["sha256"]) != pin:
            raise ValueError("Accepted recovery provenance, state or source was changed")
    manifest = base._read(report["recovery_manifest"]["path"], report["recovery_manifest"]["sha256"])[0]
    plan = base._read(report["recovery_plan"]["path"], report["recovery_plan"]["sha256"])[0]
    output = Path(plan["output"]).resolve()
    _inventory(output, manifest, {})
    if manifest["completion"] != report["completion"] or manifest["budget"] != report["budget"]:
        raise ValueError("Accepted R5 completion/budget changed")
    original = base._read(report["original_campaign"]["path"], report["original_campaign"]["sha256"])[0]
    if original.get("status") != "failed" or original.get("parents_verified") is not False:
        raise ValueError("Original failed campaign cannot be relabeled as successful")
    progress = base._read(output / "progress.json")[0]
    rebuilt_wait = _validate_wait_log(_jsonl(output / "process-guard-wait.jsonl"), progress, plan, report["budget"],
        [ORIGIN_UPDATE, *range(ORIGIN_UPDATE + 1, TOTAL_UPDATES + 1), TOTAL_UPDATES],
        {ORIGIN_UPDATE, *(p["optimizer_updates"] for p in progress["checkpoints"])})
    if rebuilt_wait != report["process_wait_audit"]:
        raise ValueError("Accepted bounded wait evidence changed")
    result = copy.deepcopy(report)
    result["processes"] = base._process_history([p["expected"] for p in report["processes"]],
                                               (system or _System()).processes(), report["protected_paths"])
    base._file(item["path"], item["sha256"])
    return result


def validate_recovered_for_request(request, *, system=None):
    """Explicit downstream dispatch support; old parent protocol stays separate."""
    report = verify_recovered_parents(request["parent_acceptance_path"], request["parent_acceptance_sha256"], system=system)
    if report["parent_plan"] != {"path": request["parent_plan_path"], "sha256": request["parent_plan_sha256"]}:
        raise ValueError("Downstream request changed the original parent plan")
    original = base._read(report["parent_plan"]["path"], report["parent_plan"]["sha256"])[0]
    for key in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir", "audit_manifest_sha256",
                "capacity_acceptance_path", "capacity_acceptance_sha256"):
        if request[key] != original[key]:
            raise ValueError("Downstream model/data/capacity differs from accepted R5 parents")
    if (not {"sft"} <= set(request["parents"]) <= {"sft", "reward"}
            or "ppo_rows" in request and request["ppo_rows"] != 1024
            or request.get("stage") == "ppo" and request.get("workload", {}).get("rows") != 1024):
        raise ValueError("Exact accepted parents and user-chosen PPO1024 scope required")
    for name, pin in request["parents"].items():
        accepted = report["stages"][name]
        if set(pin) != {"artifact_dir", "manifest_sha256"} or any(pin[k] != accepted[k] for k in pin):
            raise ValueError("Downstream requested a different parent artifact")
    recovery = base._read(report["recovery_plan"]["path"], report["recovery_plan"]["sha256"])[0]
    required = {**original["source_sha256"], **recovery["source_sha256"], **report["auditor_source_sha256"]}
    if any(request["source_sha256"].get(name) != expected for name, expected in required.items()):
        raise ValueError("Downstream source inventory omits exact independent R5 acceptance dependencies")
    return report
