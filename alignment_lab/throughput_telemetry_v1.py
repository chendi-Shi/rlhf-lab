"""Explicit one-hop PPO evidence inheritance; never rewrites an R3 artifact.

This module does not restore model/optimizer/RNG state. The caller must also use
checkpointing.load_checkpoint against the OLD identity and pinned manifest.
History is accepted only from an exact, paused, committed R3 boundary. A new
identity is attached only to newly observed work; old anchors retain their old
identity. Incomplete observed phases are rejected, never synthetically closed.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

from . import training_telemetry as legacy

PROTOCOL = "alignlab-throughput-telemetry-v1"
MIGRATION_KIND = "alignlab-throughput-telemetry-migration-v1"
HISTORY_KEYS = {"prior_progress", "metrics", "cost_events", "checkpoint"}
METRIC_KEYS = ("update", "responses", "action_tokens", "objective", "gradient_norm")
ROLLBACK_AUTHORIZATION_SHA256 = "3a60e225ebca09ffd334d5fe533647bf82cf4df91a6082ae415114cf69bf5e72"
ROLLBACK_INCIDENT_SHA256 = "144117ffb05cfaf3745e0dffc6f1cb85912dcdc6bad6445f72e02cf69b62333d"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _path(path):
    path = Path(path)
    _require(path.is_absolute(), "Evidence paths must be absolute")
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             "Evidence symlink or symlink ancestor forbidden")
    return path


def _pin(pin, *, root=None):
    _require(type(pin) is dict and set(pin) == {"path", "sha256"}, "Exact SHA pin required")
    path = _path(pin["path"])
    if root is not None:
        _require(path.resolve().is_relative_to(Path(root).resolve()), "Evidence escaped its owning stage")
    _require(legacy.sha(path) == pin["sha256"], "Evidence SHA pin mismatch")
    return path


def _jsonl(pin):
    """Read a fixed, closed ledger and bracket parsing by the same SHA anchor."""
    path = _pin(pin)
    records = []
    def unique(pairs):
        out = {}
        for key, value in pairs:
            _require(key not in out, "Duplicate ledger JSON key")
            out[key] = value
        return out
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            _require(bool(line.strip()), "Blank ledger record")
            item = json.loads(line, object_pairs_hook=unique,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite ledger JSON")))
            _require(type(item) is dict, "Ledger record must be an object")
            legacy.plain(item)
            records.append(item)
    _pin(pin)
    return records


def _observed_metric_values(record):
    """R3 drops the scalar ``responses`` while filtering rollout payloads.

    Its original response_indices and record_ids remain observed evidence of
    the exact optimizer minibatch. Derive the denominator from those fields;
    never add a fabricated scalar to the immutable historical record.
    """
    indices, ids = record.get("response_indices"), record.get("record_ids")
    _require(type(indices) is list and len(indices) == 1 and type(indices[0]) is int and indices[0] >= 0
             and type(ids) is list and len(ids) == 1 and type(ids[0]) is str,
             "Observed PPO minibatch indices/record IDs are missing or changed")
    responses = len(indices)
    _require("responses" not in record or record["responses"] == responses,
             "Observed PPO response denominator disagrees with actual minibatch indices")
    return {key: responses if key == "responses" else record[key] for key in METRIC_KEYS}


def _metrics(records, first, last, points, *, allow_observed_only=False):
    objectives = [r for r in records if r.get("event") == "objective_after_engine_return"]
    ends = [p for p in points if p.get("event") == "optimizer_end"]
    expected = list(range(first, last + 1))
    required_objectives = expected[:len(objectives)] if allow_observed_only else expected
    _require([r.get("update") for r in objectives] == required_objectives and len(objectives) <= len(expected),
             "Objective ledger must contain each real update exactly once in order")
    _require([r.get("update") for r in ends] == expected,
             "Observed optimizer ends must contain each real update exactly once")
    _require([p.get("target_update") for p in points if p.get("event") == "optimizer_start"] == expected,
             "Observed optimizer starts omit or repeat an update")
    for end in ends:
        values = _observed_metric_values(end)
        _require(values["responses"] == 1 and type(end.get("action_tokens")) is int
                 and 1 <= end["action_tokens"] <= 256
                 and all(_number(end.get(k)) for k in ("objective", "gradient_norm")),
                 "Invalid original PPO objective evidence")
    for objective, end in zip(objectives, ends):
        _require(all(objective.get(k) == v for k, v in _observed_metric_values(end).items()),
                 "Objective ledger differs from its actual observed optimizer end")
    boundaries = [r["cursor"]["optimizer_updates"] for r in records
                  if r.get("event") == "engine_boundary"]
    _require(boundaries == sorted(set(boundaries))
             and all(first <= update <= last for update in boundaries),
             "Boundary ledger repeats or escapes the committed segment")
    return objectives


def _cost_segment(points, bindings, *, started, finished, warm_started, resumed=False,
                  complete=False, allow_checkpoint_stop=False):
    """Validate actual measured phases without adding a fake completion event."""
    _require(points and _number(started) and _number(finished) and started <= finished,
             "Missing measured segment boundaries")
    epochs = [p.get("time_epoch") for p in points]
    _require(all(_number(t) and started <= t <= finished for t in epochs)
             and epochs == sorted(epochs), "Measured segment clock moved backwards or escaped bounds")
    names = [p.get("event") for p in points]
    return_count = names.count("training_engine_return")
    _require(return_count == 1 or allow_checkpoint_stop and return_count == 0,
             "One actual engine return required per segment")
    if complete:
        _require(names.count("telemetry_complete") == 1 and names[-1] == "telemetry_complete",
                 "Final telemetry completion event missing or repeated")
    else:
        _require("telemetry_complete" not in names, "Paused history must not claim completed telemetry")
    result = {"observer_seconds": 0.0, "checkpoint_seconds": 0.0, "observer_cuda_sync_seconds": 0.0}
    spans = dict(legacy.COST_SPANS)
    if resumed:
        spans["engine_wall_seconds"] = ("resume_training_start", "training_engine_return")
        _require("warm_training_start" not in names, "Original warm clock was reset")
    for label, (begin, end) in spans.items():
        if label == "engine_wall_seconds" and allow_checkpoint_stop and return_count == 0:
            _require(names.count(begin) == 1, "Original engine start missing")
            continue  # The unrecorded return time is unknown; never invent it.
        if begin in names or end in names:
            _require(names.count(begin) == names.count(end) == 1, "Measured cost span is incomplete or repeated")
            seconds = epochs[names.index(end)] - epochs[names.index(begin)]
            _require(seconds >= 0, "Negative measured cost span")
            result[label] = seconds
    active, ended = {}, {}
    pairs = {"sampling_end": "sampling_start", "frozen_scoring_end": "frozen_scoring_start",
             "optimizer_end": "optimizer_start"}
    axes = {"sampling_start": "sampling_seconds", "frozen_scoring_start": "frozen_scoring_seconds",
            "optimizer_start": "optimizer_active_seconds"}
    for point in points:
        name = point["event"]
        if name in pairs.values():
            _require(name not in active and name not in ended, "Duplicate phase start")
            key = point["target_update"] if name == "optimizer_start" else point["rollout_index"]
            active[name] = {"id": key, "boundary": point["boundary_epoch"], "start": None}
        elif name == "phase_interval_start":
            phase = point["phase"]
            _require(phase in active and active[phase]["start"] is None
                     and active[phase]["id"] == point["phase_id"]
                     and active[phase]["boundary"] <= point["started_at_epoch"] <= point["time_epoch"],
                     "Measured phase start is detached from its observed boundary")
            active[phase]["start"] = point["started_at_epoch"]
        elif name in pairs:
            phase = pairs[name]
            key = point["target_update"] if name == "optimizer_end" else point["rollout_index"]
            _require(phase in active and active[phase]["start"] is not None
                     and active[phase]["id"] == key, "Measured phase end has no matching start")
            ended[phase] = {**active.pop(phase), "end": point["boundary_epoch"]}
        if name == "measured_phase":
            phase = point["phase"]
            _require(phase in ended and point["phase_id"] == ended[phase]["id"]
                     and point["started_at_epoch"] == ended[phase]["start"]
                     and point["finished_at_epoch"] == ended[phase]["end"],
                     "Measured phase was changed, duplicated or omitted")
            result[axes[phase]] = result.get(axes[phase], 0.0) + legacy._duration(point)
            ended.pop(phase)
        elif name in ("observer_interval", "cuda_synchronization"):
            axis = "observer_seconds" if name == "observer_interval" else "observer_cuda_sync_seconds"
            result[axis] += legacy._duration(point)
    _require(not active and not ended, "Open observed phase cannot be safely inherited")
    for binding in bindings:
        begin, end = binding["save_started_at_epoch"], binding["save_finished_at_epoch"]
        _require(_number(begin) and _number(end) and started <= begin <= end <= finished,
                 "Checkpoint save interval escaped its measured segment")
        result["checkpoint_seconds"] += end - begin
    _require(_number(warm_started) and warm_started <= finished, "Original warm clock missing")
    return result


def _bindings(receipts, root, identity_sha256):
    _require(type(receipts) is list and receipts, "Committed checkpoint receipts required")
    result, seen = [], set()
    for receipt in receipts:
        directory = _path(receipt["checkpoint_path"])
        _require(directory.parent.resolve() == Path(root).resolve() and str(directory) not in seen,
                 "Checkpoint escaped its stage or repeated")
        seen.add(str(directory))
        manifest = legacy.read(_pin({"path": str(directory / "manifest.json"),
                                   "sha256": receipt["manifest_sha256"]}, root=root))
        _require(manifest.get("status") == "complete" and legacy.digest(manifest["identity"]) == identity_sha256
                 and manifest.get("identity_sha256") == identity_sha256, "Checkpoint identity mismatch")
        binding = legacy.read(_pin(receipt["telemetry_binding"], root=Path(root) / "telemetry"))
        sidecar = legacy.read(_pin(receipt["telemetry_sidecar"], root=Path(root) / "telemetry"))
        _require(binding.get("checkpoint_path") == str(directory)
                 and binding.get("manifest_sha256") == receipt["manifest_sha256"]
                 and binding.get("sidecar") == receipt["telemetry_sidecar"]
                 and sidecar.get("identity_sha256") == identity_sha256
                 and sidecar.get("cursor") == receipt["cursor"]
                 and sidecar.get("cursor_sha256") == legacy.digest(receipt["cursor"])
                 and receipt["optimizer_updates"] == receipt["cursor"]["optimizer_updates"],
                 "Checkpoint receipt, binding and sidecar disagree")
        result.append(binding)
    return result


def _rollouts(pins, *, old_count, old_identity, new_identity, old_root, new_root=None):
    records, ids = [], set()
    for index, pin in enumerate(pins):
        inherited = index < old_count
        root, identity = (old_root, old_identity) if inherited else (new_root, new_identity)
        _require(root is not None, "Unexpected new trajectory in inherited history")
        record = legacy.read(_pin(pin, root=root))
        anchor = legacy.read(_pin(record["generation_anchor"], root=root))
        _require(record.get("protocol") == legacy.PROTOCOL and anchor.get("protocol") == legacy.PROTOCOL
                 and record.get("rollout_index") == anchor.get("rollout_index") == index
                 and record.get("identity_sha256") == anchor.get("identity_sha256") == identity
                 and anchor.get("adapter_sha256") == legacy.digest(anchor["tensors"])
                 and anchor.get("optimizer_updates") == record.get("generated_at_optimizer_updates")
                 and record.get("summary") == legacy.rollout_summary(record),
                 "Original trajectory identity, sampling anchor or summary changed")
        legacy.verify_rollout_origin(record, anchor)
        _require(not ids.intersection(record["record_ids"]), "A registered prompt was sampled twice")
        ids.update(record["record_ids"])
        records.append(record)
    return records


def _lifecycle(points, records, cursor):
    total, completed = len(records), cursor["completed_rollouts"]
    _require(type(cursor.get("active_rollout")) is bool and total == completed + int(cursor["active_rollout"]),
             "Trajectory count differs from the exact committed cursor")
    for event in ("rollout_start", "rollout_collected", "sampling_start", "sampling_end",
                  "frozen_scoring_start", "frozen_scoring_end"):
        _require([p.get("rollout_index") for p in points if p.get("event") == event] == list(range(total)),
                 "Trajectory lifecycle omitted or duplicated: " + event)
    _require([p.get("rollout_index") for p in points if p.get("event") == "rollout_finished"] == list(range(completed)),
             "Completed rollout ledger differs from cursor")
    _require(cursor["offset"] == sum(r["summary"]["responses"] for r in records)
             and cursor["response_tokens"] == sum(r["summary"]["action_tokens"] for r in records),
             "Trajectory denominator differs from committed cursor")
    for record in records:
        index = record["rollout_index"]
        start = next(p for p in points if p["event"] == "rollout_start" and p["rollout_index"] == index)
        _require(start["cursor"]["optimizer_updates"] == record["generated_at_optimizer_updates"],
                 "Trajectory was regenerated after its original optimizer boundary")


def _sidecar_prefixes(receipts, rollout_pins, records):
    for receipt in receipts:
        sidecar = legacy.read(_pin(receipt["telemetry_sidecar"]))
        cursor = receipt["cursor"]
        count = cursor["completed_rollouts"] + int(cursor["active_rollout"])
        _require(sidecar["rollouts"] == rollout_pins[:count]
                 and sidecar["generation_anchors"] == [r["generation_anchor"] for r in records[:count]]
                 and sidecar.get("active_rollout") == (rollout_pins[count - 1] if cursor["active_rollout"] else None),
                 "Checkpoint sidecar substituted historical sampling evidence")


def validate_rollback_authorization(history, progress):
    """Validate the one explicitly authorized incident, without loading tensors.

    The authorization is a fixed external trust anchor, not caller-supplied
    permission. Its original log pins remain intact, including abandoned work.
    """
    if "rollback_authorization" not in history:
        return None
    pin = history["rollback_authorization"]
    _require(pin.get("sha256") == ROLLBACK_AUTHORIZATION_SHA256, "Unrecognized rollback authorization")
    auth = legacy.read(_pin(pin))
    _require(auth.get("kind") == "alignlab-explicit-rollback-replay-authorization-v1"
             and auth.get("source") == "direct_user_reply_in_this_chat"
             and auth.get("incident", {}).get("sha256") == ROLLBACK_INCIDENT_SHA256,
             "Rollback authorization provenance differs")
    for flag in ("permit_original_backend_continuation",
                 "permit_optimized_backend_only_after_actual_cuda_equivalence_and_measured_speedup",
                 "preserve_failed_history", "replayed_updates_must_be_separately_accounted",
                 "require_new_entry_safe_boundary_save_before_guard_stop", "do_not_change_original_179_sources",
                 "no_new_seed_or_data_or_protocol", "no_parallel_gpu_training", "no_deadline_extension"):
        _require(auth.get(flag) is True, "Rollback constraint missing: " + flag)
    incident = legacy.read(_pin(auth["incident"]))
    original_plan = legacy.read(_pin(auth["original_plan"]))
    _pin(auth["throughput_authorization"])
    observed, committed = auth["prior_observed_optimizer_updates"], auth["restore_optimizer_updates"]
    _require(type(observed) is type(committed) is int and 0 < committed < observed
             and auth["replay_first_update"] == committed + 1 and auth["replay_last_update"] == observed
             and auth["replay_updates"] == observed - committed
             and incident.get("observed_optimizer_updates") == observed
             and incident.get("last_saved_optimizer_updates") == committed
             and incident.get("updates_after_last_saved_checkpoint") == observed - committed
             and incident.get("saved_boundary_on_hard_block") is None
             and incident.get("campaign_status") == "failed", "Authorized incident update counts differ")
    _require(auth["original_deadline_epoch"] == progress["deadline_epoch"] == incident["original_deadline_epoch"]
             and auth["original_started_at_epoch"] + auth["wall_seconds"] == auth["original_deadline_epoch"]
             and original_plan.get("wall_seconds") == auth["wall_seconds"], "Rollback reset the original budget")
    _require(progress.get("actual_optimizer_updates") == progress.get("cursor", {}).get("optimizer_updates") == observed
             and progress.get("latest_safe_checkpoint", {}).get("optimizer_updates") == committed
             and progress.get("error") == "InterruptedError: Strict process boundary blocked: hard_blocker",
             "Rollback terminal progress is not the authorized hard block")
    checkpoint = history["checkpoint"]
    recorded = incident["checkpoint"]
    _require(all(recorded.get(k) == checkpoint[k] for k in ("path", "manifest_sha256", "sidecar", "binding")),
             "Rollback substituted the incident checkpoint")
    for name, key in (("prior_progress", "ppo42/progress.json"), ("metrics", "ppo42/metrics.jsonl"),
                      ("cost_events", "ppo42/telemetry/cost-events.jsonl")):
        _require(history[name] == incident["pins"][key], "Rollback substituted original incident evidence")
        _pin(history[name])
    for evidence in incident["pins"].values():
        _pin(evidence)
    guard = _jsonl(incident["pins"]["ppo42/process-guard.jsonl"])
    final = guard[-1]
    _require(final == progress["process_guard"]["last"] and final.get("event") == "result"
             and final.get("admission_allowed") is False and final.get("reason") == "hard_blocker"
             and final.get("optimizer_updates") == observed and final.get("saved_boundary") is None
             and final.get("boundary") == "optimizer_complete"
             and any(b.get("code") == "competing_process" for b in final.get("last_guard", {}).get("blockers", [])),
             "Rollback lacks the exact observed competing-process hard-block evidence")
    return auth


def verify_resume_evidence(history, *, expected_old_identity_sha256):
    """Independently validate closed R3 evidence; reads the pinned checkpoint on CPU.

    No state is restored and no files are written. ``metrics`` contains original
    optimizer_end observations; ``metric_values`` contains their five engine keys.
    This v1 intentionally refuses multi-hop resumptions and partially observed
    aborted phases. Process-exit authorization belongs to the caller's gate.
    """
    _require(type(history) is dict and set(history) in (HISTORY_KEYS, HISTORY_KEYS | {"rollback_authorization"}),
             "Exact resume history schema required")
    checkpoint = history["checkpoint"]
    _require(type(checkpoint) is dict and set(checkpoint) == {"path", "manifest_sha256", "sidecar", "binding"},
             "Exact checkpoint evidence schema required")
    progress_path = _pin(history["prior_progress"])
    root = progress_path.parent
    _require(progress_path.name == "progress.json", "Original stage progress path required")
    progress = legacy.read(progress_path)
    rollback = validate_rollback_authorization(history, progress)
    _require(progress.get("kind") == "alignlab-formal-alignment-stage-r3-v1"
             and progress.get("stage") == "ppo" and progress.get("status") == "paused"
             and progress.get("safe_resume_available") is True, "Only a committed paused R3 PPO stage can be inherited")
    _require(_pin(history["metrics"]).resolve() == root / "metrics.jsonl"
             and _pin(history["cost_events"]).resolve() == root / "telemetry/cost-events.jsonl",
             "History ledgers must belong to the exact stopped stage")
    latest = progress["latest_safe_checkpoint"]
    _require(latest in progress["checkpoints"] and latest["checkpoint_path"] == checkpoint["path"]
             and latest["manifest_sha256"] == checkpoint["manifest_sha256"]
             and latest["telemetry_sidecar"] == checkpoint["sidecar"]
             and latest["telemetry_binding"] == checkpoint["binding"], "Resume must use the exact final stop checkpoint")
    manifest = legacy.read(_pin({"path": str(_path(checkpoint["path"]) / "manifest.json"),
                                "sha256": checkpoint["manifest_sha256"]}, root=root))
    identity = manifest["identity"]
    _require(manifest.get("identity_sha256") == expected_old_identity_sha256
             and legacy.digest(identity) == expected_old_identity_sha256
             and progress.get("identity") == identity, "Original checkpoint and stage identities disagree")
    verified = legacy.verify_checkpoint_telemetry(checkpoint["path"],
        expected_manifest_sha256=checkpoint["manifest_sha256"], expected_sidecar_sha256=checkpoint["sidecar"]["sha256"])
    sidecar = legacy.read(_pin(checkpoint["sidecar"], root=root / "telemetry"))
    cursor = verified["cursor"]
    committed = cursor["optimizer_updates"]
    observed = rollback["prior_observed_optimizer_updates"] if rollback else committed
    _require(verified["sidecar"] == sidecar and cursor == latest["cursor"]
             and committed == latest["optimizer_updates"]
             and (rollback is not None and committed == rollback["restore_optimizer_updates"]
                  or cursor == progress.get("cursor") and committed == progress.get("actual_optimizer_updates")),
             "Stop checkpoint lags or differs from actual committed work")
    _require(progress["deadline_epoch"] == progress["request"]["deadline_epoch"]
             and identity["config"]["work"]["deadline_epoch"] == progress["deadline_epoch"],
             "Historical absolute deadline differs")
    bindings = _bindings(progress["checkpoints"], root, expected_old_identity_sha256)
    _require(all(r["optimizer_updates"] <= cursor["optimizer_updates"] for r in progress["checkpoints"]),
             "Historical checkpoint exceeds stop cursor")
    points = _jsonl(history["cost_events"])
    warm = [p["time_epoch"] for p in points if p.get("event") == "warm_training_start"]
    _require(len(warm) == 1, "Original warm-training boundary missing")
    returns = [p for p in points if p.get("event") == "training_engine_return"]
    metrics = _jsonl(history["metrics"])
    stop_boundaries = [r for r in metrics if r.get("event") == "requested_stop_boundary"]
    returned = len(returns) == 1 and returns[0].get("cursor") == cursor and returns[0].get("stop_reason") == "paused"
    checkpoint_stopped = (not returns and stop_boundaries and stop_boundaries[-1].get("cursor") == cursor
                          and stop_boundaries[-1].get("active_rollout_saved") is cursor["active_rollout"])
    _require(returned or checkpoint_stopped or rollback is not None and not returns and not stop_boundaries,
             "Neither actual engine return nor exact requested-stop checkpoint proves the pause")
    costs = _cost_segment(points, bindings, started=progress["started_at_epoch"],
        finished=progress["finished_at_epoch"], warm_started=warm[0],
        allow_checkpoint_stop=bool(checkpoint_stopped) or rollback is not None)
    objectives = _metrics(metrics, 1, observed, points, allow_observed_only=True)
    if "objective_records" in progress:
        _require(progress["objective_records"] == len(objectives), "Historical objective count differs from its ledger")
    all_points = points
    all_observed_metrics = [p for p in points if p.get("event") == "optimizer_end"]
    observed_metrics = all_observed_metrics[:committed]
    abandoned = None
    if rollback is not None:
        binding = legacy.read(_pin(checkpoint["binding"]))
        points = [p for p in all_points if p["time_epoch"] <= binding["save_started_at_epoch"]]
        _require([p["update"] for p in points if p.get("event") == "optimizer_end"] == list(range(1, committed + 1)),
                 "Checkpoint timestamp does not isolate its actual committed optimizer prefix")
        # This authorization preserves the existing active rollout only. Replaying
        # uncheckpointed generation would need a separately reviewed protocol.
        _require(cursor["active_rollout"] is True
                 and all(progress["cursor"].get(k) == cursor.get(k) for k in
                         ("config_sha256", "data_sha256", "order", "schema", "offset", "response_tokens", "completed_rollouts", "active_rollout"))
                 and not any(p.get("event") in ("rollout_start", "rollout_collected", "rollout_finished", "sampling_start")
                             for p in all_points[len(points):]), "Rollback crossed an uncheckpointed rollout boundary")
        abandoned = {"first_update": committed + 1, "last_update": observed, "updates": observed - committed,
                     "original_cost_events": copy.deepcopy(history["cost_events"]),
                     "original_metrics": copy.deepcopy(history["metrics"]),
                     "reason": "authorized_rollback_after_unsaved_hard_block", "costs_retained": True}
    records = _rollouts(sidecar["rollouts"], old_count=len(sidecar["rollouts"]),
        old_identity=expected_old_identity_sha256, new_identity=None, old_root=root / "telemetry")
    _lifecycle(points, records, cursor)
    _sidecar_prefixes(progress["checkpoints"], sidecar["rollouts"], records)
    _require(sidecar["generation_anchors"] == [r["generation_anchor"] for r in records],
             "Historical sampling anchor inventory changed")
    _require(sidecar.get("active_rollout") == (sidecar["rollouts"][-1] if cursor["active_rollout"] else None),
             "Active rollout pin differs from committed trajectory")
    _pin(history["prior_progress"])
    return {"history": copy.deepcopy(history), "history_sha256": legacy.digest(history),
        "identity": identity, "identity_sha256": expected_old_identity_sha256, "root": str(root),
        "progress": progress, "cursor": cursor, "extra_state": verified["extra_state"], "sidecar": sidecar,
        "metrics": observed_metrics, "metric_values": [_observed_metric_values(r) for r in observed_metrics],
        "metrics_source": {"kind": "actual_optimizer_end_events", "pin": copy.deepcopy(history["cost_events"]),
                           "response_denominator": "length_of_original_response_indices_cross_checked_with_record_ids"},
        "post_engine_metrics": objectives, "engine_return_observed": bool(returned),
        "committed_optimizer_updates": committed, "original_observed_optimizer_updates": observed,
        "rollback_authorization": rollback, "abandoned_segment": abandoned,
        "replay_reference_metrics": [_observed_metric_values(r) for r in all_observed_metrics[committed:]],
        "points": points, "all_points": all_points, "bindings": bindings, "costs": costs, "warm_started_at_epoch": warm[0]}


def _migration(pin, history, old_identity, new_identity):
    value = legacy.read(_pin(pin))
    fields = {"kind", "history_sha256", "old_identity_sha256", "new_identity_sha256", "original_deadline_epoch"}
    _require(type(value) is dict and set(value) == fields and value["kind"] == MIGRATION_KIND
             and value["history_sha256"] == legacy.digest(history)
             and value["old_identity_sha256"] == legacy.digest(old_identity)
             and value["new_identity_sha256"] == legacy.digest(new_identity), "Explicit identity migration pin mismatch")
    old, new = old_identity["config"], new_identity["config"]
    for key in ("stage", "scope", "synthetic_rows", "work", "registered_workload", "model", "selection", "optimizer"):
        _require(key in old and new.get(key) == old[key], "Migration changed frozen numerical configuration: " + key)
    _require(value["original_deadline_epoch"] == old["work"]["deadline_epoch"]
             and new_identity["parents"] == old_identity["parents"], "Migration changed deadline or parent lineage")
    for label, record in old_identity["inputs"].items():
        if label.startswith(("base/", "tokenizer/", "data/")):
            replacement = new_identity["inputs"].get(label, {})
            _require(all(replacement.get(k) == record[k] for k in ("sha256", "size_bytes")),
                     "Migration changed frozen model/tokenizer/data: " + label)
    return value


def _sum_costs(before, after, *, started, warm_started, finished):
    costs = {key: before.get(key, 0.0) + after.get(key, 0.0) for key in before.keys() | after.keys()}
    costs["stage_wall_seconds_to_telemetry_commit"] = finished - started
    costs["warm_training_wall_seconds_to_telemetry_commit"] = finished - warm_started
    return costs


def _update_accounting(inherited, final_update):
    committed = inherited["committed_optimizer_updates"]
    observed = inherited["original_observed_optimizer_updates"]
    replayed = observed - committed
    return {"logical_optimizer_updates": final_update,
            "original_observed_optimizer_updates": observed,
            "inherited_committed_optimizer_updates": committed,
            "new_physical_optimizer_updates": final_update - committed,
            "total_physical_optimizer_updates": observed + final_update - committed,
            "abandoned_segment": inherited["abandoned_segment"], "replayed_optimizer_updates": replayed,
            "rollback_authorization": inherited["history"].get("rollback_authorization"),
            "prior_final_state_bitwise_verified": False if replayed else None,
            "abandoned_execution_costs_retained": True}


class ResumedTrainingTelemetry(legacy.TrainingTelemetry):
    """Fresh writer with pinned, external historical evidence and a new identity."""
    def __init__(self, output, *, history, migration_pin, new_identity, clock=time.time):
        migration = legacy.read(_pin(migration_pin))
        inherited = verify_resume_evidence(history, expected_old_identity_sha256=migration["old_identity_sha256"])
        _migration(migration_pin, history, inherited["identity"], new_identity)
        output = _path(output)
        _require(not output.exists() and not output.resolve().is_relative_to(Path(inherited["root"]).resolve()),
                 "Resumed telemetry needs fresh output outside the original stage")
        now = clock()
        _require(inherited["progress"]["finished_at_epoch"] <= now < migration["original_deadline_epoch"],
                 "Resume is before the stop or beyond the original deadline")
        super().__init__(output, stage="ppo", started_at_epoch=inherited["progress"]["started_at_epoch"], clock=clock)
        self._history, self._migration_pin, self._new_identity = inherited, copy.deepcopy(migration_pin), copy.deepcopy(new_identity)
        self._segment_started, self._resume_started = now, None
        self.set_identity(new_identity)
        self.rollouts = copy.deepcopy(inherited["sidecar"]["rollouts"])
        self.anchors = {i: copy.deepcopy(pin) for i, pin in enumerate(inherited["sidecar"]["generation_anchors"])}
        self.checkpoints = [copy.deepcopy(r["telemetry_binding"]) for r in inherited["progress"]["checkpoints"]]
        self.cost = copy.deepcopy(inherited["costs"])
        self.warm_started = inherited["warm_started_at_epoch"]
        self._inheritance_pin = legacy.write_new(self.path / "inheritance.json", {
            "kind": PROTOCOL, "history": history, "migration": migration_pin,
            "new_identity": new_identity, "segment_started_at_epoch": now})
        self.point("resume_evidence_bound", history_sha256=inherited["history_sha256"],
                   old_identity_sha256=inherited["identity_sha256"], new_identity_sha256=self.identity_sha256,
                   committed_updates=inherited["cursor"]["optimizer_updates"])

    def warm_start(self):
        _require(self._resume_started is None, "Resume engine clock cannot reset")
        self._resume_started = self.clock()
        self.point("resume_training_start", inherited_warm_started_at_epoch=self.warm_started)

    def checkpoint_sidecar(self, name, cursor, extra):
        _require(cursor["optimizer_updates"] >= self._history["cursor"]["optimizer_updates"],
                 "Resumed cursor moved before inherited boundary")
        return super().checkpoint_sidecar(name, cursor, extra)

    def finish(self, *, cursor, completion, checkpoints=None):
        _require(checkpoints is not None, "Explicit cumulative checkpoint receipts required")
        _require(not self._closed and self._pending is None and not getattr(self, "_phase_starts", {})
                 and self._resume_started is not None and not cursor["active_rollout"],
                 "Cannot complete an open resumed execution")
        self.point("telemetry_complete", cursor=cursor)
        finished = self.clock()
        _require(finished <= self._history["progress"]["deadline_epoch"], "Original deadline exhausted before commit")
        metrics_path = self.path.parent / "metrics.jsonl"
        new_metrics = {"path": str(metrics_path.resolve()), "sha256": legacy.sha(metrics_path)}
        files = {p.relative_to(self.path).as_posix(): {"sha256": legacy.sha(p), "size_bytes": p.stat().st_size}
                 for p in sorted(self.path.rglob("*")) if p.is_file()}
        manifest = {"schema_version": 1, "kind": PROTOCOL, "status": "complete", "stage": "ppo",
            "identity_sha256": self.identity_sha256, "inheritance": self._inheritance_pin,
            "started_at_epoch": self.started, "warm_training_started_at_epoch": self.warm_started,
            "segment_started_at_epoch": self._segment_started, "finished_at_epoch": finished,
            "original_deadline_epoch": self._history["progress"]["deadline_epoch"],
            "final_cursor": legacy.plain(cursor), "completion": legacy.plain(completion),
            "rollouts": self.rollouts, "checkpoint_bindings": self.checkpoints,
            "checkpoint_receipts": legacy.plain(checkpoints), "files": files,
            "new_metrics": new_metrics, "costs": None,
            "update_accounting": _update_accounting(self._history, cursor["optimizer_updates"]),
            "cost_coverage": {"historical_engine_return_observed": self._history["engine_return_observed"],
                              "engine_wall_seconds": "sum_of_recorded_closed_engine_spans_only",
                              "warm_wall_origin": "original_recorded_warm_training_start_event"},
            "clock_semantics": "Original stage/deadline retained. Warm wall uses the original recorded warm_training_start event; paused R3 does not persist its separate in-memory warm clock. Phase costs sum actual measured segments; no synthetic events."}
        manifest["costs"] = _verify_manifest(self.path, manifest, self._history, expected_metrics_pin=new_metrics,
                                             expected_checkpoints=checkpoints, verify_costs=False)
        self.cost = manifest["costs"]
        result = legacy.write_new(self.path / "manifest.json", manifest)
        self._closed = True
        return result


def _verify_manifest(path, manifest, inherited, *, expected_metrics_pin=None, expected_checkpoints=None, verify_costs=True):
    m, h = manifest, inherited
    _require(m.get("kind") == PROTOCOL and m.get("status") == "complete" and m.get("stage") == "ppo",
             "Invalid resumed telemetry manifest")
    inheritance = legacy.read(_pin(m["inheritance"], root=path))
    _require(inheritance["history"] == h["history"], "Inherited history was substituted")
    _require(m.get("cost_coverage") == {"historical_engine_return_observed": h["engine_return_observed"],
                                      "engine_wall_seconds": "sum_of_recorded_closed_engine_spans_only",
                                      "warm_wall_origin": "original_recorded_warm_training_start_event"},
             "Measured versus unavailable historical clock coverage changed")
    migration = _migration(inheritance["migration"], h["history"], h["identity"], inheritance["new_identity"])
    _require(m["identity_sha256"] == migration["new_identity_sha256"]
             and m["original_deadline_epoch"] == migration["original_deadline_epoch"]
             and m["started_at_epoch"] == h["progress"]["started_at_epoch"]
             and m["warm_training_started_at_epoch"] == h["warm_started_at_epoch"]
             and m["segment_started_at_epoch"] == inheritance["segment_started_at_epoch"]
             and h["progress"]["finished_at_epoch"] <= m["segment_started_at_epoch"] <= m["finished_at_epoch"] <= m["original_deadline_epoch"],
             "Resumed stage reset an identity, clock or absolute deadline")
    actual = set()
    for item in path.rglob("*"):
        _require(not item.is_symlink(), "Resumed evidence symlink forbidden")
        if item.is_file() and item != path / "manifest.json":
            actual.add(item.relative_to(path).as_posix())
    _require(actual == set(m["files"]), "Untracked or missing resumed evidence file")
    for name, record in m["files"].items():
        item = _pin({"path": str(path / name), "sha256": record["sha256"]}, root=path)
        _require(item.stat().st_size == record["size_bytes"], "Resumed evidence size changed")
    old_receipts = h["progress"]["checkpoints"]
    receipts = m["checkpoint_receipts"]
    _require(receipts[:len(old_receipts)] == old_receipts, "Original checkpoint receipt prefix changed")
    if expected_checkpoints is not None:
        _require(receipts == legacy.plain(expected_checkpoints), "Cumulative checkpoint receipts differ")
    new_receipts = receipts[len(old_receipts):]
    new_bindings = _bindings(new_receipts, path.parent, m["identity_sha256"])
    _require(m["checkpoint_bindings"] == [r["telemetry_binding"] for r in receipts],
             "Checkpoint binding inventory changed")
    old_rollouts = h["sidecar"]["rollouts"]
    _require(m["rollouts"][:len(old_rollouts)] == old_rollouts, "Original trajectory pin prefix changed")
    records = _rollouts(m["rollouts"], old_count=len(old_rollouts), old_identity=h["identity_sha256"],
        new_identity=m["identity_sha256"], old_root=Path(h["root"]) / "telemetry", new_root=path)
    _sidecar_prefixes(receipts, m["rollouts"], records)
    new_points = _jsonl({"path": str(path / "cost-events.jsonl"), "sha256": m["files"]["cost-events.jsonl"]["sha256"]})
    cursor, completion = m["final_cursor"], m["completion"]
    work = h["identity"]["config"]["registered_workload"]
    _require(cursor["optimizer_updates"] == completion["optimizer_updates"] == work["optimizer_updates"]
             and cursor["offset"] == completion["rows_processed"] == work["rows"]
             and cursor["response_tokens"] == completion["response_tokens"]
             and cursor["completed_rollouts"] == completion["completed_rollouts"]
             and cursor["active_rollout"] is False and completion.get("stop_reason") == "data_exhausted"
             and all(completion.get(k) is True for k in ("data_pass_complete", "protocol_complete", "full_selected_dataset_pass"))
             and completion.get("full_accepted_dataset_pass") is False and completion.get("synthetic_rows") == 0
             and completion.get("scope") == work["scope"], "Incomplete or altered registered PPO work")
    _require(all(cursor.get(k) == h["cursor"].get(k) for k in ("config_sha256", "data_sha256", "order", "schema")),
             "Resumed cursor changed original configuration, data or order")
    _lifecycle(h["points"] + new_points, records, cursor)
    _require(new_receipts[-1]["cursor"] == cursor
             and new_receipts[-1]["optimizer_updates"] == cursor["optimizer_updates"],
             "Final checkpoint is not the exact final cursor")
    if expected_metrics_pin is not None:
        _require(m["new_metrics"] == expected_metrics_pin, "New objective ledger pin differs")
    _require(_pin(m["new_metrics"]).resolve() == path.parent / "metrics.jsonl", "New metrics escaped resumed stage")
    _metrics(_jsonl(m["new_metrics"]), h["cursor"]["optimizer_updates"] + 1, cursor["optimizer_updates"], new_points)
    _require(m.get("update_accounting") == _update_accounting(h, cursor["optimizer_updates"]),
             "Logical, abandoned and physical update accounting differs")
    observed_by_update = {p["update"]: p for p in new_points if p.get("event") == "optimizer_end"}
    for reference in h["replay_reference_metrics"]:
        _require(_observed_metric_values(observed_by_update.get(reference["update"], {})) == reference,
                 "Replayed optimizer metrics differ from actual abandoned observations")
    returns = [p for p in new_points if p.get("event") == "training_engine_return"]
    _require(len(returns) == 1 and returns[0].get("cursor") == cursor and returns[0].get("stop_reason") == "data_exhausted",
             "Final engine did not return the declared cursor")
    measured = _cost_segment(new_points, new_bindings, started=m["segment_started_at_epoch"],
        finished=m["finished_at_epoch"], warm_started=m["warm_training_started_at_epoch"], resumed=True, complete=True)
    costs = _sum_costs(h["costs"], measured, started=m["started_at_epoch"],
        warm_started=m["warm_training_started_at_epoch"], finished=m["finished_at_epoch"])
    if verify_costs:
        _require(m["costs"] == costs, "Cumulative cost summary differs from actual measured segments")
    return costs


def verify_resumed_training_telemetry(output, *, expected_manifest_sha256, expected_identity_sha256,
                                      expected_completion=None, checkpoints=None, expected_metrics_pin=None):
    """Recheck old/new SHA evidence, identities, trajectory origins and every update.

    The old stop payload is inspected using the original checkpoint verifier;
    old/new objective ledgers remain distinct immutable files. This is not an
    optimizer/model restore nor a replacement for the runner's final-weight
    acceptance, authorization, process-clearance or source-equivalence checks.
    """
    path = _path(output)
    manifest = legacy.read(_pin({"path": str(path / "manifest.json"), "sha256": expected_manifest_sha256}))
    _require(manifest.get("identity_sha256") == expected_identity_sha256, "Expected resumed identity differs")
    if expected_completion is not None:
        _require(manifest.get("completion") == expected_completion, "Expected resumed completion differs")
    inheritance = legacy.read(_pin(manifest["inheritance"], root=path))
    migration = legacy.read(_pin(inheritance["migration"]))
    inherited = verify_resume_evidence(inheritance["history"], expected_old_identity_sha256=migration["old_identity_sha256"])
    _verify_manifest(path, manifest, inherited, expected_metrics_pin=expected_metrics_pin, expected_checkpoints=checkpoints)
    _pin({"path": str(path / "manifest.json"), "sha256": expected_manifest_sha256})
    return manifest
