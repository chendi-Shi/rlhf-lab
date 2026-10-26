"""Explicit fresh-output continuation of the original six-branch campaign.

The original start, absolute deadline and observed waiting costs remain binding.
Only an authenticated pre-update cache prefix can cross the failed-attempt
boundary. Old source/evidence files remain immutable; no automatic retry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from . import alignment_supervisor as previous
from . import alignment_process_guard_r2 as process_guard
from .probe_dispatch import _System, _stamp, _parse, _new, _status
from .capacity_supervisor import sha, _pin, _canonical, _prerequisites, _bundle, NF4
from .model_snapshot import verify_snapshot, REPO_ID, REVISION

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-formal-alignment-campaign-r2-v1"
STAGE_KIND = "alignlab-formal-alignment-stage-r2-v1"
EXECUTION_REVISION = previous.EXECUTION_REVISION
TELEMETRY_PROTOCOL = previous.TELEMETRY_PROTOCOL
SEQUENCE = previous.SEQUENCE
NEW_SOURCE_FILES = previous.NEW_SOURCE_FILES | process_guard.SOURCE_FILES | {
    "alignment_lab/alignment_supervisor_r2.py", "alignment_lab/alignment_stage_r2.py",
    "scripts/run_alignlab_alignment_r2.py", "scripts/run_alignlab_alignment_stage_r2.py",
    "tests_v2/test_alignment_supervisor_r2.py", "tests_v2/test_alignment_stage_r2.py",
    "alignment_lab/reference_cache_recovery.py", "tests_v2/test_reference_cache_recovery.py",
    "tests_v2/test_alignment_continuation_integration.py"}
_source = previous._source
_parent_acceptance = previous._parent_acceptance
_command = previous._command
_same_identity = previous._same_identity
_waitable_peer_decision = previous._waitable_peer_decision
_parents_gone = previous._parents_gone
_cache = previous._cache
_cpu_payload = previous._cpu_payload
_checkpoint_content = previous._checkpoint_content
_verify_observation_artifacts = previous._verify_observation_artifacts
workload = previous.workload


def _pinned(record):
    if type(record) is not dict or set(record) != {"path", "sha256"}:
        raise ValueError("Exact external path/SHA evidence required")
    path = Path(record["path"])
    if (not path.is_absolute() or not path.resolve().is_relative_to(ROOT.resolve())
            or path.is_symlink() or any(p.is_symlink() for p in path.parents)):
        raise ValueError("Ordinary project-local continuation evidence required")
    _pin(path, record["sha256"])
    return path.resolve()


def validate_continuation(plan):
    """Authenticate historical failure, unchanged work, absolute budget and user intent."""
    value = plan.get("continuation")
    keys = {"prior_plan", "prior_campaign", "prior_stage_request", "prior_stage_progress", "authorization",
            "started_at_epoch", "deadline_epoch", "process_wait_spent_seconds", "cache_prefix"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError("Exact explicit alignment continuation evidence required")
    paths = {k: _pinned(value[k]) for k in keys - {
        "started_at_epoch", "deadline_epoch", "process_wait_spent_seconds", "cache_prefix"}}
    old_plan, campaign, request, progress, authorization = (
        _parse(paths[k]) for k in ("prior_plan", "prior_campaign", "prior_stage_request",
                                  "prior_stage_progress", "authorization"))
    historical_output = Path(old_plan["output"]).resolve()
    if (old_plan.get("kind") != previous.KIND
            or paths["prior_campaign"] != historical_output / "campaign.json"
            or paths["prior_stage_request"] != historical_output / "dpo42.request.json"
            or paths["prior_stage_progress"] != historical_output / "dpo42/progress.json"
            or Path(plan["output"]).resolve().is_relative_to(historical_output)):
        raise ValueError("Historical failure paths or fresh output differ")
    for key in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir", "audit_manifest_sha256",
                "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path", "parent_plan_sha256",
                "parent_acceptance_path", "parent_acceptance_sha256", "parents", "ppo_rows", "wall_seconds", "poll_seconds"):
        if plan[key] != old_plan[key]:
            raise ValueError("Continuation changed registered work or parent/model/data identity: " + key)
    if any(plan["source_sha256"].get(k) != v for k, v in old_plan["source_sha256"].items()):
        raise ValueError("Every historical frozen dependency must remain unchanged")
    start, deadline, spent = (value[k] for k in ("started_at_epoch", "deadline_epoch", "process_wait_spent_seconds"))
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, deadline, spent))
            or start <= 0 or spent < 0 or deadline != start + old_plan["wall_seconds"]
            or campaign.get("started_at_epoch") != start or campaign.get("deadline_epoch") != deadline
            or campaign.get("kind") != previous.KIND or campaign.get("status") != "failed"
            or campaign.get("plan_sha256") != value["prior_plan"]["sha256"]
            or [s.get("name") for s in campaign.get("stages", [])] != ["dpo42"]
            or progress.get("status") != "paused" or progress.get("stage") != "dpo"
            or progress.get("actual_optimizer_updates") != 0 or progress.get("latest_safe_checkpoint") is not None
            or progress.get("request") != request or request.get("seed") != 42
            or request.get("kind") != previous.STAGE_KIND
            or request.get("deadline_epoch") != deadline
            or request.get("alignment_plan_sha256") != value["prior_plan"]["sha256"]
            or Path(request.get("alignment_plan_path", "")).resolve() != paths["prior_plan"]
            or request.get("workload") != workload("dpo", plan["ppo_rows"])
            or progress.get("process_guard", {}).get("total_waited_seconds") != spent
            or campaign["stages"][0].get("request") != request
            or campaign["stages"][0].get("exit_code") in (None, 0)
            or campaign["stages"][0].get("active_processes") != []):
        raise ValueError("Continuation must retain the failed zero-update attempt and its actual deadline/wait ledger")
    if (authorization.get("kind") != "explicit-user-primary-alignment-continuation-authorization-v1"
            or authorization.get("source") != "direct_user_reply_in_this_chat"
            or authorization.get("allow_continuous_training") is not True
            or authorization.get("origin_plan_sha256") != value["prior_plan"]["sha256"]
            or authorization.get("origin_campaign_sha256") != value["prior_campaign"]["sha256"]
            or authorization.get("original_started_at_epoch") != start
            or authorization.get("original_deadline_epoch") != deadline):
        raise ValueError("Pinned user authorization must name this continuation and unchanged absolute budget")
    prefix = value["cache_prefix"]
    if (type(prefix) is not dict or prefix.get("progress") != value["prior_stage_progress"]
            or _pinned(prefix["request"]) != historical_output / "dpo42/request.json"
            or _parse(prefix["request"]["path"]) != request):
        raise ValueError("Cache prefix belongs to a different failed stage")
    from .reference_cache_recovery import audit_prefix
    audit_prefix(prefix)
    return {"origin_plan": old_plan, "origin_campaign": campaign, "prior_stage_request": request,
            "prior_stage_progress": progress, "authorization": authorization}


def _prior_processes_gone(plan, processes):
    campaign = _parse(plan["continuation"]["prior_campaign"]["path"])
    historical = [*campaign.get("supervisor_processes", []),
                  *(p for stage in campaign["stages"] for p in stage.get("known_processes", []))]
    if not historical:
        raise ValueError("Prior process exit cannot be proved without original full identities")
    by_pid = {p["pid"]: p for p in processes}
    for expected in historical:
        observed = by_pid.get(expected["pid"])
        if observed is not None and (not observed.get("created_at_utc")
                or _same_identity(observed, expected)
                or process_guard.strict.ProcessIdentity.from_record(observed).instance
                   == process_guard.strict.ProcessIdentity.from_record(expected).instance):
            raise RuntimeError("A prior campaign process instance has not verifiably exited")


class _LiveIdentity(previous._LiveIdentity):
    @classmethod
    def from_plan(cls, path, digest, plan):
        sentinel = super().from_plan(path, digest, plan)
        value = plan["continuation"]
        for key in ("prior_plan", "prior_campaign", "prior_stage_request", "prior_stage_progress", "authorization"):
            sentinel.add_file(value[key]["path"], value[key]["sha256"], always_hash=True)
        for key in ("journal", "boundary_receipt", "progress", "request"):
            pin = value["cache_prefix"][key]
            sentinel.add_file(pin["path"], pin["sha256"], always_hash=key != "journal")
        return sentinel


def _verify_prefix_migration(output, request, result, cached, cache_identity):
    """Independently compare every carried score/token/key with historical evidence."""
    from .reference_cache import cache_key, identity_sha256
    from .reference_cache_recovery import audit_prefix
    output = Path(output).resolve()
    spec = request["continuation"]["cache_prefix"]
    pin = result.get("cache_prefix_recovery_report")
    path = _pinned(pin)
    cache = result["reference_cache"]
    if (path != output / "cache-prefix-migration.json"
            or cache.get("migration_report_path") != str(path)
            or cache.get("migration_report_sha256") != pin["sha256"]):
        raise ValueError("Migration report must be pinned at the owning stage path")
    report, audit = _parse(path), audit_prefix(spec)
    old = audit.identity
    new = json.loads(_canonical(cache_identity))
    old_sources, new_sources = old["files"]["source"], new["files"]["source"]
    if not set(old_sources) <= set(new_sources) or any(new_sources[k] != v for k, v in old_sources.items()):
        raise ValueError("Migration changed an original reference source")
    new["files"]["source"] = old_sources
    if new != old:
        raise ValueError("Migration changed actual reference/protocol/data identity")
    added = {k: v for k, v in new_sources.items() if k not in old_sources}
    total, count = spec["expected_unique_sequences"], spec["scored_sequences"]
    if len(cached.entries) != total or len(audit.records) != count:
        raise ValueError("Migration does not cover the complete exact cache inventory")
    old_order, remapped = {}, []
    for key, entry in cached.entries.items():
        row = entry.to_dict()
        prompt, completion = row["prompt_token_ids"], row["completion_token_ids"]
        if cache_key(cache_identity, prompt, completion) != key:
            raise ValueError("Completed cache key differs from actual tokens")
        old_key = cache_key(old, prompt, completion)
        if old_key in old_order:
            raise ValueError("Duplicate historical token inventory")
        old_order[old_key] = row
    if [r["key"] for r in audit.records] != sorted(old_order)[:count]:
        raise ValueError("Migrated scores are not the authenticated original sorted prefix")
    for historical in audit.records:
        row = old_order[historical["key"]]
        if any(row[k] != historical[k] for k in ("prompt_token_ids", "completion_token_ids", "sum_logprob", "token_count")):
            raise ValueError("Migrated token IDs, counts or scores changed")
        remapped.append(row["key"])
    expected = {**audit.summary, "kind": "alignlab-reference-cache-prefix-migration-v1",
        "actual_loaded_reference_verified": True, "new_identity_sha256": identity_sha256(cache_identity),
        "added_sources": added, "new_record_keys_sha256": hashlib.sha256(_canonical(sorted(remapped))).hexdigest(),
        "score_values_unchanged": True, "sequence_tokens_unchanged": True,
        "identity_migration": "strict_source_inventory_superset_with_explicit_rekey",
        "new_optimizer_updates": 0, "formal_training_complete": False,
        "imported_sequences": count, "new_sequences_to_score": total - count,
        "source_journal_sha256": spec["journal"]["sha256"],
        "source_boundary_receipt_sha256": spec["boundary_receipt"]["sha256"],
        "all_original_sources_unchanged": True, "source_inventory_added": added,
        "old_reference_identity": old, "new_reference_identity": cache_identity}
    if (report != expected or cache.get("imported_sequences") != count
            or cache.get("newly_scored_sequences") != total - count):
        raise ValueError("Migration report/counts differ from independently reconstructed evidence")
    return pin


def _verify_cache_files(identity, cache_identity):
    """Reconstruct the receipt's mandatory extra source labels exactly."""
    for group in ("base", "tokenizer", "source"):
        expected = {k: v for k, v in identity["inputs"].items() if k.startswith(group + "/")}
        if group == "source":
            for filename in ("gpu_reference.py", "reference_bridge.py", "shared_qwen.py",
                             "token_logprobs.py", "training_steps.py", "reference_cache.py"):
                record = identity["inputs"]["source/alignment_lab/" + filename]
                path = ROOT / "alignment_lab" / filename
                if Path(record["path"]).resolve() != path.resolve():
                    raise ValueError("Required reference implementation path differs")
                _pin(path, record["sha256"])
                if path.stat().st_size != record["size_bytes"]:
                    raise ValueError("Required reference implementation size differs")
                expected["gpu_reference/" + filename] = record
        if cache_identity["files"][group] != expected:
            raise ValueError("Reference cache uses a different exact base/tokenizer/scoring inventory")


def _verify_live_objectives(output, stage, objectives):
    path = Path(output) / "telemetry/cost-events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    live = [e for e in events if e.get("event") == "optimizer_end"]
    keys = ("update", "objective", "gradient_norm", "rows") if stage == "dpo" else (
        "update", "objective", "gradient_norm", "responses", "action_tokens")
    if (len(live) != len(objectives)
            or any({k: row.get(k) for k in keys} != {k: item.get(k) for k in keys}
                   for row, item in zip(live, objectives))):
        raise ValueError("Live optimizer observations differ from returned actual objectives/work")


def validate_plan(path, expected_sha):
    _pin(path, expected_sha)
    plan = _parse(path)
    fields = {"schema_version", "kind", "output", "wall_seconds", "poll_seconds", "source_sha256",
              "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
              "audit_manifest_sha256", "cpu_check_report", "cpu_check_report_sha256",
              "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path",
              "parent_plan_sha256", "parent_acceptance_path", "parent_acceptance_sha256",
              "parents", "ppo_rows", "prerequisites", "process_wait_policy",
              "allowed_background_processes", "continuation"}
    if (set(plan) != fields or type(plan["schema_version"]) is not int
            or plan["schema_version"] != 1 or plan["kind"] != KIND):
        raise ValueError("Exact versioned primary-alignment plan required")
    if (type(plan["wall_seconds"]) is not int or plan["wall_seconds"] <= 0
            or type(plan["poll_seconds"]) is not int or plan["poll_seconds"] != 10
            or type(plan["ppo_rows"]) is not int or plan["ppo_rows"] != 1024):
        raise ValueError("Explicit positive shared budget, ten-second polling and user-selected PPO1024 required")
    process_guard.validate_policy(plan["process_wait_policy"])
    process_guard.validate_background(plan["allowed_background_processes"])
    output = Path(plan["output"])
    if (not output.is_absolute() or output.is_symlink()
            or output.resolve().parent != (ROOT / "runs").resolve()):
        raise ValueError("Output must be a fresh direct child of project runs")
    original, _ = _parent_acceptance(plan)
    sources = plan["source_sha256"]
    if (not isinstance(sources, dict)
            or not (set(original["source_sha256"]) | NEW_SOURCE_FILES) <= set(sources)
            or any(sources.get(k) != v for k, v in original["source_sha256"].items())):
        raise ValueError("All 59 parent dependencies and observed-v2 implementation/test files must be frozen")
    for name, digest in sources.items():
        _pin(_source(name), digest)
    _pin(plan["cpu_check_report"], plan["cpu_check_report_sha256"])
    cpu = _parse(plan["cpu_check_report"])
    if (cpu.get("status") != "cpu_software_checks_passed" or cpu.get("source_unchanged") is not True
            or cpu.get("cuda_initialized") is not False or cpu.get("device") != "cpu"
            or type(cpu.get("tests_run")) is not int or cpu["tests_run"] < 253
            or any(type(cpu.get(k)) is not int or cpu[k] != 0 for k in ("errors", "failures", "skipped"))):
        raise ValueError("Complete fresh CPU proof required without failures/skips/CUDA")
    checked = {k.replace("\\", "/"): v for k, v in cpu.get("source_before", cpu.get("source_sha256", {})).items()}
    required = set(sources)
    if not required <= set(checked) or any(checked[k] != sources[k] for k in required):
        raise ValueError("Fresh CPU proof does not cover frozen alignment dependencies")
    validate_continuation(plan)
    _prerequisites(plan["prerequisites"])
    _pin(path, expected_sha)
    return plan


def _own_processes(processes, plan_path):
    by_pid = {p["pid"]: p for p in processes}
    current, own = by_pid.get(os.getpid()), {}
    if current is None or not _command(current, "run_alignlab_alignment_r2.py", plan_path):
        raise RuntimeError("Cannot identify current supervisor PID/create-time/command")
    while current and _command(current, "run_alignlab_alignment_r2.py", plan_path):
        if current["pid"] in own:
            raise RuntimeError("Cyclic supervisor ancestry")
        _stamp(current["created_at_utc"])
        own[current["pid"]] = current
        current = by_pid.get(current.get("parent_pid"))
    return own


def _peer_decision(processes, own, known=(), background=()):
    # v2's allow matcher binds PID/create/command, but ancestry is also part of
    # this supervisor's external child proof. Never accept a changed parent.
    known = dict(known)
    actual = {p["pid"]: p for p in processes}
    for pid, expected in known.items():
        observed = actual.get(pid)
        if (observed is not None
                and process_guard.strict.ProcessIdentity.from_record(observed).instance
                == process_guard.strict.ProcessIdentity.from_record(expected).instance
                and not _same_identity(observed, expected)):
            raise RuntimeError("Known child full process identity changed")
    return process_guard.strict_admission(lambda: processes,
        owned_identities=list(own.values()),
        allowed_identities=[*background, *known.values()], deadline_epoch=None)


def _verify_stage(output, request, known, code):
    """Accept an exited child only after real files and exact work are checked."""
    if code != 0:
        raise RuntimeError(f"Alignment child exited {code}; no automatic retry")
    _pin(request["alignment_plan_path"], request["alignment_plan_sha256"])
    pinned_plan = _parse(request["alignment_plan_path"])
    if request.get("kind") != STAGE_KIND or request.get("continuation") != pinned_plan["continuation"]:
        raise ValueError("Stage continuation differs from the admitted r2 plan")
    validate_continuation(pinned_plan)
    output = Path(output).resolve()
    result = _parse(output / "progress.json")
    stage, work = request["stage"], request["workload"]
    rows, updates = work["rows"], work["optimizer_updates"]
    if (result.get("status") != "formal_alignment_stage_passed"
            or result.get("formal_training") is not True or result.get("research_complete") is not False
            or result.get("stage") != stage or result.get("request") != request
            or _parse(output / "request.json") != request or result.get("pid") not in known
            or result.get("actual_optimizer_updates") != updates
            or result.get("deadline_epoch") != request["deadline_epoch"]
            or type(result.get("started_at_epoch")) not in (int, float)
            or type(result.get("finished_at_epoch")) not in (int, float)
            or not result["started_at_epoch"] <= result["finished_at_epoch"] <= request["deadline_epoch"]):
        raise RuntimeError("Missing exact completed request/process/deadline/update evidence")
    model = result.get("model", {})
    if (model.get("device") != "cuda:0" or model.get("quantization") != NF4
            or model.get("quantized_linear_count") != 196
            or type(model.get("parameters_selected")) is not int or model["parameters_selected"] <= 0
            or model.get("snapshot_manifest_sha256") != request["model_manifest_sha256"]):
        raise RuntimeError("Child lacks actual initialized CUDA NF4 model evidence")
    samples = result.get("gpu_samples")
    if (not isinstance(samples, list) or not samples
            or any(not any(g.get("index") == 0 for g in sample.get("gpus", []))
                   or not all(type(sample.get(k)) in (int, float) and math.isfinite(sample[k])
                              and sample[k] > 0 for k in ("max_allocated_bytes", "max_reserved_bytes"))
                   for sample in samples)):
        raise RuntimeError("Child lacks positive actual GPU memory measurements")
    selection = result["selection"]
    partition = "preference_train" if stage == "dpo" else "ppo_prompts"
    if (selection.get("scope") != work["scope"] or selection.get("partition") != partition
            or any(selection.get(k) != rows for k in ("selected_rows", "real_rows"))
            or selection.get("accepted_partition_rows") != (13139 if stage == "dpo" else 12748)
            or selection.get("selection_algorithm") != ("all_accepted_rows" if stage == "dpo" else work["selection_algorithm"])
            or selection.get("selection_seed") != (None if stage == "dpo" else 42)
            or selection.get("synthetic_rows") != 0
            or selection.get("accepted_manifest_sha256") != request["audit_manifest_sha256"]):
        raise ValueError("Planned selection must contain the exact registered real work")
    selection_path = Path(selection["selection_manifest_path"]).resolve()
    if selection_path != output / "selection.json":
        raise ValueError("Selection manifest escaped child output")
    _pin(selection_path, selection["selection_manifest_sha256"])
    saved_selection = _parse(selection_path)
    summary = {k: v for k, v in selection.items()
               if k not in {"selection_manifest_path", "selection_manifest_sha256"}}
    records = saved_selection["records"]
    if (saved_selection["summary"] != summary or len(records) != rows
            or len({r["record_id"] for r in records}) != rows
            or hashlib.sha256(_canonical(records)).hexdigest() != selection["records_sha256"]
            or any(not isinstance(r["record_id"], str) or r["record_id"].startswith("synthetic-")
                   or not 1 <= r["prompt_tokens"] <= 1024
                   or not re.fullmatch("[0-9a-f]{64}", r["token_sha256"])
                   or (stage == "dpo" and not (1 <= r["chosen_tokens"] <= 768
                                               and 1 <= r["rejected_tokens"] <= 768))
                   or (stage == "ppo" and (r["chosen_tokens"] is not None or r["rejected_tokens"] is not None))
                   for r in records)):
        raise ValueError("Invalid selection identity, lengths or token summary")
    manifest = _parse(Path(request["audit_dir"]) / "manifest.json")
    partition_path = Path(request["audit_dir"]) / (partition + ".jsonl")
    partition_sha = manifest["artifacts"][partition_path.name]
    _pin(partition_path, partition_sha)
    with partition_path.open(encoding="utf-8") as stream:
        accepted = [json.loads(line) for line in stream if line.strip()]
    if stage == "ppo":
        accepted.sort(key=lambda r: (hashlib.sha256(
            f'{work["selection_seed"]}:{r["prompt_id"]}'.encode()).hexdigest(), r["prompt_id"]))
        accepted = accepted[:rows]
        if len({r["prompt_id"] for r in records}) != rows:
            raise ValueError("PPO registered prompts must be unique")
    if [(r["record_id"], r["cluster_id"], r["prompt_id"]) for r in records] != [
            (r["record_id"], r["cluster_id"], r["prompt_id"]) for r in accepted]:
        raise ValueError("Selection does not match the exact registered train IDs/order")
    if (selection.get("unique_prompt_ids") != len({r["prompt_id"] for r in records})
            or selection.get("independent_prompt_clusters") != len({r["cluster_id"] for r in records})):
        raise ValueError("Selection cluster/prompt counts differ from its actual records")
    identity = result["identity"]
    if _parse(output / "identity.json") != identity:
        raise ValueError("Saved identity differs from progress")
    config = identity["config"]
    if (config.get("stage") != stage or config.get("purpose") != STAGE_KIND
            or config.get("execution_revision") != EXECUTION_REVISION
            or config.get("telemetry_protocol") != TELEMETRY_PROTOCOL
            or config.get("scope") != work["scope"] or config.get("formal_training") is not True
            or config.get("synthetic_rows") != 0 or config.get("registered_workload") != work
            or config.get("work", {}).get("seed") != request["seed"]):
        raise ValueError("Artifact identity does not declare this registered formal branch")
    if config.get("continuation_revision") != "alignment-r2" or config.get("continuation") != request["continuation"]:
        raise ValueError("Stage identity lost its explicit original-budget continuation provenance")
    if not NEW_SOURCE_FILES <= set(request["source_sha256"]):
        raise ValueError("Completed branch lacks mandatory observed-v2 source identity")
    if set(identity["parents"]) != set(request["parents"]):
        raise ValueError("Branch parent identity differs")
    for name, pin in request["parents"].items():
        item = identity["parents"][name]
        if (item["sha256"] != pin["manifest_sha256"]
                or Path(item["path"]).resolve() != Path(pin["artifact_dir"]).resolve() / "manifest.json"):
            raise ValueError("Branch parent is not the admitted completed formal model")
    expected_imports = {"policy": ("sft", []), "reference": ("sft", [])}
    if stage == "ppo":
        expected_imports["reward"] = ("reward", ["reward"])
    imports = result.get("imports", {})
    identity_sha = hashlib.sha256(_canonical(identity)).hexdigest()
    if set(imports) != set(expected_imports):
        raise ValueError("Missing exact policy/reference/reward initialization receipts")
    for role, (parent, heads) in expected_imports.items():
        receipt = imports[role]
        if (receipt.get("stage") != parent or receipt.get("target_stage") != stage
                or receipt.get("target_role") != role or receipt.get("heads") != heads
                or receipt.get("target_identity_sha256") != identity_sha
                or receipt.get("manifest_sha256") != request["parents"][parent]["manifest_sha256"]):
            raise ValueError("Initialization receipt differs from final branch/cache identity")
    required = {"source/" + name: (ROOT / name, digest) for name, digest in request["source_sha256"].items()}
    required.update({name: (request[name + "_path"], request[name + "_sha256"])
                     for name in ("alignment_plan", "parent_plan", "parent_acceptance", "capacity_acceptance")})
    required.update({
        "data/manifest": (Path(request["audit_dir"]) / "manifest.json", request["audit_manifest_sha256"]),
        "data/partition": (partition_path, partition_sha),
        "data/selection": (selection_path, selection["selection_manifest_sha256"]),
    })
    snapshot = _parse(Path(request["model_snapshot"]) / "snapshot-manifest.json")
    required.update({"base/" + name: (Path(request["model_snapshot"]) / name, item["sha256"])
                     for name, item in snapshot["files"].items()})
    token_dir = Path(request["tokenizer_dir"])
    token_manifest = _parse(token_dir / "source-manifest.json")
    required.update({"tokenizer/" + name: (token_dir / name, item["sha256"])
                     for name, item in token_manifest["artifacts"].items()})
    required["tokenizer/source-manifest.json"] = (token_dir / "source-manifest.json", sha(token_dir / "source-manifest.json"))
    for name, (path, digest) in required.items():
        item = identity["inputs"].get(name, {})
        if item.get("sha256") != digest or Path(item.get("path", "")).resolve() != Path(path).resolve():
            raise ValueError("Frozen branch input identity mismatch: " + name)
    for name, item in identity["inputs"].items():
        if not name.startswith("base/"):
            _pin(item["path"], item["sha256"])
            if Path(item["path"]).stat().st_size != item["size_bytes"]:
                raise ValueError("Frozen branch input size changed")
    cursor = result["cursor"]
    if (result.get("outcome_stop_reason") != "data_exhausted" or cursor.get("offset") != rows
            or cursor.get("optimizer_updates") != updates or cursor.get("active_rollout") is not False
            or cursor.get("completed_rollouts") != (rows // 16 if stage == "ppo" else 0)
            or result.get("adapter_weights_changed") is not True
            or result.get("scalar_heads_changed") != ({"value": True} if stage == "ppo" else {})):
        raise RuntimeError("Incomplete registered optimizer work")
    metrics = [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
               for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    boundaries = [m for m in metrics if m.get("event") == "engine_boundary"]
    objectives = [m for m in metrics if m.get("event") == "objective_after_engine_return"]
    if (result.get("objective_logging") != "live_optimizer_end_and_after_engine_returns" or result.get("objective_records") != updates
            or [m["cursor"]["optimizer_updates"] for m in boundaries] != list(range(1, updates + 1))
            or [m.get("update") for m in objectives] != list(range(1, updates + 1))
            or any(not all(type(m.get(k)) in (int, float) and math.isfinite(m[k])
                           for k in ("objective", "gradient_norm")) for m in objectives)):
        raise RuntimeError("Missing continuous finite objective/update evidence")
    if stage == "dpo" and any(m.get("rows") != min(16, rows - i * 16) for i, m in enumerate(objectives)):
        raise ValueError("DPO final partial batch/work accounting changed")
    if stage == "ppo" and any(m.get("responses") != 1 or type(m.get("action_tokens")) is not int
                             or not 1 <= m["action_tokens"] <= 256 for m in objectives):
        raise ValueError("PPO exact per-response optimizer work changed")
    completion = result["completion"]
    if (completion.get("scope") != work["scope"] or completion.get("synthetic_rows") != 0
            or completion.get("stop_reason") != "data_exhausted"
            or any(completion.get(k) is not True for k in
                   ("data_pass_complete", "protocol_complete", "full_selected_dataset_pass"))
            or completion.get("full_accepted_dataset_pass") is not (stage == "dpo")
            or completion.get("rows_processed") != rows or completion.get("optimizer_updates") != updates
            or completion.get("response_tokens") != cursor.get("response_tokens")
            or type(cursor.get("response_tokens")) is not int or cursor["response_tokens"] <= 0):
        raise ValueError("Incomplete registered work must not claim full-partition completion")
    if (completion.get("completed_rollouts") != cursor["completed_rollouts"]
            or (stage == "dpo" and cursor["response_tokens"] != sum(r["chosen_tokens"] + r["rejected_tokens"] for r in records))
            or (stage == "ppo" and (not rows <= cursor["response_tokens"] <= rows * 256
                                   or sum(m["action_tokens"] for m in objectives) != 2 * cursor["response_tokens"]))):
        raise ValueError("Actual token/rollout counts differ from registered work")
    artifact = result["artifact"]
    if (Path(artifact["artifact_dir"]).resolve() != output / "final-artifact"
            or Path(artifact["manifest_path"]).resolve() != output / "final-artifact/manifest.json"):
        raise ValueError("Final artifact escaped child output")
    saved = _bundle(output / "final-artifact", artifact["manifest_sha256"],
                    {"weights.pt"}, "alignlab-completed-stage-v1")
    if saved.get("stage") != stage or saved["identity"] != identity or saved.get("completion") != completion:
        raise ValueError("Final artifact identity/completion differs")
    exported_weights = _cpu_payload(output / "final-artifact/weights.pt",
                                    saved["artifacts"]["weights.pt"]["sha256"])
    checkpoints = result["checkpoints"]
    indices = [c["optimizer_updates"] for c in checkpoints]
    keys = [c["cursor_sha256"] for c in checkpoints]
    if (len(keys) != len(set(keys)) or not ({1, updates} | set(range(50, updates + 1, 50))) <= set(indices)
            or result["latest_safe_checkpoint"] not in checkpoints
            or result["latest_safe_checkpoint"]["optimizer_updates"] != updates
            or result["latest_safe_checkpoint"]["cursor"] != cursor):
        raise ValueError("First/periodic/final safe checkpoints missing")
    for receipt in checkpoints:
        directory, update = Path(receipt["checkpoint_path"]).resolve(), receipt["optimizer_updates"]
        cursor_sha = hashlib.sha256(_canonical(receipt["cursor"])).hexdigest()
        if (directory.parent != output or type(update) is not int or not 0 <= update <= updates
                or receipt["cursor_sha256"] != cursor_sha
                or receipt["cursor"].get("optimizer_updates") != update
                or directory.name != f"update-{update:06d}-{cursor_sha[:16]}" or not receipt["reasons"]
                or type(receipt.get("active_elapsed_seconds")) not in (int, float)
                or not 0 <= receipt["active_elapsed_seconds"] <= result["finished_at_epoch"] - result["started_at_epoch"]):
            raise ValueError("Invalid checkpoint receipt")
        checkpoint = _bundle(directory, receipt["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint branch identity changed")
        _checkpoint_content(directory, checkpoint, receipt,
                            final=receipt == result["latest_safe_checkpoint"], artifact=exported_weights,
                            require_observation=True)
    # A CPU or metadata-only cache cannot substitute for the child's measured
    # frozen-reference guards. The recorded actual loaded tensor anchor must
    # agree with the immutable cache scoring anchor.
    roles = ["reference", "reward"] if stage == "ppo" else ["reference"]
    for role in roles:
        anchor = result.get(role + "_anchor", {})
        metadata = anchor.get("metadata", {})
        if (not anchor.get("tensors") or metadata.get("device") != "cuda:0"
                or metadata.get("precision") != "bfloat16" or metadata.get("quantization") != NF4):
            raise RuntimeError("Missing actual frozen CUDA reference/reward tensor anchor")
    audits = result.get("frozen_full_audits", [])
    if ([a.get("phase") for a in audits] != ["before_training", "after_training", "after_export"]
            or any(a.get("roles") != roles or a.get("reward_head_content_verified") is not (stage == "ppo")
                   or type(a.get("time_epoch")) not in (int, float)
                   or not result["started_at_epoch"] <= a["time_epoch"] <= result["finished_at_epoch"]
                   for a in audits)
            or [a["time_epoch"] for a in audits] != sorted(a["time_epoch"] for a in audits)):
        raise RuntimeError("Required before/after/export frozen content audits missing")
    if stage == "ppo":
        if result.get("reference_cache") is not None or not result.get("reward_head_anchor"):
            raise ValueError("PPO requires online frozen scoring and an independently frozen reward head")
        if any(a.get("cache_manifest_sha256") is not None for a in audits):
            raise ValueError("PPO must not claim an offline DPO reference cache")
    else:
        cache_pin = result["reference_cache"]
        requested = request["reference_cache"]
        if (cache_pin.get("mode") != requested["mode"]
                or Path(cache_pin["path"]).resolve() != Path(requested["path"]).resolve()
                or cache_pin.get("exact_key_coverage_verified") is not True
                or any(a.get("cache_manifest_sha256") != cache_pin["manifest_sha256"] for a in audits)
                or (requested["mode"] == "reuse"
                    and any(cache_pin[k] != requested[k] for k in ("manifest_sha256", "identity_sha256")))):
            raise ValueError("Reference cache mode/path/pins/coverage differ from the admitted cache")
        cached, cache_identity = _cache(cache_pin)
        if (cache_pin.get("unique_sequences") != len(cached.entries)
                or cache_pin.get("expected_unique_sequences") != len(cached.entries)
                or not 1 <= len(cached.entries) <= 2 * rows
                or cache_identity["accepted_dataset_manifest"]["sha256"] != request["audit_manifest_sha256"]
                or cache_identity["scoring_config"].get("frozen_reference_receipt") != result["reference_anchor"]):
            raise ValueError("Stored reference cache differs from actual loaded frozen-reference identity")
        _verify_cache_files(identity, cache_identity)
        sft_dir = Path(request["parents"]["sft"]["artifact_dir"])
        sft_manifest = _parse(sft_dir / "manifest.json")
        for label, filename, digest in (
                ("manifest", "manifest.json", request["parents"]["sft"]["manifest_sha256"]),
                ("weights", "weights.pt", sft_manifest["artifacts"]["weights.pt"]["sha256"])):
            item = cache_identity["files"]["adapter"].get(label, {})
            if item.get("sha256") != digest or Path(item.get("path", "")).resolve() != (sft_dir / filename).resolve():
                raise ValueError("Reference cache does not belong to the common completed SFT")
        for key, filename, digest in (("reference_cache/manifest", "manifest.json", cache_pin["manifest_sha256"]),
                                      ("reference_cache/identity", "identity.json", cache_pin["identity_sha256"])):
            item = identity["inputs"].get(key, {})
            if (item.get("sha256") != digest
                    or Path(item.get("path", "")).resolve() != Path(cache_pin["path"]).resolve() / filename):
                raise ValueError("Final branch identity does not pin the consumed reference cache")
        if request["seed"] == 42 and requested["mode"] == "create":
            migration_pin = _verify_prefix_migration(output, request, result, cached, cache_identity)
            item = identity["inputs"].get("reference_cache/prefix_migration", {})
            if (item.get("sha256") != migration_pin["sha256"]
                    or Path(item.get("path", "")).resolve() != Path(migration_pin["path"]).resolve()):
                raise ValueError("Final identity does not bind the cache prefix migration receipt")
        elif result.get("cache_prefix_recovery_report") is not None:
            raise ValueError("Only initial DPO42 may claim a cache prefix migration")
    _verify_observation_artifacts(output, request, result)
    _verify_live_objectives(output, stage, objectives)
    guard = result.get("process_guard", {})
    owned = guard.get("owned_identities", [])
    if (not owned or any(p["pid"] not in known or not _same_identity(p, known[p["pid"]])
                         for p in owned)
            or {p["pid"] for p in owned} != set(known)):
        raise ValueError("Child guard identities differ from the externally observed process chain")
    verified_guard = process_guard.verify_guard_log(output / "process-guard.jsonl",
                                                  request=request, progress=result)
    if (verified_guard.get("verified") is not True
            or verified_guard.get("inherited_waited_seconds") != request["process_wait_spent_seconds"]
            or verified_guard.get("total_waited_seconds") != guard.get("total_waited_seconds")):
        raise ValueError("Complete strict boundary-guard and inherited campaign wait proof required")
    return result


def run_alignment(plan_path, expected_plan_sha256, *, execute=False, system=None):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(plan_path, expected_plan_sha256)
    flags = {"primary_alignment_verified": False, "parents_verified": True,
             "capacity_verified": True, "formal_training": False, "research_complete": False}
    if not execute:
        return {"status": "plan_only", "plan": plan, **flags}
    live_identity = _LiveIdentity.from_plan(plan_path, expected_plan_sha256, plan)
    # Bracket the initial full validation with metadata observations rather
    # than silently blessing a file replacement that occurred while hashing.
    validate_plan(plan_path, expected_plan_sha256)
    live_identity.check()
    system, output = system or _System(), Path(plan["output"]).resolve()
    lock = ROOT / "reports/.alignlab-alignment.lock"
    foreign_locks = [ROOT / ("reports/" + name) for name in
                     (".alignlab-gpu.lock", ".alignlab-parents.lock", ".alignlab-capacity.lock")]
    if output.exists() or lock.exists() or any(p.exists() for p in foreign_locks):
        raise FileExistsError("Existing output or lock; never reuse outputs or delete unknown locks")
    if system.now() >= plan["continuation"]["deadline_epoch"]:
        return {"status": "budget_exhausted", "started_at_epoch": plan["continuation"]["started_at_epoch"],
                "deadline_epoch": plan["continuation"]["deadline_epoch"], **flags}
    processes = system.processes()
    _parents_gone(plan, processes)
    _prior_processes_gone(plan, processes)
    own = _own_processes(processes, plan_path)
    initial_guard = process_guard.strict_admission(system.processes,
        owned_identities=list(own.values()), allowed_identities=plan["allowed_background_processes"],
        deadline_epoch=plan["continuation"]["deadline_epoch"], clock=system.now, sleep=system.sleep)
    if not initial_guard["caller_admission_allowed"]:
        return {"status": "blocked_before_start", "process_guard": initial_guard,
                "started_at_epoch": None, **flags}
    admission = system.resources()
    if admission["blockers"]:
        return {"status": "blocked_before_start", "resources": admission, "started_at_epoch": None, **flags}
    if system.now() >= plan["continuation"]["deadline_epoch"]:
        return {"status": "budget_exhausted", **flags}
    if Path(str(plan_path) + ".stop").exists():
        return {"status": "stopped_before_start", **flags}
    token = {"owner": KIND, "pid": os.getpid(), "plan_sha256": expected_plan_sha256, "token": uuid.uuid4().hex}
    _new(lock, token)
    started = plan["continuation"]["started_at_epoch"]
    attempt_started = system.now()
    state = {"kind": KIND, "status": "starting", **flags, "formal_training": True,
             "scope": "primary_six_alignment_branches_only", "plan_path": str(plan_path),
             "plan_sha256": expected_plan_sha256, "started_at_epoch": started,
             "deadline_epoch": plan["continuation"]["deadline_epoch"], "attempt_started_at_epoch": attempt_started,
             "continuation": plan["continuation"], "supervisor_processes": list(own.values()),
              "resources_before": admission, "initial_process_guard": initial_guard,
              "total_process_wait_seconds": plan["continuation"]["process_wait_spent_seconds"], "process_wait_policy": plan["process_wait_policy"], "stages": []}
    child, child_finished = None, True

    def write(status=None, **values):
        if status:
            state["status"] = status
        state.update(values, updated_at_epoch=system.now())
        _status(output / "campaign.json", state)

    def stop_reason():
        if (output / "STOP").exists() or Path(str(plan_path) + ".stop").exists():
            return "paused"
        return "budget_exhausted" if system.now() >= state["deadline_epoch"] else None

    def signal(reason):
        if not (output / "STOP").exists():
            _new(output / "STOP", {"owner": KIND, "plan_sha256": expected_plan_sha256, "reason": reason})

    def frozen_identity():
        try:
            live_identity.check()
            return True
        except Exception as exc:
            state["identity_guard_error"] = f"{type(exc).__name__}: {exc}"
            return False

    def complete_identity():
        live_identity.check()
        validate_plan(plan_path, expected_plan_sha256)
        live_identity.check()

    def stage_admission(name):
        """No child/GPU work is active here; all waiting keeps the same ledger."""
        def event(value):
            with (output / "supervisor-process-guard.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage_name": name, **value}, allow_nan=False) + "\n")
            if value.get("event") == "wait_start":
                write("waiting_process_identity", waiting_stage=name)

        session = process_guard.GuardSession(system.processes, owned_identities=list(own.values()),
            allowed_identities=plan["allowed_background_processes"], deadline_epoch=state["deadline_epoch"],
            policy=plan["process_wait_policy"], spent_seconds=state["total_process_wait_seconds"],
            stop_requested=lambda: stop_reason() == "paused", identity_unchanged=frozen_identity,
            clock=system.now, sleep=system.sleep, on_event=event)
        receipt = lambda: {"complete": True, "optimizer_updates": 0, "kind": "before_cuda",
            "path": str(output / "plan.snapshot.json"), "sha256": sha(output / "plan.snapshot.json")}
        decision = session.check(boundary="supervisor_before_" + name, optimizer_updates=0,
                                 save_boundary=receipt)
        state["total_process_wait_seconds"] = decision["total_waited_seconds"]
        state["last_process_guard"] = decision
        if not decision["admission_allowed"]:
            reason = decision["reason"]
            status = "budget_exhausted" if reason == "deadline_exhausted" else (
                "paused" if reason == "stop_requested" else "failed")
            write(status, process_guard_failure=reason)
            return False
        return True

    try:
        output.mkdir(exist_ok=False)
        _new(output / "plan.snapshot.json", plan)
        live_identity.add_file(output / "plan.snapshot.json", sha(output / "plan.snapshot.json"), always_hash=True)
        _status(output / "live-identity-sentinel.json", live_identity.record())
        write()
        snapshot = verify_snapshot(plan["model_snapshot"], plan["model_manifest_sha256"])
        if snapshot["repo_id"] != REPO_ID or snapshot["revision"] != REVISION:
            raise ValueError("Actual model snapshot differs from registered revision")
        cache_pin = None
        for stage, seed in SEQUENCE:
            name = f"{stage}{seed}"
            while True:
                live_identity.check()
                reason = stop_reason()
                if reason:
                    write(reason)
                    return state
                if not stage_admission(name):
                    return state
                if any(p.exists() for p in foreign_locks):
                    raise FileExistsError("Foreign or retained training/GPU lock")
                admission = system.resources()
                if not admission["blockers"]:
                    break
                write("waiting_resources", resources=admission)
                remaining = state["deadline_epoch"] - system.now()
                if remaining > 0:
                    system.sleep(min(10, remaining))
            reference = None
            if stage == "dpo":
                if cache_pin is None:
                    reference = {"mode": "create", "path": str(output / name / "reference-cache")}
                else:
                    _cache(cache_pin)
                    reference = {"mode": "reuse", **cache_pin}
            request = {k: plan[k] for k in ("source_sha256", "model_snapshot", "model_manifest_sha256",
                "tokenizer_dir", "audit_dir", "audit_manifest_sha256", "capacity_acceptance_path",
                "capacity_acceptance_sha256", "parent_plan_path", "parent_plan_sha256",
                "parent_acceptance_path", "parent_acceptance_sha256", "allowed_background_processes",
                "process_wait_policy", "continuation")}
            request.update(schema_version=1, kind=STAGE_KIND, stage=stage, seed=seed,
                output=str(output / name), deadline_epoch=state["deadline_epoch"],
                alignment_plan_path=str(plan_path), alignment_plan_sha256=expected_plan_sha256,
                supervisor_processes=list(own.values()),
                process_wait_spent_seconds=state["total_process_wait_seconds"],
                parents={k: plan["parents"][k] for k in (("sft",) if stage == "dpo" else ("sft", "reward"))},
                workload=workload(stage, plan["ppo_rows"]), reference_cache=reference)
            path = output / f"{name}.request.json"
            _new(path, request)
            live_identity.add_file(path, sha(path), always_hash=True)
            _status(output / "live-identity-sentinel.json", live_identity.record())
            args = [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B", "-u",
                    "scripts/run_alignlab_alignment_stage_r2.py", "--request", str(path),
                    "--request-sha256", sha(path), "--execute"]
            record = {"name": name, "stage": stage, "seed": seed, "request": request,
                      "request_sha256": sha(path), "command": args, "launch_epoch": system.now(),
                      "known_processes": [], "status": "dispatching"}
            state["stages"].append(record)
            write("dispatching")
            with (output / f"{name}.stdout.log").open("xb") as stdout, (output / f"{name}.stderr.log").open("xb") as stderr:
                complete_identity()
                reason = stop_reason()
                if reason:
                    write(reason)
                    return state
                # The child independently guards its before-CUDA boundary. An
                # unknown process appearing after stage_admission must use that
                # bounded wait, not force a fresh request/ledger or early STOP.
                child = system.spawn(args, stdout, stderr)
                child_finished = False
                record["wrapper_pid"] = child.pid
                known, monitor_error, stopped = {}, None, None
                snapshot_uncertain_since = None
                while True:
                    try:
                        live_identity.check()
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    stopped = stopped or stop_reason()
                    if stopped or monitor_error:
                        signal(stopped or "identity_failure")
                    try:
                        live = system.processes()
                    except Exception as exc:
                        # Unknown whole-system observations are handled by the
                        # live child's bounded safe-boundary wait. A supervisor
                        # query failure cannot itself prove child exit.
                        if snapshot_uncertain_since is None:
                            snapshot_uncertain_since = system.now()
                        observation_deadline = state["deadline_epoch"]
                        record.update(pending_boundary_wait=True,
                            process_snapshot_error=f"{type(exc).__name__}: {exc}",
                            process_snapshot_uncertain_since=snapshot_uncertain_since,
                            process_snapshot_observation_deadline=observation_deadline,
                            exit_code=child.poll())
                        if system.now() >= observation_deadline:
                            failed_status = stop_reason() or "failed"
                            signal(stop_reason() or "process_snapshot_unavailable")
                            # Keep the lock and make the remaining uncertainty
                            # explicit; a wrapper exit alone is not child exit.
                            state["child_may_still_be_running"] = True
                            record["exit_identity_verified"] = False
                            write(failed_status,
                                  process_guard_failure="process_snapshot_unavailable_exit_unverified")
                            return state
                        write("stopping" if stopped or monitor_error else "pending_boundary_wait")
                        system.sleep(max(0, min(10, observation_deadline - system.now())))
                        continue
                    snapshot_uncertain_since = None
                    by_pid = {p["pid"]: p for p in live}
                    for _ in range(len(live) + 1):
                        added = False
                        for p in live:
                            if p["pid"] not in known and (p["pid"] == child.pid or p.get("parent_pid") in known):
                                if (not _command(p, "run_alignlab_alignment_stage_r2.py", path)
                                        or _stamp(p["created_at_utc"]).timestamp() < record["launch_epoch"]):
                                    monitor_error = "Child command/creation/ancestry mismatch"
                                else:
                                    known[p["pid"]] = p
                                    added = True
                        if not added:
                            break
                    try:
                        _parents_gone(plan, live)
                        peer_guard = _peer_decision(live, own, known, plan["allowed_background_processes"])
                        record["process_guard"] = peer_guard
                        pending = not peer_guard["caller_admission_allowed"]
                        record["pending_boundary_wait"] = pending and _waitable_peer_decision(peer_guard)
                        if pending and not record["pending_boundary_wait"]:
                            raise RuntimeError("Known competing training or process identity changed")
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    if monitor_error:
                        signal("identity_failure")
                    active = [p for pid, p in known.items() if pid in by_pid and _same_identity(by_pid[pid], p)]
                    code = child.poll()
                    record.update(known_processes=list(known.values()), active_processes=active,
                                  monitoring_error=monitor_error, stop_reason=stopped, exit_code=code)
                    write("stopping" if stopped or monitor_error else (
                        "pending_boundary_wait" if record.get("pending_boundary_wait") else "running"))
                    if code is not None and not active:
                        unverified = [p for p in live if
                            p["pid"] not in known and (p["pid"] == child.pid or p.get("parent_pid") in known)
                            or p["pid"] in known
                            and process_guard.strict.ProcessIdentity.from_record(p).instance
                                == process_guard.strict.ProcessIdentity.from_record(known[p["pid"]]).instance
                            and not _same_identity(p, known[p["pid"]])]
                        if not known or unverified:
                            state["child_may_still_be_running"] = True
                            record["exit_identity_verified"] = False
                            record["unverified_descendants"] = unverified
                            raise RuntimeError("Child exit identity is unverified; wrapper exit is insufficient")
                        child_finished = True
                        if monitor_error:
                            raise RuntimeError(monitor_error)
                        if stopped:
                            write(stopped)
                            return state
                        if any(p.exists() for p in foreign_locks):
                            raise FileExistsError("Exited child retained a lock; do not advance")
                        complete_identity()
                        result = _verify_stage(output / name, request, known, code)
                        verified_wait = result["process_guard"]["total_waited_seconds"]
                        if (request["process_wait_spent_seconds"] != state["total_process_wait_seconds"]
                                or not state["total_process_wait_seconds"] <= verified_wait <= plan["wall_seconds"]):
                            raise ValueError("Child reset or exceeded the shared campaign waiting ledger")
                        state["total_process_wait_seconds"] = verified_wait
                        if stage == "dpo":
                            produced = {k: result["reference_cache"][k]
                                        for k in ("path", "manifest_sha256", "identity_sha256")}
                            if cache_pin is None:
                                if Path(produced["path"]).resolve() != output / name / "reference-cache":
                                    raise ValueError("Initial DPO cache escaped its owning stage")
                                cache_pin = produced
                                # Cache becomes immutable only after full
                                # completed-stage acceptance. Later branches
                                # reuse its exact files; no live rebaselining.
                                live_identity.add_tree(cache_pin["path"])
                            elif produced != cache_pin:
                                raise ValueError("Later seeds did not reuse the original frozen reference cache")
                        record.update(status="accepted", progress_path=str(output / name / "progress.json"),
                            progress_sha256=sha(output / name / "progress.json"), artifact=result["artifact"],
                            reference_cache=result.get("reference_cache"),
                            inherited_process_wait_seconds=request["process_wait_spent_seconds"],
                            total_process_wait_seconds=verified_wait, finished_at_epoch=system.now())
                        write("stage_accepted")
                        break
                    system.sleep(10)
        complete_identity()
        if not stage_admission("final_publication"):
            return state
        if any(p.exists() for p in foreign_locks):
            raise FileExistsError("A training/GPU lock remains after child exit")
        reason = stop_reason()
        if reason:
            write(reason)
            return state
        write("primary_alignment_passed", primary_alignment_verified=True, research_complete=False)
        return state
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        if output.exists():
            if child is not None and not child_finished:
                signal("supervisor_error")
                state["child_may_still_be_running"] = True
            write("failed")
        raise
    finally:
        if output.exists():
            state.update(finished_at_epoch=system.now(), elapsed_seconds=system.now() - started,
                         attempt_elapsed_seconds=system.now() - attempt_started)
            _status(output / "campaign.json", state)
        if child_finished and lock.exists() and _parse(lock) == token:
            lock.unlink()
