"""Explicit, fresh-directory continuation of an accepted RM optimizer boundary.

The old campaign, its deadline and every old byte remain historical evidence.
The outer checkpoint identity is new; numerical work, absolute deadline and
the complete cursor stay unchanged. Parameters, Adam and RNG restore exactly.
No automatic retry, source rewrite, old-output resume or implicit budget exists.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import uuid

from . import study_stage as old
from . import parent_recovery as r1
from . import parent_recovery_r2 as r2
from . import parent_recovery_r3 as r3
from .capacity_stage import _write, read_json, sha

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-formal-parent-recovery-r4"
PLAN_KIND = "alignlab-parent-recovery-r4-plan"
REVISION = "parent-recovery-r4"
ORIGIN_UPDATE, ROWS, UPDATES, BATCH = 677, 13139, 822, 16
WALL_SECONDS = 28800
NEW_SOURCES = frozenset({"alignment_lab/parent_recovery_r4.py",
    "scripts/run_alignlab_parent_recovery_r4.py", "tests_v2/test_parent_recovery_r4.py"})

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _path(value):
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("An absolute ordinary project path is required")
    for parent in path.parents:
        if parent == ROOT.parent:
            break
        if parent.is_symlink():
            raise ValueError("Symlink ancestry is forbidden")
    return path.resolve()


def pin(path):
    path = _path(path)
    return {"path": str(path), "sha256": sha(path)}


def _pinned(record):
    if type(record) is not dict or set(record) != {"path", "sha256"}:
        raise ValueError("Expected an external path/SHA pin")
    path = _path(record["path"])
    old._pin(path, record["sha256"])
    return path


def _jsonl(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Unexpected blank metric row")
    return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            for line in lines]


def _objectives(path, first, last):
    values = [x for x in _jsonl(path) if x.get("event") == "objective_after_engine_return"]
    if [x.get("update") for x in values] != list(range(first, last + 1)):
        raise ValueError("Missing, duplicate or out-of-order objective updates")
    for item in values:
        if (type(item.get("rows")) is not int or item["rows"] <= 0
                or any(type(item.get(k)) not in (int, float) or not math.isfinite(item[k])
                       for k in ("objective", "gradient_norm"))):
            raise ValueError("Malformed or nonfinite objective evidence")
    return values


def validate_request(request, *, execute=False):
    """Bind a new dispatch to the unchanged original and safely paused R3 chain."""
    fields = {"schema_version", "kind", "stage", "output", "recovery_plan_path",
              "recovery_plan_sha256", "source_sha256", "budget", "origin", "prior_segment"}
    if (type(request) is not dict or set(request) != fields or type(request["schema_version"]) is not int
            or request["schema_version"] != 1 or request["kind"] != KIND or request["stage"] != "reward"):
        raise ValueError("Exact chained R4 recovery request required")
    output = _path(request["output"])
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        raise ValueError("Fresh independent recovery output required")
    plan = read_json(_pinned({"path": request["recovery_plan_path"], "sha256": request["recovery_plan_sha256"]}))
    if (plan.get("schema_version") != 1 or plan.get("kind") != PLAN_KIND
            or plan.get("execution_revision") != REVISION or plan.get("output") != request["output"]
            or plan.get("wall_seconds") != WALL_SECONDS or plan.get("source_sha256") != request["source_sha256"]
            or plan.get("origin") != request["origin"] or plan.get("prior_segment") != request["prior_segment"]
            or type(plan.get("allowed_background_processes")) is not list):
        raise ValueError("R4 plan/request or background identity contract differs")
    segment = request["prior_segment"]
    names = {"request", "progress", "identity", "selection", "metrics", "migration_manifest",
             "migration_state_audit", "metadata_comparison", "checkpoint"}
    if type(segment) is not dict or set(segment) != names:
        raise ValueError("Complete pinned paused R3 evidence required")
    paths = {name: _pinned(segment[name]) for name in names - {"checkpoint"}}
    prior_request = read_json(paths["request"])
    original = r3.validate_request(prior_request, execute=True)
    prior_output = Path(prior_request["output"]).resolve()
    if (request["origin"] != prior_request["origin"] or request["budget"] != prior_request["budget"]
            or plan.get("budget_authorization") != original["plan"]["budget_authorization"]):
        raise ValueError("Original origin, authorization and identical absolute budget must be retained")
    if (output == prior_output or output.is_relative_to(prior_output)
            or output.is_relative_to(Path(original["old_request"]["output"]).parent)):
        raise ValueError("R4 cannot overwrite or nest inside a historical output")
    sources = request["source_sha256"]
    if (type(sources) is not dict or not NEW_SOURCES <= set(sources)
            or any(sources.get(k) != v for k, v in prior_request["source_sha256"].items())):
        raise ValueError("All frozen R3 dependencies plus new R4 sources must be pinned")
    for name, expected in sources.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe source path")
        old._pin(_path(ROOT / name), expected)
    filenames = {"progress": "progress.json", "identity": "identity.json", "selection": "selection.json",
                 "metrics": "metrics.jsonl", "migration_manifest": "migration-manifest.json",
                 "migration_state_audit": "migration-state-audit.json", "metadata_comparison": "model-metadata-comparison.json"}
    for name, filename in filenames.items():
        if paths[name] != prior_output / filename:
            raise ValueError("Pinned R3 evidence escaped its original output")
    if read_json(prior_output / "request.json") != prior_request or (prior_output / "manifest.json").exists():
        raise ValueError("R3 evidence is not an unfinished dispatch")
    progress, identity = read_json(paths["progress"]), read_json(paths["identity"])
    if (progress.get("request") != prior_request or progress.get("status") != "paused"
            or progress.get("actual_optimizer_updates") != ORIGIN_UPDATE
            or progress.get("segment_optimizer_updates") != ORIGIN_UPDATE - r3.ORIGIN_UPDATE
            or progress.get("objective_records") != ORIGIN_UPDATE - r3.ORIGIN_UPDATE
            or progress.get("outcome_stop_reason") != "paused" or progress.get("process_guard_blocked") is not True
            or progress.get("identity_guard_error") is not None or progress.get("emergency_checkpoint_attempted") is not False
            or progress.get("safe_resume_available") is not True or progress.get("identity") != identity):
        raise ValueError("R3 is not the selected complete safe process-guard boundary")
    if (identity["config"]["purpose"] != old.KIND or identity["config"]["scope"] != old.SCOPE
            or identity["config"]["synthetic_rows"] != 0
            or identity["config"]["work"]["deadline_epoch"] != request["budget"]["deadline_epoch"]
            or identity["config"]["model"] != original["old_identity"]["config"]["model"]):
        raise ValueError("R3 model/work identity differs from the real original protocol")
    checkpoint = segment["checkpoint"]
    if type(checkpoint) is not dict or set(checkpoint) != {"path", "manifest_sha256"}:
        raise ValueError("Externally pinned checkpoint required")
    checkpoint_path = _path(checkpoint["path"])
    if checkpoint_path != prior_output / f"update-{ORIGIN_UPDATE:06d}":
        raise ValueError("Unexpected selected R3 boundary")
    old._pin(checkpoint_path / "manifest.json", checkpoint["manifest_sha256"])
    manifest = read_json(checkpoint_path / "manifest.json")
    safe = progress.get("latest_safe_checkpoint", {})
    if (manifest.get("status") != "complete" or manifest.get("format") != "alignlab-optimizer-checkpoint-v1"
            or manifest.get("identity") != identity or manifest.get("identity_sha256") != digest(identity)
            or safe.get("manifest_sha256") != checkpoint["manifest_sha256"]
            or Path(safe.get("checkpoint_path", "")).resolve() != checkpoint_path
            or safe.get("optimizer_updates") != ORIGIN_UPDATE):
        raise ValueError("R3 safe checkpoint identity is inconsistent")
    if {p.name for p in checkpoint_path.iterdir()} != {"state.pt", "manifest.json"}:
        raise ValueError("Unexpected checkpoint files")
    artifact = manifest["artifacts"]["state.pt"]
    old._pin(checkpoint_path / "state.pt", artifact["sha256"])
    if (checkpoint_path / "state.pt").stat().st_size != artifact["size_bytes"]:
        raise ValueError("Prior checkpoint size changed")
    # R3's complete normalization receipt must still bind every real model field.
    receipt = read_json(paths["metadata_comparison"])
    expected = r2.compare_metadata(receipt["actual"], original["old_identity"]["config"]["model"])
    expected["execution_revision"] = r3.REVISION
    if ({k: v for k, v in receipt.items() if k != "runtime_target_modules_type"} != expected
            or expected["all_fields_equal"] is not True or progress["model"] != expected["actual"]):
        raise ValueError("R3 metadata comparison is inconsistent")
    if (progress.get("migration_manifest") != segment["migration_manifest"]
            or progress.get("migration_state_audit") != segment["migration_state_audit"]):
        raise ValueError("Prior migration receipts changed")
    transition = read_json(paths["migration_manifest"])
    if (transition.get("kind") != "alignlab-same-budget-continuation-v1"
            or transition.get("changed_cursor_fields") != [] or transition.get("changed_work_fields") != []
            or transition.get("parameter_optimizer_rng_change_permitted") is not False
            or transition["old_cursor"] != original["old_progress"]["cursor"]
            or transition["old_work"] != original["old_identity"]["config"]["work"]
            or transition["new_work"] != identity["config"]["work"]
            or transition["old_work"] != transition["new_work"]
            or transition["old_cursor"] != transition["new_cursor"]
            or transition["origin_checkpoint"] != prior_request["prior_segment"]["checkpoint"]
            or transition["recovery_request"] != segment["request"]
            or transition["old_cursor_sha256"] != digest(transition["old_cursor"])
            or transition["new_cursor_sha256"] != digest(transition["new_cursor"])):
        raise ValueError("R3 same-budget continuation changed cursor or numerical work")
    selection = read_json(paths["selection"])
    if selection != read_json(request["origin"]["selection"]["path"]):
        raise ValueError("R3 selection differs from the original complete partition")
    records = selection["records"]
    if len(records) != ROWS or len({r["record_id"] for r in records}) != ROWS:
        raise ValueError("Prior real row inventory changed")
    historical = original["historical_metrics"] + _objectives(paths["metrics"], r3.ORIGIN_UPDATE + 1, ORIGIN_UPDATE)
    if ([m["update"] for m in historical] != list(range(1, ORIGIN_UPDATE + 1))
            or any(m["rows"] != BATCH for m in historical)
            or sum(m["rows"] for m in historical) != ORIGIN_UPDATE * BATCH):
        raise ValueError("Original plus previous recovery historical work has a gap or duplicate")
    boundaries = [x for x in _jsonl(paths["metrics"]) if x.get("event") == "optimizer_boundary"]
    if [x["cursor"]["optimizer_updates"] for x in boundaries] != list(range(r3.ORIGIN_UPDATE + 1, ORIGIN_UPDATE + 1)):
        raise ValueError("R3 boundary coverage is incomplete")
    cursor = progress["cursor"]
    from .stage_engine import _hash
    expected_config = _hash({"config": identity["config"]["work"], "role": "reward"})
    order = list(range(ROWS)); random.Random(identity["config"]["work"]["seed"]).shuffle(order)
    if (cursor["order"] != order or cursor["config_sha256"] != expected_config
            or cursor["data_sha256"] != original["old_progress"]["cursor"]["data_sha256"]):
        raise ValueError("R3 cursor work/data/order changed")
    for item in boundaries:
        c = item["cursor"]; offset = c["optimizer_updates"] * BATCH
        tokens = sum(records[i]["chosen_tokens"] + records[i]["rejected_tokens"] for i in order[:offset])
        if (c["offset"] != offset or c["response_tokens"] != tokens or c["order"] != order
                or c["config_sha256"] != expected_config or c["data_sha256"] != cursor["data_sha256"]
                or c["active_rollout"] is not False or c["completed_rollouts"] != 0):
            raise ValueError("R3 boundary cursor denominator differs")
    if not boundaries or boundaries[-1]["cursor"] != cursor or cursor["offset"] != ORIGIN_UPDATE * BATCH:
        raise ValueError("R3 final cursor differs from its last logged safe boundary")
    if execute:
        authorization = read_json(_pinned(plan.get("dispatch_authorization")))
        if (authorization.get("kind") != "explicit-user-rm-r4-dispatch-authorization-v1"
                or authorization.get("source") != "direct_user_message_in_this_chat"
                or authorization.get("user_message") != "继续啊，怎么停了" or authorization.get("allow_one_dispatch") is not True
                or authorization.get("budget") != request["budget"]
                or authorization.get("prior_request_sha256") != segment["request"]["sha256"]
                or authorization.get("resume_optimizer_update") != ORIGIN_UPDATE
                or authorization.get("target_total_optimizer_updates") != UPDATES):
            raise ValueError("Explicit user authorization for this one same-budget R4 dispatch required")
    return {"output": output, "plan": plan, "old_request": original["old_request"],
            "old_progress": progress, "old_identity": identity, "old_manifest": manifest,
            "historical_metrics": historical, "resume_checkpoint": checkpoint,
            "prior_request": prior_request, "original_context": original}


def audit_prior_migration(context):
    """Audit both prior transitions from actual payloads: 256→R2 and 671→R3."""
    r3.audit_prior_migration(context["original_context"])
    progress = context["old_progress"]
    transition = read_json(_pinned(progress["migration_manifest"]))
    expected = r3.audit_migration_checkpoints(context["prior_request"]["prior_segment"]["checkpoint"],
        progress["origin_checkpoint"], transition, context["old_identity"])
    if read_json(_pinned(progress["migration_state_audit"])) != expected:
        raise ValueError("Prior R3 migration payload audit differs from its committed receipt")
    return expected

def migrate_cursor(cursor, rows, old_config, new_config, *, role="reward"):
    """Validate exact work and attest a no-op continuation of the complete cursor.

    This pure helper neither restores tensors nor mutates the caller's cursor.
    A complete new-identity checkpoint must commit the result before training.
    """
    from .stage_engine import _hash, SupervisedConfig
    if not isinstance(old_config, SupervisedConfig) or not isinstance(new_config, SupervisedConfig):
        raise ValueError("Typed supervised configurations required")
    before, after = asdict(old_config), asdict(new_config)
    if (role != "reward" or old_config.kind != "reward" or new_config.kind != "reward"
            or {k: v for k, v in before.items() if k != "deadline_epoch"}
            != {k: v for k, v in after.items() if k != "deadline_epoch"}
            or not math.isfinite(new_config.deadline_epoch) or new_config.deadline_epoch != old_config.deadline_epoch):
        raise ValueError("R4 numerical work and the absolute deadline must remain identical")
    required = {"config_sha256", "data_sha256", "schema", "order", "offset", "optimizer_updates",
                "response_tokens", "active_rollout", "completed_rollouts"}
    if type(cursor) is not dict or set(cursor) != required:
        raise ValueError("Exact complete supervised cursor required")
    order = list(range(len(rows))); random.Random(old_config.seed).shuffle(order)
    if (cursor["config_sha256"] != _hash({"config": before, "role": role})
            or cursor["data_sha256"] != _hash(rows) or cursor["schema"] != 1 or cursor["order"] != order
            or cursor["active_rollout"] is not False or cursor["completed_rollouts"] != 0):
        raise ValueError("Original config, data order or supervised cursor identity differs")
    for name in ("offset", "optimizer_updates", "response_tokens"):
        if type(cursor[name]) is not int or cursor[name] < 0:
            raise ValueError("Invalid original cursor counter")
    offset = cursor["offset"]
    tokens = sum(len(rows[i]["chosen_ids"]) + len(rows[i]["rejected_ids"]) for i in order[:offset])
    if (not 0 < offset < len(rows) or offset % old_config.effective_batch
            or cursor["optimizer_updates"] != offset // old_config.effective_batch
            or cursor["optimizer_updates"] >= old_config.max_updates or cursor["response_tokens"] != tokens):
        raise ValueError("Cursor is not a verified complete intermediate optimizer boundary")
    migrated = copy.deepcopy(cursor)
    migrated["config_sha256"] = _hash({"config": after, "role": role})
    receipt = {"schema_version": 1, "kind": "alignlab-same-budget-continuation-v1",
        "changed_cursor_fields": [], "changed_work_fields": [],
        "old_work": before, "new_work": after, "old_cursor": copy.deepcopy(cursor), "new_cursor": migrated,
        "old_cursor_sha256": digest(cursor), "new_cursor_sha256": digest(migrated),
        "parameter_optimizer_rng_change_permitted": False,
        "meaning": "Explicit chained continuation; work, cursor and absolute deadline remain exactly unchanged."}
    return migrated, receipt


def verify_combined_completion(outcome, rows, origin_cursor, historical_metrics):
    """Prove the full shuffled pass while counting only 145 genuinely new updates."""
    end, start = outcome.cursor, origin_cursor
    total_tokens = sum(len(r.chosen_ids) + len(r.rejected_ids) for r in rows)
    total_updates = math.ceil(len(rows) / BATCH)
    remaining = total_updates - start["optimizer_updates"]
    if (outcome.stop_reason != "data_exhausted" or outcome.extra_state is not None
            or end["offset"] != len(rows) or end["optimizer_updates"] != total_updates
            or end["response_tokens"] != total_tokens or end["active_rollout"] is not False
            or end["completed_rollouts"] != 0 or end["order"] != start["order"]
            or end["data_sha256"] != start["data_sha256"]
            or [x["update"] for x in historical_metrics] != list(range(1, start["optimizer_updates"] + 1))
            or [x["update"] for x in outcome.metrics] != list(range(start["optimizer_updates"] + 1, total_updates + 1))
            or len(outcome.metrics) != remaining
            or sum(x["rows"] for x in historical_metrics) != start["offset"]
            or sum(x["rows"] for x in outcome.metrics) != len(rows) - start["offset"]):
        raise ValueError("Historical plus resumed updates do not prove the complete accepted data pass")
    for item in [*historical_metrics, *outcome.metrics]:
        expected_rows = min(BATCH, len(rows) - (item["update"] - 1) * BATCH)
        if (item["rows"] != expected_rows or any(not math.isfinite(item[k]) for k in ("objective", "gradient_norm"))):
            raise ValueError("Objective denominator or finite update evidence differs")
    return {"stop_reason": "data_exhausted", "data_pass_complete": True, "protocol_complete": True,
        "full_accepted_dataset_pass": True, "scope": old.SCOPE, "synthetic_rows": 0,
        "rows_processed": len(rows), "optimizer_updates": total_updates, "response_tokens": total_tokens,
        "historical_optimizer_updates": start["optimizer_updates"], "segment_optimizer_updates": remaining,
        "historical_rows": start["offset"], "segment_rows": len(rows) - start["offset"],
        "historical_response_tokens": start["response_tokens"],
        "segment_response_tokens": total_tokens - start["response_tokens"]}


def _checkpoint_payload(directory, expected_sha):
    """CPU-only integrity/structure reader; CUDA RNG remains inert saved bytes."""
    from . import checkpointing as ck
    import torch
    directory = _path(directory)
    if {p.name for p in directory.iterdir()} != {"manifest.json", "state.pt"}:
        raise ValueError("Incomplete or foreign checkpoint files")
    old._pin(directory / "manifest.json", expected_sha)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("format") != ck.FORMAT:
        raise ValueError("Incomplete checkpoint")
    ck._validate_identity(manifest["identity"])
    if manifest["identity_sha256"] != ck._digest(ck._canonical(manifest["identity"])):
        raise ValueError("Checkpoint identity digest mismatch")
    data = (directory / "state.pt").read_bytes()
    if manifest["artifacts"] != {"state.pt": {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}:
        raise ValueError("Checkpoint content changed")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if set(payload) != {"parameters", "optimizer", "cursor", "rng", "extra_state"}:
        raise ValueError("Incomplete optimizer boundary")
    ck._state_tree(payload, integer_keys=True)
    if ck._tensor_specs(payload) != manifest["tensor_specs"]:
        raise ValueError("Checkpoint tensor structure differs from receipt")
    return manifest, payload


def _tree_equal(left, right):
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(_tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def audit_migration_checkpoints(origin, migrated, transition, expected_new_identity):
    before_manifest, before = _checkpoint_payload(origin["path"], origin["manifest_sha256"])
    after_manifest, after = _checkpoint_payload(migrated["checkpoint_path"], migrated["manifest_sha256"])
    if (after_manifest["identity"] != expected_new_identity
            or transition.get("kind") != "alignlab-same-budget-continuation-v1"
            or transition.get("changed_cursor_fields") != [] or transition.get("changed_work_fields") != []
            or transition["old_work"] != transition["new_work"]
            or transition["old_cursor"] != transition["new_cursor"]
            or before["cursor"] != after["cursor"]
            or before["cursor"] != transition["old_cursor"] or after["cursor"] != transition["new_cursor"]
            or any(not _tree_equal(before[k], after[k]) for k in ("parameters", "optimizer", "rng", "extra_state"))
            or before_manifest["parameter_specs"] != after_manifest["parameter_specs"]
            or before_manifest["optimizer"] != after_manifest["optimizer"]
            or before_manifest["generator_specs"] != after_manifest["generator_specs"]
            or before_manifest["software"] != after_manifest["software"]):
        raise ValueError("Origin migration changed parameters, optimizer, RNG or unapproved cursor content")
    return {"parameters_bitwise_equal": True, "optimizer_bitwise_equal": True, "rng_bitwise_equal": True,
            "extra_state_equal": True, "origin_manifest_sha256": origin["manifest_sha256"],
            "new_origin_manifest_sha256": migrated["manifest_sha256"],
            "changed_cursor_fields": [], "actual_optimizer_updates_before_first_new_step": ORIGIN_UPDATE}


class _Runtime:
    def now(self):
        return time.time()

    def sleep(self, value):
        time.sleep(value)

    def processes(self):
        from .probe_dispatch import _System
        return _System().processes()

    def bind_processes(self, request_path, request_sha):
        """Only this exact entry/request and its same-command venv wrapper may self-register."""
        if os.name != "nt":
            raise RuntimeError("Production recovery requires real Windows process identities")
        import ctypes
        from ctypes import wintypes
        parse = ctypes.windll.shell32.CommandLineToArgvW
        parse.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
        parse.restype = ctypes.POINTER(wintypes.LPWSTR)
        release = ctypes.windll.kernel32.LocalFree
        release.argtypes = (ctypes.c_void_p,)
        snapshot = self.processes()
        indexed = {p["pid"]: p for p in snapshot}
        owned, current = [], os.getpid()
        while current in indexed:
            record = indexed[current]
            if not record.get("command_line"):
                raise RuntimeError("Own process command is unavailable")
            count = ctypes.c_int()
            argv = parse(record["command_line"], ctypes.byref(count))
            try:
                args = [argv[i] for i in range(count.value)]
            finally:
                release(argv)
            executable = Path(args.pop(0)).resolve()
            if executable not in {Path(__import__('sys').executable).resolve(),
                    Path(__import__('sys')._base_executable).resolve(), (ROOT / ".runtime/alignlab-v2/Scripts/python.exe").resolve()}:
                raise RuntimeError("Unexpected R4 recovery Python executable")
            while args and args[0] in {"-u", "-I", "-B"}:
                args.pop(0)
            wanted = ["--request", str(Path(request_path).resolve()), "--request-sha256", request_sha, "--execute"]
            if (not args or Path(args[0]).resolve() != ROOT / "scripts/run_alignlab_parent_recovery_r4.py"
                    or args[1:] != wanted):
                raise RuntimeError("A Python ancestor is not this exact pinned recovery command")
            owned.append(record)
            current = record["parent_pid"]
        if not owned:
            raise RuntimeError("Own process identity was not observed")
        return owned

    def resources(self):
        from rlhf_lab.runtime import gpu_inventory, available_ram_gb
        result = gpu_inventory()
        result["available_ram_gb"] = available_ram_gb()
        device = next((g for g in result["gpus"] if g["index"] == 0), None)
        result["admitted"] = (device is not None and device["free_mb"] >= 7000
            and device["utilization_percent"] <= 20 and result["available_ram_gb"] is not None
            and result["available_ram_gb"] >= 8)
        return result

    def load(self, request):
        import torch
        from .model_loader import ModelLoadConfig, load_quantized_stage
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        return load_quantized_stage(request["model_snapshot"], request["model_manifest_sha256"],
                                   request["tokenizer_dir"], ModelLoadConfig("reward", seed=42))

    def sample_gpu(self):
        import torch
        from rlhf_lab.runtime import gpu_inventory
        torch.cuda.synchronize(0)
        return {"time_epoch": self.now(), **gpu_inventory(), "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def run_recovery(request_path, expected_request_sha256, *, execute=False, runtime=None):
    request_path = _pinned({"path": str(request_path), "sha256": expected_request_sha256})
    request = read_json(request_path)
    context = validate_request(request, execute=execute)
    if not execute:
        return {"status": "plan_only", "kind": KIND, "formal_training": False, "research_complete": False}
    output = context["output"]
    if output.exists():
        raise FileExistsError("Recovery output exists; no automatic retry or overwrite")
    runtime = runtime or _Runtime()
    started, deadline = request["budget"]["started_at_epoch"], request["budget"]["deadline_epoch"]
    stop = lambda: ((output / "STOP").exists() or (output.parent / "STOP").exists()
                    or Path(str(request_path) + ".stop").exists()
                    or Path(str(request["recovery_plan_path"]) + ".stop").exists())
    from .process_guard_v2 import inspect_processes
    owned = runtime.bind_processes(request_path, expected_request_sha256)
    allowed = [*context["plan"]["allowed_background_processes"], *owned]
    guard_events = []

    def inspect():
        decision = inspect_processes(runtime.processes, allowed_identities=allowed, deadline_epoch=deadline,
                                     clock=runtime.now, sleep=runtime.sleep)
        observed = {(p.pid, p.created_at_utc, p.command_line) for p in decision.final_snapshot}
        from .process_guard_v2 import ProcessIdentity
        missing_owned = []
        for item in owned:
            item = ProcessIdentity.from_record(item)
            if (item.pid, item.created_at_utc, item.command_line) not in observed:
                missing_owned.append(item.pid)
        record = decision.to_dict()
        record["missing_or_changed_owned_pids"] = missing_owned
        record["caller_admission_allowed"] = decision.admission_allowed and not missing_owned
        guard_events.append(record)
        if output.exists():
            with (output / "process-guard.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return record["caller_admission_allowed"]

    def boundary():
        if stop():
            raise InterruptedError("Recovery STOP at a safe boundary")
        if runtime.now() >= deadline or runtime.now() < started:
            raise TimeoutError("The separate recovery segment absolute budget is exhausted or invalid")

    def small_identity_unchanged():
        # Per-update guard covers only request/plan/code bytes; the immutable
        # model and data artifacts retain their full startup/publication audits.
        try:
            _pinned({"path": str(request_path), "sha256": expected_request_sha256})
            _pinned({"path": request["recovery_plan_path"], "sha256": request["recovery_plan_sha256"]})
            for name, expected in request["source_sha256"].items():
                _pinned({"path": str(ROOT / name), "sha256": expected})
            return True
        except (OSError, ValueError) as exc:
            status["identity_guard_error"] = f"{type(exc).__name__}: {exc}"
            return False

    boundary()
    if not inspect():
        raise RuntimeError("Other or unresolved Python workload blocks recovery admission")
    resources = runtime.resources()
    if not resources["admitted"]:
        raise RuntimeError("Recovery GPU/RAM admission blocked; no empty waiting process is launched")
    lock = ROOT / "reports/.alignlab-gpu.lock"
    owner = {"owner": KIND, "pid": os.getpid(), "request_sha256": expected_request_sha256,
             "output": str(output), "processes": owned, "token": uuid.uuid4().hex}
    _write(lock, owner, exclusive=True)
    status = {"kind": KIND, "status": "starting", "stage": "reward", "request": request,
        "request_file": {"path": str(request_path), "sha256": expected_request_sha256},
        "pid": os.getpid(), "process_identity": owned, "started_at_epoch": started,
        "execution_started_at_epoch": runtime.now(), "deadline_epoch": deadline,
        "original_deadline_epoch": context["old_request"]["deadline_epoch"],
        "formal_training": True, "research_complete": False, "scope": old.SCOPE, "synthetic_rows": 0,
        "actual_optimizer_updates": ORIGIN_UPDATE, "segment_optimizer_updates": 0,
        "checkpoints": [], "latest_safe_checkpoint": None, "gpu_samples": [],
        "resources_before": resources, "objective_logging": "after_engine_returns",
        "emergency_checkpoint_attempted": False}

    def progress(phase):
        status.update(status=phase, updated_at_epoch=runtime.now(), active_elapsed_seconds=runtime.now() - started)
        _write(output / "progress.json", status)
        print(json.dumps({"stage": "reward", "phase": phase,
                          "optimizer_updates": status["actual_optimizer_updates"],
                          "segment_optimizer_updates": status["segment_optimizer_updates"]}), flush=True)

    def metric(value):
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_epoch": runtime.now(), **value}, allow_nan=False) + "\n")
            stream.flush()

    try:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "request.json", request, exclusive=True)
        (output / "metrics.jsonl").touch(exist_ok=False)
        with (output / "process-guard.jsonl").open("x", encoding="utf-8") as stream:
            for event in guard_events:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        progress("loading_verified_full_real_partition")
        from .training_data import load_training_partition
        from .stage_engine import SupervisedConfig, parameter_mapping, run_supervised, _records
        from .checkpointing import build_identity, load_checkpoint, save_checkpoint
        from .stage_artifacts import import_stage, export_stage
        import torch
        previous = context["old_request"]
        partition = load_training_partition(previous["audit_dir"], "preference_train", previous["audit_manifest_sha256"])
        selection, records = old.real_selection(partition, previous)
        expected_selection = read_json(request["origin"]["selection"]["path"])
        if expected_selection != {"summary": selection, "records": records}:
            raise ValueError("Re-encoded recovery data differs from the original selection")
        selection_path = output / "selection.json"
        _write(selection_path, expected_selection, exclusive=True)
        selection.update(selection_manifest_path=str(selection_path), selection_manifest_sha256=sha(selection_path))
        status["selection"] = selection
        boundary(); validate_request(request, execute=True)
        audit_prior_migration(context)
        if not inspect() or not runtime.resources()["admitted"]:
            raise RuntimeError("Recovery admission changed before model load")
        boundary()
        progress("loading_nf4")
        loaded = runtime.load(previous)
        metadata_receipt = r2.compare_metadata(loaded.metadata, context["old_identity"]["config"]["model"])
        metadata_receipt["execution_revision"] = REVISION
        metadata_receipt["runtime_target_modules_type"] = type(loaded.metadata.get("config", {}).get("target_modules")).__name__
        _write(output / "model-metadata-comparison.json", metadata_receipt, exclusive=True)
        if metadata_receipt["all_fields_equal"] is not True:
            raise ValueError("Actual loaded model metadata has a substantive difference")
        loaded.metadata = metadata_receipt["actual"]
        backend = loaded.backend
        if loaded.metadata != context["old_identity"]["config"]["model"]:
            raise ValueError("Actual loaded model/runtime differs from the saved stage")
        status["model"] = loaded.metadata
        status["gpu_samples"].append(runtime.sample_gpu())
        old_config = SupervisedConfig(**context["old_identity"]["config"]["work"])
        config = replace(old_config, deadline_epoch=deadline)
        encoded = _records(partition.rows, backend, preference=True, eos_id=config.eos_id)
        migrated_cursor, transition = migrate_cursor(context["old_progress"]["cursor"], encoded, old_config, config)
        if migrated_cursor["optimizer_updates"] != ORIGIN_UPDATE or migrated_cursor["offset"] != ORIGIN_UPDATE * BATCH:
            raise ValueError("Unexpected selected origin boundary")
        transition.update(origin_checkpoint=context["resume_checkpoint"],
            recovery_request={"path": str(request_path), "sha256": expected_request_sha256},
            recovery_plan={"path": request["recovery_plan_path"], "sha256": request["recovery_plan_sha256"]},
            budget_authorization=context["plan"]["budget_authorization"])
        transition_path = output / "migration-manifest.json"
        _write(transition_path, transition, exclusive=True)
        files = old._identity_files(previous, selection_path)
        files.update({"source/" + name: ROOT / name for name in request["source_sha256"]})
        files.update({"recovery_plan": Path(request["recovery_plan_path"]), "recovery_request": request_path,
            "migration_manifest": transition_path, "source_checkpoint": Path(context["resume_checkpoint"]["path"]) / "manifest.json",
            "source_checkpoint_state": Path(context["resume_checkpoint"]["path"]) / "state.pt",
            "budget_authorization": Path(context["plan"]["budget_authorization"]["path"])})
        files.update({"recovery_origin/" + name: Path(record["path"]) for name, record in request["origin"].items() if name != "checkpoint"})
        files.update({"prior_segment/" + name: Path(record["path"]) for name, record in request["prior_segment"].items() if name != "checkpoint"})
        files["dispatch_authorization"] = Path(context["plan"]["dispatch_authorization"]["path"])
        identity_config = copy.deepcopy(context["old_identity"]["config"])
        identity_config.update(work=asdict(config), selection=selection, execution_revision=REVISION,
            recovery={"source_checkpoint": context["resume_checkpoint"], "old_deadline_epoch": old_config.deadline_epoch,
                      "new_budget": request["budget"], "migration_manifest_sha256": sha(transition_path),
                      "old_identity_sha256": context["old_manifest"]["identity_sha256"]})
        identity = build_identity(files, identity_config,
            {name: Path(parent["artifact_dir"]) / "manifest.json" for name, parent in previous["parents"].items()})
        _write(output / "identity.json", identity, exclusive=True)
        status["identity"] = identity
        sft = previous["parents"]["sft"]
        status["imports"] = {"reward": import_stage(sft["artifact_dir"], backend, loaded.role,
            expected_manifest_sha256=sft["manifest_sha256"], expected_identity=identity)}
        mapping = parameter_mapping(backend, loaded.role, heads=loaded.heads)
        optimizer = torch.optim.AdamW(mapping.values(), lr=old.EXPECTED["reward"][3])
        generator = torch.Generator(device=loaded.device).manual_seed(42)
        boundary()
        restored = load_checkpoint(context["resume_checkpoint"]["path"], expected_identity=context["old_identity"],
            parameters=mapping, optimizer=optimizer, generators={"sampling": generator},
            expected_manifest_sha256=context["resume_checkpoint"]["manifest_sha256"])
        if restored["cursor"] != transition["old_cursor"] or restored["extra_state"] is not None:
            raise ValueError("Loaded state differs from the recorded transition")
        initial = {name: p.detach().cpu().clone() for name, p in mapping.items()}
        migrated = save_checkpoint(output / f"origin-{ORIGIN_UPDATE:06d}", identity=identity, parameters=mapping,
            optimizer=optimizer, cursor=migrated_cursor, generators={"sampling": generator}, extra_state=None)
        migrated_record = {k: migrated[k] for k in ("checkpoint_path", "manifest_sha256")}
        state_audit = audit_migration_checkpoints(context["resume_checkpoint"], migrated_record, transition, identity)
        state_audit_path = output / "migration-state-audit.json"
        _write(state_audit_path, state_audit, exclusive=True)
        status.update(origin_checkpoint=migrated_record, migration_manifest=pin(transition_path),
                      migration_state_audit=pin(state_audit_path), cursor=migrated_cursor,
                      latest_safe_checkpoint=migrated_record)
        schedule = old.CheckpointSchedule(UPDATES)
        blocked = False

        def save_safe(cursor, extra, *, safe_stop=False):
            if status.get("identity_guard_error"):
                # An identity drift cannot be laundered into a new checkpoint.
                # Preserve the last already committed receipt instead.
                return
            reasons = schedule.reasons(cursor["optimizer_updates"], runtime.now() - started, safe_stop=safe_stop)
            if cursor["optimizer_updates"] == ORIGIN_UPDATE:
                return
            if cursor["optimizer_updates"] == ORIGIN_UPDATE + 1:
                reasons.append("first_recovery_update")
            if reasons:
                item = schedule.save(cursor["optimizer_updates"], runtime.now() - started, reasons,
                    lambda: save_checkpoint(output / f"update-{cursor['optimizer_updates']:06d}", identity=identity,
                        parameters=mapping, optimizer=optimizer, cursor=cursor,
                        generators={"sampling": generator}, extra_state=extra))
                status.update(checkpoints=list(schedule.committed.values()), latest_safe_checkpoint=item)

        def checkpoint(cursor, extra):
            nonlocal blocked
            status.update(cursor=cursor, actual_optimizer_updates=cursor["optimizer_updates"],
                          segment_optimizer_updates=cursor["optimizer_updates"] - ORIGIN_UPDATE)
            gpu = runtime.sample_gpu()
            status["gpu_samples"].append(gpu)
            blocked = not small_identity_unchanged() or blocked
            save_safe(cursor, extra)
            metric({"event": "optimizer_boundary", "cursor": cursor,
                    "active_elapsed_seconds": runtime.now() - started, "gpu": gpu})
            if runtime.now() < deadline and not stop():
                blocked = not inspect()
            blocked = not small_identity_unchanged() or blocked
            progress("training")

        boundary()
        if not inspect():
            blocked = True
        blocked = not small_identity_unchanged() or blocked
        progress("training")
        with loaded.autocast():
            outcome = run_supervised(backend, loaded.role, optimizer, partition.rows, config,
                device=loaded.device, state=migrated_cursor, checkpoint_callback=checkpoint,
                stop_requested=lambda: stop() or (blocked and runtime.now() < deadline), clock=runtime.now)
        status.update(cursor=outcome.cursor, actual_optimizer_updates=outcome.cursor["optimizer_updates"],
            segment_optimizer_updates=outcome.cursor["optimizer_updates"] - ORIGIN_UPDATE,
            outcome_stop_reason=outcome.stop_reason, process_guard_blocked=blocked)
        for item in outcome.metrics:
            metric({"event": "objective_after_engine_return", **item})
        status["objective_records"] = len(outcome.metrics)
        save_safe(outcome.cursor, outcome.extra_state, safe_stop=True)
        if outcome.stop_reason == "paused":
            raise InterruptedError("Recovery stopped at a saved complete optimizer boundary")
        if outcome.stop_reason == "budget_exhausted":
            raise TimeoutError("New recovery absolute deadline reached; old deadline was never changed")
        if blocked:
            raise InterruptedError("Recovery identity/process admission changed at a saved complete boundary")
        completion = verify_combined_completion(outcome, partition.rows, transition["old_cursor"], context["historical_metrics"])
        status["completion"] = completion
        changed = [n for n, p in mapping.items() if not torch.equal(initial[n], p.detach().cpu())]
        status["adapter_weights_changed"] = any(".lora_" in n for n in changed)
        status["scalar_heads_changed"] = {"reward": any(n.startswith("scalar_heads.reward.") for n in changed)}
        if not status["adapter_weights_changed"] or not status["scalar_heads_changed"]["reward"]:
            raise RuntimeError("Recovery adapter and reward head must both actually update")
        boundary(); validate_request(request, execute=True)
        if not inspect():
            raise InterruptedError("Process admission changed before publication")
        artifact = export_stage(output / "final-artifact", backend, loaded.role, stage="reward",
            identity=identity, heads=loaded.heads, completion=completion)
        status["artifact"] = {k: artifact[k] for k in ("artifact_dir", "manifest_path", "manifest_sha256")}
        boundary(); validate_request(request, execute=True)
        status["finished_at_epoch"] = runtime.now()
        progress("formal_parent_recovery_passed")
        manifest = {"schema_version": 1, "kind": "alignlab-parent-recovery-r4-result-v1",
            "status": "formal_parent_recovery_passed", "formal_training": True, "research_complete": False,
            "request": status["request_file"], "budget": request["budget"], "origin": request["origin"], "prior_segment": request["prior_segment"],
            "identity_sha256": migrated["identity_sha256"], "completion": completion,
            "process_identity": owned, "started_at_epoch": started, "finished_at_epoch": status["finished_at_epoch"],
            "origin_checkpoint": migrated_record, "migration_manifest": status["migration_manifest"],
            "migration_state_audit": status["migration_state_audit"], "artifact": status["artifact"],
            "latest_safe_checkpoint": status["latest_safe_checkpoint"], "checkpoints": status["checkpoints"]}
        manifest["files"] = {p.relative_to(output).as_posix(): {"sha256": sha(p), "size_bytes": p.stat().st_size}
                             for p in sorted(output.rglob("*")) if p.is_file()}
        boundary()
        _write(output / "manifest.json", manifest, exclusive=True)
        return manifest
    except BaseException as exc:
        status.update(status="budget_exhausted" if isinstance(exc, TimeoutError) else
            "paused" if isinstance(exc, InterruptedError) else "failed", error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(), finished_at_epoch=runtime.now(),
            safe_resume_available=status["latest_safe_checkpoint"] is not None,
            recovery_policy="No automatic retry; any later recovery requires another explicit immutable plan")
        if output.exists() and not (output / "manifest.json").exists():
            _write(output / "progress.json", status)
        raise
    finally:
        if lock.exists() and read_json(lock) == owner:
            lock.unlink()


def verify_recovery_output(output, *, expected_manifest_sha256, expected_request=None):
    """Independent read-only CPU audit of receipts, payloads and cumulative work.

    Does not perform a CUDA restore or assert process exit; the accepting owner
    must independently verify recorded PID/creation/command identities exited.
    """
    import torch
    from .stage_artifacts import _read_artifact
    output = _path(output)
    old._pin(output / "manifest.json", expected_manifest_sha256)
    result = read_json(output / "manifest.json")
    if (result.get("kind") != "alignlab-parent-recovery-r4-result-v1"
            or result.get("status") != "formal_parent_recovery_passed"
            or result.get("formal_training") is not True or result.get("research_complete") is not False):
        raise ValueError("Not a complete formal recovery result")
    request = read_json(_pinned(result["request"]))
    if expected_request is not None and expected_request != request:
        raise ValueError("Recovery request differs from caller's pinned request")
    context = validate_request(request, execute=True)
    audit_prior_migration(context)
    if context["output"] != output or read_json(output / "request.json") != request:
        raise ValueError("Recovery output/request differs")
    files = result["files"]
    if (type(files) is not dict or not files
            or {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
            != set(files) | {"manifest.json"}):
        raise ValueError("Incomplete or unexpected recovery output inventory")
    for relative, evidence in files.items():
        path = output / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or _path(path).parent == ROOT.parent:
            raise ValueError("Unsafe recovery inventory path")
        old._pin(path, evidence["sha256"])
        if path.stat().st_size != evidence["size_bytes"]:
            raise ValueError("Recovery artifact size changed")
    progress = read_json(output / "progress.json")
    identity = read_json(output / "identity.json")
    receipt = read_json(output / "model-metadata-comparison.json")
    expected_receipt = r2.compare_metadata(receipt["actual"], context["old_identity"]["config"]["model"])
    expected_receipt["execution_revision"] = REVISION
    if ({k: v for k, v in receipt.items() if k != "runtime_target_modules_type"} != expected_receipt
            or expected_receipt["all_fields_equal"] is not True
            or progress["model"] != expected_receipt["actual"]):
        raise ValueError("R4 full model metadata comparison receipt differs")
    if (progress.get("status") != result["status"] or progress["request"] != request
            or result["origin"] != request["origin"] or result["prior_segment"] != request["prior_segment"]
            or result["budget"] != request["budget"]
            or progress["identity"] != identity or progress["completion"] != result["completion"]
            or progress["actual_optimizer_updates"] != UPDATES
            or progress["segment_optimizer_updates"] != UPDATES - ORIGIN_UPDATE
            or progress["adapter_weights_changed"] is not True
            or progress["scalar_heads_changed"] != {"reward": True}
            or not request["budget"]["started_at_epoch"] <= result["finished_at_epoch"] <= request["budget"]["deadline_epoch"]
            or result["finished_at_epoch"] != progress["finished_at_epoch"]
            or result["process_identity"] != progress["process_identity"]):
        raise ValueError("Recovery progress does not prove the declared segment")
    transition = read_json(_pinned(result["migration_manifest"]))
    if Path(result["migration_manifest"]["path"]).resolve() != output / "migration-manifest.json":
        raise ValueError("Migration receipt is outside recovery output")
    before, after = transition["old_cursor"], transition["new_cursor"]
    if (transition["changed_cursor_fields"] != []
            or transition["changed_work_fields"] != []
            or before != context["old_progress"]["cursor"] or before != after
            or {k: v for k, v in before.items() if k != "config_sha256"}
            != {k: v for k, v in after.items() if k != "config_sha256"}
            or transition["old_cursor_sha256"] != digest(before) or transition["new_cursor_sha256"] != digest(after)):
        raise ValueError("Undeclared cursor migration")
    from .stage_engine import _hash
    old_work, new_work = transition["old_work"], transition["new_work"]
    if (old_work != context["old_identity"]["config"]["work"] or new_work != identity["config"]["work"] or new_work != old_work
            or {k: v for k, v in old_work.items() if k != "deadline_epoch"}
            != {k: v for k, v in new_work.items() if k != "deadline_epoch"}
            or new_work["deadline_epoch"] != request["budget"]["deadline_epoch"]
            or before["config_sha256"] != _hash({"config": old_work, "role": "reward"})
            or after["config_sha256"] != _hash({"config": new_work, "role": "reward"})):
        raise ValueError("Numerical work or declared deadline migration changed")
    if Path(result["origin_checkpoint"]["checkpoint_path"]).resolve() != output / f"origin-{ORIGIN_UPDATE:06d}":
        raise ValueError("New origin checkpoint escaped recovery output")
    audited = audit_migration_checkpoints(context["resume_checkpoint"], result["origin_checkpoint"], transition, identity)
    if read_json(_pinned(result["migration_state_audit"])) != audited:
        raise ValueError("Published migration audit differs from actual checkpoint tensors")
    objectives = _objectives(output / "metrics.jsonl", ORIGIN_UPDATE + 1, UPDATES)
    historical = context["historical_metrics"]
    selected = read_json(output / "selection.json")
    if selected != read_json(request["origin"]["selection"]["path"]):
        raise ValueError("Recovery selected different records")
    records = selected["records"]
    if len(records) != ROWS or len({r["record_id"] for r in records}) != ROWS:
        raise ValueError("Recovery selection count/uniqueness differs")
    cursor = progress["cursor"]
    total_tokens = sum(r["chosen_tokens"] + r["rejected_tokens"] for r in records)
    if (cursor["order"] != before["order"] or cursor["data_sha256"] != before["data_sha256"]
            or cursor["config_sha256"] != after["config_sha256"] or cursor["offset"] != ROWS
            or cursor["optimizer_updates"] != UPDATES or cursor["response_tokens"] != total_tokens
            or cursor["active_rollout"] is not False or cursor["completed_rollouts"] != 0):
        raise ValueError("Final cursor is not the whole original one-pass dataset")
    boundaries = [x for x in _jsonl(output / "metrics.jsonl") if x.get("event") == "optimizer_boundary"]
    if [x["cursor"]["optimizer_updates"] for x in boundaries] != list(range(ORIGIN_UPDATE + 1, UPDATES + 1)):
        raise ValueError("Missing or duplicate recovery optimizer boundaries")
    processed_tokens = before["response_tokens"]
    for item, boundary_record in zip(objectives, boundaries):
        update = item["update"]
        first, last = (update - 1) * BATCH, min(update * BATCH, ROWS)
        processed_tokens += sum(records[i]["chosen_tokens"] + records[i]["rejected_tokens"] for i in before["order"][first:last])
        boundary_cursor = boundary_record["cursor"]
        if (item["rows"] != last - first or boundary_cursor["offset"] != last
                or boundary_cursor["order"] != before["order"] or boundary_cursor["response_tokens"] != processed_tokens
                or boundary_cursor["config_sha256"] != after["config_sha256"]
                or boundary_cursor["data_sha256"] != before["data_sha256"]
                or boundary_cursor["active_rollout"] is not False):
            raise ValueError("Recovery boundary record does not match actual original data order")
    if sum(x["rows"] for x in [*historical, *objectives]) != ROWS:
        raise ValueError("Combined objective denominator is incomplete")
    expected_completion = {"stop_reason": "data_exhausted", "data_pass_complete": True,
        "protocol_complete": True, "full_accepted_dataset_pass": True, "scope": old.SCOPE,
        "synthetic_rows": 0, "rows_processed": ROWS, "optimizer_updates": UPDATES,
        "response_tokens": total_tokens, "historical_optimizer_updates": ORIGIN_UPDATE,
        "segment_optimizer_updates": UPDATES - ORIGIN_UPDATE, "historical_rows": before["offset"],
        "segment_rows": ROWS - before["offset"], "historical_response_tokens": before["response_tokens"],
        "segment_response_tokens": total_tokens - before["response_tokens"]}
    if result["completion"] != expected_completion:
        raise ValueError("Published combined completion does not match the independently counted work")
    checkpoint = result["latest_safe_checkpoint"]
    if Path(checkpoint["checkpoint_path"]).resolve() != output / f"update-{UPDATES:06d}":
        raise ValueError("Final checkpoint path differs")
    final_manifest, final_payload = _checkpoint_payload(checkpoint["checkpoint_path"], checkpoint["manifest_sha256"])
    if final_manifest["identity"] != identity or final_payload["cursor"] != cursor or final_payload["extra_state"] is not None:
        raise ValueError("Final saved optimizer boundary differs")
    if any(int(s["step"].item()) != UPDATES for s in final_payload["optimizer"]["state"].values()):
        raise ValueError("Final Adam update counters differ")
    artifact = result["artifact"]
    if Path(artifact["artifact_dir"]).resolve() != output / "final-artifact":
        raise ValueError("Final model escaped output")
    model_manifest, _ = _read_artifact(artifact["artifact_dir"], artifact["manifest_sha256"])
    if model_manifest["identity"] != identity or model_manifest["completion"] != result["completion"]:
        raise ValueError("Final adapter metadata differs from optimizer checkpoint")
    weights = torch.load(output / "final-artifact/weights.pt", map_location="cpu", weights_only=True)
    normalized = {}
    for name, value in final_payload["parameters"].items():
        if name.startswith("scalar_heads.reward."):
            normalized["heads/reward/" + name.removeprefix("scalar_heads.reward.")] = value
        else:
            normalized["adapter/" + name.replace(".lora_A.reward.", ".lora_A.").replace(".lora_B.reward.", ".lora_B.")] = value
    if set(weights) != set(normalized) or any(not torch.equal(weights[k], normalized[k]) for k in weights):
        raise ValueError("Final exported adapter/head weights differ from complete optimizer boundary")
    return result
