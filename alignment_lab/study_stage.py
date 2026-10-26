"""Fresh, full-real SFT/RM parent stages, using the accepted supervised engine.

No automatic resume is exposed. Immutable checkpoints retain original identity,
deadline, optimizer and RNG for separately authorized recovery. Only a returned
clean engine boundary can be saved; an unknown backward/optimizer exception
never triggers a speculative emergency checkpoint. Objective metrics are
published after the engine returns, while cursor/cost boundaries are live.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

from .capacity_stage import sha, read_json, _write, boundary

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-formal-parent-stage-v1"
SCOPE = "full_accepted_real_training_partition"
EXPECTED = {"sft": (4000, 250, "sft_train", 1e-4),
            "reward": (13139, 822, "preference_train", 5e-5)}
PAD, EOS = 151643, 151645
# Explicit execution dependencies: later, unrelated evaluation modules must not
# silently change an already frozen parent run's source identity.
REQUIRED_SOURCES = frozenset({
    "alignment_lab/__init__.py", "alignment_lab/study_stage.py",
    "alignment_lab/capacity_stage.py", "alignment_lab/gpu_probe.py",
    "alignment_lab/model_loader.py", "alignment_lab/model_snapshot.py",
    "alignment_lab/training_data.py", "alignment_lab/chat_codec.py",
    "alignment_lab/data_audit.py", "alignment_lab/stage_engine.py",
    "alignment_lab/training_steps.py", "alignment_lab/shared_qwen.py",
    "alignment_lab/token_logprobs.py", "alignment_lab/rollout.py",
    "alignment_lab/ppo_objective.py", "alignment_lab/checkpointing.py",
    "alignment_lab/stage_artifacts.py", "rlhf_lab/runtime.py",
    "scripts/run_alignlab_study_stage.py",
})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _pin(path, digest):
    if (type(digest) is not str or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest) or sha(path) != digest):
        raise ValueError(f"Frozen file SHA mismatch: {path}")


def validate_parent(parent, request):
    """Reject a capacity-trained or uncommitted parent before CUDA admission."""
    if set(parent) != {"artifact_dir", "manifest_sha256"}:
        raise ValueError("Parent needs artifact directory and immutable manifest SHA")
    path = Path(parent["artifact_dir"]) / "manifest.json"
    _pin(path, parent["manifest_sha256"])
    manifest = read_json(path)
    config, completion = manifest.get("identity", {}).get("config", {}), manifest.get("completion", {})
    if (manifest.get("stage") != "sft" or manifest.get("status") != "complete"
            or config.get("purpose") != KIND or config.get("scope") != SCOPE
            or config.get("synthetic_rows") != 0 or config.get("formal_training") is not True
            or config.get("expected_rows") != EXPECTED["sft"][0]
            or config.get("expected_updates") != EXPECTED["sft"][1]
            or completion.get("full_accepted_dataset_pass") is not True
            or completion.get("scope") != SCOPE or completion.get("synthetic_rows") != 0
            or completion.get("rows_processed") != EXPECTED["sft"][0]
            or completion.get("optimizer_updates") != EXPECTED["sft"][1]
            or completion.get("stop_reason") != "data_exhausted"
            or completion.get("data_pass_complete") is not True
            or completion.get("protocol_complete") is not True):
        raise ValueError("Only a complete fresh formal full-real SFT parent is accepted; capacity parents forbidden")
    inputs = manifest["identity"]["inputs"]
    if (inputs.get("data/manifest", {}).get("sha256") != request["audit_manifest_sha256"]
            or inputs.get("parent_plan", {}).get("sha256") != request["parent_plan_sha256"]):
        raise ValueError("SFT parent data/plan identity differs from this formal run")
    progress = read_json(path.parent.parent / "progress.json")
    if (progress.get("status") != "formal_parent_stage_passed"
            or progress.get("formal_training") is not True
            or progress.get("research_complete") is not False
            or progress.get("artifact", {}).get("manifest_sha256") != parent["manifest_sha256"]):
        raise ValueError("SFT parent has no committed successful stage report")
    _pin(path.parent / "weights.pt", manifest["artifacts"]["weights.pt"]["sha256"])
    return manifest


def validate_request(request):
    keys = {"schema_version", "kind", "stage", "output", "deadline_epoch", "parent_plan_path",
        "parent_plan_sha256", "source_sha256", "model_snapshot", "model_manifest_sha256",
        "tokenizer_dir", "audit_dir", "audit_manifest_sha256", "capacity_acceptance_path",
        "capacity_acceptance_sha256", "parents", "expected_rows", "expected_updates", "seed"}
    if (type(request) is not dict or set(request) != keys or type(request["schema_version"]) is not int
            or request["schema_version"] != 1 or request["kind"] != KIND or request["stage"] not in EXPECTED):
        raise ValueError("Exact formal-parent stage request schema required")
    stage = request["stage"]
    if (type(request["seed"]) is not int or request["seed"] != 42
            or type(request["expected_rows"]) is not int or request["expected_rows"] != EXPECTED[stage][0]
            or type(request["expected_updates"]) is not int or request["expected_updates"] != EXPECTED[stage][1]
            or set(request["parents"]) != ({"sft"} if stage == "reward" else set())):
        raise ValueError("Formal parent rows/updates/seed/lineage changed")
    if (type(request["deadline_epoch"]) not in (int, float) or not math.isfinite(request["deadline_epoch"])
            or request["deadline_epoch"] <= 0):
        raise ValueError("Frozen absolute deadline required")
    output = Path(request["output"]).resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        raise ValueError("A fresh project runs output directory is required")
    sources = request["source_sha256"]
    if type(sources) is not dict or not REQUIRED_SOURCES <= set(sources):
        raise ValueError("Explicit complete parent execution source inventory required")
    for name, digest in sources.items():
        raw = ROOT / name
        if raw.is_symlink() or not raw.resolve().is_relative_to(ROOT):
            raise ValueError("Source must be an ordinary project file")
        _pin(raw, digest)
    for path, digest in ((request["parent_plan_path"], request["parent_plan_sha256"]),
            (request["capacity_acceptance_path"], request["capacity_acceptance_sha256"]),
            (Path(request["model_snapshot"]) / "snapshot-manifest.json", request["model_manifest_sha256"]),
            (Path(request["audit_dir"]) / "manifest.json", request["audit_manifest_sha256"])):
        _pin(path, digest)
    acceptance = read_json(request["capacity_acceptance_path"])
    if (acceptance.get("status") != "accepted_registered_full_capacity_workload_only"
            or acceptance.get("all_checks_passed") is not True or acceptance.get("capacity_verified") is not True
            or acceptance.get("formal_training") is not False
            or acceptance.get("model_snapshot", {}).get("sha256") != request["model_manifest_sha256"]
            or acceptance.get("data", {}).get("manifest", {}).get("sha256") != request["audit_manifest_sha256"]):
        raise ValueError("Independent accepted full-capacity evidence required for these exact inputs")
    snapshot = acceptance["source_snapshot"]
    _pin(snapshot["path"], snapshot["sha256"])
    for parent in request["parents"].values():
        validate_parent(parent, request)
    return output


def real_selection(partition, request):
    from .training_data import TrainingPartition, TrainingRow
    stage = request["stage"]
    if (not isinstance(partition, TrainingPartition) or partition.partition != EXPECTED[stage][2]
            or partition.manifest_sha256 != request["audit_manifest_sha256"]
            or len(partition.rows) != request["expected_rows"]):
        raise ValueError("Use the entire exact accepted training partition, without resampling")
    records, seen = [], set()
    for row in partition.rows:
        if (not isinstance(row, TrainingRow) or row.record_id in seen
                or row.record_id.startswith("synthetic-") or not row.record_id):
            raise ValueError("Formal parents cannot include duplicate or synthetic records")
        seen.add(row.record_id)
        fields = ("chosen_ids",) if stage == "sft" else ("chosen_ids", "rejected_ids")
        if not row.prompt_ids or len(row.prompt_ids) > 1024:
            raise ValueError("Formal prompt length differs from accepted protocol")
        for field in fields:
            ids = getattr(row, field)
            if not ids or len(ids) > 768 or ids[-1] != EOS or EOS in ids[:-1]:
                raise ValueError("Full real completions must preserve their one terminal EOS")
        if stage == "sft" and row.rejected_ids is not None:
            raise ValueError("SFT partition must contain its chosen target only")
        records.append({"record_id": row.record_id, "cluster_id": row.cluster_id, "prompt_id": row.prompt_id,
            "prompt_tokens": len(row.prompt_ids), "chosen_tokens": len(row.chosen_ids),
            "rejected_tokens": None if row.rejected_ids is None else len(row.rejected_ids),
            "token_sha256": _digest([row.prompt_ids, row.chosen_ids, row.rejected_ids])})
    clusters = len({r.cluster_id for r in partition.rows})
    if stage == "sft" and clusters != request["expected_rows"]:
        raise ValueError("Formal SFT requires one real row per independent cluster")
    summary = {"scope": SCOPE, "partition": partition.partition,
        "accepted_manifest_sha256": partition.manifest_sha256, "accepted_identity_sha256": partition.identity_sha256,
        "serialization_version": partition.serialization_version, "configured_rows": len(records),
        "selected_rows": len(records), "real_rows": len(records), "synthetic_rows": 0,
        "unique_prompt_ids": len({r.prompt_id for r in partition.rows}),
        "independent_prompt_clusters": clusters, "records_sha256": _digest(records)}
    return summary, records


class CheckpointSchedule:
    """Save each completed update at most once, combining all trigger reasons."""
    hours = (1, 2, 4, 8)

    def __init__(self, final_update):
        self.final_update = final_update
        self.costs_recorded, self.attempted, self.committed = set(), set(), {}

    def reasons(self, update, elapsed, *, safe_stop=False):
        reasons = []
        if update == 1:
            reasons.append("first_update")
        if update > 0 and update % 50 == 0:
            reasons.append("every_50_updates")
        if update == self.final_update:
            reasons.append("full_data_pass")
        if safe_stop:
            reasons.append("safe_engine_return")
        reasons.extend(f"cost_{hour}h" for hour in self.hours if hour not in self.costs_recorded and elapsed >= hour * 3600)
        return reasons

    def save(self, update, elapsed, reasons, writer):
        if update in self.committed:
            record = self.committed[update]
            record["reasons"] = sorted(set(record["reasons"]) | set(reasons))
            self._costs(reasons)
            return record
        if update in self.attempted:
            raise RuntimeError("An incomplete checkpoint attempt cannot be retried or overwritten")
        self.attempted.add(update)
        receipt = writer()
        record = {key: receipt[key] for key in ("checkpoint_path", "manifest_sha256")}
        record.update(optimizer_updates=update, active_elapsed_seconds=elapsed, reasons=list(reasons))
        self.committed[update] = record
        self._costs(reasons)
        return record

    def _costs(self, reasons):
        self.costs_recorded.update(hour for hour in self.hours if f"cost_{hour}h" in reasons)


def _identity_files(request, selection_path):
    snapshot, tokenizer, audit = (Path(request[k]) for k in ("model_snapshot", "tokenizer_dir", "audit_dir"))
    files = {f"source/{name}": ROOT / name for name in request["source_sha256"]}
    files.update({f"base/{name}": snapshot / name for name in read_json(snapshot / "snapshot-manifest.json")["files"]})
    files.update({f"tokenizer/{name}": tokenizer / name for name in read_json(tokenizer / "source-manifest.json")["artifacts"]})
    files.update({"tokenizer/source-manifest.json": tokenizer / "source-manifest.json",
        "data/manifest": audit / "manifest.json", "data/partition": audit / f"{EXPECTED[request['stage']][2]}.jsonl",
        "data/selection": selection_path, "protocol": ROOT / "docs/ALIGNLAB_V2.md",
        "parent_plan": Path(request["parent_plan_path"]), "capacity_acceptance": Path(request["capacity_acceptance_path"])})
    return files


def verify_completion(outcome, rows, request):
    tokens = sum(len(r.chosen_ids) + (len(r.rejected_ids) if request["stage"] == "reward" else 0) for r in rows)
    cursor = outcome.cursor
    if (outcome.stop_reason != "data_exhausted" or cursor["offset"] != request["expected_rows"]
            or cursor["optimizer_updates"] != request["expected_updates"]
            or cursor["response_tokens"] != tokens or cursor["active_rollout"] or outcome.extra_state is not None
            or len(outcome.metrics) != request["expected_updates"]):
        raise RuntimeError("Formal parent has not completed its entire accepted real data pass")
    return {"stop_reason": "data_exhausted", "data_pass_complete": True, "protocol_complete": True,
        "full_accepted_dataset_pass": True, "scope": SCOPE, "synthetic_rows": 0,
        "rows_processed": len(rows), "optimizer_updates": cursor["optimizer_updates"], "response_tokens": tokens}


def run_study_stage(request, *, execute=False):
    output = validate_request(request)
    if not execute:
        return {"status": "plan_only", "stage": request["stage"], "formal_training": False, "research_complete": False}
    if output.exists():
        raise FileExistsError("Formal stage output exists; no overwrite or automatic resume")
    from .gpu_probe import resources
    stop = lambda: (output / "STOP").exists() or (output.parent / "STOP").exists()
    deadline, stage = request["deadline_epoch"], request["stage"]
    boundary(deadline, stop)
    admission = resources()
    if admission["blockers"]:
        raise RuntimeError(f"CUDA admission blocked: {admission['blockers']}")
    lock, owner = ROOT / "reports/.alignlab-gpu.lock", {"pid": os.getpid(), "output": str(output), "token": uuid.uuid4().hex}
    _write(lock, owner, exclusive=True)
    started = time.time()
    status = {"kind": KIND, "status": "starting", "stage": stage, "request": request, "pid": os.getpid(),
        "started_at_epoch": started, "deadline_epoch": deadline, "formal_training": True, "research_complete": False,
        "training_started": False, "actual_optimizer_updates": 0, "resources_before": admission,
        "scope": SCOPE, "synthetic_rows": 0, "checkpoints": [], "latest_safe_checkpoint": None,
        "gpu_samples": [], "objective_logging": "after_engine_returns"}

    def progress(phase):
        status.update(status=phase, updated_at_epoch=time.time(), active_elapsed_seconds=time.time() - started)
        _write(output / "progress.json", status)
        print(json.dumps({"stage": stage, "phase": phase, "optimizer_updates": status["actual_optimizer_updates"],
                          "active_elapsed_seconds": status["active_elapsed_seconds"]}), flush=True)

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

    try:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "request.json", request, exclusive=True)
        progress("loading_full_real_partition")
        from .training_data import load_training_partition
        partition = load_training_partition(request["audit_dir"], EXPECTED[stage][2], request["audit_manifest_sha256"])
        selection, records = real_selection(partition, request)
        selection_path = output / "selection.json"
        _write(selection_path, {"summary": selection, "records": records}, exclusive=True)
        selection.update(selection_manifest_path=str(selection_path), selection_manifest_sha256=sha(selection_path))
        status["selection"] = selection
        boundary(deadline, stop)
        validate_request(request)
        immediate = resources()
        if immediate["blockers"]:
            raise RuntimeError(f"Resources changed before CUDA load: {immediate['blockers']}")
        import torch
        from .model_loader import ModelLoadConfig, load_quantized_stage
        from .stage_engine import SupervisedConfig, parameter_mapping, run_supervised
        from .checkpointing import build_identity, save_checkpoint
        from .stage_artifacts import import_stage, export_stage
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        progress("loading_nf4")
        loaded = load_quantized_stage(request["model_snapshot"], request["model_manifest_sha256"],
            request["tokenizer_dir"], ModelLoadConfig(stage, seed=request["seed"]))
        status["model"] = loaded.metadata
        gpu_sample()
        config = SupervisedConfig(kind=stage, seed=request["seed"], effective_batch=16, microbatch=1,
            max_updates=request["expected_updates"], deadline_epoch=deadline, pad_id=PAD, eos_id=EOS, chunk_tokens=16)
        identity = build_identity(_identity_files(request, selection_path), {
            "stage": stage, "purpose": KIND, "scope": SCOPE, "synthetic_rows": 0, "formal_training": True,
            "expected_rows": request["expected_rows"], "expected_updates": request["expected_updates"],
            "work": asdict(config), "model": loaded.metadata, "selection": selection,
            "optimizer": {"name": "AdamW", "lr": EXPECTED[stage][3], "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            "checkpoint_policy": {"first": 1, "interval_updates": 50, "cost_hours": [1, 2, 4, 8], "save_clean_return": True}},
            {name: Path(parent["artifact_dir"]) / "manifest.json" for name, parent in request["parents"].items()})
        status["identity"] = identity
        _write(output / "identity.json", identity, exclusive=True)
        backend = loaded.backend
        status["imports"] = {}
        if stage == "reward":
            parent = request["parents"]["sft"]
            validate_parent(parent, request)
            status["imports"]["reward"] = import_stage(parent["artifact_dir"], backend, loaded.role,
                expected_manifest_sha256=parent["manifest_sha256"], expected_identity=identity)
        boundary(deadline, stop)
        mapping = parameter_mapping(backend, loaded.role, heads=loaded.heads)
        optimizer = torch.optim.AdamW(mapping.values(), lr=EXPECTED[stage][3])
        generator = torch.Generator(device=loaded.device).manual_seed(request["seed"])
        initial = {name: p.detach().cpu().clone() for name, p in mapping.items()}
        schedule = CheckpointSchedule(request["expected_updates"])

        def save_safe(cursor, extra, *, safe_stop=False):
            update, elapsed = cursor["optimizer_updates"], time.time() - started
            reasons = schedule.reasons(update, elapsed, safe_stop=safe_stop)
            if reasons:
                record = schedule.save(update, elapsed, reasons, lambda: save_checkpoint(
                    output / f"update-{update:06d}", identity=identity, parameters=mapping,
                    optimizer=optimizer, cursor=cursor, generators={"sampling": generator}, extra_state=extra))
                status["checkpoints"] = list(schedule.committed.values())
                status["latest_safe_checkpoint"] = record

        def checkpoint(cursor, extra):
            status.update(actual_optimizer_updates=cursor["optimizer_updates"],
                          training_started=cursor["optimizer_updates"] > 0, cursor=cursor)
            save_safe(cursor, extra)
            metric({"event": "optimizer_boundary", "cursor": cursor,
                    "active_elapsed_seconds": time.time() - started, "gpu": gpu_sample()})
            progress("training")

        progress("training")
        with loaded.autocast():
            outcome = run_supervised(backend, loaded.role, optimizer, partition.rows, config,
                device=loaded.device, checkpoint_callback=checkpoint, stop_requested=stop)
        status.update(cursor=outcome.cursor, actual_optimizer_updates=outcome.cursor["optimizer_updates"],
                      outcome_stop_reason=outcome.stop_reason)
        for item in outcome.metrics:
            metric({"event": "objective_after_engine_return", **item})
        status["objective_records"] = len(outcome.metrics)
        # A normal return guarantees the engine is outside a backward callback,
        # with cleared gradients. Saving here is allowed even at the deadline.
        save_safe(outcome.cursor, outcome.extra_state, safe_stop=True)
        if outcome.stop_reason == "paused":
            raise InterruptedError("Parent stage paused at a saved complete optimizer boundary")
        if outcome.stop_reason == "budget_exhausted":
            raise TimeoutError("Shared parent deadline reached; latest complete boundary saved")
        status["completion"] = verify_completion(outcome, partition.rows, request)
        boundary(deadline, stop)
        changed = [name for name, p in mapping.items() if not torch.equal(initial[name], p.detach().cpu())]
        status["adapter_weights_changed"] = any(".lora_" in name for name in changed)
        status["scalar_heads_changed"] = {h: any(name.startswith(f"scalar_heads.{h}.") for name in changed) for h in loaded.heads}
        if not status["adapter_weights_changed"] or not all(status["scalar_heads_changed"].values()):
            raise RuntimeError("Every formal adapter and owned scalar head must actually change")
        validate_request(request)
        boundary(deadline, stop)
        artifact = export_stage(output / "final-artifact", backend, loaded.role, stage=stage,
            identity=identity, heads=loaded.heads, completion=status["completion"])
        status["artifact"] = {key: artifact[key] for key in ("artifact_dir", "manifest_path", "manifest_sha256")}
        gpu_sample()
        validate_request(request)
        boundary(deadline, stop)
        progress("formal_parent_stage_passed")
        return status
    except BaseException as exc:
        status.update(status="budget_exhausted" if isinstance(exc, TimeoutError) else
            "paused" if isinstance(exc, InterruptedError) else "failed",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            safe_resume_available=status["latest_safe_checkpoint"] is not None,
            emergency_checkpoint_attempted=False,
            recovery_policy="Use only an explicitly pinned previously committed checkpoint; no automatic resume")
        raise
    finally:
        status.update(finished_at_epoch=time.time(), elapsed_seconds=time.time() - started)
        if output.exists():
            _write(output / "progress.json", status)
        if lock.exists() and read_json(lock) == owner:
            lock.unlink()
