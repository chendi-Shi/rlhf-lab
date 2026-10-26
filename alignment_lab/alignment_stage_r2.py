"""Fresh formal DPO/PPO branches over explicit real training data.

Uses the accepted numerical engines unchanged. Existing outputs and implicit
resume/retry are forbidden. Frozen reference cache construction is cooperative
between individual scored sequences; no manifest is published for partial
scoring. A PPO subset is explicitly distinguished from its full accepted pool.
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

from . import study_stage as parent_stage
from .capacity_stage import sha, read_json, _write, boundary, _head_anchor
from .training_telemetry import TrainingTelemetry, PROTOCOL as TELEMETRY_PROTOCOL, REVISION as EXECUTION_REVISION

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-formal-alignment-stage-r2-v1"
FULL_SCOPE = "full_accepted_real_training_partition"
SUBSET_SCOPE = "registered_unique_real_prompt_subset"
SELECTION_RULE = "sha256_prompt_id_v1"
DPO_ROWS, PPO_POOL_ROWS = 13139, 12748
PPO_COUNTS = (1024, 4096)
EFFECTIVE_BATCH, ROLLOUT_BATCH, MAX_NEW_TOKENS = 16, 16, 256
PAD, EOS, VOCAB_SIZE = 151643, 151645, 151936
CHUNK_TOKENS = 16
PRECISION = "bfloat16"
REQUIRED_SOURCES = parent_stage.REQUIRED_SOURCES | {
    "alignment_lab/alignment_stage.py", "scripts/run_alignlab_alignment_stage.py",
    "alignment_lab/alignment_stage_r2.py", "scripts/run_alignlab_alignment_stage_r2.py",
    "alignment_lab/alignment_supervisor_r2.py", "scripts/run_alignlab_alignment_r2.py",
    "alignment_lab/alignment_process_guard_r2.py", "alignment_lab/process_guard_v4.py",
    "alignment_lab/reference_cache_recovery.py", "tests_v2/test_alignment_stage_r2.py",
    "alignment_lab/gpu_reference.py", "alignment_lab/reference_bridge.py",
    "alignment_lab/reference_cache.py",
    "alignment_lab/training_telemetry.py", "alignment_lab/stage_engine_observed.py",
    "alignment_lab/alignment_process_guard.py", "alignment_lab/process_guard_v2.py",
    "alignment_lab/process_guard_v3.py", "tests_v2/test_alignment_process_guard.py",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_workload(stage, work):
    if type(work) is not dict:
        raise ValueError("Explicit registered workload required")
    if stage == "dpo":
        expected = {"scope": FULL_SCOPE, "rows": DPO_ROWS,
                    "optimizer_updates": math.ceil(DPO_ROWS / EFFECTIVE_BATCH), "beta": 0.1}
        if (work != expected or type(work.get("rows")) is not int
                or type(work.get("optimizer_updates")) is not int or type(work.get("beta")) is not float):
            raise ValueError("DPO requires the entire accepted preference partition and beta=0.1")
    elif stage == "ppo":
        if (set(work) != {"scope", "rows", "optimizer_updates", "selection_seed", "selection_algorithm", "kl_coef"}
                or work["scope"] != SUBSET_SCOPE or type(work["rows"]) is not int or work["rows"] not in PPO_COUNTS
                or type(work["optimizer_updates"]) is not int or work["optimizer_updates"] != 2 * work["rows"]
                or type(work["selection_seed"]) is not int or work["selection_seed"] != 42
                or work["selection_algorithm"] != SELECTION_RULE
                or type(work["kl_coef"]) is not float or work["kl_coef"] != 0.05):
            raise ValueError("PPO needs an explicit registered distinct-prompt count and frozen selection/algorithm")
    else:
        raise ValueError("Only formal DPO/PPO branches are supported")


def _formal_parent(name, record, request):
    """Verify completed real parents; artifact importer also checks all payloads."""
    path = Path(record["artifact_dir"]) / "manifest.json"
    parent_stage._pin(path, record["manifest_sha256"])
    manifest = read_json(path)
    config, completion = manifest["identity"]["config"], manifest["completion"]
    rows, updates = parent_stage.EXPECTED[name][:2]
    if (manifest.get("stage") != name or manifest.get("status") != "complete"
            or config.get("purpose") != parent_stage.KIND or config.get("scope") != FULL_SCOPE
            or config.get("synthetic_rows") != 0 or config.get("formal_training") is not True
            or config.get("expected_rows") != rows or config.get("expected_updates") != updates
            or completion.get("scope") != FULL_SCOPE or completion.get("synthetic_rows") != 0
            or completion.get("full_accepted_dataset_pass") is not True
            or completion.get("data_pass_complete") is not True or completion.get("protocol_complete") is not True
            or completion.get("stop_reason") != "data_exhausted"
            or completion.get("rows_processed") != rows or completion.get("optimizer_updates") != updates):
        raise ValueError("Capacity/partial parents cannot initialize formal alignment")
    inputs = manifest["identity"]["inputs"]
    if (inputs["data/manifest"]["sha256"] != request["audit_manifest_sha256"]
            or inputs["parent_plan"]["sha256"] != request["parent_plan_sha256"]):
        raise ValueError("Parent data/common-plan identity changed")
    parent_stage._pin(path.parent / "weights.pt", manifest["artifacts"]["weights.pt"]["sha256"])
    return manifest


def validate_parent_acceptance(request):
    parent_stage._pin(request["parent_acceptance_path"], request["parent_acceptance_sha256"])
    accepted = read_json(request["parent_acceptance_path"])
    if accepted.get("kind") == "alignlab-independent-recovered-parents-r5-acceptance-v1":
        from .recovered_parent_acceptance_r5 import validate_recovered_for_request
        return validate_recovered_for_request(request)
    if accepted.get("kind") == "alignlab-independent-recovered-parents-acceptance-v1":
        from .recovered_parent_acceptance import validate_recovered_for_request
        return validate_recovered_for_request(request)
    if (type(accepted.get("schema_version")) is not int or accepted["schema_version"] != 1
            or accepted.get("kind") != "alignlab-independent-formal-parents-acceptance-v1"
            or accepted.get("status") != "accepted_full_real_formal_parents"
            or accepted.get("all_checks_passed") is not True or accepted.get("parents_verified") is not True
            or accepted.get("formal_training") is not True or accepted.get("research_complete") is not False
            or accepted.get("processes_exited") is not True or set(accepted.get("stages", {})) != {"sft", "reward"}):
        raise ValueError("Independent acceptance of both full-real formal parents is required")
    if (accepted["parent_plan"]["sha256"] != request["parent_plan_sha256"]
            or Path(accepted["parent_plan"]["path"]).resolve() != Path(request["parent_plan_path"]).resolve()):
        raise ValueError("Accepted common parent plan differs")
    parent_stage._pin(accepted["parent_plan"]["path"], accepted["parent_plan"]["sha256"])
    campaign_record = accepted["campaign"]
    parent_stage._pin(campaign_record["path"], campaign_record["sha256"])
    campaign = read_json(campaign_record["path"])
    if (campaign_record["status"] != "parents_passed" or campaign.get("status") != "parents_passed"
            or campaign.get("parents_verified") is not True or campaign.get("formal_training") is not True
            or campaign.get("research_complete") is not False
            or campaign.get("plan_sha256") != request["parent_plan_sha256"]):
        raise ValueError("Accepted parent campaign is not complete")
    campaign_stages = campaign.get("stages", [])
    if len(campaign_stages) != 2 or {s.get("stage") for s in campaign_stages} != {"sft", "reward"}:
        raise ValueError("Parent campaign lacks exactly both accepted stages")
    manifests = {}
    for name, record in accepted["stages"].items():
        rows, updates = parent_stage.EXPECTED[name][:2]
        if record.get("rows") != rows or record.get("optimizer_updates") != updates:
            raise ValueError("Accepted parent workload differs")
        parent_stage._pin(record["progress_path"], record["progress_sha256"])
        progress = read_json(record["progress_path"])
        if (progress.get("status") != "formal_parent_stage_passed" or progress.get("stage") != name
                or progress.get("formal_training") is not True or progress.get("research_complete") is not False
                or progress.get("actual_optimizer_updates") != updates
                or progress.get("artifact", {}).get("manifest_sha256") != record["manifest_sha256"]
                or Path(progress["artifact"]["artifact_dir"]).resolve() != Path(record["artifact_dir"]).resolve()):
            raise ValueError("Accepted parent progress/artifact changed")
        entry = next(s for s in campaign_stages if s["stage"] == name)
        if (entry.get("status") != "accepted" or entry.get("progress_sha256") != record["progress_sha256"]
                or entry.get("artifact", {}).get("manifest_sha256") != record["manifest_sha256"]):
            raise ValueError("Parent campaign acceptance record differs")
        if name in request["parents"]:
            requested = request["parents"][name]
            if (requested["manifest_sha256"] != record["manifest_sha256"]
                    or Path(requested["artifact_dir"]).resolve() != Path(record["artifact_dir"]).resolve()):
                raise ValueError("Requested parent is not the independently accepted parent")
        manifests[name] = _formal_parent(name, record, request)
    if manifests["reward"]["identity"]["parents"]["sft"]["sha256"] != accepted["stages"]["sft"]["manifest_sha256"]:
        raise ValueError("RM must descend from the same formal SFT parent")
    return accepted


def validate_request(request):
    common = {"schema_version", "stage", "output", "deadline_epoch", "parent_plan_path",
        "parent_plan_sha256", "source_sha256", "model_snapshot", "model_manifest_sha256",
        "tokenizer_dir", "audit_dir", "audit_manifest_sha256", "capacity_acceptance_path",
        "capacity_acceptance_sha256", "parents", "seed"}
    extra = {"kind", "alignment_plan_path", "alignment_plan_sha256", "parent_acceptance_path",
             "parent_acceptance_sha256", "workload", "reference_cache", "process_wait_policy",
             "allowed_background_processes", "supervisor_processes", "process_wait_spent_seconds", "continuation"}
    if type(request) is not dict or set(request) != common | extra or request["kind"] != KIND:
        raise ValueError("Exact formal alignment request schema required")
    from .alignment_process_guard_r2 import validate_policy, validate_background
    validate_policy(request["process_wait_policy"])
    validate_background(request["allowed_background_processes"])
    validate_background(request["supervisor_processes"])
    if (not request["supervisor_processes"] or type(request["process_wait_spent_seconds"]) not in (int, float)
            or not math.isfinite(request["process_wait_spent_seconds"])
            or not 0 <= request["process_wait_spent_seconds"] < request["process_wait_policy"]["max_total_wait_seconds"]):
        raise ValueError("Active supervisor and inherited campaign wait budget required")
    stage = request["stage"]
    validate_workload(stage, request["workload"])
    if (type(request["seed"]) is not int or request["seed"] not in (42, 43, 44)
            or type(request["parents"]) is not dict
            or set(request["parents"]) != ({"sft"} if stage == "dpo" else {"sft", "reward"})
            or any(type(p) is not dict or set(p) != {"artifact_dir", "manifest_sha256"}
                   for p in request["parents"].values())):
        raise ValueError("Exact main-experiment seed/parent lineage required")
    # Reuse frozen metadata/source/admission-independent validation. Never call
    # the parent engine or pretend that these branches are SFT stages.
    parent_request = {k: request[k] for k in common}
    parent_request.update(kind=parent_stage.KIND, stage="sft", seed=42, parents={},
        expected_rows=parent_stage.EXPECTED["sft"][0], expected_updates=parent_stage.EXPECTED["sft"][1])
    output = parent_stage.validate_request(parent_request)
    if not REQUIRED_SOURCES <= set(request["source_sha256"]):
        raise ValueError("Explicit complete alignment execution source inventory required")
    parent_stage._pin(request["alignment_plan_path"], request["alignment_plan_sha256"])
    admitted_plan = read_json(request["alignment_plan_path"])
    if any(admitted_plan.get(k) != request[k] for k in ("process_wait_policy", "allowed_background_processes")):
        raise ValueError("Child cannot change the frozen campaign wait policy or vetted identities")
    if admitted_plan.get("kind") != "alignlab-formal-alignment-campaign-r2-v1":
        raise ValueError("Explicit r2 continuation campaign required")
    if request["continuation"] != admitted_plan.get("continuation"):
        raise ValueError("Child continuation must equal the frozen authorized campaign")
    from .alignment_supervisor_r2 import validate_continuation
    validate_continuation(admitted_plan)
    if request["deadline_epoch"] != request["continuation"]["deadline_epoch"]:
        raise ValueError("Child cannot reset the inherited absolute deadline")
    validate_parent_acceptance(request)
    cache = request["reference_cache"]
    if stage == "ppo":
        if cache is not None:
            raise ValueError("PPO uses online frozen-reference scoring, not the offline DPO cache")
    elif type(cache) is not dict or cache.get("mode") not in {"create", "reuse"}:
        raise ValueError("DPO must explicitly create or pin/reuse its reference cache")
    elif cache["mode"] == "create":
        if set(cache) != {"mode", "path"} or Path(cache["path"]).resolve() != output / "reference-cache":
            raise ValueError("Create the DPO cache only in this fresh branch output")
    else:
        if set(cache) != {"mode", "path", "manifest_sha256", "identity_sha256"}:
            raise ValueError("Reuse requires externally pinned cache manifest and identity file SHAs")
        parent_stage._pin(Path(cache["path"]) / "manifest.json", cache["manifest_sha256"])
        parent_stage._pin(Path(cache["path"]) / "identity.json", cache["identity_sha256"])
    return output


def select_real_rows(partition, request):
    from .training_data import TrainingPartition, TrainingRow
    stage, work = request["stage"], request["workload"]
    pool = DPO_ROWS if stage == "dpo" else PPO_POOL_ROWS
    part = "preference_train" if stage == "dpo" else "ppo_prompts"
    if (not isinstance(partition, TrainingPartition) or partition.partition != part
            or partition.manifest_sha256 != request["audit_manifest_sha256"] or len(partition.rows) != pool):
        raise ValueError("Exact full accepted real training pool required")
    seen, prompts = set(), set()
    for row in partition.rows:
        if (not isinstance(row, TrainingRow) or any(type(v) is not str or not v for v in
                (row.record_id, row.prompt_id, row.cluster_id))
                or row.record_id in seen or row.record_id.startswith("synthetic-")):
            raise ValueError("Duplicate/synthetic/malformed training identity")
        seen.add(row.record_id)
        sequences = [row.prompt_ids] + ([] if stage == "ppo" else [row.chosen_ids, row.rejected_ids])
        if any(not seq or any(type(t) is not int or not 0 <= t < VOCAB_SIZE for t in seq) for seq in sequences):
            raise ValueError("Exact nonempty vocabulary token IDs required")
        if len(row.prompt_ids) > 1024:
            raise ValueError("No truncation of accepted prompt content")
        if stage == "dpo":
            if any(len(ids) > 768 or ids[-1] != EOS or EOS in ids[:-1] for ids in sequences[1:]):
                raise ValueError("Full preference answers must retain one terminal EOS")
        elif row.prompt_id in prompts or row.chosen_ids is not None or row.rejected_ids is not None:
            raise ValueError("PPO pool requires unique prompt IDs without offline answer labels")
        prompts.add(row.prompt_id)
    selected = partition.rows if stage == "dpo" else tuple(sorted(partition.rows,
        key=lambda row: (hashlib.sha256(f"{work['selection_seed']}:{row.prompt_id}".encode()).hexdigest(),
                         row.prompt_id))[:work["rows"]])
    records = [{"record_id": r.record_id, "prompt_id": r.prompt_id, "cluster_id": r.cluster_id,
        "prompt_tokens": len(r.prompt_ids), "chosen_tokens": None if r.chosen_ids is None else len(r.chosen_ids),
        "rejected_tokens": None if r.rejected_ids is None else len(r.rejected_ids),
        "token_sha256": digest([r.prompt_ids, r.chosen_ids, r.rejected_ids])} for r in selected]
    summary = {"scope": work["scope"], "partition": part, "accepted_partition_rows": pool,
        "selected_rows": len(selected), "real_rows": len(selected), "synthetic_rows": 0,
        "unique_prompt_ids": len({r.prompt_id for r in selected}),
        "independent_prompt_clusters": len({r.cluster_id for r in selected}),
        "accepted_manifest_sha256": partition.manifest_sha256, "accepted_identity_sha256": partition.identity_sha256,
        "serialization_version": partition.serialization_version, "records_sha256": digest(records),
        "selection_algorithm": "all_accepted_rows" if stage == "dpo" else SELECTION_RULE,
        "selection_seed": None if stage == "dpo" else work["selection_seed"]}
    return tuple(selected), summary, records


def stage_config(request):
    from .stage_engine import SupervisedConfig, PPOConfig
    work, deadline, seed = request["workload"], request["deadline_epoch"], request["seed"]
    common = dict(seed=seed, deadline_epoch=deadline, pad_id=PAD, eos_id=EOS,
                  max_updates=work["optimizer_updates"], chunk_tokens=CHUNK_TOKENS)
    if request["stage"] == "dpo":
        return SupervisedConfig(kind="dpo", effective_batch=EFFECTIVE_BATCH, microbatch=1, beta=work["beta"], **common)
    return PPOConfig(rollout_batch=ROLLOUT_BATCH, optimizer_batch=1, epochs=2,
        max_responses=work["rows"], max_new_tokens=MAX_NEW_TOKENS, kl_coef=work["kl_coef"], **common)


def scoring_config(receipt, serialization_version):
    return {"template": serialization_version,
        "serialization": "exact_accepted_token_ids_including_terminal_eos_raw_logprob_sums",
        "precision": receipt.precision, "quantization": receipt.quantization,
        "library_versions": receipt.anchor["metadata"]["library_versions"]}


def expected_cache_sequences(receipt, identity, pairs, *, check_boundary=lambda: None):
    from . import gpu_reference as gpu
    unique = {}
    for chosen, rejected in pairs:
        check_boundary()
        for batch in gpu._pairs(receipt, chosen, rejected):
            for prompt, completion in gpu._sequences(batch):
                unique[gpu.cache_key(identity, prompt, completion)] = (prompt, completion)
    if not unique:
        raise ValueError("A reference cache requires nonempty paired inputs")
    return dict(sorted(unique.items()))


def create_bounded_cache(receipt, output, pairs, *, config, deadline_epoch,
                         stop_requested=lambda: False, progress=lambda **_: None, clock=time.time,
                         record_boundary=lambda **_: None):
    """Same scoring primitives/protocol as gpu_reference, with per-sequence stops."""
    import torch
    from . import gpu_reference as gpu
    if not isinstance(receipt, gpu.FrozenReferenceReceipt):
        raise TypeError("An actual captured frozen reference receipt is required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Existing or partial cache must never be overwritten or implicitly reused")
    check = lambda: boundary(deadline_epoch, stop_requested, clock=clock)
    check()
    if any(p.grad is not None for p in receipt._owner._all_parameters()):
        raise RuntimeError("Reference scoring requires an idle backend with cleared gradients")
    receipt.verify(full=True)
    identity = gpu._cache_identity(receipt, config, CHUNK_TOKENS)
    unique = expected_cache_sequences(receipt, identity, pairs, check_boundary=check)
    autocast = receipt.anchor["metadata"]["autocast"]
    if torch.is_autocast_enabled(receipt.device.type) and (
            not autocast["enabled"] or torch.get_autocast_dtype(receipt.device.type) != gpu._DTYPES[receipt.precision]):
        raise ValueError("Ambient autocast differs from the actual frozen scoring protocol")
    records = []
    progress(scored_sequences=0, expected_unique_sequences=len(unique), complete=False)
    record_boundary(scored_sequences=0, expected_unique_sequences=len(unique), record=None)
    for key, (prompt, completion) in unique.items():
        check()
        receipt.verify(full=False)
        ids = torch.tensor([prompt + completion], dtype=torch.long, device=receipt.device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
        with torch.autocast(receipt.device.type, dtype=gpu._DTYPES[receipt.precision], enabled=autocast["enabled"]):
            scores, counts = receipt._owner.read(receipt.reference_role, lambda view:
                gpu._sequence_scores(view, batch, CHUNK_TOKENS, differentiable=False))
        records.append({"key": key, "prompt_token_ids": prompt, "completion_token_ids": completion,
                        "sum_logprob": float(scores[0].item()), "token_count": int(counts[0].item())})
        progress(scored_sequences=len(records), expected_unique_sequences=len(unique), complete=False)
        record_boundary(scored_sequences=len(records), expected_unique_sequences=len(unique), record=records[-1])
    check()
    receipt.verify(full=True)
    check()
    cache = gpu.write_or_reuse_cache(output, identity, records)
    if cache.reused:
        raise FileExistsError("Cache appeared during scoring; explicit future pinned reuse required")
    binding = gpu.bind_reference_cache(receipt, output, expected_identity=identity,
                                      expected_manifest_sha256=cache.manifest_sha256)
    check()
    progress(scored_sequences=len(records), expected_unique_sequences=len(unique), complete=True)
    return binding


def bind_complete_cache(receipt, cache, pairs, *, config, deadline_epoch, stop_requested=lambda: False):
    from . import gpu_reference as gpu
    check = lambda: boundary(deadline_epoch, stop_requested)
    check()
    path = Path(cache["path"])
    parent_stage._pin(path / "manifest.json", cache["manifest_sha256"])
    parent_stage._pin(path / "identity.json", cache["identity_sha256"])
    identity = gpu._cache_identity(receipt, config, CHUNK_TOKENS)
    # The expected identity is recomputed from actual loaded tensors/files;
    # never accept cache-provided template/precision/source declarations.
    if read_json(path / "identity.json") != identity:
        raise ValueError("Reused cache differs from the actual frozen scoring protocol/identity")
    binding = gpu.bind_reference_cache(receipt, path, expected_identity=identity,
                                      expected_manifest_sha256=cache["manifest_sha256"])
    expected = expected_cache_sequences(receipt, identity, pairs, check_boundary=check)
    if set(binding._cache.entries) != set(expected):
        raise ValueError("Reused cache must contain exactly the full selected preference sequence keys")
    parent_stage._pin(path / "identity.json", cache["identity_sha256"])
    check()
    return binding


class BoundaryCheckpoints:
    """Same optimizer count may have different newly collected PPO rollout state."""
    def __init__(self, final_update):
        self.schedule = parent_stage.CheckpointSchedule(final_update)
        self.attempted, self.records = set(), {}
        self.warm_costs = set()

    def save(self, cursor, elapsed, *, writer, safe_return=False, warm_elapsed=None, clock=time.time):
        update = cursor["optimizer_updates"]
        reasons = self.schedule.reasons(update, elapsed, safe_stop=safe_return)
        if warm_elapsed is not None:
            reasons += [f"warm_cost_{hour}h" for hour in self.schedule.hours
                        if hour not in self.warm_costs and warm_elapsed >= hour * 3600]
        if not reasons:
            return None
        observation = {"observed_at_epoch": clock(), "stage_elapsed_seconds": elapsed,
                       "warm_training_elapsed_seconds": warm_elapsed, "reasons": list(reasons),
                       "thresholds": [{"axis": axis, "target_seconds": hour * 3600,
                           "actual_seconds": value, "overshoot_seconds": value - hour * 3600}
                           for axis, prefix, value in (("stage_wall", "cost_", elapsed),
                               ("warm_training_wall", "warm_cost_", warm_elapsed))
                           if value is not None for hour in self.schedule.hours
                           if f"{prefix}{hour}h" in reasons]}
        self.warm_costs.update(hour for hour in self.schedule.hours if f"warm_cost_{hour}h" in reasons)
        key = digest(cursor)
        if key in self.records:
            self.records[key]["reasons"] = sorted(set(self.records[key]["reasons"]) | set(reasons))
            self.records[key]["cost_observations"].append({**observation, "reused_same_cursor": True})
            self.schedule._costs(reasons)
            return self.records[key]
        if key in self.attempted:
            raise RuntimeError("Cannot retry or overwrite a partial checkpoint attempt")
        self.attempted.add(key)
        before = clock()
        receipt = writer(f"update-{update:06d}-{key[:16]}")
        record = {k: receipt[k] for k in ("checkpoint_path", "manifest_sha256")}
        record.update({k: receipt[k] for k in ("telemetry_sidecar", "telemetry_binding") if k in receipt})
        record.update(optimizer_updates=update, cursor_sha256=key, cursor=cursor,
                      active_elapsed_seconds=elapsed, reasons=reasons,
                      save_started_at_epoch=before, save_finished_at_epoch=clock(),
                      cost_observations=[{**observation, "reused_same_cursor": False}])
        self.records[key] = record
        self.schedule._costs(reasons)
        return record


def verify_completion(outcome, rows, request):
    stage, work, cursor = request["stage"], request["workload"], outcome.cursor
    if (outcome.stop_reason != "data_exhausted" or cursor["offset"] != work["rows"]
            or cursor["optimizer_updates"] != work["optimizer_updates"] or cursor["active_rollout"]
            or outcome.extra_state is not None or len(outcome.metrics) != work["optimizer_updates"]):
        raise RuntimeError("Registered selected workload is not complete")
    if stage == "dpo":
        tokens = sum(len(row.chosen_ids) + len(row.rejected_ids) for row in rows)
        if cursor["response_tokens"] != tokens or cursor["completed_rollouts"] != 0:
            raise RuntimeError("DPO actual completion token/data counts differ")
    elif (cursor["completed_rollouts"] != math.ceil(work["rows"] / ROLLOUT_BATCH)
            or not work["rows"] <= cursor["response_tokens"] <= work["rows"] * MAX_NEW_TOKENS):
        raise RuntimeError("PPO selected prompts/rollouts/action counts differ")
    return {"stop_reason": "data_exhausted", "data_pass_complete": True, "protocol_complete": True,
        "full_selected_dataset_pass": True, "full_accepted_dataset_pass": stage == "dpo",
        "scope": work["scope"], "synthetic_rows": 0, "rows_processed": len(rows),
        "optimizer_updates": cursor["optimizer_updates"], "response_tokens": cursor["response_tokens"],
        "completed_rollouts": cursor["completed_rollouts"]}


def _identity_files(request, selection):
    snapshot, tokenizer, audit = (Path(request[k]) for k in ("model_snapshot", "tokenizer_dir", "audit_dir"))
    files = {f"source/{n}": ROOT / n for n in request["source_sha256"]}
    files.update({f"base/{n}": snapshot / n for n in read_json(snapshot / "snapshot-manifest.json")["files"]})
    files.update({f"tokenizer/{n}": tokenizer / n for n in read_json(tokenizer / "source-manifest.json")["artifacts"]})
    files.update({"tokenizer/source-manifest.json": tokenizer / "source-manifest.json",
        "data/manifest": audit / "manifest.json",
        "data/partition": audit / ("preference_train.jsonl" if request["stage"] == "dpo" else "ppo_prompts.jsonl"),
        "data/selection": selection, "protocol": ROOT / "docs/ALIGNLAB_V2.md"})
    for name in ("parent_plan", "alignment_plan", "parent_acceptance", "capacity_acceptance"):
        files[name] = Path(request[name + "_path"])
    continuation = request["continuation"]
    for name in ("prior_plan", "prior_campaign", "prior_stage_request", "prior_stage_progress", "authorization"):
        files["continuation/" + name] = Path(continuation[name]["path"])
    for name in ("journal", "boundary_receipt", "progress", "request"):
        files["continuation/cache_prefix/" + name] = Path(continuation["cache_prefix"][name]["path"])
    return files


def run_alignment_stage(request, *, execute=False):
    output = validate_request(request)
    if not execute:
        return {"status": "plan_only", "stage": request["stage"], "formal_training": False, "research_complete": False}
    if output.exists():
        raise FileExistsError("Fresh alignment output required; no automatic resume/retry")
    from .gpu_probe import resources
    stop = lambda: ((output / "STOP").exists() or (output.parent / "STOP").exists()
                    or Path(request["alignment_plan_path"] + ".stop").exists())
    stage, deadline = request["stage"], request["deadline_epoch"]
    boundary(deadline, stop)
    from .alignment_process_guard_r2 import strict_admission, child_identities, GuardSession
    from .probe_dispatch import _System
    system = _System()
    owned = child_identities(system.processes(), request)
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

    def progress(phase):
        status.update(status=phase, updated_at_epoch=time.time(), active_elapsed_seconds=time.time() - started)
        _write(output / "progress.json", status)
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
            for name, expected in request["source_sha256"].items():
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
            progress("waiting_process_boundary")
        elif event["event"] == "result":
            status["process_guard"].update(last=event, total_waited_seconds=event["total_waited_seconds"])

    guard = GuardSession(system.processes, owned_identities=owned, allowed_identities=allowed,
        deadline_epoch=deadline, policy=request["process_wait_policy"],
        spent_seconds=request["process_wait_spent_seconds"], stop_requested=stop,
        identity_unchanged=unchanged, on_event=guard_event)

    def require_guard(name, update, save):
        result = guard.check(boundary=name, optimizer_updates=update, save_boundary=save)
        if not result["admission_allowed"]:
            status["process_guard"]["blocked_reason"] = result["reason"]
            if result["reason"] == "deadline_exhausted":
                raise TimeoutError("Absolute alignment deadline reached at process boundary")
            raise InterruptedError("Strict process boundary blocked: " + result["reason"])
        return result

    def file_receipt(path, kind, update):
        return {"complete": True, "optimizer_updates": update, "kind": kind,
                "path": str(Path(path).resolve()), "sha256": sha(path)}

    try:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "request.json", request, exclusive=True)
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
        validate_request(request)
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
        backend = loaded.backend
        status["model"] = loaded.metadata
        gpu_sample()
        files, config = _identity_files(request, selection_path), stage_config(request)
        lr = 5e-6 if stage == "dpo" else 1e-6
        identity_config = {"stage": stage, "purpose": KIND, "formal_training": True, "research_complete": False,
            "execution_revision": EXECUTION_REVISION, "telemetry_protocol": TELEMETRY_PROTOCOL,
            "scope": request["workload"]["scope"], "synthetic_rows": 0, "work": asdict(config),
            "registered_workload": request["workload"], "model": loaded.metadata, "selection": selection,
            "continuation_revision": "alignment-r2", "continuation": request["continuation"],
            "optimizer": {"name": "AdamW", "lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            "process_guard_cache_record_interval": 16,
            "checkpoint_policy": {"first": 1, "interval_updates": 50, "cost_hours": [1, 2, 4, 8],
                "warm_training_cost_hours": [1, 2, 4, 8], "save_clean_return": True,
                "cost_axes": "stage_wall_since_admission_and_warm_wall_since_engine_entry",
                "same_cursor_labels": "merged_with_actual_observation_times_and_original_save_times"}}
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
                "source_files": {k: v for k, v in files.items() if k.startswith("source/")},
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
                def save_cache():
                    path = output / f"cache-safe-boundary-{values['scored_sequences']:06d}-{guard.calls:06d}.json"
                    _write(path, {"kind": "alignlab-safe-cache-record-boundary-v1", "complete": True,
                        "optimizer_updates": 0, "cache_journal": str(cache_journal),
                        "committed_record": row, "identity": import_identity}, exclusive=True)
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
                    from .reference_cache_recovery import create_bounded_cache_from_prefix
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

        validate_request(request)
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
                progress("requested_stop_boundary_saved")
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
        validate_request(request)
        boundary(deadline, stop)
        artifact = export_stage(output / "final-artifact", backend, loaded.role, stage=stage,
            identity=identity, heads=loaded.heads, completion=status["completion"])
        status["artifact"] = {k: artifact[k] for k in ("artifact_dir", "manifest_path", "manifest_sha256")}
        gpu_sample()
        audit_frozen(full=True, phase="after_export")
        validate_request(request)
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
        if output.exists():
            _write(output / "progress.json", status)
        if lock.exists() and read_json(lock) == owner:
            lock.unlink()
