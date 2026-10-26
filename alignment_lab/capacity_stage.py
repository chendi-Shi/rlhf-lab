"""One admitted CUDA capacity stage. No supervisor, automatic retry or resume.

The caller supplies one immutable request and the campaign's absolute deadline.
Synthetic shape stress and the short selected data pass are capacity evidence,
never a complete dataset training run or an independent quality evaluation.
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

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-full-capacity-stage-v1"
COUNTS = {"sft": 16, "reward": 16, "dpo": 320, "ppo": 128}
LEARNING_RATES = {"sft": 1e-4, "reward": 5e-5, "dpo": 5e-6, "ppo": 1e-6}
PARENTS = {"sft": set(), "reward": {"sft"}, "dpo": {"sft"},
           "ppo": {"sft", "reward"}, "eval": {"sft", "reward"}}
PARTITIONS = {"sft": "sft_train", "reward": "preference_train",
              "dpo": "preference_train", "ppo": "ppo_prompts", "eval": "ppo_prompts"}
PAD, EOS = 151643, 151645


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _write(path, value, *, exclusive=False):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary, path)


def validate_request(request):
    required = {"schema_version", "kind", "stage", "output", "deadline_epoch",
                "capacity_plan_path", "capacity_plan_sha256", "source_sha256",
                "model_snapshot", "model_manifest_sha256", "tokenizer_dir",
                "audit_dir", "audit_manifest_sha256", "parents"}
    if set(request) != required or request["schema_version"] != 1 or request["kind"] != KIND:
        raise ValueError("Exact versioned capacity-stage request required")
    stage = request["stage"]
    if stage not in PARENTS or set(request["parents"]) != PARENTS[stage]:
        raise ValueError("Stage requires its exact SFT/reward lineage")
    deadline = request["deadline_epoch"]
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("A shared finite absolute deadline is required")
    output = Path(request["output"]).resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        raise ValueError("Capacity stage needs a dedicated project runs directory")
    required_sources = {p.relative_to(ROOT).as_posix() for p in (ROOT / "alignment_lab").glob("*.py")}
    required_sources |= {"scripts/run_alignlab_capacity_stage.py", "rlhf_lab/runtime.py"}
    sources = request["source_sha256"]
    if not isinstance(sources, dict) or not required_sources <= set(sources):
        raise ValueError("Complete stage implementation inventory required")
    for name, expected in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.is_symlink() or sha(path) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for path, expected in ((request["capacity_plan_path"], request["capacity_plan_sha256"]),
                           (Path(request["audit_dir"]) / "manifest.json", request["audit_manifest_sha256"]),
                           (Path(request["model_snapshot"]) / "snapshot-manifest.json", request["model_manifest_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"Frozen input manifest changed: {path}")
    for name, parent in request["parents"].items():
        if set(parent) != {"artifact_dir", "manifest_sha256"}:
            raise ValueError("Parent must pin a completed artifact directory and manifest SHA")
        if sha(Path(parent["artifact_dir"]) / "manifest.json") != parent["manifest_sha256"]:
            raise ValueError(f"{name} parent manifest changed")
    return output


def boundary(deadline, stop_requested, *, clock=time.time):
    if stop_requested():
        raise InterruptedError("Capacity STOP requested at a cooperative boundary")
    if clock() >= deadline:
        raise TimeoutError("Shared capacity deadline exhausted")


def stage_config(stage, deadline):
    from .stage_engine import SupervisedConfig, PPOConfig
    if stage == "ppo":
        return PPOConfig(seed=42, rollout_batch=16, optimizer_batch=1, epochs=2,
            max_responses=128, max_updates=256, max_new_tokens=256,
            deadline_epoch=deadline, pad_id=PAD, eos_id=EOS, chunk_tokens=16)
    return SupervisedConfig(kind=stage, seed=42, effective_batch=16, microbatch=1,
        max_updates=COUNTS[stage] // 16, deadline_epoch=deadline,
        pad_id=PAD, eos_id=EOS, chunk_tokens=16)


def verify_outcome(stage, outcome, count):
    expected = 256 if stage == "ppo" else count // 16
    if (outcome.stop_reason != "data_exhausted" or outcome.cursor["offset"] != count
            or outcome.cursor["optimizer_updates"] != expected
            or outcome.cursor["active_rollout"] or outcome.extra_state is not None):
        raise RuntimeError(f"Capacity work incomplete: {outcome.stop_reason}, {outcome.cursor}")
    if stage == "ppo" and outcome.cursor["completed_rollouts"] != 8:
        raise RuntimeError("PPO requires all eight real rollout batches")
    if len(outcome.metrics) != expected:
        raise RuntimeError("Measured optimizer work differs from declared count")
    return {"stop_reason": "data_exhausted", "data_pass_complete": True,
            "protocol_complete": True, "optimizer_updates": expected,
            "rows_processed": count, "scope": "capacity_selection_data_pass",
            "full_accepted_dataset_pass": False, "quality_claim": False}


def compare_state(actual, expected, *, path="state", exact_float=False):
    """Compare replay parameters/optimizer numerically, all discrete state exactly."""
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"{path}: tensor metadata mismatch")
        if expected.is_floating_point() and not exact_float:
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-5, atol=2e-6, msg=path)
        elif not torch.equal(actual.cpu(), expected.cpu()):
            raise AssertionError(f"{path}: discrete tensor mismatch")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: mapping mismatch")
        for key in expected:
            compare_state(actual[key], expected[key], path=f"{path}.{key}", exact_float=exact_float)
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"{path}: sequence mismatch")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_state(a, b, path=f"{path}[{i}]", exact_float=exact_float)
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"{path}: value mismatch")


def _inventory(request):
    snapshot, tokenizer = Path(request["model_snapshot"]), Path(request["tokenizer_dir"])
    manifest = read_json(snapshot / "snapshot-manifest.json")
    token_manifest = read_json(tokenizer / "source-manifest.json")
    files = {f"source/{name}": ROOT / name for name in request["source_sha256"]}
    files.update({f"base/{name}": snapshot / name for name in manifest["files"]})
    files.update({f"tokenizer/{name}": tokenizer / name for name in token_manifest["artifacts"]})
    files.update({"tokenizer/source-manifest.json": tokenizer / "source-manifest.json",
                  "data/manifest": Path(request["audit_dir"]) / "manifest.json",
                  "data/partition": Path(request["audit_dir"]) / f"{PARTITIONS[request['stage']]}.jsonl",
                  "protocol": ROOT / "docs/ALIGNLAB_V2.md",
                  "capacity_plan": Path(request["capacity_plan_path"])})
    return files


def _import_parents(request, loaded, identity):
    from .stage_artifacts import import_stage
    stage, backend = request["stage"], loaded.backend
    receipts = {}
    destinations = [] if stage == "sft" else [("sft", loaded.role, ())]
    if stage in {"dpo", "ppo", "eval"}:
        destinations.append(("sft", "reference", ()))
    if stage in {"ppo", "eval"}:
        destinations.append(("reward", "reward", ("reward",)))
    for parent, role, heads in destinations:
        source = request["parents"][parent]
        receipts[role] = import_stage(source["artifact_dir"], backend, role,
            expected_manifest_sha256=source["manifest_sha256"], expected_identity=identity, heads=heads)
    return receipts


def _reference_files(request, files, parent):
    artifact = Path(request["parents"][parent]["artifact_dir"])
    return {"base_files": {k: v for k, v in files.items() if k.startswith("base/")},
            "tokenizer_files": {k: v for k, v in files.items() if k.startswith("tokenizer/")},
            "source_files": {k: v for k, v in files.items() if k.startswith("source/")},
            "adapter_files": {"manifest": artifact / "manifest.json", "weights": artifact / "weights.pt"},
            "dataset_manifest": Path(request["audit_dir"]) / "manifest.json"}


def _head_anchor(head):
    import torch
    result = {}
    for name, tensor in head.state_dict().items():
        cpu = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(cpu).all()):
            raise FloatingPointError("Nonfinite frozen reward head")
        result[name] = {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
                        "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def _parameter_hashes(parameters):
    import torch
    return {name: hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, parameter in parameters.items()}


def ppo_shape_stress(backend, role, optimizer, rows, *, device, special_token_ids,
                     prompt_length=1024, action_length=256, chunk_tokens=16):
    """Synthetic fixed-shape read/backward only; zero optimizer steps."""
    import torch
    from .ppo_objective import clipped_ppo_loss
    from .stage_engine import parameter_mapping
    mapping = parameter_mapping(backend, role, heads=("value",))
    before = _parameter_hashes(mapping)
    specials = set(special_token_ids)
    tokens = sorted({t for row in rows for t in (row["prompt_ids"] if isinstance(row, dict) else row.prompt_ids)
                     if t not in specials})
    if not tokens:
        raise ValueError("PPO shape fixture requires real non-special training token IDs")
    prompt = tuple(tokens[i % len(tokens)] for i in range(prompt_length))
    actions = tuple(tokens[(i + 1) % len(tokens)] for i in range(action_length))
    ids = torch.tensor([prompt + actions], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :prompt_length] = -100
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    scores, values = backend.read(role, lambda view: view.token_scores_and_values(**batch, chunk_tokens=chunk_tokens))
    start = prompt_length - 1
    old_lp, old_values = scores.logprobs[:, start:].clone(), values[:, start:].clone()
    mask, advantages, returns = torch.ones_like(old_lp, dtype=torch.bool), torch.ones_like(old_lp), old_values + 1

    def loss(view):
        new_scores, new_values = view.token_scores_and_values(**batch, chunk_tokens=chunk_tokens)
        return clipped_ppo_loss(new_scores.logprobs[:, start:], new_values[:, start:],
            old_lp, old_values, advantages, returns, mask, 0.2, 0.2, 0.5, action_length).objective_contribution

    objective = backend.backward(role, optimizer, loss, heads=("value",))
    gradients = [p.grad for p in mapping.values() if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("PPO maximum-shape backward must produce finite gradients")
    optimizer.zero_grad(set_to_none=True)
    if _parameter_hashes(mapping) != before or any(p.grad is not None for p in backend._all_parameters()):
        raise RuntimeError("No-step synthetic pressure changed parameters or retained gradients")
    return {"capacity_only": True, "synthetic": True, "optimizer_updates": 0,
            "counted_as_rollout": False, "prompt_tokens": prompt_length, "action_tokens": action_length,
            "attention_all_true": True, "objective": float(objective),
            "trainable_parameter_sha256": before, "parameters_unchanged": True}


def verify_generation(result, *, forced):
    if (result.get("quality_evaluation") is not False or result.get("capacity_only") is not True
            or result.get("termination") not in ({"forced_max_new_tokens"} if forced else {"eos", "max_new_tokens"})
            or result.get("final_action_flushed_to_kv") is not True):
        raise RuntimeError("Capacity generation interrupted or lacks its completed KV evidence")
    if forced and (result.get("generated_tokens") != 2048 or result.get("prompt_tokens") != 1024
                   or result.get("max_cache_tokens") != 3072 or result.get("ignore_eos") is not True):
        raise RuntimeError("Longest generation must actually execute 1024 + 2048 tokens")


def _selection(request):
    from transformers import AutoTokenizer
    from .training_data import load_training_partition
    from .capacity_data import select_capacity_rows
    partition = load_training_partition(request["audit_dir"], PARTITIONS[request["stage"]],
                                        request["audit_manifest_sha256"])
    tokenizer = AutoTokenizer.from_pretrained(request["tokenizer_dir"], local_files_only=True,
                                               trust_remote_code=False)
    selected = select_capacity_rows(partition, stage="ppo" if request["stage"] == "eval" else request["stage"],
        eos_id=EOS, special_token_ids=tuple(tokenizer.all_special_ids), vocab_size=151936)
    return selected, tokenizer


def run_capacity_stage(request, *, execute=False):
    output = validate_request(request)
    if not execute:
        return {"status": "plan_only", "stage": request["stage"], "output": str(output),
                "capacity_verified": False, "formal_training": False}
    if output.exists():
        raise FileExistsError("Capacity stage output exists; no overwrite or implicit resume")
    from .gpu_probe import resources
    from rlhf_lab.runtime import gpu_inventory
    lock = ROOT / "reports/.alignlab-gpu.lock"
    owner = {"pid": os.getpid(), "output": str(output), "token": uuid.uuid4().hex}
    stop = lambda: (output / "STOP").exists() or (output.parent / "STOP").exists()
    deadline, started, stage = request["deadline_epoch"], time.time(), request["stage"]
    boundary(deadline, stop)
    admission = resources()
    if admission["blockers"]:
        raise RuntimeError(f"CUDA admission blocked: {admission['blockers']}")
    _write(lock, owner, exclusive=True)
    status = {"kind": KIND, "status": "starting", "stage": stage, "request": request,
              "pid": os.getpid(), "started_at_epoch": started, "deadline_epoch": deadline,
              "capacity_verified": False, "formal_training": False, "training_started": False,
              "actual_optimizer_updates": 0, "recovery_optimizer_updates": 0,
              "resources_before": admission, "checkpoints": [], "gpu_samples": []}

    def progress(phase, **values):
        status.update(status=phase, updated_at_epoch=time.time(), **values)
        _write(output / "progress.json", status)
        print(json.dumps({"phase": phase, "stage": stage, "optimizer_updates": status["actual_optimizer_updates"]}), flush=True)

    def metric(value):
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_epoch": time.time(), **value}, allow_nan=False) + "\n")

    def gpu_sample():
        import torch
        torch.cuda.synchronize(0)
        value = {"time_epoch": time.time(), **gpu_inventory(),
                 "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                 "max_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        status["gpu_samples"].append(value)
        return value

    try:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "request.json", request, exclusive=True)
        progress("loading_capacity_selection")
        selection, tokenizer = _selection(request)
        rows = selection.rows
        status["selection"] = selection.metadata
        status["selected_record_ids"] = [r.record_id if hasattr(r, "record_id") else r["record_id"] for r in rows]
        if len(rows) != COUNTS["ppo" if stage == "eval" else stage]:
            raise ValueError("Capacity selection row count differs from registered work")
        boundary(deadline, stop)
        validate_request(request)
        immediate = resources()
        if immediate["blockers"]:
            raise RuntimeError(f"Resources changed before CUDA load: {immediate['blockers']}")
        import torch
        from .model_loader import ModelLoadConfig, load_quantized_stage, QUANTIZATION
        from .stage_engine import parameter_mapping, run_supervised, run_ppo, _batch
        from .checkpointing import build_identity, save_checkpoint, load_checkpoint
        from .stage_artifacts import export_stage
        from .gpu_reference import capture_frozen_reference, create_reference_cache
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        progress("loading_nf4")
        loaded = load_quantized_stage(request["model_snapshot"], request["model_manifest_sha256"],
            request["tokenizer_dir"], ModelLoadConfig("ppo" if stage == "eval" else stage))
        backend = loaded.backend
        status["model"] = loaded.metadata
        gpu_sample()
        files = _inventory(request)
        config = None if stage == "eval" else stage_config(stage, deadline)
        identity = build_identity(files, {"stage": "ppo" if stage == "eval" else stage,
            "purpose": KIND, "reported_stage": stage, "deadline_epoch": deadline,
            "work": None if config is None else asdict(config), "model": loaded.metadata,
            "optimizer": None if stage == "eval" else {"name": "AdamW", "lr": LEARNING_RATES[stage],
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            "selection": selection.metadata},
            {name: Path(parent["artifact_dir"]) / "manifest.json" for name, parent in request["parents"].items()})
        status["identity"] = identity
        status["imports"] = _import_parents(request, loaded, identity)
        boundary(deadline, stop)
        receipts, binding, head_anchor = {}, None, None
        if stage in {"dpo", "ppo", "eval"}:
            for role, parent in (("reference", "sft"),) + ((("reward", "reward"),) if stage in {"ppo", "eval"} else ()):
                boundary(deadline, stop)
                progress(f"capturing_{role}_receipt")
                receipts[role] = capture_frozen_reference(backend, role,
                    artifact_files=_reference_files(request, files, parent),
                    accepted_data_manifest_sha256=request["audit_manifest_sha256"],
                    precision="bfloat16", quantization=QUANTIZATION)
                status[f"{role}_anchor"] = receipts[role].anchor
            if stage in {"ppo", "eval"}:
                head_anchor = _head_anchor(backend._heads["reward"])
                status["reward_head_anchor"] = head_anchor

        def audit_frozen(*, full):
            for receipt in receipts.values():
                receipt.verify(full=full)
            if binding is not None:
                binding.verify(full=full)
            if head_anchor is not None and _head_anchor(backend._heads["reward"]) != head_anchor:
                raise RuntimeError("Frozen reward head changed")
            if full:
                status.setdefault("frozen_full_audits", []).append({"time_epoch": time.time(),
                    "roles": sorted(receipts), "cache_manifest_sha256": None if binding is None else binding.manifest_sha256,
                    "reward_head_content_verified": head_anchor is not None})

        if stage == "eval":
            from .capacity_generation import greedy_capacity_probe
            # Shape tokens derive only from accepted training prompt content.
            specials = set(tokenizer.all_special_ids)
            ordinary = sorted({t for row in rows for t in row.prompt_ids if t not in specials})
            if not ordinary:
                raise ValueError("No non-special training tokens for capacity shape")
            prompt = tuple(ordinary[i % len(ordinary)] for i in range(1024))
            boundary(deadline, stop)
            with loaded.autocast():
                ordinary_result = greedy_capacity_probe(backend, "policy", rows[0].prompt_ids,
                    eos_id=EOS, max_new_tokens=2048, device=loaded.device, ignore_eos=False,
                    stop_requested=stop, deadline_epoch=deadline)
                boundary(deadline, stop)
                verify_generation(ordinary_result, forced=False)
                result = greedy_capacity_probe(backend, "policy", prompt, eos_id=EOS,
                    max_new_tokens=2048, device=loaded.device, ignore_eos=True,
                    stop_requested=stop, deadline_epoch=deadline)
            boundary(deadline, stop)
            verify_generation(result, forced=True)
            status["generation"] = {"eos_aware": ordinary_result, "forced_shape": result}
            gpu_sample()
        else:
            mapping = parameter_mapping(backend, loaded.role, heads=loaded.heads)
            optimizer = torch.optim.AdamW(mapping.values(), lr=LEARNING_RATES[stage])
            generator = torch.Generator(device=loaded.device).manual_seed(42)
            initial = {name: p.detach().cpu().clone() for name, p in mapping.items()}
            if stage == "ppo":
                boundary(deadline, stop)
                progress("ppo_synthetic_maximum_backward")
                with loaded.autocast():
                    status["synthetic_backward"] = ppo_shape_stress(backend, loaded.role, optimizer, rows,
                        device=loaded.device, special_token_ids=tokenizer.all_special_ids)
                gpu_sample()
                boundary(deadline, stop)
            if stage == "dpo":
                progress("building_actual_reference_cache")

                def pairs():
                    for row in rows:
                        boundary(deadline, stop)
                        values = asdict(row) if hasattr(row, "__dataclass_fields__") else row
                        yield (_batch([values], "chosen_ids", PAD, loaded.device),
                               _batch([values], "rejected_ids", PAD, loaded.device))

                reference = receipts["reference"]
                with loaded.autocast():
                    binding = create_reference_cache(reference, output / "reference-cache", pairs(),
                        scoring_config={"template": selection.metadata.get("serialization_version", "accepted_audit_codec"),
                            "serialization": "exact_accepted_token_ids_including_terminal_eos_raw_logprob_sums",
                            "precision": "bfloat16", "quantization": QUANTIZATION,
                            "library_versions": reference.anchor["metadata"]["library_versions"]}, chunk_tokens=16)
                status["reference_cache_manifest_sha256"] = binding.manifest_sha256
                gpu_sample()
            audit_frozen(full=True)
            boundary(deadline, stop)
            saved = {}

            def checkpoint(cursor, extra):
                update = cursor["optimizer_updates"]
                status.update(actual_optimizer_updates=update, training_started=update > 0, cursor=cursor)
                audit_frozen(full=False)
                # Full immutable checkpoints are expensive: save the required
                # update boundaries, not hundreds of redundant base-file hashes.
                final_update = 256 if stage == "ppo" else COUNTS[stage] // 16
                if update in ({1, 2, final_update} if stage == "ppo" else {final_update}):
                    receipt = save_checkpoint(output / f"update-{update:06d}", identity=identity,
                        parameters=mapping, optimizer=optimizer, cursor=cursor,
                        generators={"sampling": generator}, extra_state=extra)
                    saved[update] = receipt
                    status["checkpoints"].append({k: receipt[k] for k in ("checkpoint_path", "manifest_sha256")})
                metric({"event": "optimizer_boundary", "cursor": cursor, "gpu": gpu_sample()})
                progress("training")

            progress("training")
            with loaded.autocast():
                if stage == "ppo":
                    outcome = run_ppo(backend, loaded.role, optimizer, rows, config, device=loaded.device,
                        generator=generator, reference_role="reference", reward_role="reward",
                        checkpoint_callback=checkpoint, stop_requested=stop)
                else:
                    outcome = run_supervised(backend, loaded.role, optimizer, rows, config, device=loaded.device,
                        reference_provider=None if binding is None else binding.pair_scores,
                        checkpoint_callback=checkpoint, stop_requested=stop)
            for value in outcome.metrics:
                metric({"event": "objective", **value})
            status["outcome_stop_reason"] = outcome.stop_reason
            completion = verify_outcome(stage, outcome, len(rows))
            boundary(deadline, stop)
            changed = [name for name, p in mapping.items() if not torch.equal(initial[name], p.detach().cpu())]
            status["changed_parameter_names"] = changed
            status["adapter_weights_changed"] = any(".lora_" in name for name in changed)
            status["scalar_heads_changed"] = {head: any(name.startswith(f"scalar_heads.{head}.") for name in changed)
                                             for head in loaded.heads}
            if not status["adapter_weights_changed"] or not all(status["scalar_heads_changed"].values()):
                raise RuntimeError("Stage did not change both its LoRA adapter and every owned scalar head")
            audit_frozen(full=True)
            validate_request(request)
            boundary(deadline, stop)
            artifact = export_stage(output / "final-artifact", backend, loaded.role, stage=stage,
                identity=identity, heads=loaded.heads, completion=completion)
            status["artifact"] = {k: artifact[k] for k in ("artifact_dir", "manifest_path", "manifest_sha256")}
            if stage == "ppo":
                boundary(deadline, stop)
                progress("verifying_ppo_midrollout_recovery")
                audit_frozen(full=True)
                restored = load_checkpoint(saved[1]["checkpoint_path"], expected_identity=identity,
                    parameters=mapping, optimizer=optimizer, generators={"sampling": generator},
                    expected_manifest_sha256=saved[1]["manifest_sha256"])
                if not restored["cursor"]["active_rollout"] or restored["extra_state"] is None:
                    raise RuntimeError("First PPO checkpoint must retain the active rollout and old statistics")
                replay_records = []
                with loaded.autocast():
                    replay = run_ppo(backend, loaded.role, optimizer, rows, config, device=loaded.device,
                        generator=generator, reference_role="reference", reward_role="reward",
                        state=restored["cursor"], extra_state=restored["extra_state"],
                        checkpoint_callback=lambda c, e: replay_records.append((c, e)),
                        stop_requested=lambda: stop() or bool(replay_records))
                status["recovery_optimizer_updates"] = len(replay.metrics)
                if len(replay.metrics) != 1 or len(replay_records) != 1 or replay.stop_reason != "paused":
                    raise RuntimeError("PPO recovery must replay exactly one next optimizer update")
                boundary(deadline, stop)
                replay_receipt = save_checkpoint(output / "replayed-update-000002", identity=identity,
                    parameters=mapping, optimizer=optimizer, cursor=replay.cursor,
                    generators={"sampling": generator}, extra_state=replay.extra_state)
                for receipt in (saved[2], replay_receipt):
                    directory = Path(receipt["checkpoint_path"])
                    if (sha(directory / "manifest.json") != receipt["manifest_sha256"]
                            or sha(directory / "state.pt") != receipt["artifacts"]["state.pt"]["sha256"]):
                        raise RuntimeError("Continuous/replayed checkpoint changed before comparison")
                expected = torch.load(Path(saved[2]["checkpoint_path"]) / "state.pt", map_location="cpu", weights_only=True)
                actual = torch.load(Path(replay_receipt["checkpoint_path"]) / "state.pt", map_location="cpu", weights_only=True)
                for key in ("parameters", "optimizer"):
                    compare_state(actual[key], expected[key], path=key)
                for key in ("cursor", "extra_state", "rng"):
                    compare_state(actual[key], expected[key], path=key, exact_float=True)
                status["recovery"] = {"verified": True, "continuous_checkpoint": saved[2]["manifest_sha256"],
                    "resume_checkpoint": saved[1]["manifest_sha256"],
                    "replay_checkpoint": replay_receipt["manifest_sha256"],
                    "replay_checkpoint_path": replay_receipt["checkpoint_path"],
                    "cursor_oldstats_rng_exact": True, "parameters_optimizer_rtol": 2e-5,
                    "parameters_optimizer_atol": 2e-6, "final_artifact_exported_before_replay": True}
                gpu_sample()
        audit_frozen(full=True)
        validate_request(request)
        boundary(deadline, stop)
        status["stage_capacity_passed"] = True
        status["scope"] = "Selected capacity workload only; no formal training or quality claim"
        progress("capacity_stage_passed")
        return status
    except BaseException as exc:
        status.update(status="budget_exhausted" if isinstance(exc, TimeoutError) else
                      "paused" if isinstance(exc, InterruptedError) else "failed",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                      stage_capacity_passed=False)
        raise
    finally:
        status.update(finished_at_epoch=time.time(), elapsed_seconds=time.time() - started)
        if output.exists():
            _write(output / "progress.json", status)
        if lock.exists() and read_json(lock) == owner:
            lock.unlink()
