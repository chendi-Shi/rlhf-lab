"""Explicit observed-v2 loops over the frozen stage engine's numeric primitives.

The old engine is not patched. Config classes, shuffle, sampling, objectives,
denominators and optimizer steps are the same. Events add CPU copies and callback
overhead; the runner owns CUDA synchronization and wall-clock accounting.

Callbacks receive recursively immutable plain values, never training tensors.
They must not mutate model parameters or consume training RNG. Version/RNG
guards reject such observable side effects. Full tensor-content audits remain
the runner's responsibility (Tensor.data writes do not bump version counters).
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import asdict
import hashlib
import math
import random
import time
from types import MappingProxyType

import torch

from . import stage_engine as old

SupervisedConfig = old.SupervisedConfig
PPOConfig = old.PPOConfig
StageOutcome = old.StageOutcome
parameter_mapping = old.parameter_mapping
EVENT_SCHEMA = 1
ORIGIN_FIELDS = {"schema_version", "rollout_index", "cursor_before", "record_ids",
                 "generation_model_versions", "model_versions", "rng_before_hex", "rng_after_hex",
                 "rng_before_sha256", "rng_after_sha256", "origin_sha256"}


def _immutable(value):
    if isinstance(value, torch.Tensor):
        return _immutable(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("Observation mappings require string keys")
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Observation must contain finite plain data")


def event_to_plain(value):
    """Independent JSON-compatible copy of a recursively immutable event."""
    if isinstance(value, Mapping):
        return {key: event_to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [event_to_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("Only immutable engine event values may be converted")


def _versions(backend):
    return tuple((id(p), p._version) for p in backend._all_parameters())


def _named_versions(backend):
    names = {id(p): name for name, p in backend._model.named_parameters()}
    for head_name, head in backend._heads.items():
        names.update({id(p): f"scalar_heads.{head_name}.{name}" for name, p in head.named_parameters()})
    return tuple((names[id(p)], p._version) for p in backend._all_parameters())


def _rng(generator):
    return bytes(generator.get_state().tolist())


def _emit(callback, backend, name, stage, cursor, *, generator=None, **payload):
    if callback is None:
        return
    value = _immutable({"schema_version": EVENT_SCHEMA, "event": name, "stage": stage,
                        "cursor": cursor, **payload})
    versions = _versions(backend)
    cpu_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    cuda_rng = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else None
    sampling_rng = None if generator is None else _rng(generator)
    callback(value)
    if (_versions(backend) != versions or not torch.equal(torch.get_rng_state(), cpu_rng)
            or random.getstate() != python_rng
            or cuda_rng is None and torch.cuda.is_initialized()
            or cuda_rng is not None and any(not torch.equal(a, b) for a, b in
                                            zip(cuda_rng, torch.cuda.get_rng_state_all()))
            or generator is not None and _rng(generator) != sampling_rng):
        raise RuntimeError("Observer changed model versions or training RNG; discard this run")


def _origin(cursor, subset, backend, generator):
    before = _rng(generator)
    return {"schema_version": 1, "rollout_index": cursor["completed_rollouts"],
            "cursor_before": copy.deepcopy(cursor), "record_ids": [r["record_id"] for r in subset],
            "generation_model_versions": _named_versions(backend), "model_versions": _versions(backend),
            "rng_before_hex": before.hex(), "rng_before_sha256": hashlib.sha256(before).hexdigest()}


def _complete_origin(origin, rollout):
    if (origin["model_versions"] != rollout.model_versions
            or origin["rng_before_hex"] != rollout.rng_before.hex()):
        raise RuntimeError("Sampling changed its original observation anchor")
    origin = copy.deepcopy(origin)
    origin.update(rng_after_hex=rollout.rng_after.hex(),
                  rng_after_sha256=hashlib.sha256(rollout.rng_after).hexdigest())
    origin["origin_sha256"] = old._hash(origin)
    return origin


def _validate_origin(active, cursor, rows, config, backend, generator):
    origin = active.get("observation_origin")
    if not isinstance(origin, dict) or set(origin) != ORIGIN_FIELDS or origin["schema_version"] != 1:
        raise ValueError("Observed active rollout requires its original generation/RNG anchors")
    if origin["origin_sha256"] != old._hash({k: v for k, v in origin.items() if k != "origin_sha256"}):
        raise ValueError("Observed rollout origin identity changed")
    before = origin["cursor_before"]
    old._validate_ppo_cursor(before, None, rows, config)
    if (before.get("active_rollout") is not False or origin["rollout_index"] != before["completed_rollouts"]
            or cursor["completed_rollouts"] != origin["rollout_index"]
            or before["offset"] + len(active["responses"]) != cursor["offset"]
            or before["response_tokens"] + sum(len(r["action_ids"]) for r in active["responses"]) != cursor["response_tokens"]
            or origin["record_ids"] != active["record_ids"]
            or any(before.get(k) != cursor.get(k) for k in ("config_sha256", "data_sha256", "schema", "order"))):
        raise ValueError("Observed rollout origin differs from the recorded stage cursor")
    names = tuple(name for name, _ in _named_versions(backend))
    named = origin["generation_model_versions"]
    versions = origin["model_versions"]
    if (not isinstance(named, (list, tuple)) or not isinstance(versions, (list, tuple))
            or len(named) != len(names) or len(versions) != len(names)
            or tuple(pair[0] for pair in named) != names
            or any(len(pair) != 2 or type(pair[1]) is not int or pair[1] < 0 for pair in named)
            or any(len(pair) != 2 or any(type(v) is not int for v in pair) or pair[0] <= 0 or pair[1] < 0 for pair in versions)
            or [pair[1] for pair in named] != [pair[1] for pair in versions]):
        raise ValueError("Malformed original generation model versions")
    for phase in ("before", "after"):
        raw = bytes.fromhex(origin[f"rng_{phase}_hex"])
        if not raw or hashlib.sha256(raw).hexdigest() != origin[f"rng_{phase}_sha256"]:
            raise ValueError("Malformed original sampling RNG anchor")
    if _rng(generator).hex() != origin["rng_after_hex"]:
        raise ValueError("Resume generator differs from the original completed sampling RNG")


def run_supervised(backend, role, optimizer, examples, config, *, device, state=None,
                   reference_provider=None, checkpoint_callback=None, stop_requested=lambda: False,
                   clock=time.time, event_callback=None):
    """Same bounded numeric pass, with preparation and optimizer observations."""
    if not isinstance(config, SupervisedConfig):
        raise ValueError("SupervisedConfig required")
    rows = old._records(examples, backend, preference=config.kind != "sft", eos_id=config.eos_id)
    cursor, parameters, frozen, versions = old._prepare(backend, role, optimizer, config, rows, state, {})
    processed = [rows[i] for i in cursor["order"][:cursor["offset"]]]
    processed_tokens = sum(len(r["chosen_ids"]) +
                           (len(r["rejected_ids"]) if config.kind != "sft" else 0) for r in processed)
    if (cursor["active_rollout"] or cursor["completed_rollouts"] != 0
            or cursor["optimizer_updates"] != math.ceil(cursor["offset"] / config.effective_batch)
            or (cursor["offset"] < len(rows) and cursor["offset"] % config.effective_batch)
            or cursor["response_tokens"] != processed_tokens):
        raise ValueError("Invalid supervised update boundary")
    if config.kind == "dpo" and reference_provider is None:
        raise ValueError("DPO requires a verified reference-score provider")
    metrics = []
    while cursor["offset"] < len(rows):
        reason = old._boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, None, reason, tuple(metrics))
        selected = [rows[i] for i in cursor["order"][cursor["offset"]:cursor["offset"] + config.effective_batch]]
        info = {"target_update": cursor["optimizer_updates"] + 1,
                "record_ids": [row["record_id"] for row in selected]}
        _emit(event_callback, backend, "preparation_start", config.kind, cursor, **info)
        reason = old._boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, None, reason, tuple(metrics))
        batches = []
        for start in range(0, len(selected), config.microbatch):
            subset = selected[start:start + config.microbatch]
            chosen = old._batch(subset, "chosen_ids", config.pad_id, device)
            rejected = None if config.kind == "sft" else old._batch(subset, "rejected_ids", config.pad_id, device)
            reference = reference_provider(chosen, rejected) if config.kind == "dpo" else None
            batches.append((chosen, rejected, reference))
        _emit(event_callback, backend, "preparation_end", config.kind, cursor, **info)
        reason = old._boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, None, reason, tuple(metrics))
        tokens = sum(len(row["chosen_ids"]) for row in selected)
        denominator = tokens if config.kind == "sft" else len(selected)
        _emit(event_callback, backend, "optimizer_start", config.kind, cursor,
              denominator=denominator, **info)
        reason = old._boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, None, reason, tuple(metrics))
        objective = 0.0
        for chosen, rejected, reference in batches:
            if config.kind == "sft":
                result = old.sft_backward(backend, role, optimizer, chosen, token_denominator=denominator,
                                          chunk_tokens=config.chunk_tokens)
            elif config.kind == "reward":
                result = old.reward_pair_backward(backend, role, optimizer, chosen, rejected, pair_denominator=denominator)
            else:
                result = old.dpo_pair_backward(backend, role, optimizer, chosen, rejected, reference,
                            beta=config.beta, pair_denominator=denominator, chunk_tokens=config.chunk_tokens)
            objective += float(result.objective_contribution)
        norm = old._step(optimizer, parameters, frozen, versions, config.max_grad_norm)
        cursor["offset"] += len(selected)
        cursor["optimizer_updates"] += 1
        cursor["response_tokens"] += tokens + (sum(len(r["rejected_ids"]) for r in selected) if config.kind != "sft" else 0)
        metric = {"update": cursor["optimizer_updates"], "rows": len(selected),
                  "objective": objective, "gradient_norm": norm}
        metrics.append(metric)
        _emit(event_callback, backend, "optimizer_end", config.kind, cursor, **metric, **info)
        if checkpoint_callback:
            checkpoint_callback(copy.deepcopy(cursor), None)
    return StageOutcome(cursor, None, "data_exhausted", tuple(metrics))


def run_ppo(backend, role, optimizer, examples, config, *, device, generator, reference_role,
            reward_role, state=None, extra_state=None, checkpoint_callback=None,
            stop_requested=lambda: False, clock=time.time, event_callback=None):
    """Same on-policy loops; persist original collection anchors for active resume."""
    if not isinstance(config, PPOConfig):
        raise ValueError("PPOConfig required")
    rows = old._records(examples, backend, prompts_only=True)
    cursor, parameters, frozen, versions = old._prepare(backend, role, optimizer, config, rows, state,
                            {"reference_role": reference_role, "reward_role": reward_role})
    backend.require_frozen_role(reference_role)
    backend.require_frozen_role(reward_role)
    if cursor["active_rollout"] != (extra_state is not None):
        raise ValueError("Saved active PPO rollout is missing or unexpected")
    active = copy.deepcopy(extra_state)
    if active is not None:
        old._restore_rollout(active, backend, role, config)
        _validate_origin(active, cursor, rows, config, backend, generator)
    old._validate_ppo_cursor(cursor, active, rows, config)
    metrics = []
    while True:
        if active is None and cursor["offset"] >= min(len(rows), config.max_responses):
            reason = "data_exhausted" if cursor["offset"] == len(rows) else "response_limit"
            return StageOutcome(cursor, None, reason, tuple(metrics))
        reason = old._boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, active, reason, tuple(metrics))
        if active is None:
            count = min(config.rollout_batch, len(rows) - cursor["offset"], config.max_responses - cursor["offset"])
            subset = [rows[i] for i in cursor["order"][cursor["offset"]:cursor["offset"] + count]]
            origin = _origin(cursor, subset, backend, generator)
            start_info = {"rollout_index": origin["rollout_index"], "record_ids": origin["record_ids"],
                          "observation_origin": origin}
            _emit(event_callback, backend, "rollout_start", "ppo", cursor, generator=generator, **start_info)
            _emit(event_callback, backend, "sampling_start", "ppo", cursor, generator=generator, **start_info)
            reason = old._boundary_reason(cursor, config, stop_requested, clock)
            if reason:
                return StageOutcome(cursor, None, reason, tuple(metrics))
            rollout = old.sample_rollout(backend, role, [r["prompt_ids"] for r in subset], eos_id=config.eos_id,
                     pad_id=config.pad_id, max_new_tokens=config.max_new_tokens, generator=generator, device=device)
            origin = _complete_origin(origin, rollout)
            phase = {"rollout_index": origin["rollout_index"], "record_ids": origin["record_ids"],
                     "observation_origin": origin}
            _emit(event_callback, backend, "sampling_end", "ppo", cursor, generator=generator, **phase)
            packed = old.pack_rollout(rollout, device=device)
            _emit(event_callback, backend, "frozen_scoring_start", "ppo", cursor, generator=generator, **phase)
            reference, rewards = old.score_frozen_rollout(backend, rollout, reference_role=reference_role,
                                reward_role=reward_role, device=device, chunk_tokens=config.chunk_tokens)
            _emit(event_callback, backend, "frozen_scoring_end", "ppo", cursor, generator=generator, **phase)
            shaped = old.shaped_rewards(packed["old_logprobs"], reference, rewards, packed["action_mask"], config.kl_coef)
            advantages, returns = old.masked_gae(shaped, packed["old_values"], packed["action_mask"], config.gamma, config.lam)
            advantages = old.whiten_advantages(advantages, packed["action_mask"])
            active = {"responses": [asdict(r) for r in rollout.responses], "record_ids": [r["record_id"] for r in subset],
                      "advantages": advantages.cpu().clone(), "returns": returns.cpu().clone(),
                      "reference_logprobs": reference.cpu().clone(), "reward_scores": rewards.cpu().clone(),
                      "epoch": 0, "offset": 0, "observation_origin": origin}
            cursor["offset"] += count
            cursor["response_tokens"] += int(packed["action_mask"].sum())
            cursor["active_rollout"] = True
            _emit(event_callback, backend, "rollout_collected", "ppo", cursor, generator=generator,
                  rollout_index=origin["rollout_index"], cursor_before=origin["cursor_before"],
                  **active, old_logprobs=packed["old_logprobs"], old_values=packed["old_values"],
                  action_mask=packed["action_mask"],
                  rollout_metadata={"policy_role": rollout.policy_role, "eos_id": rollout.eos_id,
                                    "pad_id": rollout.pad_id, "vocab_size": rollout.vocab_size,
                                    "max_new_tokens": rollout.max_new_tokens})
            reason = old._boundary_reason(cursor, config, stop_requested, clock)
            if reason:
                if checkpoint_callback:
                    checkpoint_callback(copy.deepcopy(cursor), copy.deepcopy(active))
                return StageOutcome(cursor, active, reason, tuple(metrics))
        rollout, packed = old._restore_rollout(active, backend, role, config)
        order = list(range(len(rollout.responses)))
        random.Random(config.seed + cursor["completed_rollouts"] * config.epochs + active["epoch"]).shuffle(order)
        indices = order[active["offset"]:active["offset"] + config.optimizer_batch]
        denominator = sum(len(rollout.responses[i].action_ids) for i in indices)
        step_info = {"rollout_index": active["observation_origin"]["rollout_index"],
                     "record_ids": [active["record_ids"][i] for i in indices], "response_indices": indices,
                     "epoch": active["epoch"], "rollout_offset": active["offset"],
                     "target_update": cursor["optimizer_updates"] + 1}
        _emit(event_callback, backend, "optimizer_start", "ppo", cursor, generator=generator,
              denominator=denominator, **step_info)
        reason = old._boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, active, reason, tuple(metrics))
        objective = 0.0
        for index in indices:
            response = rollout.responses[index]
            length = len(response.prompt_ids) + len(response.action_ids)
            inputs = {k: packed[k][index:index + 1, :length].to(device) for k in ("input_ids", "attention_mask", "labels")}
            n, start = len(response.action_ids), len(response.prompt_ids) - 1
            old_lp = packed["old_logprobs"][index:index + 1, :n].to(device)
            old_values = packed["old_values"][index:index + 1, :n].to(device)
            advantages = active["advantages"][index:index + 1, :n].to(device)
            returns = active["returns"][index:index + 1, :n].to(device)
            mask = torch.ones_like(old_lp, dtype=torch.bool)
            def loss(view):
                scores, values = view.token_scores_and_values(**inputs, chunk_tokens=config.chunk_tokens)
                result = old.clipped_ppo_loss(scores.logprobs[:, start:start + n], values[:, start:start + n],
                    old_lp, old_values, advantages, returns, mask, config.policy_clip, config.value_clip,
                    config.value_coef, denominator)
                return result.objective_contribution
            objective += float(backend.backward(role, optimizer, loss, heads=("value",)))
        norm = old._step(optimizer, parameters, frozen, versions, config.max_grad_norm)
        cursor["optimizer_updates"] += 1
        active["offset"] += len(indices)
        finished = None
        if active["offset"] == len(rollout.responses):
            active["epoch"] += 1
            active["offset"] = 0
            if active["epoch"] == config.epochs:
                finished = {"rollout_index": active["observation_origin"]["rollout_index"],
                            "record_ids": active["record_ids"], "observation_origin": active["observation_origin"]}
                cursor["completed_rollouts"] += 1
                cursor["active_rollout"] = False
                active = None
        metric = {"update": cursor["optimizer_updates"], "responses": len(indices),
                  "action_tokens": denominator, "objective": objective, "gradient_norm": norm}
        metrics.append(metric)
        _emit(event_callback, backend, "optimizer_end", "ppo", cursor, generator=generator, **metric, **step_info)
        if finished is not None:
            _emit(event_callback, backend, "rollout_finished", "ppo", cursor, generator=generator, **finished)
        if checkpoint_callback:
            checkpoint_callback(copy.deepcopy(cursor), copy.deepcopy(active))
