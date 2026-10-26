"""Bounded optimizer loops over already encoded data; caller owns model loading.

This is not a campaign CLI or GPU admission check. Configuration, source/data
and parent identities must be frozen by the outer runner. Deadline is absolute
and remains in the resume identity. Checkpoints occur only at complete optimizer
updates, with cleared gradients. Stop/deadline never dispatch another update.
One shuffled data pass is permitted; exhaustion never silently cycles data.
Caller owns device/autocast and supplies a separately constructed optimizer.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import time

import torch

from .training_steps import sft_backward, reward_pair_backward, dpo_pair_backward
from .rollout import RolloutBatch, SampledResponse, sample_rollout, pack_rollout, score_frozen_rollout
from .ppo_objective import shaped_rewards, masked_gae, whiten_advantages, clipped_ppo_loss


@dataclass(frozen=True)
class SupervisedConfig:
    kind: str
    seed: int
    effective_batch: int
    microbatch: int
    max_updates: int
    deadline_epoch: float
    pad_id: int
    eos_id: int
    max_grad_norm: float = 1.0
    beta: float = 0.1
    chunk_tokens: int = 32


@dataclass(frozen=True)
class PPOConfig:
    seed: int
    rollout_batch: int
    optimizer_batch: int
    epochs: int
    max_responses: int
    max_updates: int
    max_new_tokens: int
    deadline_epoch: float
    pad_id: int
    eos_id: int
    max_grad_norm: float = 1.0
    kl_coef: float = 0.05
    gamma: float = 1.0
    lam: float = 0.95
    policy_clip: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    chunk_tokens: int = 32


@dataclass(frozen=True)
class StageOutcome:
    cursor: dict
    extra_state: dict | None
    stop_reason: str
    metrics: tuple[dict, ...]


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def parameter_mapping(backend, role, *, heads=()):
    """Names in exact optimizer parameter order for adapter-only checkpoints."""
    named = {id(p): name for name, p in backend._model.named_parameters()}
    for head_name, head in backend._heads.items():
        named.update({id(p): f"scalar_heads.{head_name}.{name}" for name, p in head.named_parameters()})
    return {named[id(p)]: p for p in backend.parameters_for(role, heads=heads)}


def _field(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def _records(examples, backend, *, preference=False, prompts_only=False, eos_id=None):
    result, seen = [], set()
    vocab = backend._base.config.vocab_size
    for row in examples:
        record_id = _field(row, "record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError("Unique nonempty record IDs required")
        seen.add(record_id)
        value = {"record_id": record_id}
        fields = ["prompt_ids"] + ([] if prompts_only else ["chosen_ids"])
        if preference:
            fields.append("rejected_ids")
        for name in fields:
            ids = tuple(_field(row, name))
            if not ids or any(type(t) is not int or not 0 <= t < vocab for t in ids):
                raise ValueError("Encoded data must contain exact vocabulary token IDs")
            if name != "prompt_ids" and (ids[-1] != eos_id or eos_id in ids[:-1]):
                raise ValueError("Offline completions require exactly one terminal EOS")
            value[name] = ids
        for name in fields[1:]:
            if len(value["prompt_ids"]) + len(value[name]) > backend._base.config.max_position_embeddings:
                raise ValueError("Encoded sequence exceeds model context; no silent truncation")
        result.append(value)
    if not result:
        raise ValueError("Training data cannot be empty")
    return tuple(result)


def _validate_config(config, backend):
    fields = asdict(config)
    for name in ("seed", "pad_id", "eos_id"):
        if type(fields[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    for name in ("effective_batch", "microbatch", "max_updates", "chunk_tokens", "rollout_batch", "optimizer_batch", "epochs", "max_responses", "max_new_tokens"):
        if name in fields and (type(fields[name]) is not int or fields[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("deadline_epoch", "max_grad_norm", "beta", "kl_coef", "gamma", "lam", "policy_clip", "value_clip", "value_coef"):
        if name in fields and (isinstance(fields[name], bool) or not isinstance(fields[name], (int, float)) or not math.isfinite(fields[name]) or fields[name] < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.deadline_epoch <= 0 or config.max_grad_norm <= 0:
        raise ValueError("Deadline and gradient norm bound must be positive")
    if any(not 0 <= t < backend._base.config.vocab_size for t in (config.pad_id, config.eos_id)):
        raise ValueError("Special tokens outside vocabulary")
    if isinstance(config, SupervisedConfig) and (config.kind not in ("sft", "reward", "dpo") or config.beta <= 0):
        raise ValueError("Unknown supervised stage or invalid DPO beta")
    if isinstance(config, PPOConfig) and any(getattr(config, n) > 1 for n in ("gamma", "lam", "policy_clip")):
        raise ValueError("PPO discount/lambda/policy clip must be <=1")


def _batch(rows, field, pad_id, device):
    length = max(len(r["prompt_ids"]) + len(r[field]) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    attention, labels = torch.zeros_like(ids), torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        start, end = len(row["prompt_ids"]), len(row["prompt_ids"]) + len(row[field])
        ids[i, :end] = torch.tensor(row["prompt_ids"] + row[field], device=device)
        attention[i, :end] = 1
        labels[i, start:end] = ids[i, start:end]
    return dict(input_ids=ids, attention_mask=attention, labels=labels)


def _prepare(backend, role, optimizer, config, rows, state, extra_identity):
    _validate_config(config, backend)
    heads = ("value",) if isinstance(config, PPOConfig) else (("reward",) if config.kind == "reward" else ())
    parameters = backend.parameters_for(role, heads=heads)
    supplied = [p for group in optimizer.param_groups for p in group["params"]]
    if len(supplied) != len(parameters) or set(map(id, supplied)) != set(map(id, parameters)):
        raise ValueError("Optimizer must exactly cover stage adapter/head parameters")
    if any(p.grad is not None for p in backend._all_parameters()):
        raise RuntimeError("Stage entry requires a clean optimizer boundary")
    selected = set(map(id, parameters))
    frozen = tuple(p for p in backend._all_parameters() if id(p) not in selected)
    versions = tuple((id(p), p._version) for p in frozen)
    identity = {"config_sha256": _hash({"config": asdict(config), "role": role, **extra_identity}),
                "data_sha256": _hash(rows), "schema": 1}
    order = list(range(len(rows)))
    random.Random(config.seed).shuffle(order)
    if state is None:
        state = {**identity, "order": order, "offset": 0, "optimizer_updates": 0,
                 "response_tokens": 0, "active_rollout": False, "completed_rollouts": 0}
    else:
        state = copy.deepcopy(state)
        if any(state.get(k) != v for k, v in identity.items()) or state.get("order") != order:
            raise ValueError("Resume config/deadline/data/order identity changed")
        for key in ("offset", "optimizer_updates", "response_tokens", "completed_rollouts"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError("Invalid saved stage counters")
        if state["offset"] > len(rows) or state["optimizer_updates"] > config.max_updates or type(state.get("active_rollout")) is not bool:
            raise ValueError("Saved stage counters exceed work limits")
    return state, parameters, frozen, versions


def _boundary_reason(state, config, stop_requested, clock):
    if stop_requested():
        return "paused"
    if clock() >= config.deadline_epoch:
        return "budget_exhausted"
    if state["optimizer_updates"] >= config.max_updates:
        return "update_limit"
    return None


def _step(optimizer, parameters, frozen, versions, max_norm):
    if tuple((id(p), p._version) for p in frozen) != versions:
        raise RuntimeError("Frozen parameters changed during stage")
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError("No stage gradients produced")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError("Optimizer produced nonfinite parameters; discard failed update")
    if tuple((id(p), p._version) for p in frozen) != versions:
        raise RuntimeError("Optimizer changed frozen parameters")
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def run_supervised(backend, role, optimizer, examples, config, *, device, state=None,
                   reference_provider=None, checkpoint_callback=None, stop_requested=lambda: False, clock=time.time):
    """One bounded SFT/RM/DPO pass; callback receives (JSON cursor, None).

    DPO provider(chosen,rejected) returns verified ReferencePairScores. Compute
    all effective-batch reference records before accumulating policy gradients.
    Passing an online provider is useful for CPU tests; production uses a cache.
    """
    if not isinstance(config, SupervisedConfig):
        raise ValueError("SupervisedConfig required")
    rows = _records(examples, backend, preference=config.kind != "sft", eos_id=config.eos_id)
    cursor, parameters, frozen, versions = _prepare(backend, role, optimizer, config, rows, state, {})
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
        reason = _boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, None, reason, tuple(metrics))
        selected = [rows[i] for i in cursor["order"][cursor["offset"]:cursor["offset"] + config.effective_batch]]
        batches = []
        for start in range(0, len(selected), config.microbatch):
            subset = selected[start:start + config.microbatch]
            chosen = _batch(subset, "chosen_ids", config.pad_id, device)
            rejected = None if config.kind == "sft" else _batch(subset, "rejected_ids", config.pad_id, device)
            reference = reference_provider(chosen, rejected) if config.kind == "dpo" else None
            batches.append((chosen, rejected, reference))
        # Cache validation/scoring may be slow. No gradients exist yet, so a
        # stop received during preparation must prevent this update dispatch.
        reason = _boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, None, reason, tuple(metrics))
        tokens = sum(len(row["chosen_ids"]) for row in selected)
        denominator = tokens if config.kind == "sft" else len(selected)
        objective = 0.0
        for chosen, rejected, reference in batches:
            if config.kind == "sft":
                result = sft_backward(backend, role, optimizer, chosen, token_denominator=denominator, chunk_tokens=config.chunk_tokens)
            elif config.kind == "reward":
                result = reward_pair_backward(backend, role, optimizer, chosen, rejected, pair_denominator=denominator)
            else:
                result = dpo_pair_backward(backend, role, optimizer, chosen, rejected, reference,
                            beta=config.beta, pair_denominator=denominator, chunk_tokens=config.chunk_tokens)
            objective += float(result.objective_contribution)
        norm = _step(optimizer, parameters, frozen, versions, config.max_grad_norm)
        cursor["offset"] += len(selected)
        cursor["optimizer_updates"] += 1
        cursor["response_tokens"] += tokens + (sum(len(r["rejected_ids"]) for r in selected) if config.kind != "sft" else 0)
        metrics.append({"update": cursor["optimizer_updates"], "rows": len(selected), "objective": objective, "gradient_norm": norm})
        if checkpoint_callback:
            checkpoint_callback(copy.deepcopy(cursor), None)
    return StageOutcome(cursor, None, "data_exhausted", tuple(metrics))


def _restore_rollout(active, backend, role, config):
    responses = tuple(SampledResponse(**{k: tuple(v) if k != "termination" else v for k, v in r.items()})
                      for r in active["responses"])
    # Empty versions deliberately forbid re-scoring an old rollout with updated
    # policy weights after resume. Frozen old statistics below are checkpointed.
    rollout = RolloutBatch(responses, role, config.eos_id, config.pad_id, config.max_new_tokens,
                           backend._base.config.vocab_size, (), b"", b"", backend)
    packed = pack_rollout(rollout, device="cpu")
    mask = packed["action_mask"]
    for key in ("advantages", "returns", "reference_logprobs"):
        tensor = active[key]
        if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.device.type != "cpu" or tensor.requires_grad
                or tensor.shape != mask.shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError("Invalid checkpointed rollout statistics")
    if type(active["epoch"]) is not int or not 0 <= active["epoch"] < config.epochs:
        raise ValueError("Invalid saved PPO epoch")
    if type(active["offset"]) is not int or not 0 <= active["offset"] < len(responses) or active["offset"] % config.optimizer_batch:
        raise ValueError("Invalid saved PPO optimizer boundary")
    return rollout, packed


def _validate_ppo_cursor(cursor, active, rows, config):
    limit = min(len(rows), config.max_responses)
    if cursor["offset"] > limit:
        raise ValueError("Saved response cursor exceeds its original budget")
    n = len(active["responses"]) if active is not None else 0
    completed = cursor["offset"] - n
    if (completed < 0
            or (active is not None and (n != min(config.rollout_batch, limit - completed)
                                       or n < 1 or completed % config.rollout_batch))
            or (active is None and completed < limit and completed % config.rollout_batch)):
        raise ValueError("Invalid saved rollout size/cursor")
    batches = [min(config.rollout_batch, completed - i) for i in range(0, completed, config.rollout_batch)]
    updates = sum(math.ceil(size / config.optimizer_batch) * config.epochs for size in batches)
    if active is not None:
        recorded_rows = [rows[i] for i in cursor["order"][completed:cursor["offset"]]]
        expected_ids = [r["record_id"] for r in recorded_rows]
        if (active.get("record_ids") != expected_ids
                or any(tuple(response["prompt_ids"]) != row["prompt_ids"]
                       for response, row in zip(active["responses"], recorded_rows))):
            raise ValueError("Saved rollout does not match the recorded dataset cursor")
        updates += active["epoch"] * math.ceil(n / config.optimizer_batch) + active["offset"] // config.optimizer_batch
    if cursor["completed_rollouts"] != len(batches) or cursor["optimizer_updates"] != updates:
        raise ValueError("Saved PPO optimizer counters do not match the rollout position")


def run_ppo(backend, role, optimizer, examples, config, *, device, generator, reference_role,
            reward_role, state=None, extra_state=None, checkpoint_callback=None,
            stop_requested=lambda: False, clock=time.time):
    """On-policy rollout/epoch/minibatch loop, checkpointable after every update.

    Callback receives cursor plus active rollout containing sampled IDs, fixed
    old stats, returns, advantages and epoch position. Sampling generator must
    be checkpointed by the outer owner. Microbatch is one response; optimizer
    batch may contain several responses with one token-weighted denominator.
    """
    if not isinstance(config, PPOConfig):
        raise ValueError("PPOConfig required")
    rows = _records(examples, backend, prompts_only=True)
    cursor, parameters, frozen, versions = _prepare(backend, role, optimizer, config, rows, state,
                            {"reference_role": reference_role, "reward_role": reward_role})
    backend.require_frozen_role(reference_role)
    backend.require_frozen_role(reward_role)
    if cursor["active_rollout"] != (extra_state is not None):
        raise ValueError("Saved active PPO rollout is missing or unexpected")
    active = copy.deepcopy(extra_state)
    if active is not None:
        _restore_rollout(active, backend, role, config)
    _validate_ppo_cursor(cursor, active, rows, config)
    metrics = []
    while True:
        if active is None and cursor["offset"] >= min(len(rows), config.max_responses):
            reason = "data_exhausted" if cursor["offset"] == len(rows) else "response_limit"
            return StageOutcome(cursor, None, reason, tuple(metrics))
        reason = _boundary_reason(cursor, config, stop_requested, clock)
        if reason:
            return StageOutcome(cursor, active, reason, tuple(metrics))
        if active is None:
            count = min(config.rollout_batch, len(rows) - cursor["offset"], config.max_responses - cursor["offset"])
            subset = [rows[i] for i in cursor["order"][cursor["offset"]:cursor["offset"] + count]]
            rollout = sample_rollout(backend, role, [r["prompt_ids"] for r in subset], eos_id=config.eos_id,
                     pad_id=config.pad_id, max_new_tokens=config.max_new_tokens, generator=generator, device=device)
            packed = pack_rollout(rollout, device=device)
            reference, rewards = score_frozen_rollout(backend, rollout, reference_role=reference_role,
                                reward_role=reward_role, device=device, chunk_tokens=config.chunk_tokens)
            shaped = shaped_rewards(packed["old_logprobs"], reference, rewards, packed["action_mask"], config.kl_coef)
            advantages, returns = masked_gae(shaped, packed["old_values"], packed["action_mask"], config.gamma, config.lam)
            advantages = whiten_advantages(advantages, packed["action_mask"])
            active = {"responses": [asdict(r) for r in rollout.responses], "record_ids": [r["record_id"] for r in subset],
                      "advantages": advantages.cpu().clone(), "returns": returns.cpu().clone(),
                      "reference_logprobs": reference.cpu().clone(), "reward_scores": rewards.cpu().clone(), "epoch": 0, "offset": 0}
            cursor["offset"] += count
            cursor["response_tokens"] += int(packed["action_mask"].sum())
            cursor["active_rollout"] = True
            # A stop during collection is honored before the first optimizer update.
            reason = _boundary_reason(cursor, config, stop_requested, clock)
            if reason:
                if checkpoint_callback:
                    checkpoint_callback(copy.deepcopy(cursor), copy.deepcopy(active))
                return StageOutcome(cursor, active, reason, tuple(metrics))
        rollout, packed = _restore_rollout(active, backend, role, config)
        order = list(range(len(rollout.responses)))
        random.Random(config.seed + cursor["completed_rollouts"] * config.epochs + active["epoch"]).shuffle(order)
        indices = order[active["offset"]:active["offset"] + config.optimizer_batch]
        denominator = sum(len(rollout.responses[i].action_ids) for i in indices)
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
                result = clipped_ppo_loss(scores.logprobs[:, start:start + n], values[:, start:start + n],
                    old_lp, old_values, advantages, returns, mask, config.policy_clip, config.value_clip,
                    config.value_coef, denominator)
                return result.objective_contribution
            objective += float(backend.backward(role, optimizer, loss, heads=("value",)))
        norm = _step(optimizer, parameters, frozen, versions, config.max_grad_norm)
        cursor["optimizer_updates"] += 1
        active["offset"] += len(indices)
        if active["offset"] == len(rollout.responses):
            active["epoch"] += 1
            active["offset"] = 0
            if active["epoch"] == config.epochs:
                cursor["completed_rollouts"] += 1
                cursor["active_rollout"] = False
                active = None
        metrics.append({"update": cursor["optimizer_updates"], "responses": len(indices),
                        "action_tokens": denominator, "objective": objective, "gradient_norm": norm})
        if checkpoint_callback:
            checkpoint_callback(copy.deepcopy(cursor), copy.deepcopy(active))
