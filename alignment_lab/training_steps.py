"""CPU-testable backward primitives for one guarded shared Qwen3 backbone.

These functions accumulate gradients only: no step, zero_grad, clipping, model
loading, device selection, or capacity claim. The caller supplies the *whole
effective batch* token count (SFT) or pair count (RM/DPO) on every microbatch.
RM/DPO compute the actual loss without a graph, then replay and backpropagate
one side at a time with detached analytic derivatives. Only one backbone graph
is live; the linear surrogate's value is deliberately not reported as the loss.

DPO uses raw completion log-probability SUMS, including every labelled EOS.
Compute reference_pair_scores for all microbatches before accumulating policy
gradients: SharedQwen3 rightly prohibits adapter switches with pending grads.
The in-memory reference record binds exact inputs, its owner, and frozen
base/reference parameter versions. Persistent caches additionally need actual
checkpoint identities; this record does not substitute for file hashes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class BackwardResult:
    """Detached true loss and an explicitly separate backward-only scalar.

    objective_sum / units is the local mean. objective_contribution is already
    divided by the caller's effective-batch denominator. For paired objectives,
    backward_surrogate is a linearization and generally has a different value.
    """

    objective_sum: Tensor
    objective_mean: Tensor
    objective_contribution: Tensor
    backward_surrogate: Tensor | None
    units: int
    normalization_denominator: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferencePairScores:
    chosen: Tensor
    rejected: Tensor
    chosen_counts: Tensor
    rejected_counts: Tensor
    reference_role: str
    input_fingerprints: tuple[str, str]
    frozen_versions: tuple[tuple[int, int], ...]
    _owner: object = field(repr=False, compare=False)
    _tensor_versions: tuple[int, ...] = field(repr=False)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _denominator(value, units, name):
    numeric = _positive(value, name)
    if int(numeric) != numeric or numeric < units:
        raise ValueError(f"{name} must be an integer effective-batch count >= this microbatch's count")
    return int(numeric)


def _chunk_size(value):
    if type(value) is not int or value < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    return value


def _batch(batch):
    if not isinstance(batch, Mapping) or not {"input_ids", "attention_mask", "labels"}.issubset(batch):
        raise ValueError("batch needs input_ids, attention_mask, and completion labels")
    ids, attention, labels = (batch[key] for key in ("input_ids", "attention_mask", "labels"))
    if any(not isinstance(value, Tensor) for value in (ids, attention, labels)):
        raise ValueError("batch values must be tensors")
    if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 2 or ids.dtype != torch.long:
        raise ValueError("input_ids must be nonempty int64 [batch, sequence>=2]")
    if attention.shape != ids.shape or labels.shape != ids.shape or labels.dtype != torch.long:
        raise ValueError("attention_mask and int64 labels must match input_ids")
    if attention.device != ids.device or labels.device != ids.device:
        raise ValueError("batch tensors must share one device")
    if not bool(((attention == 0) | (attention == 1)).all()):
        raise ValueError("attention_mask must be binary")
    attended, labelled = attention.bool(), labels.ne(-100)
    if not bool(attended[:, 0].all()) or bool((attended[:, 1:] & ~attended[:, :-1]).any()):
        raise ValueError("Scoring requires nonempty right-padded sequences")
    if bool(labelled[:, 0].any()) or bool((labelled & ~attended).any()):
        raise ValueError("First-token and padding labels must be -100")
    if not torch.equal(labels[labelled], ids[labelled]) or not bool(labelled.any(-1).all()):
        raise ValueError("Each completion must label its actual input tokens")
    for row, mask in zip(labelled, attended):
        first = int(row.nonzero()[0].item())
        if not torch.equal(row[first:], mask[first:]):
            raise ValueError("Completion labels must form a contiguous attended suffix")
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def _paired_batches(chosen, rejected):
    chosen, rejected = _batch(chosen), _batch(rejected)
    if chosen["input_ids"].shape[0] != rejected["input_ids"].shape[0]:
        raise ValueError("Chosen/rejected batches must contain aligned pairs")
    if chosen["input_ids"].device != rejected["input_ids"].device:
        raise ValueError("Chosen/rejected pairs must use one device")
    for ci, cl, ri, rl in zip(chosen["input_ids"], chosen["labels"], rejected["input_ids"], rejected["labels"]):
        cstart, rstart = int(cl.ne(-100).nonzero()[0].item()), int(rl.ne(-100).nonzero()[0].item())
        if not torch.equal(ci[:cstart], ri[:rstart]):
            raise ValueError("Each chosen/rejected pair must have identical prompt token IDs")
    return chosen, rejected


def _fingerprint(batch):
    # Hash only small integer inputs, never backbone tensors or vocabulary logits.
    payload = {key: value.detach().cpu().tolist() for key, value in batch.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _vector(value, size, *, differentiable):
    if not isinstance(value, Tensor) or value.shape != (size,) or not value.is_floating_point():
        raise ValueError("Scorer must return one floating-point scalar per sequence")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Nonfinite sequence score")
    if value.requires_grad != differentiable:
        raise ValueError("Training scores must require gradients; read/reference scores must be detached")
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _sequence_scores(view, batch, chunk_tokens, *, differentiable):
    result = view.sequence_scores(**batch, chunk_tokens=chunk_tokens)
    expected = batch["labels"][:, 1:].ne(-100).sum(-1)
    if not isinstance(result.counts, Tensor) or not torch.equal(result.counts, expected):
        raise ValueError("Sequence score counts must equal labelled completion token counts")
    return _vector(result.sums, len(expected), differentiable=differentiable), expected


class _ReplayGuard:
    """Read-only version guard; SharedQwen3 owns role/graph/trainability guards.

    _all_parameters is the sole private backend dependency. Version counters
    catch supported in-place updates; callers must not bypass PyTorch via .data.
    """

    def __init__(self, backend, role, optimizer, heads, batches):
        selected = backend.parameters_for(role, heads=heads)
        supplied = [p for group in optimizer.param_groups for p in group["params"]]
        if len(supplied) != len(set(map(id, supplied))) or set(map(id, supplied)) != set(map(id, selected)):
            raise ValueError("Optimizer must exactly match this adapter and selected heads")
        self.backend = backend
        self.parameters = tuple(backend._all_parameters())
        self.versions = tuple(p._version for p in self.parameters)
        self.inputs = tuple(value for batch in batches for value in batch.values())
        self.input_versions = tuple(value._version for value in self.inputs)

    def check(self):
        if (tuple(map(id, self.backend._all_parameters())) != tuple(map(id, self.parameters))
                or tuple(p._version for p in self.parameters) != self.versions):
            raise RuntimeError("Parameters changed between loss scoring and paired replay")
        if tuple(value._version for value in self.inputs) != self.input_versions:
            raise RuntimeError("Input tensors changed during loss scoring or replay")


def _result(objective_sum, units, denominator, surrogate=None, **metrics):
    objective_sum = objective_sum.detach().clone()
    if not bool(torch.isfinite(objective_sum)):
        raise FloatingPointError("Nonfinite true objective")
    return BackwardResult(objective_sum, objective_sum / units, objective_sum / denominator,
                          None if surrogate is None else surrogate.detach().clone(),
                          units, denominator, {key: float(value) for key, value in metrics.items()})


def _replay_pair(backend, role, optimizer, batches, initial, coefficients, denominator,
                 scorer, heads, guard):
    surrogates = []
    for batch, expected, coefficient in zip(batches, initial, coefficients):
        guard.check()

        def loss(view):
            guard.check()
            actual = scorer(view, batch, True)
            guard.check()
            if actual.device != expected.device or not torch.allclose(actual.detach(), expected, rtol=1e-5, atol=1e-6):
                raise RuntimeError("Replay score differs from no-gradient scoring; gradients are invalid, discard accumulation")
            return (coefficient * actual).sum() / denominator

        surrogates.append(backend.backward(role, optimizer, loss, heads=heads))
        guard.check()
    return sum(surrogates)


def sft_backward(backend, role, optimizer, batch, *, token_denominator, chunk_tokens=32):
    """Accumulate completion-token cross entropy, normalized across the effective batch."""
    batch, chunk_tokens = _batch(batch), _chunk_size(chunk_tokens)
    units = int(batch["labels"][:, 1:].ne(-100).sum().item())
    denominator = _denominator(token_denominator, units, "token_denominator")
    guard = _ReplayGuard(backend, role, optimizer, (), (batch,))
    objectives = []

    def loss(view):
        guard.check()
        scores, _ = _sequence_scores(view, batch, chunk_tokens, differentiable=True)
        objective = -scores.sum()
        objectives.append(objective.detach().clone())
        guard.check()
        return objective / denominator

    backend.backward(role, optimizer, loss)
    guard.check()
    return _result(objectives[0], units, denominator)


def reward_pair_backward(backend, role, optimizer, chosen, rejected, *, pair_denominator, head="reward"):
    """Exact Bradley--Terry sum softplus(-(r_chosen-r_rejected)), one graph at a time.

    No reward centering or score clipping is applied. If either replay fails,
    previously accumulated gradients are not rolled back; discard the entire
    accumulation and rebuild the backend after a guarded-backward failure.
    """
    chosen, rejected = _paired_batches(chosen, rejected)
    units = len(chosen["input_ids"])
    denominator = _denominator(pair_denominator, units, "pair_denominator")
    guard = _ReplayGuard(backend, role, optimizer, (head,), (chosen, rejected))

    def score(view, batch, differentiable):
        return _vector(view.rewards(**batch, head=head), units, differentiable=differentiable)

    initial = []
    for batch in (chosen, rejected):
        guard.check()
        initial.append(backend.read(role, lambda view: score(view, batch, False)).detach().clone())
        guard.check()
    with torch.no_grad():
        margin = initial[0] - initial[1]
        objective = F.softplus(-margin).sum()
        derivative = -torch.sigmoid(-margin)
    if not bool(torch.isfinite(margin).all()) or not bool(torch.isfinite(objective)):
        raise FloatingPointError("Nonfinite Bradley--Terry objective")
    surrogate = _replay_pair(backend, role, optimizer, (chosen, rejected), initial,
                             (derivative, -derivative), denominator, score, (head,), guard)
    return _result(objective, units, denominator, surrogate, reward_margin=margin.mean(),
                   preference_accuracy=(margin > 0).float().mean())


def reference_pair_scores(backend, reference_role, chosen, rejected, *, chunk_tokens=32):
    """Read frozen-reference completion sums before any policy gradient accumulation."""
    backend.require_frozen_role(reference_role)  # Metadata never substitutes for this check.
    chosen, rejected = _paired_batches(chosen, rejected)
    chunk_tokens = _chunk_size(chunk_tokens)
    versions = backend.frozen_role_versions(reference_role)
    fingerprints = (_fingerprint(chosen), _fingerprint(rejected))
    tensors = []
    for batch in (chosen, rejected):
        if backend.frozen_role_versions(reference_role) != versions:
            raise RuntimeError("Frozen reference parameters changed during scoring")
        scores, counts = backend.read(reference_role, lambda view:
                                     _sequence_scores(view, batch, chunk_tokens, differentiable=False))
        tensors.extend((scores.detach().clone(), counts.detach().clone()))
    if backend.frozen_role_versions(reference_role) != versions:
        raise RuntimeError("Frozen reference parameters changed during scoring")
    if fingerprints != (_fingerprint(chosen), _fingerprint(rejected)):
        raise RuntimeError("Reference inputs changed during scoring")
    c, cc, r, rc = tensors
    return ReferencePairScores(c, r, cc, rc, reference_role, fingerprints, versions, backend,
                               tuple(t._version for t in (c, r, cc, rc)))


def _validate_reference(reference, backend, role, chosen, rejected):
    if not isinstance(reference, ReferencePairScores) or reference._owner is not backend:
        raise ValueError("Reference scores belong to a different backend")
    backend.require_frozen_role(reference.reference_role)
    if role == reference.reference_role:
        raise ValueError("Policy and reference roles must be distinct")
    tensors = (reference.chosen, reference.rejected, reference.chosen_counts, reference.rejected_counts)
    if any(t.requires_grad for t in tensors):
        raise ValueError("Reference scores must be detached")
    if tuple(t._version for t in tensors) != reference._tensor_versions:
        raise RuntimeError("Reference score tensors were modified")
    if backend.frozen_role_versions(reference.reference_role) != reference.frozen_versions:
        raise RuntimeError("Frozen base/reference parameters changed; reference scores are stale")
    if (_fingerprint(chosen), _fingerprint(rejected)) != reference.input_fingerprints:
        raise ValueError("Reference scores do not match these exact chosen/rejected inputs")
    for batch, scores, counts in ((chosen, reference.chosen, reference.chosen_counts),
                                  (rejected, reference.rejected, reference.rejected_counts)):
        _vector(scores, len(batch["input_ids"]), differentiable=False)
        if scores.device != batch["input_ids"].device or not torch.equal(counts, batch["labels"][:, 1:].ne(-100).sum(-1)):
            raise ValueError("Reference scores/counts must match the policy inputs and device")


def dpo_pair_backward(backend, role, optimizer, chosen, rejected, reference_scores, *, beta,
                      pair_denominator, chunk_tokens=32):
    """Exact reference-based DPO using completion SUMS and serial paired replay.

    L = softplus(-beta*((pi_c-pi_r)-(ref_c-ref_r))). No length normalization,
    clipping, label smoothing, reference-free option, or reference gradients.
    Reference records remain valid across policy updates, not reference updates.
    """
    chosen, rejected = _paired_batches(chosen, rejected)
    beta, chunk_tokens = _positive(beta, "beta"), _chunk_size(chunk_tokens)
    units = len(chosen["input_ids"])
    denominator = _denominator(pair_denominator, units, "pair_denominator")
    _validate_reference(reference_scores, backend, role, chosen, rejected)
    guard = _ReplayGuard(backend, role, optimizer, (), (chosen, rejected))

    def score(view, batch, differentiable):
        return _sequence_scores(view, batch, chunk_tokens, differentiable=differentiable)[0]

    initial = []
    for batch in (chosen, rejected):
        guard.check()
        initial.append(backend.read(role, lambda view: score(view, batch, False)).detach().clone())
        guard.check()
    with torch.no_grad():
        chosen_reward = beta * (initial[0] - reference_scores.chosen)
        rejected_reward = beta * (initial[1] - reference_scores.rejected)
        margin = beta * ((initial[0] - initial[1]) - (reference_scores.chosen - reference_scores.rejected))
        objective = F.softplus(-margin).sum()
        derivative = -beta * torch.sigmoid(-margin)
    if (not all(bool(torch.isfinite(value).all()) for value in (margin, chosen_reward, rejected_reward, derivative))
            or not bool(torch.isfinite(objective))):
        raise FloatingPointError("Nonfinite DPO objective or derivative")
    surrogate = _replay_pair(backend, role, optimizer, (chosen, rejected), initial,
                             (derivative, -derivative), denominator, score, (), guard)
    _validate_reference(reference_scores, backend, role, chosen, rejected)
    return _result(objective, units, denominator, surrogate, chosen_reward=chosen_reward.mean(),
                   rejected_reward=rejected_reward.mean(), reward_margin=margin.mean(),
                   preference_accuracy=(margin > 0).float().mean())
