"""PPO sampling and frozen scoring primitives; not a campaign or stage trainer.

Finite-response objective: EOS and the configured response horizon both end
the episode (zero value bootstrap). They remain distinct observed outcomes.
Sampled token IDs, including a sampled pad ID, are never decoded/re-tokenized.
"""
from dataclasses import dataclass, field
import math
from numbers import Real
import torch
from torch.nn import functional as F
from .shared_qwen import SharedQwen3


@dataclass(frozen=True)
class SampledResponse:
    prompt_ids: tuple[int, ...]
    action_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    old_values: tuple[float, ...]
    termination: str


@dataclass(frozen=True)
class RolloutBatch:
    responses: tuple[SampledResponse, ...]
    policy_role: str
    eos_id: int
    pad_id: int
    max_new_tokens: int
    vocab_size: int
    model_versions: tuple[tuple[int, int], ...]
    rng_before: bytes
    rng_after: bytes
    _owner: object = field(repr=False, compare=False)


def _versions(backend):
    return tuple((id(p), p._version) for p in backend._all_parameters())


def sample_rollout(backend, policy_role, prompts, *, eos_id, pad_id, max_new_tokens, generator, device):
    """Sample a whole rollout before any optimizer update or pending gradients.

    Caller controls random generator and device; no pretrained model loading,
    downloads, optimizer work, or global RNG manipulation takes place here.
    """
    backend.parameters_for(policy_role, heads=("value",))
    if any(p.grad is not None for p in backend._all_parameters()):
        raise RuntimeError("Clear optimizer gradients before collecting a new rollout")
    vocab = backend._base.config.vocab_size
    if type(pad_id) is not int or not 0 <= pad_id < vocab:
        raise ValueError("Padding must be an explicit vocabulary token ID")
    prompts = tuple(tuple(p) for p in prompts)
    if not prompts or any(not p or any(type(t) is not int or not 0 <= t < vocab for t in p) for p in prompts):
        raise ValueError("Prompts must contain nonempty exact token-ID sequences")
    if not isinstance(generator, torch.Generator) or generator.device != torch.device(device):
        raise ValueError("Explicit generator must match the rollout device")
    versions = _versions(backend)
    before = bytes(generator.get_state().tolist())
    responses = []
    for prompt in prompts:
        if _versions(backend) != versions:
            raise RuntimeError("Weights changed during rollout collection")
        sample = backend.read(policy_role, lambda view: view.sample_response(
            torch.tensor(prompt, dtype=torch.long, device=device), eos_id=eos_id,
            max_new_tokens=max_new_tokens, generator=generator))
        responses.append(SampledResponse(prompt_ids=prompt, **sample))
    if _versions(backend) != versions:
        raise RuntimeError("Weights changed during rollout collection")
    return RolloutBatch(tuple(responses), policy_role, eos_id, pad_id, max_new_tokens, vocab,
                        versions, before, bytes(generator.get_state().tolist()), backend)


def pack_rollout(rollout, *, device):
    """Right-pad model inputs; compact action masks depend only on lengths."""
    if not isinstance(rollout, RolloutBatch) or not rollout.responses:
        raise ValueError("Expected a nonempty owned rollout")
    if (not isinstance(rollout._owner, SharedQwen3)
            or rollout.policy_role not in rollout._owner._trainable_roles
            or rollout.vocab_size != rollout._owner._base.config.vocab_size):
        raise ValueError("Rollout owner, policy role and vocabulary must match the loaded backend")
    if (type(rollout.vocab_size) is not int or rollout.vocab_size < 1
            or type(rollout.max_new_tokens) is not int or rollout.max_new_tokens < 1
            or any(type(t) is not int or not 0 <= t < rollout.vocab_size for t in (rollout.eos_id, rollout.pad_id))
            or not isinstance(rollout.policy_role, str) or not rollout.policy_role):
        raise ValueError("Invalid explicit rollout vocabulary, role, horizon or special-token metadata")
    for response in rollout.responses:
        if (not isinstance(response, SampledResponse)
                or any(type(ids) is not tuple or any(type(t) is not int or not 0 <= t < rollout.vocab_size for t in ids)
                       for ids in (response.prompt_ids, response.action_ids))):
            raise ValueError("Every prompt/action ID must be an exact in-vocabulary integer")
        if any(type(stats) is not tuple or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in stats)
               for stats in (response.old_logprobs, response.old_values)):
            raise ValueError("Old policy statistics must be finite real numbers")
    full_length = max(len(r.prompt_ids) + len(r.action_ids) for r in rollout.responses)
    action_length = max(len(r.action_ids) for r in rollout.responses)
    batch = len(rollout.responses)
    ids = torch.full((batch, full_length), rollout.pad_id, dtype=torch.long, device=device)
    attention, labels = torch.zeros_like(ids), torch.full_like(ids, -100)
    mask = torch.zeros((batch, action_length), dtype=torch.bool, device=device)
    old_logprobs = torch.zeros(mask.shape, dtype=torch.float32, device=device)
    old_values = torch.zeros_like(old_logprobs)
    for index, response in enumerate(rollout.responses):
        p, n = len(response.prompt_ids), len(response.action_ids)
        if p < 1 or not 1 <= n <= rollout.max_new_tokens or n != len(response.old_logprobs) or n != len(response.old_values):
            raise ValueError("Invalid prompt/action/statistic lengths")
        eos = response.action_ids[-1] == rollout.eos_id
        if rollout.eos_id in response.action_ids[:-1] or response.termination != ("eos" if eos else "max_new_tokens"):
            raise ValueError("Recorded termination does not match actual sampled EOS")
        if not eos and n != rollout.max_new_tokens:
            raise ValueError("Non-EOS response must reach its recorded horizon")
        ids[index, :p + n] = torch.tensor(response.prompt_ids + response.action_ids, device=device)
        attention[index, :p + n] = 1
        labels[index, p:p + n] = ids[index, p:p + n]
        mask[index, :n] = True
        old_logprobs[index, :n] = torch.tensor(response.old_logprobs, device=device)
        old_values[index, :n] = torch.tensor(response.old_values, device=device)
    if not bool(torch.isfinite(old_logprobs).all()) or not bool(torch.isfinite(old_values).all()) or bool((old_logprobs[mask] > 0).any()):
        raise ValueError("Old policy statistics must be finite log probabilities and values")
    return dict(input_ids=ids, attention_mask=attention, labels=labels,
                action_mask=mask, old_logprobs=old_logprobs, old_values=old_values)


def compact_actions(full_shifted_scores, rollout):
    """Select actual action positions from model scores whose column t predicts t+1."""
    if full_shifted_scores.ndim != 2 or len(full_shifted_scores) != len(rollout.responses):
        raise ValueError("Shifted scores must have one row per rollout response")
    width = max(len(r.action_ids) for r in rollout.responses)
    rows = []
    for index, response in enumerate(rollout.responses):
        start, count = len(response.prompt_ids) - 1, len(response.action_ids)
        if start + count > full_shifted_scores.shape[1]:
            raise ValueError("Shifted score sequence does not contain all sampled actions")
        rows.append(F.pad(full_shifted_scores[index, start:start + count], (0, width - count)))
    return torch.stack(rows)


def score_frozen_rollout(backend, rollout, *, reference_role, reward_role, device, chunk_tokens=32):
    """Serial no-grad reference/RM scoring before policy updates; microbatch one.

    All model and scalar-head version counters must still equal sampling time.
    Once policy optimization begins these frozen statistics must be reused,
    not recomputed with a later policy. Persistent checkpoints also need hashes.
    """
    if rollout._owner is not backend or _versions(backend) != rollout.model_versions:
        raise RuntimeError("Score a rollout only on its unchanged owning backend")
    if len({rollout.policy_role, reference_role, reward_role}) != 3:
        raise ValueError("Policy, reference and reward must be separate adapter roles")
    backend.require_frozen_role(reference_role)
    backend.require_frozen_role(reward_role)
    if any("reward" in names for names in backend._trainable_heads.values()):
        raise ValueError("PPO reward head must be frozen for every training role")
    packed = pack_rollout(rollout, device=device)
    width = packed["action_mask"].shape[1]
    references, rewards = [], []
    for index, response in enumerate(rollout.responses):
        length = len(response.prompt_ids) + len(response.action_ids)
        single = {k: packed[k][index:index + 1, :length]
                  for k in ("input_ids", "attention_mask", "labels")}
        scores = backend.read(reference_role, lambda view: view.token_scores(**single, chunk_tokens=chunk_tokens))
        start, count = len(response.prompt_ids) - 1, len(response.action_ids)
        references.append(F.pad(scores.logprobs[0, start:start + count].clone(), (0, width - count)))
        rewards.append(backend.read(reward_role, lambda view: view.rewards(**single)).clone())
        if _versions(backend) != rollout.model_versions:
            raise RuntimeError("Weights changed during frozen rollout scoring")
    reference = torch.stack(references).detach()
    reward = torch.cat(rewards).detach()
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite frozen rollout scores")
    return reference, reward
