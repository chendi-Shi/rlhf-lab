"""Finite-answer PPO mathematics on compact action tensors [batch, actions].

The boolean mask is a nonempty left-aligned contiguous prefix for each row.
Action 0 is the first generated token: prompts have already been removed and
old_values[:, t] is the value of the state *before* action t. This module does
not shift token positions, add EOS, generate responses, or train a model.

Finite-answer horizon protocol: the caller records real EOS and reaching
max_new_tokens as distinct termination reasons. BOTH end this finite response
episode, with zero bootstrap after its last real generated action. A length
cap does not create an EOS token or an unagreed missing-EOS reward penalty.

Old-policy/reference log probabilities, old values, rewards, advantages and
returns must be detached. Padding is excluded before arithmetic, so arbitrary
padding values (including NaN) do not affect results or gradients. Half/bfloat16
inputs are promoted to float32 for objective arithmetic. No CUDA/device setup,
optimizer operations, entropy bonus, or implicit gradient clipping occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import Tensor


@dataclass(frozen=True)
class PPOResult:
    """Differentiable sums and one effective-batch-normalized contribution.

    value_sum already includes the conventional 0.5. Exactly once:
    objective_contribution = (policy_sum + value_coef * value_sum) /
                             token_denominator.
    Diagnostics are detached local valid-token means, not additive objectives.
    """

    policy_sum: Tensor
    value_sum: Tensor
    objective_contribution: Tensor
    valid_tokens: int
    token_denominator: int
    diagnostics: dict[str, Tensor]


def _mask(mask):
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("mask must be boolean [batch, actions]")
    if mask.shape[0] < 1 or mask.shape[1] < 1:
        raise ValueError("mask must have nonempty batch and action dimensions")
    if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("Each mask row must be a nonempty left-aligned contiguous action prefix")
    return int(mask.sum().item())


def _scalar(value, name, *, lower=0.0, upper=None, strict_lower=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if value < lower or (strict_lower and value == lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its valid range")
    return value


def _denominator(value, count):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < count:
        raise ValueError("token_denominator must be an integer effective-batch token count >= local valid tokens")
    if value > torch.iinfo(torch.int64).max:
        raise ValueError("token_denominator exceeds a representable token count")
    return int(value)


def _action(value, mask, name, *, gradients=False):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.shape != mask.shape or value.device != mask.device:
        raise ValueError(f"{name} must match the mask shape and device")
    if value.requires_grad != gradients:
        required = "require gradients" if gradients else "be detached (requires_grad=False)"
        raise ValueError(f"{name} must {required}")
    if not bool(torch.isfinite(value[mask]).all()):
        raise FloatingPointError(f"{name} contains nonfinite valid actions")
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    # Mask BEFORE subtraction/exp/square, not only when reducing the result.
    return torch.where(mask, value, 0.0)


def _common_dtype(*values):
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    return tuple(value.to(dtype=dtype) for value in values)


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("Nonfinite intermediate/result in PPO arithmetic; no implicit clipping is applied")


def shaped_rewards(old_logprobs, reference_logprobs, sequence_rewards, mask, kl_coef):
    """Detached -kl_coef*(old_logp-ref_logp), plus RM score at last action only.

    The per-action sampled log ratio is signed; individual samples need not be
    nonnegative KL estimates. This is frozen-reference reward shaping, distinct
    from the new-vs-old policy diagnostics in clipped_ppo_loss.
    """
    _mask(mask)
    coefficient = _scalar(kl_coef, "kl_coef")
    old = _action(old_logprobs, mask, "old_logprobs")
    reference = _action(reference_logprobs, mask, "reference_logprobs")
    if (not isinstance(sequence_rewards, Tensor) or not sequence_rewards.is_floating_point()
            or sequence_rewards.shape != (mask.shape[0],) or sequence_rewards.device != mask.device):
        raise ValueError("sequence_rewards must be floating-point [batch] on the mask device")
    if sequence_rewards.requires_grad:
        raise ValueError("sequence_rewards must be detached (requires_grad=False)")
    if not bool(torch.isfinite(sequence_rewards).all()):
        raise FloatingPointError("sequence_rewards must be finite")
    scores = sequence_rewards.float() if sequence_rewards.dtype in (torch.float16, torch.bfloat16) else sequence_rewards
    with torch.no_grad():
        old, reference, scores = _common_dtype(old, reference, scores)
        reward = -coefficient * (old - reference)
        last = mask.sum(-1) - 1
        reward = reward.scatter_add(1, last[:, None], scores[:, None])
        reward = reward.masked_fill(~mask, 0)
        _finite(reward)
        return reward


def masked_gae(rewards, old_values, mask, gamma, lam):
    """Detached GAE and unwhitened lambda returns with terminal bootstrap zero.

    delta_t = reward_t + gamma * V_(t+1) - V_t; both next-value and next-GAE
    terms are zero beyond the last valid action, for EOS and length-cap ends.
    returns = raw_advantages + old_values. Whiten advantages separately so
    normalizing the policy signal never changes critic targets.
    """
    _mask(mask)
    gamma = _scalar(gamma, "gamma", upper=1.0)
    lam = _scalar(lam, "lam", upper=1.0)
    rewards = _action(rewards, mask, "rewards")
    old_values = _action(old_values, mask, "old_values")
    with torch.no_grad():
        rewards, old_values = _common_dtype(rewards, old_values)
        advantages = torch.zeros_like(rewards)
        carry = torch.zeros_like(rewards[:, 0])
        for t in range(mask.shape[1] - 1, -1, -1):
            if t + 1 < mask.shape[1]:
                next_value = old_values[:, t + 1]
                next_carry = torch.where(mask[:, t + 1], carry, 0.0)
            else:
                next_value, next_carry = torch.zeros_like(carry), torch.zeros_like(carry)
            delta = rewards[:, t] + gamma * next_value - old_values[:, t]
            carry = torch.where(mask[:, t], delta + gamma * lam * next_carry, 0.0)
            advantages[:, t] = carry
        returns = torch.where(mask, advantages + old_values, 0.0)
        _finite(advantages, returns)
        return advantages, returns


def whiten_advantages(advantages, mask, epsilon=1e-8):
    """Detached global-rollout whitening using valid-token population variance.

    Call once on the whole rollout, BEFORE slicing optimizer microbatches. A
    singleton or constant rollout yields zero centered advantages, never NaN.
    This function does not whiten rewards or mutate advantages/critic returns.
    """
    _mask(mask)
    epsilon = _scalar(epsilon, "epsilon", strict_lower=True)
    values = _action(advantages, mask, "advantages")
    with torch.no_grad():
        selected = values[mask]
        mean = selected.mean()
        variance = (selected - mean).square().mean()
        normalized = (values - mean) * torch.rsqrt(variance + epsilon)
        result = normalized.masked_fill(~mask, 0)
        _finite(variance, result)
        return result


def clipped_ppo_loss(new_logprobs, new_values, old_logprobs, old_values, advantages,
                     returns, mask, policy_clip, value_clip, value_coef, token_denominator):
    """Clipped policy/value sums with a single caller-owned token denominator.

    Policy: max(-A*r, -A*clip(r, 1-policy_clip, 1+policy_clip)).
    Value: 0.5*max((V-R)^2, (V_old+clip(V-V_old,+/-value_clip)-R)^2).
    Old statistics remain fixed across all PPO epochs for this rollout. New
    logprobs/values must be differentiable; this function does not backward.
    """
    count = _mask(mask)
    denominator = _denominator(token_denominator, count)
    policy_clip = _scalar(policy_clip, "policy_clip", upper=1.0)
    value_clip = _scalar(value_clip, "value_clip")
    value_coef = _scalar(value_coef, "value_coef")
    new_lp = _action(new_logprobs, mask, "new_logprobs", gradients=True)
    new_v = _action(new_values, mask, "new_values", gradients=True)
    old_lp = _action(old_logprobs, mask, "old_logprobs")
    old_v = _action(old_values, mask, "old_values")
    adv = _action(advantages, mask, "advantages")
    targets = _action(returns, mask, "returns")
    new_lp, new_v, old_lp, old_v, adv, targets = _common_dtype(new_lp, new_v, old_lp, old_v, adv, targets)
    log_ratio = new_lp - old_lp
    ratio = torch.exp(log_ratio)
    _finite(log_ratio, ratio)
    clipped_ratio = ratio.clamp(1.0 - policy_clip, 1.0 + policy_clip)
    policy_terms = torch.maximum(-adv * ratio, -adv * clipped_ratio)
    value_delta = new_v - old_v
    clipped_values = old_v + value_delta.clamp(-value_clip, value_clip)
    value_terms = 0.5 * torch.maximum((new_v - targets).square(), (clipped_values - targets).square())
    policy_sum = policy_terms[mask].sum()
    value_sum = value_terms[mask].sum()
    contribution = (policy_sum + value_coef * value_sum) / denominator
    _finite(policy_terms, value_terms, policy_sum, value_sum, contribution)
    with torch.no_grad():
        lr = log_ratio.detach()[mask]
        diagnostics = {
            "policy_mean": policy_sum.detach() / count,
            "value_mean": value_sum.detach() / count,
            "ratio_mean": ratio.detach()[mask].mean(),
            "policy_clip_fraction": (torch.abs(ratio.detach()[mask] - 1.0) > policy_clip).to(ratio.dtype).mean(),
            "value_clip_fraction": (torch.abs(value_delta.detach()[mask]) > value_clip).to(new_v.dtype).mean(),
            # k3 estimator of KL(old || new); old-policy samples, no reference.
            "approx_kl": (torch.expm1(lr) - lr).mean(),
            "sampled_policy_kl": (-lr).mean(),
            "advantage_mean": adv[mask].mean(),
            "return_mean": targets[mask].mean(),
        }
        _finite(*diagnostics.values())
    return PPOResult(policy_sum, value_sum, contribution, count, denominator, diagnostics)
