"""Completion-only log probabilities without full-sequence vocabulary logits.

This provides sequence scores for SFT/DPO and token scores for PPO, not
a trainer or a CUDA capacity result. Callers must keep the active adapter,
head mode and weights unchanged through forward and backward.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SequenceLogProbs(NamedTuple):
    sums: Tensor
    counts: Tensor


class TokenLogProbs(NamedTuple):
    # Column t scores the label at original position t+1 using hidden state t.
    logprobs: Tensor
    mask: Tensor
    counts: Tensor


def completion_token_logprobs(
    hidden_states: Tensor,
    lm_head: nn.Module,
    labels: Tensor,
    *,
    chunk_tokens: int = 32,
    recompute_logits: bool = True,
) -> TokenLogProbs:
    """Score unmasked next-token labels using a head with raw policy logits.

    `hidden_states[b, t]` predicts `labels[b, t + 1]`. Labels for prompt,
    padding and other unscored tokens must be -100. Counts include EOS only
    when EOS is explicitly present and unmasked in labels. There is no length
    normalization, temperature transform, top-k or top-p truncation here.

    Only completion positions pass through lm_head. Recomputing each chunk's
    logits during backward avoids retaining vocabulary-sized activations for
    all chunks; disabling recomputation is for reference/diagnostic use and
    does not preserve that memory bound. The caller must obtain hidden states without first computing
    the full model's vocabulary logits, or that memory saving is defeated.
    """
    if hidden_states.ndim != 3 or labels.shape != hidden_states.shape[:2]:
        raise ValueError("Expected hidden states [batch, time, width] and labels [batch, time]")
    if hidden_states.shape[1] < 2 or chunk_tokens < 1:
        raise ValueError("Need at least two time positions and a positive chunk size")
    if labels.dtype != torch.long or labels.device != hidden_states.device:
        raise ValueError("Labels must be int64 on the hidden-state device")
    shifted_labels = labels[:, 1:]
    positions = shifted_labels.ne(-100).nonzero(as_tuple=False)
    batch_indices = positions[:, 0]
    counts = torch.bincount(batch_indices, minlength=hidden_states.shape[0])
    if bool((counts == 0).any()):
        raise ValueError("Each sequence must have at least one unmasked predicted completion token")
    selected_hidden = hidden_states[:, :-1][positions[:, 0], positions[:, 1]]
    selected_labels = shifted_labels[positions[:, 0], positions[:, 1]]
    chunks = []

    def token_scores(rows: Tensor, targets: Tensor) -> Tensor:
        return -F.cross_entropy(lm_head(rows).float(), targets, reduction="none")

    for start in range(0, positions.shape[0], chunk_tokens):
        end = start + chunk_tokens
        rows = selected_hidden[start:end]
        targets = selected_labels[start:end]
        if recompute_logits and torch.is_grad_enabled():
            scores = checkpoint(token_scores, rows, targets, use_reentrant=False)
        else:
            scores = token_scores(rows, targets)
        chunks.append(scores)
    # Retain only scalar token scores, never concatenate vocabulary logits.
    # Padding and prompt positions are exactly zero and carry no loss gradient.
    flat_positions = batch_indices * shifted_labels.shape[1] + positions[:, 1]
    token_scores_flat = torch.zeros(shifted_labels.numel(), dtype=torch.float32, device=hidden_states.device)
    logprobs = token_scores_flat.index_copy(0, flat_positions, torch.cat(chunks)).reshape_as(shifted_labels)
    return TokenLogProbs(logprobs, shifted_labels.ne(-100), counts)


def completion_logprobs(
    hidden_states: Tensor,
    lm_head: nn.Module,
    labels: Tensor,
    *,
    chunk_tokens: int = 32,
    recompute_logits: bool = True,
) -> SequenceLogProbs:
    """Unnormalized completion sums; same shift/mask rules as token scoring."""
    result = completion_token_logprobs(hidden_states, lm_head, labels,
                                     chunk_tokens=chunk_tokens, recompute_logits=recompute_logits)
    return SequenceLogProbs(result.logprobs.sum(-1), result.counts)
