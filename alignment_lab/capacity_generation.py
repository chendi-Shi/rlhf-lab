"""Greedy EOS-aware and explicitly forced-continuation capacity probes.

Forced continuation deliberately ignores generated EOS. It is a KV/shape
stress workload, not a conversational sample or a quality measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import torch


def greedy_capacity_probe(backend, role, prompt_ids, *, eos_id, max_new_tokens=2048,
                          ignore_eos=False, device, stop_requested=None,
                          deadline_epoch=None, clock=time.time):
    """Generate exact original action IDs inside one owned read callback.

    Project only the final hidden row to vocabulary logits. The decoder receives
    the prompt once, then one action at a time with an entirely attended cache.
    A final decoder-only flush includes the final action in KV (no extra action).
    STOP/deadline is checked before every decoder call, including that flush.
    """
    if (type(prompt_ids) is not tuple or not prompt_ids
            or any(type(t) is not int for t in prompt_ids)):
        raise ValueError("Prompt requires a nonempty tuple of exact token IDs")
    if type(max_new_tokens) is not int or max_new_tokens < 1 or type(ignore_eos) is not bool:
        raise ValueError("Positive integer horizon and boolean ignore_eos required")
    if deadline_epoch is not None and (type(deadline_epoch) not in (int, float)
            or not math.isfinite(deadline_epoch) or deadline_epoch <= 0):
        raise ValueError("Deadline must be a finite positive epoch")
    if not callable(clock) or (stop_requested is not None and not callable(stop_requested)):
        raise ValueError("Clock and optional STOP predicate must be callable")
    if backend._phase != "idle" or any(p.grad is not None for p in backend._all_parameters()):
        raise RuntimeError("Generation requires idle backend and cleared gradients")
    config = backend._base.config
    if (type(eos_id) is not int or not 0 <= eos_id < config.vocab_size
            or any(not 0 <= t < config.vocab_size for t in prompt_ids)):
        raise ValueError("Prompt/EOS outside actual model vocabulary")
    if len(prompt_ids) + max_new_tokens > config.max_position_embeddings:
        raise ValueError("Prompt plus horizon exceeds configured context")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"} or (target.type == "cuda" and target.index is None):
        raise ValueError("Use cpu or an explicit indexed CUDA device")
    if backend._base.get_input_embeddings().weight.device != target:
        raise ValueError("Input device must match the loaded embedding device")
    versions = tuple((id(p), p._version) for p in backend._all_parameters())
    started = time.perf_counter()

    def boundary():
        if stop_requested is not None:
            result = stop_requested()
            if type(result) is not bool:
                raise ValueError("STOP predicate must return bool")
            if result:
                return "stop_requested"
        current = clock()
        if type(current) not in (int, float) or not math.isfinite(current):
            raise ValueError("Clock must return finite epoch seconds")
        return "deadline" if deadline_epoch is not None and current >= deadline_epoch else None

    def generate(view):
        view._check()
        if backend._phase != "read" or torch.is_grad_enabled():
            raise RuntimeError("Generation requires no-gradient read ownership")
        cache, actions, calls, max_cache = None, [], 0, 0
        current = torch.tensor([prompt_ids], dtype=torch.long, device=target)
        reason, flushed = None, False

        def decode(length):
            nonlocal cache, calls, max_cache
            view._check()
            positions = torch.arange(length - current.shape[1], length, device=target)[None, :]
            result = backend._base.model(input_ids=current,
                attention_mask=torch.ones((1, length), dtype=torch.long, device=target),
                position_ids=positions, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            if cache is None or not hasattr(cache, "get_seq_length"):
                raise RuntimeError("Decoder must return an inspectable local KV cache")
            actual = int(cache.get_seq_length())
            if actual != length:
                raise RuntimeError("KV cache length differs from attended token count")
            max_cache = max(max_cache, actual)
            calls += 1
            return result.last_hidden_state[:, -1, :]

        for step in range(max_new_tokens):
            reason = boundary()
            if reason:
                break
            hidden = decode(len(prompt_ids) + step)
            logits = backend._base.lm_head(hidden).float()
            if logits.shape != (1, config.vocab_size) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Expected finite single-position vocabulary logits")
            token = int(logits.argmax(dim=-1).item())
            actions.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=target)
            if token == eos_id and not ignore_eos:
                reason = "eos"
                break
        else:
            reason = "forced_max_new_tokens" if ignore_eos else "max_new_tokens"
        if actions and reason not in {"stop_requested", "deadline"}:
            interrupt = boundary()
            if interrupt:
                reason = interrupt
            else:
                decode(len(prompt_ids) + len(actions))
                flushed = True
        view._check()
        return {"action_ids": tuple(actions), "termination": reason,
            "generated_tokens": len(actions), "generated_eos_count": actions.count(eos_id),
            "max_cache_tokens": max_cache, "decoder_calls": calls,
            "final_action_flushed_to_kv": flushed}

    result = backend.read(role, generate)
    if versions != tuple((id(p), p._version) for p in backend._all_parameters()):
        raise RuntimeError("Parameters changed during read-only capacity generation")
    if any(p.grad is not None or p.requires_grad for p in backend._all_parameters()):
        raise RuntimeError("Generation left gradients or trainable parameters")
    result.update({"schema_version": 1, "capacity_only": True, "quality_evaluation": False,
        "role": role, "prompt_ids": prompt_ids, "prompt_tokens": len(prompt_ids),
        "max_new_tokens": max_new_tokens, "eos_id": eos_id, "ignore_eos": ignore_eos,
        "mode": "forced_continuation_kv_stress" if ignore_eos else "eos_aware_greedy",
        "sampling": "argmax_full_vocabulary_temperature_one", "attention_all_true": True,
        "synthetic_eos_added": False, "elapsed_seconds": time.perf_counter() - started,
        "deadline_epoch": deadline_epoch,
        "tokens_sha256": hashlib.sha256(json.dumps([prompt_ids, result["action_ids"]],
                                        separators=(",", ":")).encode()).hexdigest()})
    return result
