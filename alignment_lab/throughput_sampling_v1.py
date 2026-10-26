"""Explicit, opt-in scalar-copy candidate; no loader, dispatch, or global patch.

The legacy numerical operations and prompt-major random draw order are retained.
This module makes no equivalence or speed claim until the caller runs its gates.
An already loaded, idle backend can be adopted with ``make_sampling_backend``.
Production callers must then use only the returned owner. Benchmarking may use
both owners sequentially, never concurrently or inside one another's callbacks.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict
import hashlib
import json
import statistics
import time

import torch
from torch import Tensor

from .shared_qwen import ModelView, SharedQwen3


IMPLEMENTATION = "alignlab-deferred-sampling-scalars-v1"
GUARD_MODES = ("original", "cached_topology")
_MISSING = object()


def _same_bindings(actual, expected):
    return (len(actual) == len(expected)
            and all(actual.get(name, _MISSING) is value for name, value in expected))


class _ReadTopology:
    """Cache traversal results, but reject any changed registered topology.

    Per-token mode, adapter and trainability checks remain. Registration maps
    are checked by key/object identity, so replacing or adding a child/parameter
    cannot hide behind a cached module/parameter tuple. This extra topology work
    can cost more than it saves; it is a separately benchmarked opt-in candidate.
    """

    def __init__(self, owner):
        self.model = owner._model
        self.heads = tuple(owner._heads.items())
        modules, seen = [], set()
        for root in (owner._model, *owner._heads.values()):
            for module in root.modules():
                if id(module) not in seen:
                    modules.append(module)
                    seen.add(id(module))
        self.modules = tuple((module, type(module), tuple(module._modules.items()),
                              tuple(module._parameters.items()), tuple(module._buffers.items()),
                              hasattr(module, "lora_A")) for module in modules)
        self.lora_modules = tuple(row[0] for row in self.modules if row[-1])
        self.parameters = tuple(owner._all_parameters())

    def verify(self, owner):
        if owner._phase != "read":
            raise RuntimeError("Cached topology is valid only inside its read callback")
        if owner._model is not self.model or not _same_bindings(owner._heads, self.heads):
            raise RuntimeError("Model/head topology changed inside its owning callback")
        if owner._model.training:
            raise RuntimeError("Model mode changed inside its owning callback")
        if any(head.training for head in owner._heads.values()):
            raise RuntimeError("Scalar head mode changed inside its owning callback")
        for module, cls, children, parameters, buffers, has_lora in self.modules:
            if (type(module) is not cls or not _same_bindings(module._modules, children)
                    or not _same_bindings(module._parameters, parameters)
                    or not _same_bindings(module._buffers, buffers)
                    or hasattr(module, "lora_A") != has_lora):
                raise RuntimeError("Registered model topology changed inside its owning callback")
            if module.training:
                raise RuntimeError("Submodule mode changed inside its owning callback")
        for module in self.lora_modules:
            if list(module.active_adapters) != [owner._active_role] or module.disable_adapters:
                raise RuntimeError("Adapter changed outside the guarded switch API")
        actual_grad_ids = {id(parameter) for parameter in self.parameters if parameter.requires_grad}
        if actual_grad_ids != owner._expected_grad_ids:
            raise RuntimeError("Parameter trainability changed inside the owning callback")


class ThroughputModelView(ModelView):
    def sample_response(self, prompt_ids, *, eos_id, max_new_tokens, generator, head="value"):
        """Legacy batch-one decoder with deferred logprob/value scalar copies.

        Keep both finite checks and the immediate action/EOS read in their
        original order. Only the two output-statistic host copies move to the
        end. Retained tensors are fresh no-grad scalar outputs, not hidden-state
        or vocabulary-sized views. A scalar-shape failure still fails closed.
        """
        self._check()
        if self._owner._phase != "read" or torch.is_grad_enabled():
            raise RuntimeError("Sampling requires an owned no-gradient read callback")
        vocab = self._owner._base.config.vocab_size
        if (not isinstance(prompt_ids, Tensor) or prompt_ids.dtype != torch.long
                or prompt_ids.ndim != 1 or prompt_ids.numel() == 0
                or bool(((prompt_ids < 0) | (prompt_ids >= vocab)).any())):
            raise ValueError("Prompt must be a nonempty unpadded int64 token vector")
        if type(eos_id) is not int or not 0 <= eos_id < vocab:
            raise ValueError("EOS must be a vocabulary token ID")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if len(prompt_ids) + max_new_tokens > self._owner._base.config.max_position_embeddings:
            raise ValueError("Prompt plus response limit exceeds the configured context")
        if not isinstance(generator, torch.Generator) or generator.device != prompt_ids.device:
            raise ValueError("Sampling needs an explicit generator on the input device")
        if head not in self._owner._trainable_heads.get(self._owner._active_role, set()):
            raise ValueError("Sampled value head must belong to the policy training role")
        cache, actions, logprobs, values = None, [], [], []
        current = prompt_ids[None, :]
        prefix_length = len(prompt_ids)
        for step in range(max_new_tokens):
            self._check()
            length = prefix_length + step
            positions = torch.arange(length - current.shape[1], length, device=current.device)[None, :]
            result = self._owner._base.model(input_ids=current,
                attention_mask=torch.ones((1, length), dtype=torch.long, device=current.device),
                position_ids=positions, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            if cache is None:
                raise RuntimeError("Decoder did not return the requested local KV cache")
            hidden = result.last_hidden_state[:, -1, :]
            logits = self._owner._base.lm_head(hidden).float()
            value = self._scalar(head, hidden)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Nonfinite policy logits or sampled state value")
            distribution = torch.log_softmax(logits, dim=-1)
            action = torch.multinomial(distribution.exp(), 1, generator=generator)
            actions.append(int(action.item()))
            logprob = distribution.gather(-1, action)
            if logprob.numel() != 1:
                raise RuntimeError("Sampled logprob must contain exactly one element")
            logprobs.append(logprob.detach())
            if value.numel() != 1:
                raise RuntimeError("Sampled value must contain exactly one element")
            values.append(value.detach())
            if actions[-1] == eos_id:
                break
            current = action
        self._check()
        old_logprobs = torch.stack(logprobs).reshape(-1).cpu().tolist()
        old_values = torch.stack(values).reshape(-1).cpu().tolist()
        if len(old_logprobs) != len(actions) or len(old_values) != len(actions):
            raise RuntimeError("Sampling statistics must have one scalar per actual action")
        return {"action_ids": tuple(actions), "old_logprobs": tuple(old_logprobs),
                "old_values": tuple(old_values),
                "termination": "eos" if actions[-1] == eos_id else "max_new_tokens"}


class ThroughputSharedQwen3(SharedQwen3):
    sampling_implementation = IMPLEMENTATION

    def __init__(self, *args, guard_mode="original", **kwargs):
        if guard_mode not in GUARD_MODES:
            raise ValueError("Explicit supported sampling guard mode required")
        self.sampling_guard_mode = guard_mode
        self._read_topology = None
        super().__init__(*args, **kwargs)

    def _assert_state(self):
        if self._phase == "read" and self._read_topology is not None:
            self._read_topology.verify(self)
        else:
            super()._assert_state()

    def read(self, role, callback):
        self._activate(role, training=False)
        self._phase = "read"
        try:
            # Validate before constructing a cache, and never reuse one across
            # callbacks or adapter switches. Original mode keeps the legacy
            # validation schedule; backward stays entirely inherited.
            if self.sampling_guard_mode == "cached_topology":
                super()._assert_state()
                self._read_topology = _ReadTopology(self)
            with torch.no_grad():
                return callback(ThroughputModelView(self, self._generation))
        finally:
            self._read_topology = None
            self._phase = "idle"


def make_sampling_backend(previous, *, guard_mode="original"):
    """Return an explicit new owner over identical model/head parameter objects.

    Call after loading and before constructing mappings/restoring a checkpoint.
    No tensor weights, optimizer state, RNG, or old global classes are replaced.
    Like the original constructor, this establishes idle frozen trainability.
    Existing optimizer objects retain their parameter references, but callers
    must follow their normal complete-checkpoint restore/admission procedure.
    """
    if not isinstance(previous, SharedQwen3) or previous._phase != "idle":
        raise ValueError("Adoption requires an idle SharedQwen3 owner")
    if any(parameter.grad is not None for parameter in previous._all_parameters()):
        raise RuntimeError("Cannot adopt a backend with pending gradients")
    before = tuple(id(parameter) for parameter in previous._all_parameters())
    result = ThroughputSharedQwen3(previous._model,
        trainable_roles=set(previous._trainable_roles), heads=dict(previous._heads),
        trainable_heads={role: tuple(heads) for role, heads in previous._trainable_heads.items()},
        guard_mode=guard_mode)
    if tuple(id(parameter) for parameter in result._all_parameters()) != before:
        raise RuntimeError("Sampling adoption changed parameter object identities")
    return result


def _response_digest(rollout):
    payload = json.dumps([asdict(response) for response in rollout.responses],
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def benchmark_sampling_equivalence(previous, prompts, *, role, eos_id, pad_id,
                                   max_new_tokens, generator_state, device,
                                   guard_mode="original", warmups=1, repeats=3,
                                   autocast_factory=contextlib.nullcontext,
                                   allow_cuda=False, clock=time.perf_counter):
    """Measure an already admitted model; never loads/dispatches or writes files.

    The caller supplies a pinned real model, fixed prompts and an exact saved
    generator state. CUDA requires explicit allow_cuda=True and prior external
    admission. Both variants replay the same state in alternating AB/BA order.
    Results are measurements only: the caller must freeze its acceptance margin,
    memory budget and repetition count before running this function.
    """
    from .rollout import sample_rollout

    if type(previous) is not SharedQwen3:
        raise ValueError("Legacy benchmark owner must be the original SharedQwen3 class")
    device = torch.device(device)
    if device.type == "cuda" and (not allow_cuda or device.index is None):
        raise ValueError("CUDA benchmark requires explicit indexed-device admission")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or CUDA benchmarking is supported")
    if type(warmups) is not int or warmups < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("Explicit nonnegative warmups and positive repeats required")
    if (not isinstance(generator_state, Tensor) or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1 or generator_state.device.type != "cpu"):
        raise ValueError("Generator state must be the exact one-dimensional CPU uint8 state")
    if any(parameter.device != device for parameter in previous._all_parameters()):
        raise ValueError("Benchmark model and explicit device must match")
    prompts = tuple(tuple(prompt) for prompt in prompts)
    initial_state = generator_state.clone()
    candidate = make_sampling_backend(previous, guard_mode=guard_mode)
    versions = tuple((id(parameter), parameter._version) for parameter in previous._all_parameters())

    def run(owner):
        generator = torch.Generator(device=device)
        generator.set_state(initial_state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        before = clock()
        with autocast_factory():
            rollout = sample_rollout(owner, role, prompts, eos_id=eos_id, pad_id=pad_id,
                max_new_tokens=max_new_tokens, generator=generator, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = clock() - before
        if versions != tuple((id(parameter), parameter._version) for parameter in previous._all_parameters()):
            raise RuntimeError("Benchmark changed model parameter identity/version")
        memory = ({"max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "max_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                  if device.type == "cuda" else None)
        return {"seconds": elapsed, "response_sha256": _response_digest(rollout), "cuda_memory": memory,
                "rng_before_sha256": hashlib.sha256(rollout.rng_before).hexdigest(),
                "rng_after_sha256": hashlib.sha256(rollout.rng_after).hexdigest(),
                "response_tokens": sum(len(response.action_ids) for response in rollout.responses)}

    warmup_results, trials = [], []
    for index in range(warmups + repeats):
        order = ("legacy", "candidate") if index % 2 == 0 else ("candidate", "legacy")
        observations = {name: run(previous if name == "legacy" else candidate) for name in order}
        equal = all(observations["legacy"][key] == observations["candidate"][key]
                    for key in ("response_sha256", "rng_before_sha256", "rng_after_sha256", "response_tokens"))
        row = {"order": list(order), "observations": observations, "exactly_equivalent": equal}
        (warmup_results if index < warmups else trials).append(row)
    baseline = statistics.median(row["observations"]["legacy"]["seconds"] for row in trials)
    optimized = statistics.median(row["observations"]["candidate"]["seconds"] for row in trials)
    return {"kind": "alignlab-sampling-throughput-measurement-v1", "implementation": IMPLEMENTATION,
            "guard_mode": guard_mode, "device": str(device), "prompt_count": len(prompts),
            "max_new_tokens": max_new_tokens, "warmups": warmup_results, "trials": trials,
            "all_exactly_equivalent": all(row["exactly_equivalent"] for row in (*warmup_results, *trials)),
            "legacy_median_seconds": baseline, "candidate_median_seconds": optimized,
            "measured_speedup": baseline / optimized if optimized > 0 else None,
            "parameter_check_scope": "object identities and version counters; independent content audit required",
            "performance_acceptance": "not_decided_by_this_function", "training_dispatched": False}
