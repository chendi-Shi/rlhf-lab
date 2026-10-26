"""Serial, guarded LoRA access to one frozen Qwen3 backbone.

This is a model-access primitive, not a trainer, generation loop, or GPU
capacity result. The caller loads a PEFT LoRA model and supplies explicit role
names; no model downloads, device selection, or environment changes happen here.
Training callbacks own one complete forward/backward before adapters can switch.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import torch
from torch import Tensor, nn

from .token_logprobs import completion_logprobs, completion_token_logprobs

T = TypeVar("T")


class ModelView:
    """Valid only inside its owner's synchronous read/backward callback."""

    def __init__(self, owner: "SharedQwen3", generation: int):
        self._owner = owner
        self._generation = generation

    def _check(self):
        if self._owner._phase == "idle" or self._generation != self._owner._generation:
            raise RuntimeError("Model view is no longer inside its owning callback")
        self._owner._assert_state()

    def hidden(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        self._check()
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64 [batch, sequence]")
        if attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device:
            raise ValueError("attention mask must match input shape and device")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention mask must be binary")
        valid = attention_mask.bool()
        if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("This scoring path requires nonempty right-padded sequences")
        positions = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
        # Calling the backbone avoids producing full-sequence vocabulary logits.
        return self._owner._base.model(input_ids=input_ids, attention_mask=attention_mask,
                                      position_ids=positions, use_cache=False).last_hidden_state

    def _validate_labels(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor):
        self._check()
        if labels.shape != input_ids.shape or labels.device != input_ids.device or labels.dtype != torch.long:
            raise ValueError("labels must be int64 on the input shape/device")
        valid = labels.ne(-100)
        if bool(valid[:, 0].any()) or bool((valid & ~attention_mask.bool()).any()):
            raise ValueError("Unpredicted first token and padding cannot be completion labels")
        if not torch.equal(labels[valid], input_ids[valid]):
            raise ValueError("Completion labels must be the actual input token IDs")
        if not bool(valid.any(-1).all()):
            raise ValueError("Every sequence needs at least one completion token")
        # Completion labels form the attended suffix, including real EOS if present.
        for row, attended in zip(valid, attention_mask.bool()):
            first = int(row.nonzero()[0])
            if not torch.equal(row[first:], attended[first:]):
                raise ValueError("Completion labels must form one contiguous attended suffix")

    def sequence_scores(self, input_ids, attention_mask, labels, *, chunk_tokens=32):
        self._validate_labels(input_ids, attention_mask, labels)
        return completion_logprobs(self.hidden(input_ids, attention_mask), self._owner._base.lm_head,
                                   labels, chunk_tokens=chunk_tokens)

    def token_scores(self, input_ids, attention_mask, labels, *, chunk_tokens=32):
        self._validate_labels(input_ids, attention_mask, labels)
        return completion_token_logprobs(self.hidden(input_ids, attention_mask),
                self._owner._base.lm_head, labels, chunk_tokens=chunk_tokens)

    def sample_response(self, prompt_ids, *, eos_id, max_new_tokens, generator, head="value"):
        """Sample one unpadded response with a callback-local KV cache.

        Raw temperature-one categorical sampling over the full vocabulary.
        EOS and horizon stops are distinguished; neither adds a synthetic token.
        The local cache never escapes this method or crosses an adapter switch.
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
            logprobs.append(float(distribution.gather(-1, action).item()))
            values.append(float(value.item()))
            if actions[-1] == eos_id:
                break
            current = action
        self._check()
        return {"action_ids": tuple(actions), "old_logprobs": tuple(logprobs),
                "old_values": tuple(values), "termination": "eos" if actions[-1] == eos_id else "max_new_tokens"}

    def token_scores_and_values(self, input_ids, attention_mask, labels, *, head="value", chunk_tokens=32):
        self._validate_labels(input_ids, attention_mask, labels)
        hidden = self.hidden(input_ids, attention_mask)
        scores = completion_token_logprobs(hidden, self._owner._base.lm_head, labels,
                                          chunk_tokens=chunk_tokens)
        # State t predicts action t+1: value indexing must match the log-prob axis.
        values = self._scalar(head, hidden[:, :-1])
        return scores, values.masked_fill(~scores.mask, 0)

    def rewards(self, input_ids, attention_mask, labels, *, head="reward"):
        self._validate_labels(input_ids, attention_mask, labels)
        hidden = self.hidden(input_ids, attention_mask)
        last = attention_mask.long().sum(-1) - 1
        rows = torch.arange(len(input_ids), device=input_ids.device)
        return self._scalar(head, hidden[rows, last])

    def _scalar(self, name: str, hidden: Tensor):
        self._check()
        head = self._owner._heads[name]
        return head(hidden.to(head.weight.dtype)).squeeze(-1)


class SharedQwen3:
    """Control adapter trainability and keep each graph on one adapter.

    `trainable_roles` is an explicit allowlist. A reference/RM scoring role must
    not be in it. All adapters must be plain LoRA with no trainable base biases,
    modules_to_save, DoRA, or dropout. A new instance/phase may have a different
    allowlist after SFT or RM is frozen. No optimizer is shared between roles.
    """

    def __init__(self, peft_model, *, trainable_roles: set[str], heads: Mapping[str, nn.Linear] | None = None,
                 trainable_heads: Mapping[str, tuple[str, ...]] | None = None):
        self._model = peft_model
        self._base = peft_model.get_base_model()
        if getattr(self._base.config, "model_type", None) != "qwen3":
            raise ValueError("This backbone path is verified only for Qwen3")
        configs = peft_model.peft_config
        if not trainable_roles.issubset(configs):
            raise ValueError("Unknown trainable adapter role")
        for cfg in configs.values():
            if (str(cfg.peft_type) not in {"LORA", "PeftType.LORA"} or cfg.bias != "none"
                    or cfg.modules_to_save or cfg.lora_dropout or getattr(cfg, "use_dora", False)):
                raise ValueError("Only plain LoRA, bias=none, no modules_to_save/DoRA/dropout is supported")
        if getattr(self._base.config, "attention_dropout", 0) != 0:
            raise ValueError("Attention dropout must be zero for consistent policy scoring")
        self._roles = set(configs)
        self._trainable_roles = set(trainable_roles)
        self._heads = dict(heads or {})
        if any(not isinstance(h, nn.Linear) or h.out_features != 1
               or h.in_features != self._base.config.hidden_size for h in self._heads.values()):
            raise ValueError("Reward/value heads must be explicit scalar Linear layers")
        self._trainable_heads = {role: set(names) for role, names in (trainable_heads or {}).items()}
        if not set(self._trainable_heads).issubset(self._trainable_roles):
            raise ValueError("Only trainable roles can have a scalar-head allowlist")
        if any(not names.issubset(self._heads) for names in self._trainable_heads.values()):
            raise ValueError("Unknown scalar head in the training allowlist")
        seen_head_params = set(map(id, peft_model.parameters()))
        for head in self._heads.values():
            current = set(map(id, head.parameters()))
            if seen_head_params & current:
                raise ValueError("Scalar heads and base model must have disjoint parameter objects")
            seen_head_params.update(current)
        self._parameters = {}
        seen = set()
        for role in sorted(self._roles):
            params = [p for name, p in peft_model.named_parameters()
                      if any(f".{kind}.{role}." in name for kind in ("lora_A", "lora_B"))]
            if not params or any(id(p) in seen for p in params):
                raise ValueError("Adapter parameter sets must be nonempty and disjoint")
            self._parameters[role] = params
            seen.update(map(id, params))
        self._phase = "idle"
        self._generation = 0
        self._active_role = None
        self._expected_grad_ids = set()
        peft_model.requires_grad_(False)
        for head in self._heads.values():
            head.requires_grad_(False)
        # Non-reentrant checkpointing works with frozen embeddings; training mode
        # remains enabled inside backward, unlike eval-mode pseudo-checkpointing.
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._base.config.use_cache = False

    def parameters_for(self, role: str, *, heads: tuple[str, ...] = ()) -> list[nn.Parameter]:
        if role not in self._trainable_roles:
            raise ValueError("Frozen or unknown role cannot own an optimizer")
        if len(heads) != len(set(heads)):
            raise ValueError("Duplicate scalar heads")
        if not set(heads).issubset(self._trainable_heads.get(role, set())):
            raise ValueError("Requested scalar heads are not trainable for this adapter role")
        params = list(self._parameters[role])
        for name in heads:
            params.extend(self._heads[name].parameters())
        return params

    def require_frozen_role(self, role: str) -> None:
        """Reject an unknown role or a role eligible for updates in this phase."""
        if role not in self._roles or role in self._trainable_roles:
            raise ValueError("Reference scoring requires a known frozen adapter role")

    def frozen_role_versions(self, role: str) -> tuple[tuple[int, int], ...]:
        """In-process guard for base + this adapter; persistent caches need SHA.

        Other adapters and scalar heads are excluded so policy updates do not
        invalidate frozen-reference scores. Tensor version counters only detect
        tracked in-place mutations, not arbitrary external storage corruption.
        """
        self.require_frozen_role(role)
        adapter_ids = {id(p) for params in self._parameters.values() for p in params}
        frozen = [p for p in self._model.parameters() if id(p) not in adapter_ids]
        frozen += self._parameters[role]
        return tuple((id(p), p._version) for p in frozen)

    def _activate(self, role, *, training, heads=()):
        if self._phase != "idle":
            raise RuntimeError("Cannot switch adapter during an active forward/backward or read callback")
        if role not in self._roles:
            raise ValueError("Unknown adapter role")
        # Never silently erase accumulated gradients to switch adapters.
        if self._active_role != role and any(p.grad is not None for p in self._all_parameters()):
            raise RuntimeError("Clear completed optimizer gradients before switching adapter")
        selected = self.parameters_for(role, heads=heads) if training else []
        self._model.set_adapter(role)
        self._model.requires_grad_(False)
        for head in self._heads.values():
            head.requires_grad_(False)
        for param in selected:
            param.requires_grad_(True)
        self._model.train(training)
        for head in self._heads.values():
            head.train(training)
        self._active_role = role
        self._expected_grad_ids = set(map(id, selected))
        self._generation += 1
        return selected

    def _assert_state(self):
        if self._phase not in {"read", "backward"}:
            raise RuntimeError("Model callback is no longer usable")
        expected_training = self._phase == "backward"
        if self._model.training != expected_training:
            raise RuntimeError("Model mode changed inside its owning callback")
        if any(h.training != expected_training for h in self._heads.values()):
            raise RuntimeError("Scalar head mode changed inside its owning callback")
        for module in self._model.modules():
            if module.training != expected_training:
                raise RuntimeError("Submodule mode changed inside its owning callback")
            if hasattr(module, "lora_A"):
                if list(module.active_adapters) != [self._active_role] or module.disable_adapters:
                    raise RuntimeError("Adapter changed outside the guarded switch API")
        actual_grad_ids = {id(p) for p in self._all_parameters() if p.requires_grad}
        if actual_grad_ids != self._expected_grad_ids:
            raise RuntimeError("Parameter trainability changed inside the owning callback")

    def _all_parameters(self):
        yield from self._model.parameters()
        for head in self._heads.values():
            yield from head.parameters()

    def read(self, role: str, callback: Callable[[ModelView], T]) -> T:
        self._activate(role, training=False)
        self._phase = "read"
        try:
            with torch.no_grad():
                return callback(ModelView(self, self._generation))
        finally:
            self._phase = "idle"

    def backward(self, role: str, optimizer, loss_callback: Callable[[ModelView], Tensor],
                 *, heads: tuple[str, ...] = ()) -> Tensor:
        """Accumulate one loss's gradients; caller controls zero_grad/step.

        The callback returns one finite scalar (already divided by the intended
        accumulation denominator). Step and clear gradients before switching
        roles. This method does not claim a complete optimizer update or train run.
        """
        selected = self._activate(role, training=True, heads=heads)
        supplied = [p for group in optimizer.param_groups for p in group["params"]]
        if len(supplied) != len(set(map(id, supplied))) or set(map(id, supplied)) != set(map(id, selected)):
            raise ValueError("Optimizer parameters must exactly match the selected adapter and heads")
        self._phase = "backward"
        versions = [p._version for p in selected]
        touched = set()
        handles = []
        for parameter in selected:
            def record_gradient(gradient, key=id(parameter)):
                touched.add(key)
                return gradient
            handles.append(parameter.register_hook(record_gradient))
        try:
            loss = loss_callback(ModelView(self, self._generation))
            if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise ValueError("Callback must return a finite differentiable scalar loss")
            if versions != [p._version for p in selected]:
                raise RuntimeError("Parameters changed while a forward graph was active")
            self._assert_state()
            loss.backward()
            if not touched:
                raise RuntimeError("Loss graph did not reach any selected adapter/head parameters")
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in selected):
                raise FloatingPointError("Nonfinite adapter/head gradients")
            selected_ids = set(map(id, selected))
            if any(p.grad is not None for p in self._all_parameters() if id(p) not in selected_ids):
                raise RuntimeError("Frozen parameters unexpectedly have gradients")
            result = loss.detach()
        except BaseException:
            # A callback can retain an incomplete graph after an exception. Do
            # not allow this instance to switch adapters and reuse that graph.
            self._phase = "failed"
            raise
        else:
            self._phase = "idle"
            return result
        finally:
            for handle in handles:
                handle.remove()
