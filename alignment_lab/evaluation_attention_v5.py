"""RM-only full-context attention with bounded query working storage.

The existing Qwen mask builder, projections, Q/K norms, RoPE and output
projection are retained. Only owned reward-model attention instances change
dispatch: explicit repeated KV -> efficient SDPA when supported, otherwise
128-query blocks against ALL keys using PyTorch's FP32-intermediate math SDPA.
No global attention registry, weight, dtype, input or CUDA allocator limit changes.
"""
from __future__ import annotations

import copy
from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb
from .evaluation_attention_receipt_v5 import KIND, QUERY_BLOCK_TOKENS


CONTRACT = {
    "kind": KIND, "scope": "reward_only", "query_block_tokens": QUERY_BLOCK_TOKENS,
    "preferred_backend": "torch_efficient_sdpa", "enable_gqa": False,
    "unsupported_backend": "torch_math_sdpa_query_blocks_full_kv",
    "fallback_on_execution_error": False, "input_truncation": False,
    "mask_builder": "unchanged_qwen3", "weights_and_precision_changed": False,
    "sdpa_input_cast": "existing_bfloat16_autocast_at_sdpa_boundary",
}


def _sdpa_autocast_inputs(query, key, value):
    # The accepted loader deliberately keeps all RMSNorm weights in FP32.
    # Installed Qwen3 therefore produces FP32 Q/K after norm/RoPE while V is
    # BF16. Original F.sdpa casts these at its autocast dispatcher. Materialize
    # that SAME cast before capability inspection, without moving it before
    # norm/RoPE or changing any parameter/buffer precision.
    if torch.is_autocast_enabled(query.device.type):
        target = torch.get_autocast_dtype(query.device.type)
        if target != torch.bfloat16 or any(t.dtype not in (torch.float32, torch.bfloat16)
                                          for t in (query, key, value)):
            raise ValueError("Reward attention preserves only the accepted BF16 autocast dispatch")
        return tuple(t.to(dtype=target) for t in (query, key, value))
    return query, key, value


def _validate_tensors(query, key, value, attention_mask):
    if (query.ndim != 4 or key.ndim != 4 or value.ndim != 4
            or key.shape != value.shape or query.shape[0] != key.shape[0]
            or query.shape[2:] != key.shape[2:] or min(query.shape) <= 0
            or query.shape[1] % key.shape[1] != 0):
        raise ValueError("RM attention needs complete equal-length dense Q/K/V with valid GQA heads")
    if (query.dtype != key.dtype or key.dtype != value.dtype
            or query.device != key.device or key.device != value.device
            or any(t.requires_grad for t in (query, key, value))):
        raise ValueError("RM attention requires equal dtype/device and detached Q/K/V")
    if attention_mask is not None:
        if (attention_mask.device != query.device or attention_mask.requires_grad
                or attention_mask.dtype not in (torch.bool, query.dtype, torch.float32)
                or attention_mask.ndim not in (2, 4)):
            raise ValueError("Expected the existing boolean/additive Qwen attention mask")
        shape = (1, 1, *attention_mask.shape) if attention_mask.ndim == 2 else attention_mask.shape
        if (shape[0] not in (1, query.shape[0]) or shape[1] not in (1, query.shape[1])
                or shape[2] not in (1, query.shape[2]) or shape[3] != key.shape[2]):
            raise ValueError("Attention mask must cover every original key and query")


def _query_mask(attention_mask, *, start, end, length, device):
    """Absolute query positions; never use upper-left is_causal on a later block."""
    if attention_mask is None:
        return (torch.arange(length, device=device)[None, :]
                <= torch.arange(start, end, device=device)[:, None])
    # The original Qwen builder already includes causality plus padding/bias.
    # Preserve its values and every key column, slicing only the query axis.
    if attention_mask.shape[-2] == 1:
        return attention_mask
    return attention_mask[..., start:end, :]


def full_kv_query_blocked_math(query, key, value, attention_mask, *, scaling):
    """Same full-row softmax; query blocks do not partition/renormalize keys."""
    length = query.shape[2]
    output = torch.empty_like(query)
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, length, QUERY_BLOCK_TOKENS):
            end = min(start + QUERY_BLOCK_TOKENS, length)
            mask = _query_mask(attention_mask, start=start, end=end,
                               length=key.shape[2], device=query.device)
            block = F.scaled_dot_product_attention(
                query[:, :, start:end, :], key, value, attn_mask=mask,
                dropout_p=0.0, scale=scaling, is_causal=False, enable_gqa=False)
            output[:, :, start:end, :].copy_(block)
    return output


class RewardAttention:
    def __init__(self, modules, *, cpu_fixture):
        self._modules = tuple(modules)
        self._cpu_fixture = cpu_fixture
        self._observations = {}
        self._efficient_checks = self._efficient_supported = self._efficient_executed = 0
        self._blocked_executed = self._max_block_rows = 0

    @property
    def contract(self):
        return copy.deepcopy(CONTRACT)

    def assert_installed(self, base):
        current = tuple(layer.self_attn for layer in base.model.layers)
        if len(current) != len(self._modules) or any(a is not b for a, b in zip(current, self._modules)):
            raise RuntimeError("Reward attention module identity changed")
        for module in current:
            if (type(module) is not Qwen3Attention
                    or getattr(module.forward, "__func__", None) is not reward_attention_forward
                    or getattr(module.forward, "__self__", None) is not module
                    or getattr(module, "_alignlab_reward_attention_v5", None) is not self
                    or module.training or module.attention_dropout != 0.0
                    or module.sliding_window is not None or module.is_causal is not True
                    or module.scaling != module.head_dim ** -0.5):
                raise RuntimeError("Reward attention binding or mathematical settings changed")

    def _supports_efficient(self, query, key, value, attention_mask, causal):
        if query.device.type != "cuda":
            return False
        params = torch.backends.cuda.SDPAParams(
            query, key, value, attention_mask, 0.0, causal, False)
        supported = bool(torch.backends.cuda.can_use_efficient_attention(
            params, debug=self._efficient_checks == 0))
        self._efficient_checks += 1
        self._efficient_supported += int(supported)
        return supported

    def attend(self, query, key, value, attention_mask, *, scaling):
        if torch.is_grad_enabled():
            raise RuntimeError("Reward attention must execute under no_grad")
        query, key, value = _sdpa_autocast_inputs(query, key, value)
        _validate_tensors(query, key, value, attention_mask)
        if self._cpu_fixture:
            if query.device.type != "cpu" or query.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("Explicit CPU attention fixture uses float32/BF16 only")
        elif query.device != torch.device("cuda:0") or query.dtype != torch.bfloat16:
            raise ValueError("Production reward attention requires unchanged BF16 CUDA:0")
        original_kv_heads = key.shape[1]
        groups = query.shape[1] // original_kv_heads
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
        causal = attention_mask is None and query.shape[2] > 1
        supported = self._supports_efficient(query, key, value, attention_mask, causal)
        if supported:
            # No exception-to-fallback path: an OOM or kernel failure is evidence.
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                result = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                    scale=scaling, is_causal=causal, enable_gqa=False)
            self._efficient_executed += 1
            backend = "torch_efficient_sdpa"
        else:
            result = full_kv_query_blocked_math(query, key, value, attention_mask, scaling=scaling)
            self._blocked_executed += 1
            self._max_block_rows = max(self._max_block_rows, min(QUERY_BLOCK_TOKENS, query.shape[2]))
            backend = "torch_math_sdpa_query_blocks_full_kv"
        signature = (query.shape[0], query.shape[2], key.shape[2], query.shape[1],
                     original_kv_heads, query.shape[-1], str(query.dtype), str(query.device),
                     "implicit_full_causal" if attention_mask is None else str(attention_mask.dtype), backend)
        self._observations[signature] = self._observations.get(signature, 0) + 1
        return result

    def receipt(self):
        fields = ("batch", "query_tokens", "key_tokens", "query_heads", "original_kv_heads",
                  "head_dim", "dtype", "device", "mask", "backend")
        return {"kind": KIND, "enabled": True, "scope": "reward_only",
                "cpu_fixture": self._cpu_fixture, "efficient_support_checks": self._efficient_checks,
                "efficient_supported": self._efficient_supported,
                "efficient_executed": self._efficient_executed,
                "blocked_math_executed": self._blocked_executed,
                "max_block_rows": self._max_block_rows, "query_block_limit": QUERY_BLOCK_TOKENS,
                "input_truncated": False, "all_key_columns_retained": True,
                "observations": [{**dict(zip(fields, key)), "calls": count}
                                 for key, count in sorted(self._observations.items())]}


def reward_attention_forward(module, hidden_states, position_embeddings, attention_mask,
                             past_key_values=None, **kwargs):
    """Qwen3's projection/norm/RoPE/output order, with owned RM dispatch only."""
    if (past_key_values is not None or kwargs.get("use_cache", False)
            or kwargs.get("output_attentions", False) or module.training
            or kwargs.get("is_causal", True) is not True
            or module.attention_dropout != 0.0 or module.sliding_window is not None):
        raise ValueError("Reward attention only supports unchanged full-context inference without cache/dropout")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    result = module._alignlab_reward_attention_v5.attend(
        query, key, value, attention_mask, scaling=module.scaling)
    result = result.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    return module.o_proj(result), None


def install_reward_attention(base, *, stage, cpu_fixture):
    """Bind only owned reward instances; original config/mask dispatch is unchanged."""
    if stage != "reward":
        return None
    if type(cpu_fixture) is not bool:
        raise ValueError("Explicit fixture flag required")
    modules = tuple(layer.self_attn for layer in base.model.layers)
    if not modules or len(modules) != base.config.num_hidden_layers:
        raise ValueError("Complete Qwen reward attention layer inventory required")
    for module in modules:
        if (type(module) is not Qwen3Attention or "forward" in module.__dict__
                or hasattr(module, "_alignlab_reward_attention_v5")
                or module.attention_dropout != 0.0 or module.sliding_window is not None
                or module.is_causal is not True
                or module.config._attn_implementation not in ("sdpa", "eager")):
            raise ValueError("Unsupported or already modified Qwen reward attention")
        if not cpu_fixture and (module.head_dim != 128 or module.num_key_value_groups != 2
                or module.config.num_attention_heads != 16 or module.config.num_key_value_heads != 8
                or module.config._attn_implementation != "sdpa"):
            raise ValueError("Production reward attention must retain accepted Qwen3-1.7B layout")
    owner = RewardAttention(modules, cpu_fixture=cpu_fixture)
    for module in modules:
        module._alignlab_reward_attention_v5 = owner
        module.forward = MethodType(reward_attention_forward, module)
    owner.assert_installed(base)
    return owner
