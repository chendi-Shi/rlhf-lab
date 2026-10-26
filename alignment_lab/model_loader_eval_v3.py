"""Evaluation-only Qwen3 loading with strict completeness and bounded initialization.

The historical loader remains unchanged. Only an already initialized Linear
weight/bias skips the upstream initialization call that eagerly casts the weight
to FP32. Missing parameters are rejected before initialization; nonpersistent
buffers (including RoPE) retain the pinned Transformers initialization path.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from transformers import BitsAndBytesConfig, Qwen3ForCausalLM

from . import model_loader as original
from .model_loader import LoadedStage, ModelLoadConfig


_TIED_PAIR = {"lm_head.weight", "model.embed_tokens.weight"}
_TIED_MAPPING = {"lm_head.weight": "model.embed_tokens.weight"}


def _validate_loading_info(model, loading_info, *, allow_tied_missing):
    """Reject incomplete accepted checkpoints before any fallback initialization."""
    for field in ("unexpected_keys", "mismatched_keys", "error_msgs", "conversion_errors"):
        value = getattr(loading_info, field, None)
        if value is None or value:
            raise ValueError(f"Strict evaluation checkpoint loading rejected {field}: {value!r}")
    missing = getattr(loading_info, "missing_keys", None)
    if not isinstance(missing, set):
        raise ValueError("Strict evaluation requires explicit missing-key evidence")
    if (not model.config.tie_word_embeddings
            or model.all_tied_weights_keys != _TIED_MAPPING):
        raise ValueError("Strict evaluation requires the pinned Qwen3 tied embedding mapping")
    if missing:
        if not allow_tied_missing or not missing <= _TIED_PAIR or len(missing) != 1:
            raise ValueError(f"Strict evaluation checkpoint is missing accepted weights: {sorted(missing)}")
        source_name = next(iter(_TIED_PAIR - missing))
        source = model.get_parameter(source_name)
        if source.device.type == "meta" or not getattr(source, "_is_hf_initialized", False):
            raise ValueError("Missing tied alias has no actually loaded source parameter")
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in missing:
            continue
        if parameter.device.type == "meta" or not getattr(parameter, "_is_hf_initialized", False):
            raise ValueError(f"Accepted parameter was not actually loaded: {name}")


class EvaluationQwen3ForCausalLM(Qwen3ForCausalLM):
    """Local subclass; no global monkey-patching or changes to model mathematics."""

    def _init_weights(self, module):
        if (isinstance(module, nn.Linear)
                and module.weight is not None
                and getattr(module.weight, "_is_hf_initialized", False)
                and (module.bias is None or getattr(module.bias, "_is_hf_initialized", False))):
            return
        return super()._init_weights(module)

    @staticmethod
    def _finalize_model_loading(model, load_config, loading_info):
        # Check raw diagnostics before Transformers can filter ignored keys or
        # initialize absent tensors. One genuinely loaded tied alias is enough.
        _validate_loading_info(model, loading_info, allow_tied_missing=True)
        result = Qwen3ForCausalLM._finalize_model_loading(model, load_config, loading_info)
        _validate_loading_info(model, result, allow_tied_missing=False)
        if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
            raise ValueError("Evaluation input/output embeddings are not actually tied")
        return result


def load_quantized_stage(snapshot, expected_manifest_sha256, tokenizer_dir, config):
    """Same NF4/BF16 CUDA construction and checks as the pinned historical loader."""
    original.validate_config(config)
    if config.stage not in {"sft", "reward"}:
        raise ValueError("Evaluation loader only constructs policy or reward layouts")
    versions = original._versions()
    from .model_snapshot import verify_snapshot
    manifest = verify_snapshot(snapshot, expected_manifest_sha256)
    original._tokenizer_matches(snapshot, tokenizer_dir)
    raw_config = json.loads((Path(snapshot) / "config.json").read_text(encoding="utf-8"))
    if (raw_config.get("model_type") != "qwen3" or raw_config.get("hidden_size") != 2048
            or raw_config.get("num_hidden_layers") != 28 or raw_config.get("vocab_size") != 151936):
        raise ValueError("Snapshot architecture differs from pinned Qwen3-1.7B")
    device = torch.device(config.device)
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError("BF16/NF4 probe requires Ampere or newer")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8, llm_int8_skip_modules=["lm_head"],
        llm_int8_enable_fp32_cpu_offload=False)
    base, loading_info = EvaluationQwen3ForCausalLM.from_pretrained(
        str(Path(snapshot).resolve()), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, quantization_config=bnb,
        device_map={"": config.device}, attn_implementation=config.attention,
        output_loading_info=True)
    if any(loading_info.get(field) for field in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Strict evaluation loading diagnostics are not empty")
    quantized_layers = original._check_quantized(base, device)
    backend, role, heads = original._attach_roles(base, config, device)
    original._check_quantized(base, device)
    if verify_snapshot(snapshot, expected_manifest_sha256) != manifest:
        raise ValueError("Snapshot changed while loading")
    original._tokenizer_matches(snapshot, tokenizer_dir)
    metadata = {"model": "Qwen/Qwen3-1.7B", "revision": original.REVISION,
                "snapshot_manifest_sha256": expected_manifest_sha256,
                "config": asdict(config), "versions": versions,
                "quantization": dict(original.QUANTIZATION), "quantized_linear_count": quantized_layers,
                "device": str(device), "gpu": torch.cuda.get_device_name(device),
                "base_embedding_dtype": "bfloat16", "normalization_dtype": "float32",
                "adapter_head_dtype": "float32", "autocast": "bfloat16",
                "parameters_selected": sum(p.numel() for p in backend.parameters_for(role, heads=heads)),
                "parents_imported": False, "capacity_verified": False,
                "evaluation_loader": "strict_initialized_linear_v3",
                "strict_checkpoint_loading_complete": True}
    return LoadedStage(backend, role, heads, metadata, device)
