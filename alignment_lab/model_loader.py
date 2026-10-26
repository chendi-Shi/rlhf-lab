"""Offline, explicit single-CUDA Qwen3 NF4 construction for bounded probes.

Loading is not GPU admission, parent-stage import, or a capacity acceptance.
Callers must check resources before calling and import verified parent artifacts
before training reward/DPO/PPO. No automatic model fallback or CPU offload.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .shared_qwen import SharedQwen3


REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PINNED = {"torch": "2.7.1+cu118", "transformers": "5.8.1", "peft": "0.18.0",
          "bitsandbytes": "0.50.2", "accelerate": "1.12.0"}
STAGES = {
    "sft": (("sft",), "sft", ()),
    "reward": (("reward",), "reward", ("reward",)),
    "dpo": (("policy", "reference"), "policy", ()),
    "ppo": (("policy", "reference", "reward"), "policy", ("value",)),
}
QUANTIZATION = {"method": "bitsandbytes", "bits": 4, "quant_type": "nf4",
                "double_quant": True, "compute_dtype": "bfloat16"}


@dataclass(frozen=True)
class ModelLoadConfig:
    stage: str
    seed: int = 42
    device: str = "cuda:0"
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj")
    attention: str = "sdpa"


@dataclass
class LoadedStage:
    backend: SharedQwen3
    role: str
    heads: tuple[str, ...]
    metadata: dict
    device: torch.device

    def autocast(self):
        if self.device.type != "cuda":
            raise ValueError("Production loaded stage must be on explicit CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def validate_config(config):
    if not isinstance(config, ModelLoadConfig) or config.stage not in STAGES:
        raise ValueError("An explicit supported stage is required")
    if type(config.seed) is not int or not 0 <= config.seed < 2**63:
        raise ValueError("Seed must be a nonnegative signed 64-bit integer")
    if not re.fullmatch(r"cuda:\d+", config.device):
        raise ValueError("Use an explicit indexed CUDA device; CPU/auto/offload is forbidden")
    if any(type(v) is not int or v < 1 for v in (config.lora_r, config.lora_alpha)):
        raise ValueError("Positive integer LoRA rank/alpha required")
    supported = set(ModelLoadConfig("sft").target_modules)
    if (type(config.target_modules) is not tuple or not config.target_modules
            or len(set(config.target_modules)) != len(config.target_modules)
            or not set(config.target_modules) <= supported):
        raise ValueError("Explicit unique Qwen3 projection targets required")
    if config.attention not in {"sdpa", "eager"}:
        raise ValueError("Unsupported attention implementation")
    return config


def _versions():
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise ValueError(f"Project dependency identity changed: {versions}")
    return versions


def _tokenizer_matches(snapshot, tokenizer_dir):
    snapshot, tokenizer_dir = Path(snapshot), Path(tokenizer_dir)
    manifest = json.loads((tokenizer_dir / "source-manifest.json").read_text(encoding="utf-8"))
    if manifest["spec"]["revision"] != REVISION or manifest["spec"]["repo"] != "Qwen/Qwen3-1.7B":
        raise ValueError("Tokenizer repository/revision differs from model")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        expected = manifest["artifacts"][name]["sha256"]
        for directory in (snapshot, tokenizer_dir):
            path = directory / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Model and accepted tokenizer content differ: {name}")


def _attach_roles(base, config, device):
    """Construction helper also exercised by tiny CPU fixtures (never admission)."""
    if getattr(base.config, "model_type", None) != "qwen3" or base.config.attention_dropout != 0:
        raise ValueError("Qwen3 with zero attention dropout required")
    roles, role, trainable_heads = STAGES[config.stage]
    base.requires_grad_(False)
    # PEFT's generic preparation casts all nonquantized parameters to FP32,
    # including this model's large tied embedding/LM head. Keep those BF16;
    # only normalization is FP32. SharedQwen3 installs non-reentrant GC.
    for module in base.modules():
        if type(module).__name__ == "Qwen3RMSNorm":
            module.to(dtype=torch.float32)
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        adapter = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
                             target_modules=list(config.target_modules), lora_dropout=0.0,
                             bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, adapter, adapter_name=roles[0], autocast_adapter_dtype=True,
                               revision=REVISION)
        for name in roles[1:]:
            model.add_adapter(name, copy.deepcopy(adapter))
        # Construction-only dtype conversion, before receipt capture/optimizers.
        for name, parameter in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                parameter.data = parameter.data.to(dtype=torch.float32)
        head_names = ("reward", "value") if config.stage == "ppo" else trainable_heads
        heads = {name: nn.Linear(base.config.hidden_size, 1, device=device, dtype=torch.float32)
                 for name in head_names}
        with torch.no_grad():
            for name, head in heads.items():
                if name == "value":
                    head.weight.zero_()
                else:
                    head.weight.normal_(0.0, 0.02)
                head.bias.zero_()
    backend = SharedQwen3(model, trainable_roles={role}, heads=heads,
                         trainable_heads={role: trainable_heads})
    selected = backend.parameters_for(role, heads=trainable_heads)
    if any(p.dtype != torch.float32 or p.device != device for p in selected):
        raise ValueError("All trainable adapter/head parameters must be FP32 on the target device")
    return backend, role, trainable_heads


def _check_quantized(base, device):
    from bitsandbytes.nn import Linear4bit, Params4bit
    if getattr(base, "is_loaded_in_4bit", False) is not True:
        raise ValueError("Model did not load in real 4-bit mode")
    layers = [(name, module) for name, module in base.named_modules() if isinstance(module, Linear4bit)]
    if not layers:
        raise ValueError("No actual bitsandbytes Linear4bit layers")
    for name, module in layers:
        weight = module.weight
        if (not isinstance(weight, Params4bit) or not weight.bnb_quantized
                or weight.quant_state is None or weight.quant_type != "nf4"
                or not weight.compress_statistics or module.compute_dtype != torch.bfloat16):
            raise ValueError(f"Unexpected or incomplete NF4 quantization: {name}")
    if any(p.device != device for p in base.parameters()) or any(b.device != device for b in base.buffers()):
        raise ValueError("Model was offloaded or placed on another device")
    embedding = base.get_input_embeddings().weight
    output = base.get_output_embeddings().weight
    if embedding.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("Tied input/output embedding must remain BF16")
    if not base.config.tie_word_embeddings or embedding.data_ptr() != output.data_ptr():
        raise ValueError("Expected actual tied Qwen3 embedding/output storage")
    return len(layers)


def load_quantized_stage(snapshot, expected_manifest_sha256, tokenizer_dir, config):
    """Load only an audited local snapshot. Calls CUDA; caller must admit first."""
    validate_config(config)
    versions = _versions()
    from .model_snapshot import verify_snapshot
    manifest = verify_snapshot(snapshot, expected_manifest_sha256)
    _tokenizer_matches(snapshot, tokenizer_dir)
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
    base = AutoModelForCausalLM.from_pretrained(
        str(Path(snapshot).resolve()), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, quantization_config=bnb,
        device_map={"": config.device}, attn_implementation=config.attention)
    quantized_layers = _check_quantized(base, device)
    backend, role, heads = _attach_roles(base, config, device)
    _check_quantized(base, device)
    # Detect snapshot/tokenizer edits while transformers was reading them.
    if verify_snapshot(snapshot, expected_manifest_sha256) != manifest:
        raise ValueError("Snapshot changed while loading")
    _tokenizer_matches(snapshot, tokenizer_dir)
    metadata = {"model": "Qwen/Qwen3-1.7B", "revision": REVISION,
                "snapshot_manifest_sha256": expected_manifest_sha256,
                "config": asdict(config), "versions": versions,
                "quantization": dict(QUANTIZATION), "quantized_linear_count": quantized_layers,
                "device": str(device), "gpu": torch.cuda.get_device_name(device),
                "base_embedding_dtype": "bfloat16", "normalization_dtype": "float32",
                "adapter_head_dtype": "float32", "autocast": "bfloat16",
                "parameters_selected": sum(p.numel() for p in backend.parameters_for(role, heads=heads)),
                "parents_imported": False, "capacity_verified": False}
    return LoadedStage(backend, role, heads, metadata, device)
