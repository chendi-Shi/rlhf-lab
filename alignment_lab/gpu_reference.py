"""Frozen-reference receipt and immutable DPO cache bridge.

CUDA code is implemented but is NOT GPU/BNB validated by CPU fixture tests.
Only plain tensors and CUDA:0 bitsandbytes uint8 NF4/FP4 Params4bit with the
explicitly checked QuantState schema are supported. CPU is an unquantized
float32 fixture path, never a quantized-kernel substitute.

Capture and verify(full=True) hash actual base/reference/buffer/quant-state
contents using bounded CPU transfers. verify(full=False), including ordinary
cache lookup, checks object/storage identities, PyTorch versions, actual
metadata/quant-state structure, and frozen file inode/size/mtime. It does NOT
rehash tensor or file contents: .data, external storage writes, timestamp
forgery and concurrent mutation are forbidden. The caller MUST run binding
verify(full=True) at stage start/end and before checkpoint recovery, and discard
stage results if either fails. Receipt-only users must do the same on receipt.
There is no automatic optimizer hook or stage-completion claim here.

Artifact inventories are hashed, not accepted as supplied digests. The loaded
anchor separately identifies actual scored tensors; this is not a checkpoint
loader or proof that arbitrary declared files were loaded by a particular
loader. Keep the cache manifest SHA in the consuming run identity externally.
Policy adapters and independent reward/value heads are deliberately excluded.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import stat

import torch

from .reference_bridge import _canonical, _class_name, _model_metadata, _sequences
from .reference_cache import build_identity, cache_key, load_cache, validate_identity, write_or_reuse_cache
from .shared_qwen import SharedQwen3
from .training_steps import ReferencePairScores, _fingerprint, _paired_batches, _sequence_scores


VERSION = "alignlab-frozen-reference-receipt-v1"
ANCHOR_KEY = "frozen_reference_receipt"
PROTOCOL_KEY = "frozen_reference_scoring"
_TOKEN = object()
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_ARTIFACT_KEYS = {"base_files", "adapter_files", "tokenizer_files", "source_files", "dataset_manifest"}
_STATE_KEYS = {"absmax", "shape", "code", "dtype", "blocksize", "quant_type", "offset", "state2", "nested"}


def _file_stats(paths):
    result = {}
    for raw in sorted(set(map(str, paths))):
        path = Path(raw)
        item = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(item.st_mode):
            raise ValueError("Frozen artifacts must remain ordinary files")
        result[raw] = {"device": item.st_dev, "inode": item.st_ino, "size": item.st_size,
                       "mtime_ns": item.st_mtime_ns}
    return _canonical(result)


def _artifact_paths(identity):
    paths = [record["path"] for group in identity["files"].values() for record in group.values()]
    return paths + [identity["cache_implementation"]["path"], identity["accepted_dataset_manifest"]["path"]]


def _cache_stats(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Frozen cache directory must remain an ordinary directory")
    members = list(path.iterdir())
    if {item.name for item in members} != {"manifest.json", "identity.json", "scores.jsonl"}:
        raise ValueError("Cache file inventory changed or contains untracked artifacts")
    return _file_stats(members)


def _quantization(value, precision):
    if value == "none":
        return "none"
    keys = {"method", "bits", "quant_type", "double_quant", "compute_dtype"}
    if (not isinstance(value, dict) or set(value) != keys
            or value["method"] != "bitsandbytes" or type(value["bits"]) is not int or value["bits"] != 4
            or value["quant_type"] not in ("nf4", "fp4") or type(value["double_quant"]) is not bool
            or value["compute_dtype"] != precision):
        raise ValueError("Only explicit bitsandbytes 4-bit NF4/FP4 configuration is supported")
    return json.loads(_canonical(value))


def _bnb_types():
    # Lazy: plain CPU fixture tests neither import BNB nor initialize CUDA.
    from bitsandbytes.nn import Params4bit, Linear4bit
    from bitsandbytes.functional import QuantState
    return Params4bit, Linear4bit, QuantState


def _tensor_guard(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or tensor.is_quantized or not tensor.is_contiguous()
            or tensor.device.type not in ("cpu", "cuda")):
        raise ValueError("Only contiguous ordinary CPU/CUDA tensor storage is supported")
    if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.uint8, torch.int64,
                            torch.int32, torch.int16, torch.int8, torch.bool):
        raise ValueError(f"Unsupported frozen tensor dtype: {tensor.dtype}")
    return {"object_id": id(tensor), "version": tensor._version, "storage_pointer": tensor.data_ptr(),
            "storage_offset": tensor.storage_offset(), "shape": list(tensor.shape),
            "stride": list(tensor.stride()), "dtype": str(tensor.dtype), "device": str(tensor.device)}


def _copy_chunk_to_cpu(chunk):
    """Small override point for transfer-size tests; never accepts a full model."""
    return chunk.to(device="cpu", copy=True, non_blocking=False)


def _tensor_hash(tensor, chunk_bytes):
    before = _tensor_guard(tensor)
    flat = tensor.detach().as_subclass(torch.Tensor).view(-1)
    step = max(1, chunk_bytes // tensor.element_size())
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), step):
        cpu = _copy_chunk_to_cpu(flat.narrow(0, start, min(step, flat.numel() - start)))
        if cpu.is_floating_point() and not bool(torch.isfinite(cpu).all()):
            raise FloatingPointError("Frozen tensor contains nonfinite values")
        digest.update(memoryview(cpu.view(torch.uint8).numpy()).cast("B"))
    if _tensor_guard(tensor) != before:
        raise RuntimeError("Frozen tensor changed during content hashing")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "sha256": digest.hexdigest(), "size_bytes": tensor.numel() * tensor.element_size()}


def _quant_state(state, *, name, tensors, guards, expected_type, nested=False):
    """Inspect real QuantState fields, never an as_dict caller assertion."""
    if type(state) is not expected_type or set(vars(state)) != _STATE_KEYS:
        raise ValueError("Unknown/missing bitsandbytes QuantState schema")
    guards[name] = id(state)
    if (type(state.blocksize) is not int or state.blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            or type(state.nested) is not bool or state.nested != (state.state2 is not None)
            or state.dtype not in _DTYPES.values()):
        raise ValueError("Malformed bitsandbytes quantization metadata")
    if not isinstance(state.absmax, torch.Tensor) or not isinstance(state.code, torch.Tensor):
        raise ValueError("QuantState absmax/code tensors are required")
    if (not state.absmax.numel() or state.code.dtype != torch.float32
            or state.code.numel() != (256 if nested else 16)):
        raise ValueError("Invalid bitsandbytes absmax/code layout")
    tensors[name + ".absmax"] = state.absmax
    tensors[name + ".code"] = state.code
    if nested:
        if (state.quant_type is not None or state.shape is not None or state.nested
                or state.offset is not None or state.absmax.dtype != torch.float32
                or state.dtype != torch.float32):
            raise ValueError("Unsupported nested blockwise quantization state")
    elif (state.quant_type not in ("nf4", "fp4") or not isinstance(state.shape, (tuple, list, torch.Size))
          or len(state.shape) != 2 or any(type(n) is not int or n < 1 for n in state.shape)):
        raise ValueError("Malformed 4-bit original shape or quantization type")
    if state.nested:
        if (not isinstance(state.offset, torch.Tensor) or state.offset.numel() != 1
                or state.offset.dtype != torch.float32 or state.absmax.dtype != torch.uint8):
            raise ValueError("Double quantization needs uint8 absmax and a float32 offset tensor")
        tensors[name + ".offset"] = state.offset
        state2 = _quant_state(state.state2, name=name + ".state2", tensors=tensors, guards=guards,
                              expected_type=expected_type, nested=True)
        if state.state2.absmax.numel() != math.ceil(state.absmax.numel() / state.state2.blocksize):
            raise ValueError("Nested quantization scale count mismatch")
    else:
        if state.offset is not None or state.absmax.dtype != torch.float32:
            raise ValueError("Non-nested quantization must use float32 absmax and no offset")
        state2 = None
    return {"class": _class_name(state), "shape": None if state.shape is None else list(state.shape),
            "dtype": str(state.dtype), "blocksize": state.blocksize, "quant_type": state.quant_type,
            "nested": state.nested, "state2": state2}


def _inspect_quantized_weight(parameter, module, quantization, *, name, tensors, guards, types):
    params_type, linear_type, state_type = types
    if type(parameter) is not params_type or not isinstance(module, linear_type):
        raise ValueError("Unknown quantized parameter/layer implementation")
    if (parameter.quant_state is None or module.quant_state is not parameter.quant_state
            or parameter.bnb_quantized is not True or parameter.dtype != torch.uint8
            or parameter.quant_storage != torch.uint8 or module.quant_storage != torch.uint8
            or parameter.quant_type != quantization["quant_type"]
            or parameter.compress_statistics != quantization["double_quant"]
            or module.compute_dtype != _DTYPES[quantization["compute_dtype"]]
            or module.compute_type_is_set is not True):
        raise ValueError("Packed weight/QuantState/compute dtype does not match declared 4-bit protocol")
    metadata = _quant_state(parameter.quant_state, name=name + "::quant_state", tensors=tensors,
                           guards=guards, expected_type=state_type)
    state = parameter.quant_state
    original_count = math.prod(state.shape)
    if (state.quant_type != quantization["quant_type"] or state.nested != quantization["double_quant"]
            or parameter.blocksize != state.blocksize
            or tuple(state.shape) != (module.out_features, module.in_features)
            or parameter.numel() != math.ceil(original_count / 2)
            or state.absmax.numel() != math.ceil(original_count / state.blocksize)):
        raise ValueError("Packed weight shape or scale count differs from QuantState")
    for key, tensor in tensors.items():
        if key.startswith(name + "::quant_state") and tensor.device != parameter.device:
            raise ValueError("QuantState and packed weights must use the same explicit device")
    return {"class": _class_name(module), "compute_dtype": str(module.compute_dtype),
            "quant_storage": str(module.quant_storage), "state": metadata}


def _inventory(backend, role, precision, quantization):
    if not isinstance(backend, SharedQwen3):
        raise TypeError("Frozen reference receipt requires SharedQwen3")
    backend.require_frozen_role(role)
    if backend._phase != "idle":
        raise RuntimeError("Reference receipt checks must occur outside active model callbacks")
    adapter_ids = {id(p) for params in backend._parameters.values() for p in params}
    reference_ids = {id(p) for p in backend._parameters[role]}
    parameters = {name: p for name, p in backend._model.named_parameters()
                  if id(p) not in adapter_ids or id(p) in reference_ids}
    if not reference_ids <= {id(p) for p in parameters.values()} or any(p.requires_grad for p in parameters.values()):
        raise ValueError("Actual base/reference parameter inventory must be complete and frozen")
    other_roles = backend._roles - {role}
    tensors = dict(parameters)
    tensors.update({name: value for name, value in backend._model.named_buffers()
                    if not any(f".{other}." in name for other in other_roles)})
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("Frozen reference requires one explicit device; offload/device maps are unsupported")
    device = next(iter(devices))
    if device not in (torch.device("cpu"), torch.device("cuda:0")):
        raise ValueError("Only explicit cuda:0 or CPU fixture device is supported")
    if device.type == "cpu" and (quantization != "none" or precision != "float32"):
        raise ValueError("CPU reference receipt is only an unquantized float32 fixture path")
    quantized = quantization != "none"
    flags = [bool(getattr(model, "is_loaded_in_4bit", False)) for model in (backend._model, backend._base)]
    if any(bool(getattr(model, "is_loaded_in_8bit", False)) for model in (backend._model, backend._base)):
        raise ValueError("8-bit/unknown quantization is unsupported")
    if any(flags) != quantized:
        raise ValueError("Actual model quantization flags differ from requested protocol")
    modules = dict(backend._model.named_modules())
    quant_layers, object_guards, reference_layers = {}, {}, {}
    types = _bnb_types() if quantized else None
    for name, parameter in parameters.items():
        if types and type(parameter) is types[0]:
            module = modules.get(name.rsplit(".", 1)[0])
            quant_layers[name] = _inspect_quantized_weight(
                parameter, module, quantization, name=name, tensors=tensors, guards=object_guards, types=types)
        elif (type(parameter) not in (torch.nn.Parameter, torch.Tensor)
              or getattr(parameter, "quant_state", None) is not None
              or not parameter.is_floating_point()):
            raise ValueError("Unknown/undeclared quantized parameter type")
        elif device.type == "cpu" and parameter.dtype != torch.float32:
            raise ValueError("CPU fixture parameters must be float32")
        elif parameter.dtype not in (torch.float32, _DTYPES[precision]):
            raise ValueError("Unquantized parameter dtype differs from requested compute precision")
    if quantized and not quant_layers:
        raise ValueError("4-bit protocol requires actual initialized packed parameters")
    for embedding in (backend._base.get_input_embeddings(), backend._base.get_output_embeddings()):
        if embedding is None or embedding.weight.dtype != _DTYPES[precision]:
            raise ValueError("Embedding and LM-head dtype must match the declared scoring precision")
    for name, module in modules.items():
        if not hasattr(module, "lora_A") or role not in module.lora_A:
            continue
        if getattr(module, "merged_adapters", []) or getattr(module.lora_dropout[role], "p", 0):
            raise ValueError("Only unmerged plain LoRA with zero dropout can be a reference")
        reference_layers[name] = {"class": _class_name(module), "r": module.r[role],
                                 "alpha": module.lora_alpha[role], "scaling": module.scaling[role],
                                 "dropout": _class_name(module.lora_dropout[role])}
    if not reference_layers:
        raise ValueError("Reference LoRA layer inventory is empty")
    libraries = ["torch", "transformers", "peft"] + (["bitsandbytes"] if quantized else [])
    metadata = {
        "version": VERSION, "reference_role": role, "device": str(device),
        "precision": precision, "quantization": quantization,
        "autocast": {"enabled": device.type == "cuda" and precision != "float32",
                     "device_type": device.type, "dtype": precision},
        "model_class": _class_name(backend._base),
        "model_config": _model_metadata(backend._base.config.to_dict()),
        "attention_implementation": backend._base.config._attn_implementation,
        "reference_adapter_config": _model_metadata(backend._model.peft_config[role].to_dict()),
        "reference_layers": reference_layers, "quantized_layers": quant_layers,
        "library_versions": {name: importlib.metadata.version(name) for name in libraries},
    }
    guard = {"metadata": metadata, "quant_state_objects": object_guards,
             "tensors": {name: _tensor_guard(tensor) for name, tensor in tensors.items()}}
    return tensors, json.loads(_canonical(metadata)), _canonical(guard), device


def _full_anchor(tensors, metadata, chunk_bytes):
    return {"metadata": metadata, "tensors": {name: _tensor_hash(tensor, chunk_bytes)
                                             for name, tensor in tensors.items()}}


def _artifact_identity(artifact_files, accepted_sha, precision, quantization, libraries):
    if not isinstance(artifact_files, Mapping) or set(artifact_files) != _ARTIFACT_KEYS:
        raise ValueError(f"artifact_files must contain exactly {sorted(_ARTIFACT_KEYS)}")
    args = dict(artifact_files)
    sources = dict(args["source_files"])
    for filename in ("gpu_reference.py", "reference_bridge.py", "shared_qwen.py",
                     "token_logprobs.py", "training_steps.py", "reference_cache.py"):
        label, path = "gpu_reference/" + filename, Path(__file__).with_name(filename).resolve()
        if label in sources and Path(sources[label]).resolve() != path:
            raise ValueError("Cannot replace required reference implementation source")
        sources[label] = path
    args["source_files"] = sources
    return build_identity(
        **args, accepted_dataset_manifest_sha256=accepted_sha,
        scoring_config={"template": "receipt_inventory_only",
                        "serialization": "receipt_inventory_only",
                        "precision": precision, "quantization": quantization,
                        "library_versions": libraries})


@dataclass(frozen=True, init=False, repr=False)
class FrozenReferenceReceipt:
    _owner: SharedQwen3
    reference_role: str
    device: torch.device
    precision: str
    _quantization_bytes: bytes
    _artifact_bytes: bytes
    _anchor_bytes: bytes
    _guard_bytes: bytes
    _file_guard_bytes: bytes
    _chunk_bytes: int

    def __init__(self, token, backend, role, device, precision, quantization,
                 artifact_identity, anchor, guard, chunk_bytes):
        if token is not _TOKEN:
            raise TypeError("Use capture_frozen_reference")
        for name, value in {
            "_owner": backend, "reference_role": role, "device": device, "precision": precision,
            "_quantization_bytes": _canonical(quantization), "_artifact_bytes": _canonical(artifact_identity),
            "_anchor_bytes": _canonical(anchor), "_guard_bytes": guard, "_chunk_bytes": chunk_bytes,
            "_file_guard_bytes": _file_stats(_artifact_paths(artifact_identity)),
        }.items():
            object.__setattr__(self, name, value)

    @property
    def anchor(self):
        return json.loads(self._anchor_bytes)

    @property
    def artifact_identity(self):
        return json.loads(self._artifact_bytes)

    @property
    def quantization(self):
        return json.loads(self._quantization_bytes)

    def verify(self, *, full=False):
        """False checks versions/structure/stat, not .data or same-stat file writes."""
        if type(full) is not bool:
            raise ValueError("full must be an explicit boolean")
        artifacts = self.artifact_identity
        if _file_stats(_artifact_paths(artifacts)) != self._file_guard_bytes:
            raise RuntimeError("Frozen artifact file stat changed")
        tensors, metadata, guard, _ = _inventory(self._owner, self.reference_role, self.precision, self.quantization)
        if guard != self._guard_bytes:
            raise RuntimeError("Frozen reference identity/version/quantization structure changed")
        if full:
            validate_identity(artifacts)
            if _canonical(_full_anchor(tensors, metadata, self._chunk_bytes)) != self._anchor_bytes:
                raise RuntimeError("Frozen reference contents changed; discard stage results")
            if _inventory(self._owner, self.reference_role, self.precision, self.quantization)[2] != guard:
                raise RuntimeError("Frozen reference changed during full verification")
            if _file_stats(_artifact_paths(artifacts)) != self._file_guard_bytes:
                raise RuntimeError("Frozen artifacts changed during full verification")
        return {"verified": True, "mode": "full_content" if full else "identity_versions_structure_file_stat",
                "untracked_content_writes_checked": full, "file_contents_rehashed": full}


def capture_frozen_reference(backend, reference_role, *, artifact_files, accepted_data_manifest_sha256,
                             precision, quantization, hash_chunk_bytes=4 * 1024 * 1024):
    if precision not in _DTYPES:
        raise ValueError("precision must be float32, float16 or bfloat16")
    if type(hash_chunk_bytes) is not int or not 16 <= hash_chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("hash_chunk_bytes must be an integer between 16 bytes and 64 MiB")
    quantization = _quantization(quantization, precision)
    tensors, metadata, guard, device = _inventory(backend, reference_role, precision, quantization)
    artifacts = _artifact_identity(artifact_files, accepted_data_manifest_sha256, precision,
                                   quantization, metadata["library_versions"])
    anchor = _full_anchor(tensors, metadata, hash_chunk_bytes)
    if _inventory(backend, reference_role, precision, quantization)[2] != guard:
        raise RuntimeError("Reference changed while capturing loaded receipt")
    validate_identity(artifacts)
    receipt = FrozenReferenceReceipt(_TOKEN, backend, reference_role, device, precision, quantization,
                                      artifacts, anchor, guard, hash_chunk_bytes)
    receipt.verify(full=True)
    return receipt


def _cache_identity(receipt, scoring_config, chunk_tokens):
    config = json.loads(_canonical(scoring_config))
    if ANCHOR_KEY in config or PROTOCOL_KEY in config:
        raise ValueError("Receipt anchor and scoring protocol are captured here, not caller assertions")
    if (config.get("precision") != receipt.precision or config.get("quantization") != receipt.quantization
            or config.get("library_versions") != receipt.anchor["metadata"]["library_versions"]):
        raise ValueError("Scoring precision/quantization/library versions differ from actual receipt")
    config[ANCHOR_KEY] = receipt.anchor
    config[PROTOCOL_KEY] = {"version": VERSION, "chunk_tokens": chunk_tokens,
                            "sequence_protocol": "exact_unpadded_tokens_microbatch_one",
                            "autocast": receipt.anchor["metadata"]["autocast"]}
    identity = receipt.artifact_identity
    identity["scoring_config"] = config
    validate_identity(identity)
    return identity


def _check_cache_identity(receipt, identity):
    config = identity["scoring_config"]
    protocol = config.get(PROTOCOL_KEY, {})
    if (not isinstance(protocol, dict) or set(protocol) != {"version", "chunk_tokens", "sequence_protocol", "autocast"}
            or protocol["version"] != VERSION or type(protocol["chunk_tokens"]) is not int
            or protocol["chunk_tokens"] < 1 or protocol["sequence_protocol"] != "exact_unpadded_tokens_microbatch_one"
            or protocol["autocast"] != receipt.anchor["metadata"]["autocast"]):
        raise ValueError("Cache lacks the supported actual scoring protocol")
    clean = dict(config)
    anchor = clean.pop(ANCHOR_KEY, None)
    clean.pop(PROTOCOL_KEY, None)
    if anchor != receipt.anchor or _cache_identity(receipt, clean, protocol["chunk_tokens"]) != identity:
        raise ValueError("Cache identity differs from actual loaded reference receipt")


def _pairs(receipt, chosen, rejected):
    chosen, rejected = _paired_batches(chosen, rejected)
    for batch in (chosen, rejected):
        if batch["input_ids"].device != receipt.device:
            raise ValueError("Paired inputs must already use the receipt's explicit device")
        vocab = receipt._owner._base.config.vocab_size
        if bool(((batch["input_ids"] < 0) | (batch["input_ids"] >= vocab)).any()):
            raise ValueError("Input IDs are outside the model vocabulary")
    return chosen, rejected


@dataclass(frozen=True, init=False, repr=False)
class ReferenceCacheBinding:
    receipt: FrozenReferenceReceipt
    path: Path
    manifest_sha256: str
    _identity_bytes: bytes
    _cache: object
    _disk_guard_bytes: bytes

    def __init__(self, token, receipt, cache):
        if token is not _TOKEN:
            raise TypeError("Use create_reference_cache or bind_reference_cache")
        for name, value in {"receipt": receipt, "path": cache.path, "manifest_sha256": cache.manifest_sha256,
                            "_identity_bytes": _canonical(cache.identity), "_cache": cache,
                            "_disk_guard_bytes": _cache_stats(cache.path)}.items():
            object.__setattr__(self, name, value)

    @property
    def identity(self):
        return json.loads(self._identity_bytes)

    def verify(self, *, full=False):
        result = self.receipt.verify(full=full)
        if _cache_stats(self.path) != self._disk_guard_bytes:
            raise RuntimeError("Frozen cache file stat changed")
        if full:
            load_cache(self.path, self.identity, expected_manifest_sha256=self.manifest_sha256)
            if _cache_stats(self.path) != self._disk_guard_bytes:
                raise RuntimeError("Cache changed during full verification")
        return result

    def pair_scores(self, chosen, rejected):
        """No reference forward/adapter switch; caller must also audit stage end."""
        self.verify(full=False)
        chosen, rejected = _pairs(self.receipt, chosen, rejected)
        fingerprints = (_fingerprint(chosen), _fingerprint(rejected))
        tensors = []
        for batch in (chosen, rejected):
            try:
                entries = [self._cache.lookup(prompt, completion) for prompt, completion in _sequences(batch)]
            except KeyError as exc:
                raise ValueError("Exact prompt/completion token IDs are missing from cache") from exc
            tensors.extend((torch.tensor([e.sum_logprob for e in entries], dtype=torch.float32,
                                         device=self.receipt.device),
                            torch.tensor([e.token_count for e in entries], dtype=torch.long,
                                         device=self.receipt.device)))
        self.verify(full=False)
        if fingerprints != (_fingerprint(chosen), _fingerprint(rejected)):
            raise RuntimeError("Paired inputs changed while looking up reference scores")
        c, cc, r, rc = tensors
        owner, role = self.receipt._owner, self.receipt.reference_role
        return ReferencePairScores(c, r, cc, rc, role, fingerprints, owner.frozen_role_versions(role),
                                   owner, tuple(t._version for t in (c, r, cc, rc)))


def bind_reference_cache(receipt, output_dir, *, expected_identity, expected_manifest_sha256):
    if not isinstance(receipt, FrozenReferenceReceipt):
        raise TypeError("A captured actual frozen reference receipt is required")
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("An externally retained cache manifest SHA-256 is required")
    receipt.verify(full=True)
    cache = load_cache(output_dir, expected_identity, expected_manifest_sha256=expected_manifest_sha256)
    _check_cache_identity(receipt, cache.identity)
    receipt.verify(full=True)
    binding = ReferenceCacheBinding(_TOKEN, receipt, cache)
    binding.verify(full=True)
    return binding


def create_reference_cache(receipt, output_dir, pairs, *, scoring_config, chunk_tokens=32):
    """Actually frozen-score unique sequences. Existing paths require pinned bind."""
    if not isinstance(receipt, FrozenReferenceReceipt):
        raise TypeError("A captured actual frozen reference receipt is required")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Cache already exists; use bind with identity and manifest SHA")
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if any(p.grad is not None for p in receipt._owner._all_parameters()):
        raise RuntimeError("Clear optimizer gradients before reference scoring/adapter switch")
    receipt.verify(full=True)
    identity = _cache_identity(receipt, scoring_config, chunk_tokens)
    unique = {}
    for chosen, rejected in pairs:
        for batch in _pairs(receipt, chosen, rejected):
            for prompt, completion in _sequences(batch):
                unique[cache_key(identity, prompt, completion)] = (prompt, completion)
    if not unique:
        raise ValueError("Cache needs nonempty paired token inputs")
    autocast = receipt.anchor["metadata"]["autocast"]
    if torch.is_autocast_enabled(receipt.device.type) and (
            not autocast["enabled"] or torch.get_autocast_dtype(receipt.device.type) != _DTYPES[receipt.precision]):
        raise ValueError("Ambient autocast differs from frozen scoring precision")
    records = []
    for key, (prompt, completion) in sorted(unique.items()):
        receipt.verify(full=False)
        ids = torch.tensor([prompt + completion], dtype=torch.long, device=receipt.device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
        with torch.autocast(receipt.device.type, dtype=_DTYPES[receipt.precision], enabled=autocast["enabled"]):
            scores, counts = receipt._owner.read(receipt.reference_role, lambda view:
                                                 _sequence_scores(view, batch, chunk_tokens, differentiable=False))
        records.append({"key": key, "prompt_token_ids": prompt, "completion_token_ids": completion,
                        "sum_logprob": float(scores[0].item()), "token_count": int(counts[0].item())})
    receipt.verify(full=True)
    cache = write_or_reuse_cache(output, identity, records)
    if cache.reused:
        raise FileExistsError("Cache appeared during scoring; bind it explicitly with an external manifest pin")
    receipt.verify(full=True)
    binding = ReferenceCacheBinding(_TOKEN, receipt, cache)
    binding.verify(full=True)
    return binding
