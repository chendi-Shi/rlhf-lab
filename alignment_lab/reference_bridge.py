"""Strict CPU/float32 bridge from actual frozen scoring to immutable disk cache.

This is a software primitive, not a GPU/quantized loader or training stage.
The loaded-reference anchor hashes actual base/reference parameters and buffers,
plus relevant model/adapter metadata. Policy adapters and independent scalar
heads are excluded. Binding and lookup both recheck actual CPU tensor contents;
the latter deliberately trades performance for detection of even .data writes.
No GPU tensor is copied to CPU. Quantized/large-GPU models need a separately
designed, verified loaded-artifact receipt before this bridge can support them.

The caller supplies the complete original artifact inventory and accepted data
manifest digest. The anchor independently records which loaded tensors really
produced the scores; it is not a general checkpoint-file loader. Manifest pins
must be retained outside the cache in the consuming run's frozen identity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import torch

from .reference_cache import build_identity, cache_key, load_cache, write_or_reuse_cache
from .shared_qwen import SharedQwen3
from .training_steps import ReferencePairScores, _fingerprint, _paired_batches, _sequence_scores


BRIDGE_VERSION = "alignlab-reference-bridge-cpu-f32-v1"
ANCHOR_KEY = "loaded_reference_anchor"
PROTOCOL_KEY = "reference_bridge_protocol"
_TOKEN = object()


def _metadata(value):
    """Normalize actual HF/PEFT config objects, including enum/set fields."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Reference metadata object keys must be strings")
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_metadata(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, torch.dtype):
        return str(value)
    # PEFT enum values subclass str; JSON handles them without self-reported IDs.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported reference metadata value: {type(value).__name__}")


def _canonical(value):
    return json.dumps(_metadata(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _model_metadata(value):
    """Preserve typed config keys (e.g. HF id2label uses integer keys)."""
    if isinstance(value, Mapping):
        items = {}
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ValueError("Model configuration keys must be strings or integers")
            items[f"{type(key).__name__}:{key}"] = _model_metadata(item)
        return items
    if isinstance(value, (tuple, list)):
        return [_model_metadata(item) for item in value]
    return _metadata(value)


def _class_name(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _tensor_identity(tensor):
    if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("Reference bridge supports only ordinary nonquantized CPU tensors")
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        raise ValueError("Reference bridge currently supports float32 parameters/buffers only")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("Frozen reference tensor contains nonfinite values")
    before = tensor._version
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    payload = memoryview(raw.numpy()).cast("B")
    h = hashlib.sha256()
    for start in range(0, len(payload), 1024 * 1024):
        h.update(payload[start:start + 1024 * 1024])
    if tensor._version != before:
        raise RuntimeError("Frozen tensor changed during hashing")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": h.hexdigest(), "size_bytes": len(payload)}


def _selected(backend, reference_role):
    if not isinstance(backend, SharedQwen3):
        raise TypeError("Reference bridge requires the guarded SharedQwen3 backend")
    backend.require_frozen_role(reference_role)
    if backend._phase != "idle":
        raise RuntimeError("Bind/lookup outside active read/backward callbacks")
    if torch.is_autocast_enabled("cpu"):
        raise ValueError("CPU reference bridge requires autocast disabled for float32 scoring")
    for model in (backend._model, backend._base):
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            raise ValueError("Quantized models need a dedicated verified loaded-artifact receipt")
    all_adapter_ids = {id(p) for params in backend._parameters.values() for p in params}
    reference_ids = {id(p) for p in backend._parameters[reference_role]}
    parameters = {name: p for name, p in backend._model.named_parameters()
                  if id(p) not in all_adapter_ids or id(p) in reference_ids}
    if not reference_ids <= {id(p) for p in parameters.values()}:
        raise ValueError("Frozen reference parameter inventory is incomplete")
    if any(p.requires_grad for p in parameters.values()):
        raise ValueError("Base/reference parameters must actually be frozen")
    other_roles = backend._roles - {reference_role}
    buffers = {name: buffer for name, buffer in backend._model.named_buffers()
               if not any(f".{role}." in name for role in other_roles)}
    return parameters, buffers


def _loaded_anchor(backend, reference_role):
    parameters, buffers = _selected(backend, reference_role)
    versions = backend.frozen_role_versions(reference_role)
    buffer_versions = tuple((name, id(value), value._version) for name, value in buffers.items())
    layers = {}
    for name, module in backend._model.named_modules():
        if not hasattr(module, "lora_A") or reference_role not in module.lora_A:
            continue
        if getattr(module, "merged_adapters", []):
            raise ValueError("Reference bridge does not accept merged LoRA adapters")
        dropout = module.lora_dropout[reference_role]
        if getattr(dropout, "p", 0) != 0:
            raise ValueError("Frozen reference scoring requires zero adapter dropout")
        layers[name] = {"class": _class_name(module), "r": module.r[reference_role],
                        "alpha": module.lora_alpha[reference_role], "scaling": module.scaling[reference_role],
                        "dropout_class": _class_name(dropout), "dropout_p": getattr(dropout, "p", 0)}
    if not layers:
        raise ValueError("No actual reference LoRA layers found")
    anchor = {"version": BRIDGE_VERSION, "reference_role": reference_role,
              "device": "cpu", "precision": "float32", "quantization": "none",
              "model_class": _class_name(backend._base), "model_config": _model_metadata(backend._base.config.to_dict()),
              "attention_implementation": backend._base.config._attn_implementation,
              "reference_adapter_config": _model_metadata(backend._model.peft_config[reference_role].to_dict()),
              "reference_layers": layers,
              "parameters": {name: _tensor_identity(value) for name, value in parameters.items()},
              "buffers": {name: _tensor_identity(value) for name, value in buffers.items()},
              "library_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}}
    fresh_parameters, fresh_buffers = _selected(backend, reference_role)
    if (backend.frozen_role_versions(reference_role) != versions
            or tuple((name, id(value)) for name, value in fresh_parameters.items()) != tuple((name, id(value)) for name, value in parameters.items())
            or tuple((name, id(value), value._version) for name, value in fresh_buffers.items()) != buffer_versions):
        raise RuntimeError("Frozen reference changed while capturing its loaded identity")
    return json.loads(_canonical(anchor))


def _require_anchor(identity, backend, reference_role):
    config = identity.get("scoring_config", {})
    protocol = config.get(PROTOCOL_KEY)
    if (not isinstance(protocol, dict) or set(protocol) != {"version", "chunk_tokens", "scoring"}
            or protocol["version"] != BRIDGE_VERSION or protocol["scoring"] != "exact_unpadded_tokens_microbatch_one"
            or type(protocol["chunk_tokens"]) is not int or protocol["chunk_tokens"] < 1):
        raise ValueError("Cache lacks the actual reference-bridge scoring protocol")
    current = _loaded_anchor(backend, reference_role)
    if config.get(ANCHOR_KEY) != current:
        raise ValueError("Loaded frozen base/reference contents or metadata differ from the cache anchor")
    return current


def _new_identity(backend, reference_role, identity_kwargs, chunk_tokens):
    if not isinstance(identity_kwargs, Mapping):
        raise ValueError("identity_kwargs must contain reference_cache.build_identity arguments")
    args = dict(identity_kwargs)
    config = json.loads(_canonical(args.get("scoring_config", {})))
    if ANCHOR_KEY in config or PROTOCOL_KEY in config:
        raise ValueError("Loaded-reference anchors are computed here, never accepted as caller assertions")
    anchor = _loaded_anchor(backend, reference_role)
    if config.get("precision") != "float32" or config.get("quantization") not in ("none", {"mode": "none"}):
        raise ValueError("CPU bridge requires explicit precision=float32 and quantization=none")
    for name, version in anchor["library_versions"].items():
        if config.get("library_versions", {}).get(name) != version:
            raise ValueError(f"scoring_config must record the actual installed {name} version")
    config[ANCHOR_KEY] = anchor
    config[PROTOCOL_KEY] = {"version": BRIDGE_VERSION, "chunk_tokens": chunk_tokens,
                            "scoring": "exact_unpadded_tokens_microbatch_one"}
    sources = dict(args.get("source_files", {}))
    for filename in ("reference_bridge.py", "shared_qwen.py", "token_logprobs.py", "training_steps.py", "reference_cache.py"):
        label, path = f"bridge/{filename}", Path(__file__).with_name(filename).resolve()
        if label in sources and Path(sources[label]).resolve() != path:
            raise ValueError("Cannot override required bridge implementation files")
        sources[label] = path
    args.update(scoring_config=config, source_files=sources)
    return build_identity(**args)


def _pairs(backend, chosen, rejected):
    chosen, rejected = _paired_batches(chosen, rejected)
    vocab = backend._base.config.vocab_size
    for batch in (chosen, rejected):
        if batch["input_ids"].device.type != "cpu":
            raise ValueError("Reference bridge accepts CPU batches only; no implicit device transfer")
        if bool(((batch["input_ids"] < 0) | (batch["input_ids"] >= vocab)).any()):
            raise ValueError("Input IDs must belong to the current model vocabulary")
    return chosen, rejected


def _sequences(batch):
    for ids, attention, labels in zip(batch["input_ids"], batch["attention_mask"], batch["labels"]):
        first = int(labels.ne(-100).nonzero()[0].item())
        length = int(attention.sum().item())
        yield tuple(ids[:first].tolist()), tuple(ids[first:length].tolist())


@dataclass(frozen=True, init=False, repr=False)
class ReferenceCacheBinding:
    """Factory-created binding; pair_scores does no model forward or role switch."""
    _backend: SharedQwen3
    reference_role: str
    path: Path
    manifest_sha256: str
    _identity_bytes: bytes
    _versions: tuple

    def __init__(self, token, backend, reference_role, cache):
        if token is not _TOKEN:
            raise TypeError("Use create_reference_cache or bind_reference_cache")
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "reference_role", reference_role)
        object.__setattr__(self, "path", cache.path)
        object.__setattr__(self, "manifest_sha256", cache.manifest_sha256)
        object.__setattr__(self, "_identity_bytes", _canonical(cache.identity))
        object.__setattr__(self, "_versions", backend.frozen_role_versions(reference_role))

    @property
    def identity(self):
        return json.loads(self._identity_bytes)

    def pair_scores(self, chosen, rejected):
        backend = self._backend
        chosen, rejected = _pairs(backend, chosen, rejected)
        fingerprints = (_fingerprint(chosen), _fingerprint(rejected))
        if backend.frozen_role_versions(self.reference_role) != self._versions:
            raise RuntimeError("Frozen base/reference versions changed after cache binding")
        identity = self.identity
        _require_anchor(identity, backend, self.reference_role)
        cache = load_cache(self.path, identity, expected_manifest_sha256=self.manifest_sha256)
        tensors = []
        for batch in (chosen, rejected):
            entries = []
            for prompt, completion in _sequences(batch):
                try:
                    entries.append(cache.lookup(prompt, completion))
                except KeyError as exc:
                    raise ValueError("Exact prompt/completion token IDs are missing from the reference cache") from exc
            tensors.extend((torch.tensor([entry.sum_logprob for entry in entries], dtype=torch.float32),
                            torch.tensor([entry.token_count for entry in entries], dtype=torch.long)))
        if fingerprints != (_fingerprint(chosen), _fingerprint(rejected)):
            raise RuntimeError("DPO inputs changed while resolving cached scores")
        _require_anchor(identity, backend, self.reference_role)
        if backend.frozen_role_versions(self.reference_role) != self._versions:
            raise RuntimeError("Frozen reference changed during cached score lookup")
        c, cc, r, rc = tensors
        return ReferencePairScores(c, r, cc, rc, self.reference_role, fingerprints, self._versions, backend,
                                   tuple(value._version for value in (c, r, cc, rc)))


def bind_reference_cache(backend, reference_role, output_dir, *, expected_identity, expected_manifest_sha256):
    """Bind only an externally pinned cache to matching actual CPU tensors."""
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("Binding requires an explicit trusted cache manifest SHA-256")
    expected_identity = json.loads(_canonical(expected_identity))
    cache = load_cache(output_dir, expected_identity, expected_manifest_sha256=expected_manifest_sha256)
    _require_anchor(cache.identity, backend, reference_role)
    return ReferenceCacheBinding(_TOKEN, backend, reference_role, cache)


def create_reference_cache(backend, reference_role, output_dir, pairs, *, identity_kwargs, chunk_tokens=32):
    """Actually score unique exact sequences, publish once, and return a binding.

    ``pairs`` contains (chosen_batch, rejected_batch) mappings accepted by
    training_steps. Scores use microbatch one; duplicate exact sequences are
    deduplicated before scoring. Existing paths are never rewritten or rescored:
    explicitly use bind_reference_cache with the stored identity and SHA pin.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Reference cache already exists; bind with explicit identity and manifest pin")
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if not isinstance(backend, SharedQwen3):
        raise TypeError("Reference bridge requires the guarded SharedQwen3 backend")
    if any(parameter.grad is not None for parameter in backend._all_parameters()):
        raise RuntimeError("Clear all optimizer gradients before reference scoring/adapter switches")
    identity = _new_identity(backend, reference_role, identity_kwargs, chunk_tokens)
    versions = backend.frozen_role_versions(reference_role)
    unique = {}
    for chosen, rejected in pairs:
        chosen, rejected = _pairs(backend, chosen, rejected)
        for batch in (chosen, rejected):
            for prompt, completion in _sequences(batch):
                unique[cache_key(identity, prompt, completion)] = (prompt, completion)
    if not unique:
        raise ValueError("Reference cache creation needs nonempty exact paired inputs")
    records = []
    for key, (prompt, completion) in sorted(unique.items()):
        if backend.frozen_role_versions(reference_role) != versions:
            raise RuntimeError("Frozen reference changed during cache scoring")
        ids = torch.tensor([prompt + completion], dtype=torch.long)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
        scores, counts = backend.read(reference_role, lambda view:
                                      _sequence_scores(view, batch, chunk_tokens, differentiable=False))
        records.append({"key": key, "prompt_token_ids": prompt, "completion_token_ids": completion,
                        "sum_logprob": float(scores[0].item()), "token_count": int(counts[0].item())})
    _require_anchor(identity, backend, reference_role)
    if backend.frozen_role_versions(reference_role) != versions:
        raise RuntimeError("Frozen reference changed during cache scoring")
    cache = write_or_reuse_cache(output, identity, records)
    if cache.reused:
        raise FileExistsError("Output appeared during scoring; bind separately with an external manifest pin")
    return bind_reference_cache(backend, reference_role, output, expected_identity=identity,
                                expected_manifest_sha256=cache.manifest_sha256)
