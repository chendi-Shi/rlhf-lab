"""Immutable completed-stage adapter/head artifacts, distinct from resume state.

Identity uses checkpointing.build_identity. Its config.stage is the exporting
stage; import's expected_identity instead describes the DESTINATION run. Inputs
must contain nonempty base/* and tokenizer/* inventories, equal by logical name,
SHA and byte size across stages. Parent manifests are actual hashed files under
the exact stage keys: sft={}, reward={sft}, dpo={sft}, ppo={sft,reward}.

Allowed transfers: SFT -> reward adapter; SFT -> DPO/PPO policy and reference;
completed reward adapter+head -> PPO reward. DPO -> PPO is deliberately invalid.
The caller's verified loader must bind the supplied backend to those actual base
and tokenizer files; this module neither loads nor copies the frozen base.

Only stop_reason=data_exhausted with BOTH explicit completion flags true is
implemented. Registered dose completion needs a separate dose verifier and is
rejected here, not silently treated as a full data pass. Completion is caller
evidence, not a claim that this serialization helper observed model training.

No adapter switch, model mode/trainability change, optimizer/RNG operation, or
download occurs. Backend must be idle and every adapter/head gradient cleared.
Imports validate everything before copying only the named adapter/head tensors.
"""
from __future__ import annotations

import copy
from enum import Enum
import io
from pathlib import Path

import torch
from torch import nn

from . import checkpointing as ckpt


FORMAT = "alignlab-completed-stage-v1"
FILES = frozenset({"weights.pt", "manifest.json"})
PARENTS = {"sft": set(), "reward": {"sft"}, "dpo": {"sft"}, "ppo": {"sft", "reward"}}
STAGE_HEADS = {"sft": (), "reward": ("reward",), "dpo": (), "ppo": ("value",)}


def _plain(value):
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=lambda item: ckpt._canonical(item))
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return ckpt._json(value)


def _completion(completion):
    if type(completion) is not dict:
        raise ValueError("completion must explicitly describe protocol and data-pass status")
    for key in ("data_pass_complete", "protocol_complete"):
        if type(completion.get(key)) is not bool:
            raise ValueError(f"completion.{key} must be an explicit boolean")
    if completion.get("stop_reason") == "dose_complete":
        raise ValueError("Registered-dose verification is not implemented; cannot publish dose completion")
    if (completion.get("stop_reason") != "data_exhausted" or not completion["data_pass_complete"]
            or not completion["protocol_complete"]):
        raise ValueError("Only a protocol-complete exhausted data pass can be exported as completed")
    return ckpt._json(completion)


def _stage_identity(identity, stage=None):
    ckpt._validate_identity(identity)
    actual = identity["config"].get("stage")
    if actual not in PARENTS or stage is not None and actual != stage:
        raise ValueError("Identity config.stage must name the correct sft/reward/dpo/ppo stage")
    if set(identity["parents"]) != PARENTS[actual]:
        raise ValueError(f"Stage {actual} requires exact parents {sorted(PARENTS[actual])}")
    _inventory(identity)
    return actual


def _inventory(identity):
    result = {}
    for prefix in ("base/", "tokenizer/"):
        records = {name: {key: record[key] for key in ("sha256", "size_bytes")}
                   for name, record in identity["inputs"].items() if name.startswith(prefix) and len(name) > len(prefix)}
        if not records:
            raise ValueError("Stage identity needs nonempty base/* and tokenizer/* file inventories")
        result[prefix] = records
    return result


def _compatible_inputs(left, right):
    if _inventory(left) != _inventory(right):
        raise ValueError("Source and destination base/tokenizer inventories differ")


def _boundary(backend):
    # SharedQwen3 currently exposes these state guards internally. Isolate the
    # dependency here; never use set_adapter or alter its trainability allowlist.
    if backend._phase != "idle":
        raise RuntimeError("Stage artifact access requires an idle backend, outside every read/backward callback")
    if any(parameter.grad is not None for parameter in backend._all_parameters()):
        raise RuntimeError("Pending gradients: finish the optimizer step and clear gradients before stage artifact access")


def _versions(backend):
    return tuple((id(parameter), parameter._version) for parameter in backend._all_parameters())


def _lora_config(backend, role):
    config = backend._model.peft_config[role]
    if (str(config.peft_type) not in {"LORA", "PeftType.LORA"} or config.bias != "none"
            or config.modules_to_save or config.lora_dropout or getattr(config, "use_dora", False)):
        raise ValueError("Only plain zero-dropout LoRA without trainable base weights is supported")
    result = _plain(config.to_dict())
    # These identify a path or role trainability, not LoRA's scoring function.
    for key in ("base_model_name_or_path", "revision", "inference_mode"):
        result.pop(key, None)
    runtime = {}
    for name, module in backend._model.named_modules():
        if hasattr(module, "lora_A") and role in module.lora_A:
            dropout = module.lora_dropout[role]
            if not isinstance(dropout, (nn.Identity, nn.Dropout)) or getattr(dropout, "p", 0) != 0:
                raise ValueError("Actual LoRA modules must also use zero dropout")
            runtime[name] = {"rank": int(module.r[role]), "alpha": float(module.lora_alpha[role]),
                             "scaling": float(module.scaling[role]), "fan_in_fan_out": bool(module.fan_in_fan_out),
                             "use_dora": bool(module.use_dora[role])}
    if not runtime:
        raise ValueError("No actual LoRA modules for the selected role")
    return {"config": result, "runtime": ckpt._json(runtime)}


def _base_signature(backend):
    fields = ("model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
              "rope_theta", "rope_scaling", "attention_bias", "attention_dropout", "hidden_act",
              "rms_norm_eps", "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id")
    config = {key: _plain(getattr(backend._base.config, key, None)) for key in fields}
    adapter_ids = {id(parameter) for values in backend._parameters.values() for parameter in values}
    dtypes = sorted({str(parameter.dtype) for parameter in backend._model.parameters() if id(parameter) not in adapter_ids})
    return {"model_config": config, "base_parameter_dtypes": dtypes}


def _parameters(backend, role, heads):
    _boundary(backend)
    if role not in backend._roles or type(heads) is not tuple or len(set(heads)) != len(heads):
        raise ValueError("Known adapter role and unique tuple of scalar heads required")
    if any(type(name) is not str or name not in backend._heads for name in heads):
        raise ValueError("Unknown scalar head")
    result = {}
    for name, parameter in backend._model.named_parameters():
        for kind in ("lora_A", "lora_B"):
            marker = f".{kind}.{role}."
            if marker in name:
                normalized = "adapter/" + name.replace(marker, f".{kind}.")
                if normalized in result:
                    raise ValueError("Adapter names collide after role normalization")
                result[normalized] = parameter
    if not result or {id(p) for p in result.values()} != {id(p) for p in backend._parameters[role]}:
        raise ValueError("Normalized adapter mapping does not match the guarded role's parameter set")
    for name in heads:
        head = backend._heads[name]
        if not isinstance(head, nn.Linear) or head.out_features != 1 or head.in_features != backend._base.config.hidden_size:
            raise ValueError("Stage heads must be explicit scalar Linear layers")
        for local_name, parameter in head.named_parameters():
            result[f"heads/{name}/{local_name}"] = parameter
    result = dict(sorted(result.items()))
    if len({id(p) for p in result.values()}) != len(result):
        raise ValueError("Stage parameters must be disjoint")
    storage_owners = {}
    for parameter in backend._all_parameters():
        storage_owners.setdefault((str(parameter.device), parameter.untyped_storage().data_ptr()), set()).add(id(parameter))
    for parameter in result.values():
        if len(storage_owners[(str(parameter.device), parameter.untyped_storage().data_ptr())]) != 1:
            raise ValueError("Stage parameters must not share storage with another adapter/head/base parameter")
        if (not parameter.is_floating_point() or parameter.is_meta or parameter.layout != torch.strided
                or not bool(torch.isfinite(parameter).all())):
            raise ValueError("Stage adapter/head parameters must be finite real dense tensors")
    return result


def _specs(parameters):
    return [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in parameters.items()]


def _parent_artifacts(identity, stack):
    parents = {}
    for stage, record in identity["parents"].items():
        path = Path(record["path"])
        if path.name != "manifest.json":
            raise ValueError("Stage parents must name completed-stage manifest.json files")
        parent, receipt = _read_artifact(path.parent, record["sha256"], stack)
        if parent["stage"] != stage:
            raise ValueError("Parent identity stage does not match its declared lineage key")
        _compatible_inputs(identity, parent["identity"])
        parents[stage] = (parent, receipt)
    if identity["config"]["stage"] == "ppo":
        reward_parent = parents["reward"][0]["identity"]["parents"]["sft"]["sha256"]
        if reward_parent != identity["parents"]["sft"]["sha256"]:
            raise ValueError("PPO policy and reward model must descend from the same SFT artifact")
    return parents


def _read_artifact(directory, expected_sha, stack=()):
    directory = Path(directory)
    expected_sha = ckpt._sha(expected_sha)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Stage artifact must be a complete regular directory")
    children = list(directory.iterdir())
    if {path.name for path in children} != FILES or any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("Stage artifact is incomplete or contains extra files")
    manifest_path = directory / "manifest.json"
    key = (str(manifest_path.resolve()), expected_sha)
    if key in stack:
        raise ValueError("Cyclic stage parent lineage")
    manifest_bytes = manifest_path.read_bytes()
    if ckpt._digest(manifest_bytes) != expected_sha:
        raise ValueError("Stage manifest SHA anchor mismatch")
    manifest = ckpt._parse(manifest_bytes)
    required = {"format", "status", "stage", "source_role", "heads", "identity", "identity_sha256",
                "completion", "lora_config", "base_signature", "tensor_specs", "implementation", "artifacts"}
    if type(manifest) is not dict or set(manifest) != required or manifest["format"] != FORMAT or manifest["status"] != "complete":
        raise ValueError("Unsupported/incomplete stage manifest")
    stage = _stage_identity(manifest["identity"], manifest["stage"])
    _completion(manifest["completion"])
    if (manifest["heads"] != list(STAGE_HEADS[stage]) or type(manifest["source_role"]) is not str
            or not manifest["source_role"] or manifest["identity_sha256"] != ckpt._digest(ckpt._canonical(manifest["identity"]))):
        raise ValueError("Stage identity/head metadata mismatch")
    if manifest["implementation"] != ckpt._file(__file__):
        raise ValueError("Stage artifact implementation identity changed")
    record = ckpt._file(directory / "weights.pt")
    if manifest["artifacts"] != {"weights.pt": {field: record[field] for field in ("sha256", "size_bytes")}}:
        raise ValueError("Stage weights hash/size mismatch")
    _parent_artifacts(manifest["identity"], (*stack, key))
    receipt = {"stage": stage, "manifest_sha256": expected_sha, "manifest_path": str(manifest_path.resolve()),
               "artifact_dir": str(directory.resolve()), "identity_sha256": manifest["identity_sha256"],
               "completion": copy.deepcopy(manifest["completion"])}
    return manifest, receipt


def _required_imports(backend, role, stage, identity):
    if stage == "sft":
        return
    expected = {role: ("sft", ())}
    if stage in {"dpo", "ppo"}:
        backend.require_frozen_role("reference")
        expected["reference"] = ("sft", ())
    if stage == "ppo":
        backend.require_frozen_role("reward")
        if any("reward" in names for names in backend._trainable_heads.values()):
            raise ValueError("PPO reward head must be frozen in every training role")
        expected["reward"] = ("reward", ("reward",))
    sources = getattr(backend, "_stage_artifact_sources", {})
    for target_role, (source_stage, heads) in expected.items():
        receipt = sources.get(target_role)
        if (not isinstance(receipt, dict) or receipt.get("stage") != source_stage
                or receipt.get("manifest_sha256") != identity["parents"][source_stage]["sha256"]
                or receipt.get("target_stage") != stage or tuple(receipt.get("heads", ())) != heads):
            raise ValueError(f"Role {target_role} lacks verified initialization from the declared {source_stage} parent")
        if receipt.get("target_identity_sha256") != ckpt._digest(ckpt._canonical(identity)):
            raise ValueError("Stage initialization receipt belongs to a different destination run identity")


def export_stage(output, backend, role, *, stage, identity, heads=(), completion):
    """Publish only a completed adapter/head, without optimizer or RNG state."""
    _boundary(backend)
    _stage_identity(identity, stage)
    identity = ckpt._json(identity)
    completion = _completion(completion)
    if heads != STAGE_HEADS[stage]:
        raise ValueError("Exported scalar heads do not match the stage")
    parameters = _parameters(backend, role, heads)
    if role not in backend._trainable_roles:
        raise ValueError("Only the owning training role can export a completed stage")
    if not set(heads).issubset(backend._trainable_heads.get(role, set())):
        raise ValueError("Export head is not owned by the stage training role")
    parents = _parent_artifacts(identity, ())
    _required_imports(backend, role, stage, identity)
    lora_config, base_signature = _lora_config(backend, role), _base_signature(backend)
    for parent, _ in parents.values():
        if parent["lora_config"] != lora_config or parent["base_signature"] != base_signature:
            raise ValueError("Parent and exporting stage LoRA/base configurations differ")
    versions = _versions(backend)
    payload = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    directory = Path(output).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "weights.pt").open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    _boundary(backend)
    if _versions(backend) != versions or _lora_config(backend, role) != lora_config or _base_signature(backend) != base_signature:
        raise RuntimeError("Backend changed while exporting stage weights")
    _stage_identity(identity, stage)
    record = ckpt._file(directory / "weights.pt")
    manifest = {"format": FORMAT, "status": "complete", "stage": stage, "source_role": role, "heads": list(heads),
                "identity": identity, "identity_sha256": ckpt._digest(ckpt._canonical(identity)),
                "completion": completion, "lora_config": lora_config, "base_signature": base_signature,
                "tensor_specs": _specs(parameters), "implementation": ckpt._file(__file__),
                "artifacts": {"weights.pt": {field: record[field] for field in ("sha256", "size_bytes")}}}
    manifest_bytes = ckpt._canonical(manifest)
    ckpt._write_bytes(directory / "manifest.json", manifest_bytes)
    return {**manifest, "manifest_sha256": ckpt._digest(manifest_bytes),
            "manifest_path": str((directory / "manifest.json").resolve()), "artifact_dir": str(directory.resolve())}


def import_stage(path, backend, target_role, *, expected_manifest_sha256, expected_identity, heads=()):
    """Import a verified parent into an idle destination run without switching roles.

    expected_identity is the destination run identity, not the source identity.
    Build/import all parents before constructing/restoring a stage optimizer.
    This initializer intentionally does not restore DPO/PPO artifacts as an SFT
    starting point; final-policy evaluation needs a separately defined identity.
    """
    _boundary(backend)
    destination_stage = _stage_identity(expected_identity)
    expected_identity = ckpt._json(expected_identity)
    parameters = _parameters(backend, target_role, heads)
    versions = _versions(backend)
    target_lora, target_base, target_specs = _lora_config(backend, target_role), _base_signature(backend), _specs(parameters)
    manifest, receipt = _read_artifact(path, expected_manifest_sha256)
    source_stage = manifest["stage"]
    if source_stage == "sft":
        allowed = {"reward": {"reward"}, "dpo": {"policy", "reference"}, "ppo": {"policy", "reference"}}
        if target_role not in allowed.get(destination_stage, set()) or heads != ():
            raise ValueError("SFT parent is not a valid initializer for this destination stage/role/heads")
        if target_role == "reference":
            backend.require_frozen_role(target_role)
        elif target_role not in backend._trainable_roles:
            raise ValueError("SFT policy/reward initialization requires the destination training role")
    elif not (source_stage == "reward" and destination_stage == "ppo" and target_role == "reward" and heads == ("reward",)):
        raise ValueError("Invalid stage transition: PPO policy must start separately from SFT, never DPO")
    else:
        backend.require_frozen_role("reward")
        if any("reward" in names for names in backend._trainable_heads.values()):
            raise ValueError("PPO reward head must remain frozen")
    parent = expected_identity["parents"].get(source_stage)
    if parent is None or parent != ckpt._file(receipt["manifest_path"]) or parent["sha256"] != expected_manifest_sha256:
        raise ValueError("Destination parent identity does not anchor this imported artifact")
    _compatible_inputs(manifest["identity"], expected_identity)
    # Validate the complete destination lineage, including PPO RM's SFT parent.
    _parent_artifacts(expected_identity, ())
    if manifest["lora_config"] != target_lora or manifest["base_signature"] != target_base:
        raise ValueError("Stage LoRA/base configuration does not match target backend")
    if manifest["heads"] != list(heads) or manifest["tensor_specs"] != target_specs:
        raise ValueError("Stage tensor names/shape/dtype/heads do not match target")
    state_bytes = (Path(path) / "weights.pt").read_bytes()
    if ckpt._digest(state_bytes) != manifest["artifacts"]["weights.pt"]["sha256"]:
        raise ValueError("Stage weights changed after validation")
    payload = torch.load(io.BytesIO(state_bytes), map_location="cpu", weights_only=True)
    if type(payload) is not dict or list(payload) != list(parameters):
        raise ValueError("Stage payload tensor names/order mismatch")
    for name, parameter in parameters.items():
        tensor = ckpt._tensor(payload[name])
        if tensor.device.type != "cpu" or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise ValueError("Stage payload shape/dtype mismatch")
    _boundary(backend)
    if (_versions(backend) != versions or _lora_config(backend, target_role) != target_lora
            or _base_signature(backend) != target_base or _specs(_parameters(backend, target_role, heads)) != target_specs):
        raise RuntimeError("Backend changed during stage import validation")
    _stage_identity(expected_identity, destination_stage)
    backup = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(payload[name])
    except BaseException:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(backup[name])
        raise
    receipt.update(target_stage=destination_stage, target_role=target_role, heads=list(heads),
                   target_identity_sha256=ckpt._digest(ckpt._canonical(expected_identity)))
    sources = copy.deepcopy(getattr(backend, "_stage_artifact_sources", {}))
    sources[target_role] = copy.deepcopy(receipt)
    backend._stage_artifact_sources = sources
    return receipt
