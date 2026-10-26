"""Immutable optimizer-boundary checkpoints for explicit adapter/head parameters.

Call after optimizer.step() AND zero_grad(set_to_none=True), never during an
accumulation. The ordered parameter mapping is the sole saved model payload;
the caller must construct that mapping from its adapter/head allowlist. Frozen
base weights are identified by input files, not copied into the checkpoint.

build_identity hashes actual files, including this implementation. The caller
must inventory all source/data/base/parent artifacts relevant to its run; hashing
a parent's manifest does not independently verify files omitted from inventory.
Save/load re-hash this inventory. A fixed expected_manifest_sha256 is required
by a formal run protocol: an unanchored manifest detects corruption, not a party
who rewrites both payload and manifest. Incomplete/new directories are preserved.

Only built-in torch AdamW, Adam and SGD have validated state schemas here. This
is not validation of paged/8-bit/custom optimizers or pretrained/GPU capacity.
Load uses weights_only=True and validates everything before committing changes.
CUDA RNG is accessed only if CUDA was already initialized; no initialization,
downloads, global seeding, training steps, or device selection occurs here.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch
from torch import nn


FORMAT = "alignlab-optimizer-checkpoint-v1"
IDENTITY_FORMAT = "alignlab-checkpoint-identity-v1"
FILES = frozenset({"state.pt", "manifest.json"})
SUPPORTED_OPTIMIZERS = (torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD)


def _json(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_json(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json(item) for key, item in value.items()}
    raise ValueError("Expected finite plain JSON values with string keys")


def _canonical(value):
    return json.dumps(_json(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(payload):
    return json.loads(payload, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")))


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA256 digest")
    return value


def _file(path):
    original = Path(path)
    if original.is_symlink():
        raise ValueError("Identity/checkpoint files must not be symlinks")
    path = original.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError(f"File changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _inventory(files, *, allow_empty=False):
    if not isinstance(files, Mapping) or not files and not allow_empty:
        raise ValueError("Identity needs an explicit nonempty file mapping")
    if any(type(name) is not str or not name.strip() for name in files):
        raise ValueError("File labels must be nonempty strings")
    return {name: _file(files[name]) for name in sorted(files)}


def build_identity(files, config, parent_manifests=None):
    """Capture actual file hashes; caller-written digest strings are not inputs."""
    if type(config) is not dict:
        raise ValueError("config must be a JSON dictionary")
    parents = _inventory({} if parent_manifests is None else parent_manifests, allow_empty=True)
    for record in parents.values():
        if type(_parse(Path(record["path"]).read_bytes())) is not dict:
            raise ValueError("Parent manifests must be JSON objects")
    identity = {"format": IDENTITY_FORMAT, "inputs": _inventory(files), "config": _json(config),
                "parents": parents, "implementation": _file(__file__)}
    _validate_identity(identity)
    return identity


def _validate_identity(identity):
    if type(identity) is not dict or set(identity) != {"format", "inputs", "config", "parents", "implementation"}:
        raise ValueError("Identity must be produced from an explicit hashed file inventory")
    if identity["format"] != IDENTITY_FORMAT or type(identity["config"]) is not dict:
        raise ValueError("Unsupported checkpoint identity")
    _json(identity)
    if type(identity["inputs"]) is not dict or not identity["inputs"] or type(identity["parents"]) is not dict:
        raise ValueError("Identity inputs/parents must be explicit file inventories")
    for group in (identity["inputs"], identity["parents"]):
        if any(type(name) is not str or not name.strip() for name in group):
            raise ValueError("Identity file labels must be nonempty")
    records = list(identity["inputs"].values()) + list(identity["parents"].values()) + [identity["implementation"]]
    for record in records:
        if type(record) is not dict or set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Malformed identity file record")
        _sha(record["sha256"])
        if type(record["path"]) is not str or type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
            raise ValueError("Malformed identity file path/size")
        if _file(record["path"]) != record:
            raise ValueError(f"Input identity changed: {record['path']}")
    if identity["implementation"] != _file(__file__):
        raise ValueError("Checkpoint implementation identity changed")


def _tensor(value):
    if type(value) is not torch.Tensor or value.layout != torch.strided or value.is_meta or value.is_complex():
        raise ValueError("State tensors must be plain dense real tensors")
    if value.requires_grad:
        raise ValueError("Checkpoint state tensors must be detached")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite checkpoint tensor")
    return value


def _state_tree(value, *, copy_tensors=False, integer_keys=False):
    if isinstance(value, torch.Tensor):
        value = _tensor(value)
        return value.detach().cpu().clone() if copy_tensors else value
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        items = [_state_tree(item, copy_tensors=copy_tensors, integer_keys=integer_keys) for item in value]
        return tuple(items) if type(value) is tuple else items
    if type(value) is dict and all(type(key) is str or integer_keys and type(key) is int for key in value):
        return {key: _state_tree(item, copy_tensors=copy_tensors, integer_keys=integer_keys) for key, item in value.items()}
    raise ValueError("State must contain only finite JSON values and plain detached tensors")


def _software():
    return {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(), "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _bindings(parameters, optimizer):
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a nonempty ordered name->Parameter mapping")
    names, params = list(parameters), list(parameters.values())
    if any(type(name) is not str or not name.strip() for name in names):
        raise ValueError("Parameter names must be nonempty strings")
    if any(not isinstance(p, nn.Parameter) or p.is_meta or not p.is_floating_point() or p.layout != torch.strided for p in params):
        raise ValueError("Only explicit real dense Parameters may be checkpointed")
    if len(params) != len(set(map(id, params))):
        raise ValueError("Duplicate parameter objects")
    storages = [(str(p.device), p.untyped_storage().data_ptr()) for p in params]
    if len(storages) != len(set(storages)):
        raise ValueError("Checkpoint parameters must not share storage")
    if any(p.grad is not None for p in params):
        raise ValueError("Pending gradients: checkpoint only after a complete step and zero_grad(set_to_none=True)")
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError("Nonfinite parameter")
    if type(optimizer) not in SUPPORTED_OPTIMIZERS:
        raise ValueError("Only built-in AdamW, Adam and SGD optimizer states are supported")
    if getattr(optimizer, "_optimizer_load_state_dict_pre_hooks", {}) or getattr(optimizer, "_optimizer_load_state_dict_post_hooks", {}):
        raise ValueError("Optimizer load hooks are incompatible with validated transactional restore")
    flattened = [p for group in optimizer.param_groups for p in group["params"]]
    if list(map(id, flattened)) != list(map(id, params)):
        raise ValueError("Ordered parameter mapping must exactly match optimizer parameter/group order")
    by_id = {id(p): name for name, p in parameters.items()}
    groups = [{"names": [by_id[id(p)] for p in group["params"]],
               "options": _json({key: value for key, value in group.items() if key != "params"})}
              for group in optimizer.param_groups]
    descriptor = {"class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}", "groups": groups}
    specs = [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in parameters.items()]
    return specs, descriptor


def _optimizer_state(state, optimizer, parameters):
    if type(state) is not dict or set(state) != {"state", "param_groups"} or type(state["state"]) is not dict:
        raise ValueError("Malformed optimizer state")
    current = optimizer.state_dict()
    groups = state["param_groups"]
    if type(groups) is not list or len(groups) != len(current["param_groups"]):
        raise ValueError("Optimizer group count changed")
    for saved, target in zip(groups, current["param_groups"]):
        if type(saved) is not dict or _canonical(saved) != _canonical(target):
            raise ValueError("Optimizer order or hyperparameters changed")
    param_ids = [index for group in groups for index in group["params"]]
    if len(param_ids) != len(set(param_ids)) or any(type(index) is not int for index in param_ids):
        raise ValueError("Malformed optimizer parameter indices")
    indexed = dict(zip(param_ids, parameters.values()))
    if not set(state["state"]).issubset(indexed):
        raise ValueError("Optimizer contains unexpected parameter state")
    for group in groups:
        for index in group["params"]:
            item = state["state"].get(index)
            if item is None:
                continue  # A never-updated parameter legitimately has no state.
            if type(item) is not dict:
                raise ValueError("Per-parameter optimizer state must be a dictionary")
            if type(optimizer) in (torch.optim.AdamW, torch.optim.Adam):
                expected = {"step", "exp_avg", "exp_avg_sq"}
                if group.get("amsgrad", False):
                    expected.add("max_exp_avg_sq")
                if set(item) != expected:
                    raise ValueError("Incomplete/unexpected Adam optimizer state")
                step = _tensor(item["step"])
                if step.ndim != 0 or step.dtype not in (torch.float32, torch.float64) or step.item() < 0 or int(step.item()) != step.item():
                    raise ValueError("Adam step must be a finite nonnegative scalar integer tensor")
            else:
                expected = {"momentum_buffer"} if group.get("momentum", 0) != 0 else set()
                if set(item) != expected:
                    raise ValueError("Unexpected SGD optimizer state")
            for key, value in item.items():
                value = _tensor(value)
                if key == "step":
                    continue
                parameter = indexed[index]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("Optimizer tensor shape/dtype does not match its parameter")
                if key in {"exp_avg_sq", "max_exp_avg_sq"} and bool((value < 0).any()):
                    raise ValueError("Adam squared moments must be nonnegative")
    _state_tree(state, integer_keys=True)


def _generator_specs(generators):
    if not isinstance(generators, Mapping) or any(type(name) is not str or not name.strip() for name in generators):
        raise ValueError("generators must be an explicit name->torch.Generator mapping")
    if any(not isinstance(generator, torch.Generator) for generator in generators.values()):
        raise ValueError("Explicit generators must be torch.Generator instances")
    if len(set(map(id, generators.values()))) != len(generators):
        raise ValueError("Explicit generator names must refer to distinct objects")
    return {name: str(generators[name].device) for name in sorted(generators)}


def _capture_rng(generators):
    numpy = np.random.get_state()
    return {"python": _json(random.getstate()),
            "numpy": {"algorithm": numpy[0], "keys": torch.tensor(numpy[1].astype(np.int64)),
                      "position": int(numpy[2]), "has_gauss": int(numpy[3]), "cached_gaussian": float(numpy[4])},
            "torch_cpu": torch.get_rng_state().clone(),
            "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None,
            "generators": {name: generator.get_state().cpu().clone() for name, generator in generators.items()}}


def _python_state(value):
    if type(value) is not list or len(value) != 3 or value[0] != 3 or type(value[1]) is not list or len(value[1]) != 625:
        raise ValueError("Malformed Python random state")
    if any(type(v) is not int or not 0 <= v < 2 ** 32 for v in value[1][:-1]) or type(value[1][-1]) is not int or not 0 <= value[1][-1] <= 624:
        raise ValueError("Invalid Python MT19937 state")
    if value[2] is not None and (type(value[2]) is not float or not math.isfinite(value[2])):
        raise ValueError("Invalid Python cached Gaussian")
    state = (value[0], tuple(value[1]), value[2])
    random.Random(0).setstate(state)
    return state


def _numpy_state(value):
    if type(value) is not dict or set(value) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise ValueError("Malformed NumPy random state")
    keys = _tensor(value["keys"])
    if (value["algorithm"] != "MT19937" or keys.dtype != torch.int64 or keys.shape != (624,)
            or bool(((keys < 0) | (keys >= 2 ** 32)).any())
            or type(value["position"]) is not int or not 0 <= value["position"] <= 624
            or type(value["has_gauss"]) is not int or value["has_gauss"] not in (0, 1)
            or type(value["cached_gaussian"]) is not float or not math.isfinite(value["cached_gaussian"])):
        raise ValueError("Invalid NumPy MT19937 state")
    state = ("MT19937", keys.cpu().numpy().astype(np.uint32), value["position"], value["has_gauss"], value["cached_gaussian"])
    np.random.RandomState(0).set_state(state)
    return state


def _rng_tensor(value, device):
    value = _tensor(value)
    if value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
        raise ValueError("RNG states must be CPU uint8 vectors")
    if torch.device(device).type == "cuda" and not torch.cuda.is_initialized():
        raise ValueError("CUDA RNG restore requires already-initialized CUDA; initialization is forbidden here")
    temporary = torch.Generator(device=device)
    temporary.set_state(value)


def _validate_rng(rng, generators):
    if type(rng) is not dict or set(rng) != {"python", "numpy", "torch_cpu", "cuda", "generators"}:
        raise ValueError("Incomplete RNG state")
    _python_state(rng["python"])
    _numpy_state(rng["numpy"])
    _rng_tensor(rng["torch_cpu"], "cpu")
    if type(rng["generators"]) is not dict or set(rng["generators"]) != set(generators):
        raise ValueError("Explicit RNG generator names changed")
    for name, generator in generators.items():
        _rng_tensor(rng["generators"][name], generator.device)
    if rng["cuda"] is None:
        if torch.cuda.is_initialized():
            raise ValueError("Checkpoint has no CUDA RNG, but target CUDA is initialized")
    else:
        if not torch.cuda.is_initialized() or type(rng["cuda"]) is not list or len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("Initialized CUDA devices do not match checkpoint RNG state")
        for index, state in enumerate(rng["cuda"]):
            _rng_tensor(state, f"cuda:{index}")


def _restore_rng(rng, generators):
    random.setstate(_python_state(rng["python"]))
    np.random.set_state(_numpy_state(rng["numpy"]))
    torch.set_rng_state(rng["torch_cpu"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    for name, generator in generators.items():
        generator.set_state(rng["generators"][name])


def _write_bytes(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _tensor_specs(value, path=""):
    if isinstance(value, torch.Tensor):
        return {path: {"shape": list(value.shape), "dtype": str(value.dtype)}}
    result = {}
    if type(value) is dict:
        for key, child in value.items():
            result.update(_tensor_specs(child, path + "/" + json.dumps(key)))
    elif type(value) in (list, tuple):
        for index, child in enumerate(value):
            result.update(_tensor_specs(child, path + "/[" + str(index) + "]"))
    return result


def _versions(value):
    if isinstance(value, torch.Tensor):
        return ((id(value), value._version),)
    if isinstance(value, Mapping):
        return tuple(item for child in value.values() for item in _versions(child))
    if type(value) in (list, tuple):
        return tuple(item for child in value for item in _versions(child))
    return ()


def save_checkpoint(output_dir, *, identity, parameters, optimizer, cursor, generators, extra_state=None):
    """Create one new directory; manifest is written last. Never overwrite.

    Returns the persisted manifest plus manifest_sha256 and checkpoint_path as
    response-only fields. A failure after directory creation preserves evidence.
    """
    _validate_identity(identity)
    identity = _json(identity)  # Own the metadata snapshot throughout this save.
    specs, descriptor = _bindings(parameters, optimizer)
    parameters = dict(parameters)
    state = optimizer.state_dict()
    _optimizer_state(state, optimizer, parameters)
    versions = (_versions(parameters), _versions(state))
    if type(cursor) is not dict:
        raise ValueError("cursor must be a finite JSON dictionary")
    cursor = _json(cursor)
    generator_specs = _generator_specs(generators)
    generators = dict(generators)
    rng = _capture_rng(generators)
    _validate_rng(rng, generators)
    payload = {"parameters": {name: p.detach().cpu().clone() for name, p in parameters.items()},
               "optimizer": _state_tree(state, copy_tensors=True, integer_keys=True), "cursor": cursor,
               "rng": rng, "extra_state": _state_tree(extra_state, copy_tensors=True)}
    _state_tree(payload, integer_keys=True)
    directory = Path(output_dir).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "state.pt").open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Recheck bindings/identity after I/O: source or live state must not change
    # during a synchronous save. Do not bypass tracked updates with .data.
    _validate_identity(identity)
    if (_bindings(parameters, optimizer) != (specs, descriptor)
            or (_versions(parameters), _versions(optimizer.state_dict())) != versions):
        raise ValueError("Parameters/optimizer changed during checkpoint save")
    saved_record = _file(directory / "state.pt")
    manifest = {"format": FORMAT, "status": "complete", "identity": _json(identity),
                "identity_sha256": _digest(_canonical(identity)), "parameter_specs": specs,
                "optimizer": descriptor, "generator_specs": generator_specs, "software": _software(),
                "tensor_specs": _tensor_specs(payload),
                "artifacts": {"state.pt": {key: saved_record[key] for key in ("sha256", "size_bytes")}}}
    manifest_bytes = _canonical(manifest)
    _write_bytes(directory / "manifest.json", manifest_bytes)
    return {**manifest, "manifest_sha256": _digest(manifest_bytes), "checkpoint_path": str(directory.resolve())}


def _validate_payload(payload, parameters, optimizer, generators, specs, manifest):
    if type(payload) is not dict or set(payload) != {"parameters", "optimizer", "cursor", "rng", "extra_state"}:
        raise ValueError("Incomplete/unexpected checkpoint payload")
    _state_tree(payload, integer_keys=True)
    if type(payload["parameters"]) is not dict or list(payload["parameters"]) != list(parameters):
        raise ValueError("Saved parameter names/order changed")
    for spec, parameter in zip(specs, parameters.values()):
        value = _tensor(payload["parameters"][spec["name"]])
        if value.shape != parameter.shape or value.dtype != parameter.dtype or value.device.type != "cpu":
            raise ValueError("Saved parameter shape/dtype mismatch")
    _optimizer_state(payload["optimizer"], optimizer, parameters)
    if type(payload["cursor"]) is not dict:
        raise ValueError("Saved cursor must be a JSON dictionary")
    _json(payload["cursor"])
    _state_tree(payload["extra_state"])
    _validate_rng(payload["rng"], generators)
    if _tensor_specs(payload) != manifest["tensor_specs"]:
        raise ValueError("Checkpoint tensor shape/dtype schema changed")


def load_checkpoint(path, *, expected_identity, parameters, optimizer, generators, expected_manifest_sha256=None):
    """Validate all files/state first, then restore the complete step boundary.

    Target optimizer configuration must already match saved group hyperparameters
    (including scheduler-adjusted LR). Rejection during validation changes no
    target parameter, optimizer state or RNG. Unexpected commit failures attempt
    rollback. Pin expected_manifest_sha256 in a formal resume protocol.
    """
    _validate_identity(expected_identity)
    expected_identity = _json(expected_identity)
    specs, descriptor = _bindings(parameters, optimizer)
    parameters = dict(parameters)
    _optimizer_state(optimizer.state_dict(), optimizer, parameters)
    generator_specs = _generator_specs(generators)
    generators = dict(generators)
    target_versions = (_versions(parameters), _versions(optimizer.state_dict()))
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Checkpoint must be a regular complete directory")
    children = list(directory.iterdir())
    if {child.name for child in children} != FILES or any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError("Incomplete checkpoint or unexpected/extra files")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest_sha = _digest(manifest_bytes)
    if expected_manifest_sha256 is not None and manifest_sha != _sha(expected_manifest_sha256):
        raise ValueError("Checkpoint manifest SHA anchor mismatch")
    manifest = _parse(manifest_bytes)
    required = {"format", "status", "identity", "identity_sha256", "parameter_specs", "optimizer",
                "generator_specs", "software", "tensor_specs", "artifacts"}
    if type(manifest) is not dict or set(manifest) != required or manifest["format"] != FORMAT or manifest["status"] != "complete":
        raise ValueError("Unsupported/incomplete checkpoint manifest")
    if (_canonical(manifest["identity"]) != _canonical(expected_identity)
            or manifest["identity_sha256"] != _digest(_canonical(expected_identity))):
        raise ValueError("Checkpoint identity/config/parent mismatch")
    if manifest["parameter_specs"] != specs or manifest["optimizer"] != descriptor:
        raise ValueError("Checkpoint parameter order/shape/dtype or optimizer hyperparameters changed")
    if manifest["generator_specs"] != generator_specs or manifest["software"] != _software():
        raise ValueError("Checkpoint generator/runtime configuration changed")
    record = _file(directory / "state.pt")
    if manifest["artifacts"] != {"state.pt": {key: record[key] for key in ("sha256", "size_bytes")}}:
        raise ValueError("Checkpoint artifact hash/size mismatch")
    # Deserialize from the exact bytes whose digest was checked, preventing a
    # pathname swap between hashing and torch.load. Only adapters/state live here.
    import io
    state_bytes = (directory / "state.pt").read_bytes()
    if _digest(state_bytes) != record["sha256"]:
        raise ValueError("Checkpoint changed between hash verification and loading")
    payload = torch.load(io.BytesIO(state_bytes), map_location="cpu", weights_only=True)
    _validate_payload(payload, parameters, optimizer, generators, specs, manifest)
    shadow = copy.deepcopy(optimizer)
    shadow.load_state_dict(copy.deepcopy(payload["optimizer"]))
    _optimizer_state(shadow.state_dict(), shadow, {name: p for name, p in
                     zip(parameters, [p for group in shadow.param_groups for p in group["params"]])})
    del shadow
    _validate_identity(expected_identity)
    if (_bindings(parameters, optimizer) != (specs, descriptor)
            or (_versions(parameters), _versions(optimizer.state_dict())) != target_versions):
        raise ValueError("Target parameters/optimizer changed during checkpoint validation")
    before_parameters = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_rng = _capture_rng(generators)
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(payload["parameters"][name])
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload["rng"], generators)
    except BaseException:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(before_parameters[name])
        optimizer.load_state_dict(before_optimizer)
        _restore_rng(before_rng, generators)
        raise
    return {"cursor": payload["cursor"], "extra_state": payload["extra_state"],
            "manifest_sha256": manifest_sha, "checkpoint_path": str(directory.resolve()),
            "identity_sha256": manifest["identity_sha256"]}
