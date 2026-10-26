"""Read-only final-policy/RM evaluation; no training-parent transition changes.

Production loading uses the pinned local NF4 CUDA:0 loader. The separate CPU
binding function is a tiny, nonquantized fixture interface, not GPU admission.
Normal EOS-aware greedy decoding shares the previously checked local KV path;
there is no forced continuation, truncation, sampling, backward or optimizer API.

Full audits hash actual tensors (including BNB quantization state) in bounded
chunks and rehash all source/artifact files. Between full boundary audits,
version/structure/file-stat guards are cheaper and do not detect .data writes
or same-stat file corruption. Such writes/concurrent model access are forbidden;
the execution owner must discard results unless its final full audit succeeds.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import io
from pathlib import Path

import torch

from . import checkpointing as ckpt
from . import gpu_reference as frozen
from . import stage_artifacts as artifacts
from . import model_loader_eval_v3 as evaluation_loader
from .capacity_generation import greedy_capacity_probe
from .model_loader_eval_v3 import ModelLoadConfig, load_quantized_stage
from .shared_qwen import ModelView, SharedQwen3


ACCEPTANCE_KIND = "alignlab-independent-evaluation-model-acceptance-v2"
PARENT_KIND = "alignlab-formal-parent-stage-v1"
ALIGNMENT_KIND = "alignlab-formal-alignment-stage-r3-v1"
FULL_SCOPE = "full_accepted_real_training_partition"
SUBSET_SCOPE = "registered_unique_real_prompt_subset"
RECORD_FIELDS = {"model_id", "stage", "artifact_dir", "manifest_sha256",
                 "acceptance_path", "acceptance_sha256"}
STAGES = {"base", "sft", "reward", "dpo", "ppo"}


def _pin(path, digest):
    record = ckpt._file(path)
    if record["sha256"] != ckpt._sha(digest):
        raise ValueError(f"Evaluation file SHA mismatch: {path}")
    return record


def _json_file(path, digest):
    record = _pin(path, digest)
    raw = Path(path).read_bytes()
    if ckpt._digest(raw) != record["sha256"]:
        raise ValueError("File changed while reading evaluation evidence")
    return ckpt._parse(raw), record


def _same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def _formal(manifest):
    """Validate the completed R5/r3/resume lineage without changing model math."""
    from .evaluation_acceptance_v2 import validate_formal_artifact
    validate_formal_artifact(manifest)

def _read_tree(directory, digest, files, seen=None):
    seen = set() if seen is None else seen
    key = (str(Path(directory).resolve()), digest)
    if key in seen:
        return
    manifest, _ = artifacts._read_artifact(directory, digest)
    _formal(manifest)
    seen.add(key)
    for path in (Path(directory) / "manifest.json", Path(directory) / "weights.pt"):
        files[str(path.resolve())] = ckpt._file(path)
    identity = manifest["identity"]
    for record in [*identity["inputs"].values(), *identity["parents"].values(), identity["implementation"]]:
        files[record["path"]] = record
    for parent in identity["parents"].values():
        _read_tree(Path(parent["path"]).parent, parent["sha256"], files, seen)
    return manifest


def _validate_evidence(record, inputs, snapshot_sha):
    if (type(record) is not dict or set(record) != RECORD_FIELDS
            or type(record["model_id"]) is not str or not record["model_id"]
            or record["stage"] not in STAGES):
        raise ValueError("Exact evaluation model record required")
    record = copy.deepcopy(record)
    stage = record["stage"]
    accepted, accepted_file = _json_file(record["acceptance_path"], record["acceptance_sha256"])
    if (accepted.get("schema_version") != 2 or type(accepted.get("schema_version")) is not int
            or accepted.get("kind") != ACCEPTANCE_KIND
            or accepted.get("status") != "accepted_for_evaluation"
            or accepted.get("all_checks_passed") is not True
            or accepted.get("processes_exited") is not True
            or accepted.get("snapshot_manifest_sha256") != ckpt._sha(snapshot_sha)):
        raise ValueError("Independent completed-model acceptance and actual snapshot pin required")
    row = accepted.get("models", {}).get(record["model_id"])
    if (type(row) is not dict
            or set(row) != {"stage", "artifact_dir", "manifest_sha256", "progress_path", "progress_sha256"}
            or row["stage"] != stage):
        raise ValueError("Acceptance does not identify this model")
    identity = ckpt.build_identity(inputs, {"purpose": "evaluation-read-only"})
    artifacts._inventory(identity)
    files = {r["path"]: r for r in identity["inputs"].values()}
    files[accepted_file["path"]] = accepted_file
    files[identity["implementation"]["path"]] = identity["implementation"]
    if stage == "base":
        if (any(record[k] is not None for k in ("artifact_dir", "manifest_sha256"))
                or any(row[k] is not None for k in ("artifact_dir", "manifest_sha256", "progress_path", "progress_sha256"))):
            raise ValueError("Base evaluation must not have an adapter artifact")
        return None, files, identity
    if (not _same_path(record["artifact_dir"], row["artifact_dir"])
            or record["manifest_sha256"] != row["manifest_sha256"]):
        raise ValueError("Accepted artifact differs from requested model")
    manifest = _read_tree(record["artifact_dir"], record["manifest_sha256"], files)
    if manifest["stage"] != stage:
        raise ValueError("Artifact stage differs from requested model")
    artifacts._compatible_inputs(manifest["identity"], identity)
    progress, progress_file = _json_file(row["progress_path"], row["progress_sha256"])
    from .evaluation_acceptance_v2 import validate_formal_progress
    validate_formal_progress(progress, manifest, model_id=record["model_id"])
    artifact = progress.get("artifact", {})
    if (progress.get("formal_training") is not True
            or progress.get("research_complete") is not False
            or progress.get("identity") != manifest["identity"]
            or progress.get("completion") != manifest["completion"]
            or progress.get("actual_optimizer_updates") != manifest["completion"]["optimizer_updates"]
            or artifact.get("manifest_sha256") != record["manifest_sha256"]
            or not _same_path(artifact.get("artifact_dir", ""), record["artifact_dir"])
            or not _same_path(artifact.get("manifest_path", ""), Path(record["artifact_dir"]) / "manifest.json")):
        raise ValueError("Accepted progress/artifact/identity/completion evidence differs")
    files[progress_file["path"]] = progress_file
    return manifest, files, identity


def _import_final(backend, role, manifest, directory):
    """Independent final-artifact importer; never grants training provenance."""
    artifacts._boundary(backend)
    stage = manifest["stage"]
    heads = ("reward",) if stage == "reward" else ()
    parameters = artifacts._parameters(backend, role, heads)
    if (manifest["lora_config"] != artifacts._lora_config(backend, role)
            or manifest["base_signature"] != artifacts._base_signature(backend)):
        raise ValueError("Final artifact LoRA/base configuration differs from loaded backend")
    specs = artifacts._specs(parameters)
    if stage == "ppo":
        # Validate the complete exported value head, but never install/use it.
        specs += [{"name": "heads/value/bias", "shape": [1], "dtype": "torch.float32"},
                  {"name": "heads/value/weight", "shape": [1, backend._base.config.hidden_size],
                   "dtype": "torch.float32"}]
    specs.sort(key=lambda item: item["name"])
    if manifest["tensor_specs"] != specs:
        raise ValueError("Final artifact tensor inventory/shape/dtype mismatch")
    raw = (Path(directory) / "weights.pt").read_bytes()
    if ckpt._digest(raw) != manifest["artifacts"]["weights.pt"]["sha256"]:
        raise ValueError("Final weights changed after manifest verification")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(payload) is not dict or list(payload) != [item["name"] for item in specs]:
        raise ValueError("Final weights tensor names/order differ")
    for spec in specs:
        tensor = ckpt._tensor(payload[spec["name"]])
        if (tensor.device.type != "cpu" or list(tensor.shape) != spec["shape"]
                or str(tensor.dtype) != spec["dtype"]):
            raise ValueError("Final weights contain an invalid tensor")
    backup = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(payload[name])
    except BaseException:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(backup[name])
        raise


class _ReadOnlyQwen(SharedQwen3):
    """Fixed-role read owner; disabled LoRA is a separately guarded base mode."""
    def __init__(self, model, *, role, heads, base_only):
        super().__init__(model, trainable_roles=set(), heads=heads)
        self._evaluation_role, self._base_only = role, base_only
        self._model.eval()
        for head in self._heads.values():
            head.eval()
        if base_only:
            # PEFT's context-manager exit can re-enable adapter trainability.
            # This dedicated base-only owner therefore disables once and never
            # re-enables/switches adapters for its entire evaluation lifetime.
            self._model.base_model.disable_adapter_layers()

    def backward(self, *args, **kwargs):
        raise RuntimeError("Evaluation model cannot train or update")

    def _assert_state(self):
        if not self._base_only:
            return super()._assert_state()
        if (self._phase != "read" or self._model.training
                or any(module.training for module in self._model.modules())
                or any(head.training for head in self._heads.values())
                or any(p.requires_grad or p.grad is not None for p in self._all_parameters())):
            raise RuntimeError("Base evaluation mode/trainability changed")
        for module in self._model.modules():
            if hasattr(module, "lora_A") and (not module.disable_adapters or module.merged_adapters):
                raise RuntimeError("Base generation requires genuinely disabled unmerged adapters")

    def read(self, role, callback):
        if role != self._evaluation_role:
            raise ValueError("Evaluation cannot switch to another adapter")
        if not self._base_only:
            return super().read(role, callback)
        if self._phase != "idle" or any(p.grad is not None for p in self._all_parameters()):
            raise RuntimeError("Base evaluation requires idle model without gradients")
        self._generation += 1
        self._phase = "read"
        try:
            with torch.no_grad():
                self._assert_state()
                result = callback(ModelView(self, self._generation))
                self._assert_state()
                return result
        finally:
            self._phase = "idle"


class EvaluationModel:
    """Only generation, scalar scoring and frozen audits are public operations."""
    def __init__(self, loaded, record, *, input_files, cpu_fixture):
        artifacts._boundary(loaded.backend)
        self._record = copy.deepcopy(record)
        self.model_id, self.stage = record["model_id"], record["stage"]
        self.device = torch.device(loaded.device)
        self._precision = "float32" if cpu_fixture else "bfloat16"
        self._quantization = "none" if cpu_fixture else copy.deepcopy(loaded.metadata["quantization"])
        self._autocast = nullcontext if cpu_fixture else loaded.autocast
        self.runtime = {**copy.deepcopy(loaded.metadata), "inference_only": True,
                        "cpu_fixture": cpu_fixture, "model_id": self.model_id,
                        "stage": self.stage, "base_adapters_disabled": self.stage == "base",
                        "ppo_value_head_used": False}
        manifest, files, identity = _validate_evidence(record, input_files,
            loaded.metadata["snapshot_manifest_sha256"])
        self._inputs, self._snapshot_sha = copy.deepcopy(input_files), loaded.metadata["snapshot_manifest_sha256"]
        self._manifest = manifest
        self.role = loaded.role
        if manifest is not None:
            _import_final(loaded.backend, self.role, manifest, record["artifact_dir"])
        self._backend = _ReadOnlyQwen(loaded.backend._model, role=self.role,
            heads=loaded.backend._heads, base_only=self.stage == "base")
        self._files = files
        # This inventory is explicit; later unrelated modules cannot alter it.
        for module in (__file__, evaluation_loader.__file__, artifacts.__file__, frozen.__file__):
            item = ckpt._file(module)
            self._files[item["path"]] = item
        self._file_guard = frozen._file_stats(self._files)
        tensors, metadata, guard = self._inventory()
        self._guard = guard
        self._anchor = frozen._full_anchor(tensors, metadata, 4 * 1024 * 1024)
        self._source_sha = ckpt._digest(ckpt._canonical(self._files))
        self.frozen_audit(full=True)

    def _inventory(self):
        backend = self._backend
        artifacts._boundary(backend)
        if (any(p.requires_grad for p in backend._all_parameters()) or backend._trainable_roles
                or any(module.training for module in backend._model.modules())
                or any(h.training for h in backend._heads.values())):
            raise RuntimeError("Evaluation must remain inference-only")
        if any(bool(module.disable_adapters) != (self.stage == "base")
               for module in backend._model.modules() if hasattr(module, "lora_A")):
            raise RuntimeError("Evaluation adapter enabled/disabled state changed")
        tensors, metadata, core_guard, _ = frozen._inventory(backend, self.role, self._precision, self._quantization)
        # Reference receipt intentionally omits other adapters/heads. Evaluation
        # promises all loaded weights unchanged, so guard and hash every one.
        for name, parameter in backend._model.named_parameters():
            tensors.setdefault(name, parameter)
        for name, buffer in backend._model.named_buffers():
            tensors.setdefault(name, buffer)
        for head_name, head in backend._heads.items():
            for name, parameter in head.named_parameters():
                tensors[f"heads/{head_name}/{name}"] = parameter
            for name, buffer in head.named_buffers():
                tensors[f"heads/{head_name}/{name}"] = buffer
        metadata = {**metadata, "evaluation_stage": self.stage,
                    "all_lora": {role: artifacts._lora_config(backend, role) for role in sorted(backend._roles)}}
        guard = ckpt._canonical({"core": core_guard.hex(), "metadata": metadata,
                                "all_tensors": {n: frozen._tensor_guard(p) for n, p in tensors.items()}})
        return tensors, metadata, guard

    @property
    def metadata(self):
        return copy.deepcopy(self.runtime)

    def frozen_audit(self, *, full=True):
        if type(full) is not bool:
            raise ValueError("full must be an explicit bool")
        if frozen._file_stats(self._files) != self._file_guard:
            raise RuntimeError("Evaluation source/artifact file stat changed")
        tensors, metadata, guard = self._inventory()
        if guard != self._guard:
            raise RuntimeError("Evaluation weights/metadata/version/quant-state changed")
        if full:
            for path, expected in self._files.items():
                if ckpt._file(path) != expected:
                    raise RuntimeError("Evaluation file contents changed")
            if frozen._full_anchor(tensors, metadata, 4 * 1024 * 1024) != self._anchor:
                raise RuntimeError("Evaluation tensor contents changed")
            if self._inventory()[2] != guard or frozen._file_stats(self._files) != self._file_guard:
                raise RuntimeError("Evaluation changed during full audit")
        groups = {"adapter": {}, "base": {}, "head": {}}
        for name, value in self._anchor["tensors"].items():
            group = "head" if name.startswith("heads/") else "adapter" if ".lora_" in name else "base"
            groups[group][name] = value
        return {"type": "alignlab-evaluation-frozen-audit-v1", "unchanged": True, "full": full,
                "model_id": self.model_id, "source_sha256": self._source_sha,
                **{name + "_sha256": ckpt._digest(ckpt._canonical(values)) for name, values in groups.items()},
                "untracked_content_writes_checked": full, "file_contents_rehashed": full}

    def greedy_generate(self, prompt_ids, *, eos_id, max_new_tokens=2048,
                        deadline_epoch=None, stop_requested=None):
        if self.stage == "reward":
            raise ValueError("Reward model is only a scalar scorer")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
            raise ValueError("Evaluation generation horizon must be 1..2048")
        if eos_id != self._backend._base.config.eos_token_id:
            raise ValueError("EOS must match the loaded model configuration")
        self.frozen_audit(full=False)
        with self._autocast():
            result = greedy_capacity_probe(self._backend, self.role, prompt_ids, eos_id=eos_id,
                max_new_tokens=max_new_tokens, ignore_eos=False, device=self.device,
                deadline_epoch=deadline_epoch, stop_requested=stop_requested)
        self.frozen_audit(full=False)
        result.pop("capacity_only")
        result.pop("quality_evaluation")
        result["termination"] = "length" if result["termination"] == "max_new_tokens" else result["termination"]
        return {**result, "inference_only": True, "model_id": self.model_id, "stage": self.stage,
                "generated_token_ids": result["action_ids"],
                "complete": result["termination"] in {"eos", "length"},
                "prompt_truncated": False}

    def reward_score(self, prompt_ids, completion_ids):
        if self.stage != "reward":
            raise ValueError("Scalar reward scoring requires the accepted reward artifact")
        vocab = self._backend._base.config.vocab_size
        for tokens in (prompt_ids, completion_ids):
            if (type(tokens) is not tuple or not tokens
                    or any(type(token) is not int or not 0 <= token < vocab for token in tokens)):
                raise ValueError("Scoring needs nonempty tuples of exact vocabulary token IDs")
        if len(prompt_ids) + len(completion_ids) > self._backend._base.config.max_position_embeddings:
            raise ValueError("Reward context exceeds model capacity; truncation is forbidden")
        self.frozen_audit(full=False)
        ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.device)
        labels = ids.clone()
        labels[:, :len(prompt_ids)] = -100
        with self._autocast():
            score = self._backend.read(self.role, lambda view: view.rewards(
                ids, torch.ones_like(ids), labels, head="reward"))
        if score.shape != (1,) or not bool(torch.isfinite(score).all()) or score.requires_grad:
            raise FloatingPointError("Reward must be one finite detached scalar")
        self.frozen_audit(full=False)
        return float(score.item())


def bind_evaluation_model(loaded, record, *, input_files):
    """Unquantized CPU-only fixture binding; never substitutes for actual loader."""
    if (torch.device(loaded.device) != torch.device("cpu")
            or any(p.device.type != "cpu" for p in loaded.backend._all_parameters())
            or getattr(loaded.backend._base, "is_loaded_in_4bit", False)):
        raise ValueError("Injected binding is restricted to unquantized CPU fixtures")
    return EvaluationModel(loaded, record, input_files=input_files, cpu_fixture=True)


def load_evaluation_model(record, *, model_snapshot, model_manifest_sha256, tokenizer_dir):
    """Load accepted local NF4 policy/RM on cuda:0, after caller resource admission."""
    from .model_snapshot import verify_snapshot
    manifest = verify_snapshot(model_snapshot, model_manifest_sha256)
    snapshot, tokenizer = Path(model_snapshot), Path(tokenizer_dir)
    token_manifest = ckpt._parse((tokenizer / "source-manifest.json").read_bytes())
    inputs = {f"base/{name}": snapshot / name for name in manifest["files"]}
    inputs.update({f"tokenizer/{name}": tokenizer / name for name in token_manifest["artifacts"]})
    inputs["tokenizer/source-manifest.json"] = tokenizer / "source-manifest.json"
    inputs["snapshot_manifest"] = snapshot / "snapshot-manifest.json"
    # Reject missing/changed acceptance before any CUDA model allocation.
    _validate_evidence(record, inputs, model_manifest_sha256)
    layout = "reward" if record["stage"] == "reward" else "sft"
    loaded = load_quantized_stage(model_snapshot, model_manifest_sha256, tokenizer_dir,
                                 ModelLoadConfig(layout, seed=42, device="cuda:0"))
    return EvaluationModel(loaded, record, input_files=inputs, cpu_fixture=False)


def greedy_generate(model, prompt_ids, **kwargs):
    return model.greedy_generate(prompt_ids, **kwargs)


def reward_score(model, prompt_ids, completion_ids):
    return model.reward_score(prompt_ids, completion_ids)


def frozen_audit(model, *, full=True):
    return model.frozen_audit(full=full)
