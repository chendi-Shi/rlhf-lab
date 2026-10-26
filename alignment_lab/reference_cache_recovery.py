"""Read-only authentication and explicit identity migration of interrupted cache scores.

A cache boundary is not an optimizer checkpoint. This module never resumes an
optimizer and never writes the old run. Added execution sources cause new cache
keys; only exact frozen reference/protocol/data equality permits carrying scores
into those keys. The original complete output remains incomplete historical work.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from . import reference_cache as storage

SOURCE_FILES = ("alignment_lab/reference_cache_recovery.py", "tests_v2/test_reference_cache_recovery.py")
PIN_NAMES = frozenset({"journal", "boundary_receipt", "progress", "request"})
FIELDS = PIN_NAMES | {"scored_sequences", "expected_unique_sequences"}
SERIALIZATION = "exact_accepted_token_ids_including_terminal_eos_raw_logprob_sums"


def _canonical(value):
    return storage._canonical(storage._json_value(value))


def _equal(left, right):
    return _canonical(left) == _canonical(right)


def _ordinary(path):
    path = Path(path)
    if not path.is_absolute() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Prefix evidence must use absolute ordinary paths without symlinks")
    return path


def _pin(spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("Evidence pin requires exactly path/sha256")
    path = _ordinary(spec["path"])
    record = storage._file_record(path)
    if record["sha256"] != storage._sha(spec["sha256"]):
        raise ValueError(f"Prefix evidence SHA mismatch: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("Prefix evidence changed while reading")
    return payload


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != FIELDS:
        raise ValueError("Cache prefix specification has unknown or missing fields")
    n, total = spec["scored_sequences"], spec["expected_unique_sequences"]
    if type(n) is not int or type(total) is not int or not 0 < n < total:
        raise ValueError("A recovery prefix must be nonempty and strictly partial")
    for name in PIN_NAMES:
        pin = spec[name]
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            raise ValueError("Prefix evidence pin fields changed")
        _ordinary(pin["path"])
        storage._sha(pin["sha256"])
    return json.loads(_canonical(spec))


def _validate_record(record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Historical input record fields changed")
    actual = storage._file_record(_ordinary(record["path"]))
    if actual != record:
        raise ValueError(f"Historical input changed: {record['path']}")


def _old_identity(boundary, progress, request):
    identity = boundary["identity"]
    if (not isinstance(identity, dict) or set(identity) != {"format", "inputs", "config", "parents", "implementation"}
            or identity["format"] != "alignlab-checkpoint-identity-v1"):
        raise ValueError("Historical import identity is malformed")
    inputs, cfg = identity["inputs"], identity["config"]
    if (cfg.get("stage") != "dpo" or cfg.get("synthetic_rows") != 0
            or cfg.get("formal_training") is not True or cfg.get("work", {}).get("deadline_epoch") != request["deadline_epoch"]
            or cfg["work"].get("chunk_tokens") != 16
            or not _equal(cfg.get("model"), progress.get("model"))
            or not _equal(cfg.get("selection"), progress.get("selection"))):
        raise ValueError("Historical model, selection, deadline or scoring protocol changed")
    for record in [*inputs.values(), *identity["parents"].values(), identity["implementation"]]:
        _validate_record(record)
    source = {key: row["path"] for key, row in inputs.items() if key.startswith("source/")}
    if (set(source) != {"source/" + name for name in request["source_sha256"]}
            or any(inputs["source/" + name]["sha256"] != sha for name, sha in request["source_sha256"].items())):
        raise ValueError("Historical frozen source inventory differs from request")
    for filename in ("gpu_reference.py", "reference_bridge.py", "shared_qwen.py", "token_logprobs.py", "training_steps.py", "reference_cache.py"):
        path = Path(__file__).with_name(filename).resolve()
        source["gpu_reference/" + filename] = path
    dataset = inputs["data/manifest"]
    parent = request["parents"]["sft"]
    manifest = Path(parent["artifact_dir"]) / "manifest.json"
    if (dataset["sha256"] != request["audit_manifest_sha256"]
            or Path(dataset["path"]).resolve() != (Path(request["audit_dir"]) / "manifest.json").resolve()
            or identity["parents"]["sft"]["sha256"] != parent["manifest_sha256"]
            or Path(identity["parents"]["sft"]["path"]).resolve() != manifest.resolve()):
        raise ValueError("Historical dataset or parent chain differs from request")
    anchor = progress["reference_anchor"]
    metadata = anchor["metadata"]
    config = {"template": cfg["selection"]["serialization_version"], "serialization": SERIALIZATION,
        "precision": metadata["precision"], "quantization": metadata["quantization"],
        "library_versions": metadata["library_versions"], "frozen_reference_receipt": anchor,
        "frozen_reference_scoring": {"version": "alignlab-frozen-reference-receipt-v1", "chunk_tokens": 16,
            "sequence_protocol": "exact_unpadded_tokens_microbatch_one", "autocast": metadata["autocast"]}}
    return storage.build_identity(base_files={k: v["path"] for k, v in inputs.items() if k.startswith("base/")},
        adapter_files={"manifest": manifest, "weights": manifest.parent / "weights.pt"},
        tokenizer_files={k: v["path"] for k, v in inputs.items() if k.startswith("tokenizer/")}, source_files=source,
        dataset_manifest=dataset["path"], accepted_dataset_manifest_sha256=dataset["sha256"], scoring_config=config)


@dataclass(frozen=True)
class PrefixAudit:
    """Authenticated evidence only; actual new loaded reference still needs verification."""
    _identity: bytes
    _records: bytes
    _summary: bytes

    @property
    def identity(self):
        return json.loads(self._identity)

    @property
    def records(self):
        return json.loads(self._records)

    @property
    def summary(self):
        return json.loads(self._summary)

    def to_dict(self):
        return self.summary


def audit_prefix(spec):
    """Read-only standard-library audit. Does not load torch, score, or create output."""
    spec = validate_spec(spec)
    raw = {name: _pin(spec[name]) for name in PIN_NAMES}
    boundary, progress, request = (storage._parse(raw[name]) for name in ("boundary_receipt", "progress", "request"))
    output = _ordinary(request["output"])
    if (request.get("stage") != "dpo" or request.get("seed") != 42
            or progress.get("status") != "paused" or progress.get("actual_optimizer_updates") != 0
            or progress.get("latest_safe_checkpoint") is not None or progress.get("checkpoints") not in ([], None)
            or not _equal(progress.get("request"), request)):
        raise ValueError("Prefix origin must be a paused zero-optimizer DPO42 stage with exact request")
    if set(boundary) != {"kind", "complete", "optimizer_updates", "cache_journal", "committed_record", "identity"}:
        raise ValueError("Cache boundary receipt fields changed")
    if (boundary["kind"] != "alignlab-safe-cache-record-boundary-v1" or boundary["complete"] is not True
            or type(boundary["optimizer_updates"]) is not int or boundary["optimizer_updates"] != 0):
        raise ValueError("Cache boundary is not a complete zero-update scoring receipt")
    expected_paths = {"journal": output / "reference-cache-records.jsonl", "progress": output / "progress.json", "request": output / "request.json"}
    if (any(Path(spec[name]["path"]).resolve() != path.resolve() for name, path in expected_paths.items())
            or Path(spec["boundary_receipt"]["path"]).parent.resolve() != output.resolve()
            or Path(boundary["cache_journal"]).resolve() != expected_paths["journal"].resolve()):
        raise ValueError("Prefix files and receipt do not belong to the same original stage")
    cache = request["reference_cache"]
    if (cache.get("mode") != "create" or Path(cache["path"]).resolve() != (output / "reference-cache").resolve()
            or Path(cache["path"]).exists() or Path(cache["path"]).is_symlink()):
        raise ValueError("Prefix recovery must not replace or reuse an existing final/partial cache directory")
    n, total = spec["scored_sequences"], spec["expected_unique_sequences"]
    observed = progress.get("cache_scoring_progress", {})
    if (observed.get("scored_sequences") != n or observed.get("expected_unique_sequences") != total or observed.get("complete") is not False):
        raise ValueError("Prefix counts differ from last actual progress")
    identity = _old_identity(boundary, progress, request)
    identity_sha = storage.identity_sha256(identity)
    lines = raw["journal"].splitlines()
    if len(lines) != n + 1 or any(not line.strip() for line in lines):
        raise ValueError("Prefix journal is incomplete, extended or contains empty rows")
    chain, records, last_key, final_row = "0" * 64, [], None, None
    for index, line in enumerate(lines):
        row = storage._parse(line)
        if not isinstance(row, dict) or set(row) != {"schema_version", "previous_sha256", "scored_sequences", "expected_unique_sequences", "record", "record_sha256"}:
            raise ValueError("Prefix journal row fields changed")
        payload = {k: v for k, v in row.items() if k != "record_sha256"}
        actual = hashlib.sha256(_canonical(payload)).hexdigest()
        if (type(row["schema_version"]) is not int or row["schema_version"] != 1
                or row["previous_sha256"] != chain or row["record_sha256"] != actual
                or type(row["scored_sequences"]) is not int or row["scored_sequences"] != index
                or type(row["expected_unique_sequences"]) is not int or row["expected_unique_sequences"] != total
                or (row["record"] is None) != (index == 0)):
            raise ValueError("Prefix journal hash chain/count/endpoint differs")
        if index:
            entry = next(iter(storage._entries([row["record"]], identity_sha, require_key=True).values()))
            if last_key is not None and entry.key <= last_key:
                raise ValueError("Historical prefix scoring order is not strictly increasing by original key")
            records.append(entry.to_dict())
            last_key = entry.key
        chain, final_row = actual, row
    if not _equal(boundary["committed_record"], final_row):
        raise ValueError("Safe boundary receipt does not authenticate the exact journal endpoint")
    # Inputs remain unchanged across the read; pins externally authenticate the evidence.
    for name in PIN_NAMES:
        if _pin(spec[name]) != raw[name]:
            raise ValueError("Prefix evidence changed during audit")
    summary = {"kind": "alignlab-reference-cache-prefix-audit-v1", "verified": True,
        "optimizer_checkpoint": False, "origin_optimizer_updates": 0, "pins": spec,
        "original_output": str(output), "original_deadline_epoch": request["deadline_epoch"],
        "scored_sequences": n, "expected_unique_sequences": total,
        "remaining_sequences": total - n, "old_identity_sha256": identity_sha, "journal_last_sha256": chain,
        "actual_loaded_reference_verified": False}
    return PrefixAudit(_canonical(identity), _canonical(records), _canonical(summary))


def verified_prefix_records(spec, receipt, unique, *, config, chunk_tokens=16):
    """Verify the actual reference, then explicitly rekey immutable old finite scores."""
    from . import gpu_reference as gpu
    if not isinstance(receipt, gpu.FrozenReferenceReceipt):
        raise TypeError("An actual captured frozen reference receipt is required")
    if type(chunk_tokens) is not int or chunk_tokens != 16:
        raise ValueError("Historical scoring chunk_tokens must remain 16")
    receipt.verify(full=True)
    audit = audit_prefix(spec)
    old = audit.identity
    new = gpu._cache_identity(receipt, config, chunk_tokens)
    old_sources, new_sources = old["files"]["source"], new["files"]["source"]
    if any(new_sources.get(k) != v for k, v in old_sources.items()):
        raise ValueError("Every historical source must remain byte-identical under its original label")
    reduced = json.loads(_canonical(new))
    reduced["files"]["source"] = old_sources
    if not _equal(old, reduced):
        raise ValueError("Actual frozen reference, base, adapter, tokenizer, data, or scoring protocol differs")
    if not isinstance(unique, dict) or len(unique) != spec["expected_unique_sequences"]:
        raise ValueError("Actual complete unique sequence denominator differs")
    old_order = {}
    for key, (prompt, completion) in unique.items():
        prompt, completion = storage._tokens(prompt, "prompt"), storage._tokens(completion, "completion")
        if storage.cache_key(new, prompt, completion) != key:
            raise ValueError("Actual unique sequence key does not match new identity")
        old_key = storage.cache_key(old, prompt, completion)
        if old_key in old_order:
            raise ValueError("Actual sequence inventory contains a duplicate")
        old_order[old_key] = (key, prompt, completion)
    historical = audit.records
    if [r["key"] for r in historical] != sorted(old_order)[:len(historical)]:
        raise ValueError("Historical journal is not the exact initial prefix of the actual complete sequence inventory")
    remapped = []
    for record in historical:
        key, prompt, completion = old_order[record["key"]]
        if tuple(record["prompt_token_ids"]) != prompt or tuple(record["completion_token_ids"]) != completion:
            raise ValueError("Historical sequence tokens differ from actual accepted input")
        remapped.append({**record, "key": key})
    remapped.sort(key=lambda r: r["key"])
    receipt.verify(full=True)
    report = {**audit.summary, "kind": "alignlab-reference-cache-prefix-migration-v1",
        "actual_loaded_reference_verified": True, "new_identity_sha256": storage.identity_sha256(new),
        "added_sources": {k: v for k, v in new_sources.items() if k not in old_sources},
        "new_record_keys_sha256": hashlib.sha256(_canonical([r["key"] for r in remapped])).hexdigest(),
        "score_values_unchanged": True, "sequence_tokens_unchanged": True,
        "identity_migration": "strict_source_inventory_superset_with_explicit_rekey",
        "new_optimizer_updates": 0, "formal_training_complete": False,
        "imported_sequences": len(remapped), "new_sequences_to_score": len(unique) - len(remapped),
        "source_journal_sha256": spec["journal"]["sha256"],
        "source_boundary_receipt_sha256": spec["boundary_receipt"]["sha256"],
        "all_original_sources_unchanged": True,
        "source_inventory_added": {k: v for k, v in new_sources.items() if k not in old_sources},
        "old_reference_identity": old, "new_reference_identity": new}
    return remapped, report


def create_bounded_cache_from_prefix(receipt, output, pairs, *, prefix_spec, config, deadline_epoch,
        stop_requested=lambda: False, progress=lambda **_: None, clock=time.time,
        record_boundary=lambda **_: None, migration_callback=lambda report: None):
    """Score only the missing sequences into a fresh standard complete cache."""
    import torch
    from . import gpu_reference as gpu
    from .alignment_stage import boundary, expected_cache_sequences
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("A recovery cache output must be entirely new")
    check = lambda: boundary(deadline_epoch, stop_requested, clock=clock)
    check()
    if not isinstance(receipt, gpu.FrozenReferenceReceipt):
        raise TypeError("An actual captured frozen reference receipt is required")
    if any(p.grad is not None for p in receipt._owner._all_parameters()):
        raise RuntimeError("Reference scoring requires cleared gradients and an idle backend")
    receipt.verify(full=True)
    identity = gpu._cache_identity(receipt, config, 16)
    unique = expected_cache_sequences(receipt, identity, pairs, check_boundary=check)
    records, migration = verified_prefix_records(prefix_spec, receipt, unique, config=config, chunk_tokens=16)
    if output.resolve().is_relative_to(Path(migration["original_output"]).resolve()):
        raise ValueError("Recovery output must be outside the historical run directory")
    if deadline_epoch != migration["original_deadline_epoch"]:
        raise ValueError("Cache continuation must retain the exact original absolute deadline")
    autocast = receipt.anchor["metadata"]["autocast"]
    if torch.is_autocast_enabled(receipt.device.type) and (
            not autocast["enabled"] or torch.get_autocast_dtype(receipt.device.type) != gpu._DTYPES[receipt.precision]):
        raise ValueError("Ambient autocast differs from actual frozen scoring protocol")
    check()
    migration_callback(migration)
    imported = {row["key"] for row in records}
    replay = records
    records = []
    progress(scored_sequences=0, expected_unique_sequences=len(unique), complete=False)
    record_boundary(scored_sequences=0, expected_unique_sequences=len(unique), record=None)
    for record in replay:
        check()
        records.append(record)
        progress(scored_sequences=len(records), expected_unique_sequences=len(unique), complete=False)
        record_boundary(scored_sequences=len(records), expected_unique_sequences=len(unique), record=record)
    for key, (prompt, completion) in unique.items():
        if key in imported:
            continue
        check()
        receipt.verify(full=False)
        ids = torch.tensor([prompt + completion], dtype=torch.long, device=receipt.device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
        with torch.autocast(receipt.device.type, dtype=gpu._DTYPES[receipt.precision], enabled=autocast["enabled"]):
            scores, counts = receipt._owner.read(receipt.reference_role, lambda view:
                gpu._sequence_scores(view, batch, 16, differentiable=False))
        record = {"key": key, "prompt_token_ids": prompt, "completion_token_ids": completion,
            "sum_logprob": float(scores[0].item()), "token_count": int(counts[0].item())}
        records.append(record)
        progress(scored_sequences=len(records), expected_unique_sequences=len(unique), complete=False)
        record_boundary(scored_sequences=len(records), expected_unique_sequences=len(unique), record=record)
    check()
    receipt.verify(full=True)
    # Historical inputs cannot silently drift while a suffix is being scored.
    for name in PIN_NAMES:
        _pin(prefix_spec[name])
    check()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Recovery cache appeared during scoring")
    cache = gpu.write_or_reuse_cache(output, identity, records)
    if cache.reused:
        raise FileExistsError("Recovery cache appeared concurrently")
    binding = gpu.bind_reference_cache(receipt, output, expected_identity=identity, expected_manifest_sha256=cache.manifest_sha256)
    check()
    progress(scored_sequences=len(records), expected_unique_sequences=len(unique), complete=True)
    return binding


