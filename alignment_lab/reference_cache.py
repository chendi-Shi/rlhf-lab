"""Immutable, standard-library storage for caller-supplied reference log-probs.

This module does not load a model, import torch, run scoring, or establish the
numerical correctness of supplied scores. A scorer must supply unnormalized
completion log-probability sums and the exact, unpadded token IDs it scored.
Every completion token must be scored, including EOS when present.

File lists must include every base/adapter/tokenizer/scoring-source file used by
the scorer. Adapter files may be explicitly empty for a base-only reference.
``scoring_config`` must contain the complete scoring protocol, not just the
required template/serialization/precision/quantization/library-version fields.
The explicit accepted dataset SHA is supplied by the upstream audit owner;
this module checks that digest and readiness flags, not the audit's methodology.

Existing directories are only read and verified. Partial outputs are preserved
and rejected. Pin the returned manifest SHA in the downstream run identity:
artifact hashes alone cannot authenticate a manifest an attacker also rewrites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


FORMAT = "alignlab-reference-logprobs-v1"
FILES = frozenset({"identity.json", "scores.jsonl", "manifest.json"})
REQUIRED_CONFIG = frozenset({"template", "serialization", "precision", "quantization", "library_versions"})
GROUPS = ("base", "adapter", "tokenizer", "source")
SCORE_SEMANTICS = "sum_of_unmodified_conditional_log_probabilities_of_all_completion_tokens"


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError("Identity/config must contain only finite JSON values and string object keys")


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(payload: bytes) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value: {value}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UTF-8/JSON cache artifact") from exc


def _file_record(path: str | os.PathLike) -> dict[str, Any]:
    original = Path(path)
    if original.is_symlink():
        raise ValueError(f"Input must be a regular file, not a symbolic link: {original}")
    path = original.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Input must be a regular file: {path}")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        after = os.fstat(stream.fileno())
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError(f"Input changed while hashing: {path}")
    return {"path": str(path), "sha256": h.hexdigest(), "size_bytes": after.st_size}


def _group(files: Mapping[str, str | os.PathLike], name: str) -> dict[str, Any]:
    if not isinstance(files, Mapping) or (not files and name != "adapter"):
        raise ValueError(f"{name}_files must be an explicit {'possibly empty ' if name == 'adapter' else 'nonempty '}mapping")
    if any(not isinstance(label, str) or not label.strip() for label in files):
        raise ValueError("File labels must be nonempty strings")
    return {label: _file_record(files[label]) for label in sorted(files)}


def _config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or not REQUIRED_CONFIG <= set(value):
        raise ValueError(f"Complete scoring_config must explicitly contain {sorted(REQUIRED_CONFIG)}")
    config = _json_value(value)
    for key in ("template", "serialization", "precision", "quantization"):
        if (not isinstance(config[key], (str, dict)) or not config[key]
                or isinstance(config[key], str) and not config[key].strip()):
            raise ValueError(f"scoring_config.{key} must be an explicit nonempty string/object")
    libraries = config["library_versions"]
    if not isinstance(libraries, dict) or not libraries or any(not key.strip() or not isinstance(version, str)
                                                              or not version.strip() for key, version in libraries.items()):
        raise ValueError("library_versions must explicitly map libraries to nonempty versions")
    return config


def _accepted_manifest(path, expected_sha256):
    expected_sha256 = _sha(expected_sha256)
    record = _file_record(path)
    if record["sha256"] != expected_sha256:
        raise ValueError("Dataset manifest SHA differs from the explicitly accepted digest")
    payload = Path(record["path"]).read_bytes()
    if _digest(payload) != expected_sha256:
        raise ValueError("Dataset manifest changed during acceptance verification")
    manifest = _parse(payload)
    if (not isinstance(manifest, dict) or manifest.get("status") != "audit_complete"
            or manifest.get("data_ready") is not True or manifest.get("readiness_blockers") != []
            or manifest.get("training_started") is not False):
        raise ValueError("Dataset manifest must be an accepted, ready audit with no blockers")
    return record


def build_identity(*, base_files: Mapping[str, str | os.PathLike],
                   adapter_files: Mapping[str, str | os.PathLike],
                   tokenizer_files: Mapping[str, str | os.PathLike],
                   source_files: Mapping[str, str | os.PathLike],
                   dataset_manifest: str | os.PathLike, accepted_dataset_manifest_sha256: str,
                   scoring_config: dict[str, Any]) -> dict[str, Any]:
    """Capture actual bytes; caller-provided hashes cannot substitute for files."""
    files = {name: _group(values, name) for name, values in
             (("base", base_files), ("adapter", adapter_files), ("tokenizer", tokenizer_files), ("source", source_files))}
    identity = {"format": FORMAT, "schema_version": 1, "files": files,
                "cache_implementation": _file_record(__file__),
                "accepted_dataset_manifest": _accepted_manifest(dataset_manifest, accepted_dataset_manifest_sha256),
                "scoring_config": _config(scoring_config), "score_semantics": SCORE_SEMANTICS}
    validate_identity(identity)
    return identity


def validate_identity(identity: dict[str, Any]) -> None:
    """Recheck actual input files, even when a caller passes a stored identity."""
    expected = {"format", "schema_version", "files", "cache_implementation", "accepted_dataset_manifest",
                "scoring_config", "score_semantics"}
    if (not isinstance(identity, dict) or set(identity) != expected or identity.get("format") != FORMAT
            or type(identity.get("schema_version")) is not int or identity["schema_version"] != 1
            or identity.get("score_semantics") != SCORE_SEMANTICS):
        raise ValueError("Invalid or unsupported reference-cache identity")
    _config(identity["scoring_config"])
    groups = identity["files"]
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError("Identity must include base, adapter, tokenizer and source file groups")
    records = []
    for name, group in groups.items():
        if not isinstance(group, dict) or (not group and name != "adapter"):
            raise ValueError(f"Missing {name} file inventory")
        if any(not isinstance(label, str) or not label.strip() for label in group):
            raise ValueError("Invalid file inventory label")
        records.extend(group.values())
    records.extend([identity["cache_implementation"], identity["accepted_dataset_manifest"]])
    for record in records:
        if (not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}
                or not isinstance(record["path"], str) or not Path(record["path"]).is_absolute()
                or type(record["size_bytes"]) is not int or record["size_bytes"] < 0):
            raise ValueError("Malformed file identity")
        _sha(record["sha256"])
        if _file_record(record["path"]) != record:
            raise ValueError(f"Frozen input file changed: {record['path']}")
    if identity["cache_implementation"] != _file_record(__file__):
        raise ValueError("Cache implementation differs from the frozen identity")
    accepted = identity["accepted_dataset_manifest"]
    _accepted_manifest(accepted["path"], accepted["sha256"])


def identity_sha256(identity: dict[str, Any]) -> str:
    return _digest(_canonical(identity))


def _tokens(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be nonempty unpadded token IDs")
    if any(type(token) is not int or token < 0 for token in values):
        raise ValueError(f"{name} must contain nonnegative integers, not bools")
    return tuple(values)


def _token_key(identity_hash: str, prompt, completion) -> str:
    return _digest(_canonical({"identity_sha256": _sha(identity_hash),
                               "prompt_token_ids": _tokens(prompt, "prompt_token_ids"),
                               "completion_token_ids": _tokens(completion, "completion_token_ids")}))


def cache_key(identity: dict[str, Any], prompt_token_ids: Sequence[int], completion_token_ids: Sequence[int]) -> str:
    """Key includes the full identity and exact ordered prompt/completion IDs."""
    return _token_key(identity_sha256(identity), prompt_token_ids, completion_token_ids)


@dataclass(frozen=True)
class CacheEntry:
    key: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    sum_logprob: float
    token_count: int

    def to_dict(self):
        return {"key": self.key, "prompt_token_ids": list(self.prompt_token_ids),
                "completion_token_ids": list(self.completion_token_ids),
                "sum_logprob": self.sum_logprob, "token_count": self.token_count}


def _entries(records: Iterable[Mapping[str, Any] | CacheEntry], identity_hash: str, *, require_key=False) -> dict[str, CacheEntry]:
    result = {}
    for record in records:
        record = record.to_dict() if isinstance(record, CacheEntry) else record
        required = {"prompt_token_ids", "completion_token_ids", "sum_logprob", "token_count"}
        if require_key:
            required.add("key")
        if not isinstance(record, Mapping) or not required <= set(record) or set(record) - required - {"key"}:
            raise ValueError("Cache record must contain exact token IDs, sum_logprob and token_count")
        prompt, completion = _tokens(record["prompt_token_ids"], "prompt_token_ids"), _tokens(record["completion_token_ids"], "completion_token_ids")
        score, count = record["sum_logprob"], record["token_count"]
        if type(score) not in (int, float):
            raise ValueError("sum_logprob must be a finite, nonpositive number")
        try:
            score = float(score)
        except (OverflowError, ValueError) as exc:
            raise ValueError("sum_logprob must be finite") from exc
        if not math.isfinite(score) or score > 0:
            raise ValueError("sum_logprob must be a finite, nonpositive number")
        if type(count) is not int or count != len(completion):
            raise ValueError("token_count must equal the nonempty completion token count")
        key = _token_key(identity_hash, prompt, completion)
        if "key" in record and record["key"] != key:
            raise ValueError("Cache key differs from frozen identity and actual token IDs")
        if key in result:
            raise ValueError(f"Duplicate cache key: {key}")
        result[key] = CacheEntry(key, prompt, completion, score, count)
    if not result:
        raise ValueError("A completed reference cache must contain at least one score")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class ReferenceCache:
    path: Path
    identity_sha256: str
    manifest_sha256: str
    entries: Mapping[str, CacheEntry]
    reused: bool
    _identity_json: bytes = field(repr=False)

    @property
    def identity(self):
        return _parse(self._identity_json)  # A detached copy; callers cannot mutate the stored identity.

    def lookup(self, prompt_token_ids: Sequence[int], completion_token_ids: Sequence[int]) -> CacheEntry:
        return self.entries[_token_key(self.identity_sha256, prompt_token_ids, completion_token_ids)]


def _directory(output: Path):
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Cache output must be a real directory")
    children = list(output.iterdir())
    if {path.name for path in children} != FILES or any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("Cache has incomplete, untracked or non-regular artifacts; refusing reuse")


def load_cache(output_dir: str | os.PathLike, expected_identity: dict[str, Any], *,
               expected_manifest_sha256: str | None = None) -> ReferenceCache:
    """Read-only verification, including every artifact and original input file."""
    expected_identity = _parse(_canonical(expected_identity))
    validate_identity(expected_identity)
    output = Path(output_dir).absolute()
    _directory(output)
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest_hash = _digest(manifest_bytes)
    if expected_manifest_sha256 is not None and manifest_hash != _sha(expected_manifest_sha256):
        raise ValueError("Cache manifest differs from the pinned SHA-256")
    manifest = _parse(manifest_bytes)
    expected_fields = {"format", "schema_version", "status", "identity_sha256", "record_count", "artifacts", "score_origin"}
    if (not isinstance(manifest, dict) or set(manifest) != expected_fields or manifest.get("format") != FORMAT
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("status") != "cache_complete" or manifest.get("score_origin") != "caller_supplied"):
        raise ValueError("Cache completion manifest is malformed or unsupported")
    if type(manifest["record_count"]) is not int or manifest["record_count"] <= 0:
        raise ValueError("Cache record_count must be positive")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != FILES - {"manifest.json"}:
        raise ValueError("Cache manifest lacks the exact artifact inventory")
    payloads = {}
    for name, expected in artifacts.items():
        if (not isinstance(expected, dict) or set(expected) != {"sha256", "size_bytes"}
                or type(expected["size_bytes"]) is not int or expected["size_bytes"] <= 0):
            raise ValueError("Malformed artifact identity")
        payload = (output / name).read_bytes()
        if len(payload) != expected["size_bytes"] or _digest(payload) != _sha(expected["sha256"]):
            raise ValueError(f"Cache artifact hash/size mismatch: {name}")
        payloads[name] = payload
    identity = _parse(payloads["identity.json"])
    identity_hash = identity_sha256(expected_identity)
    if identity != expected_identity or identity_sha256(identity) != identity_hash or manifest["identity_sha256"] != identity_hash:
        raise ValueError("Cache has a different scoring identity; use a new output directory")
    lines = payloads["scores.jsonl"].splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Cache contains empty score rows")
    entries = _entries((_parse(line) for line in lines), identity_hash, require_key=True)
    if len(entries) != manifest["record_count"]:
        raise ValueError("Cache record count differs from complete artifact")
    validate_identity(expected_identity)
    _directory(output)
    if (output / "manifest.json").read_bytes() != manifest_bytes or any((output / name).read_bytes() != payload
                                                                 for name, payload in payloads.items()):
        raise ValueError("Cache changed while verifying")
    return ReferenceCache(output, identity_hash, manifest_hash, MappingProxyType(entries), True, _canonical(identity))


def write_or_reuse_cache(output_dir: str | os.PathLike, identity: dict[str, Any],
                         records: Iterable[Mapping[str, Any] | CacheEntry] | None = None, *,
                         expected_manifest_sha256: str | None = None) -> ReferenceCache:
    """Create once, or validate and read an existing cache without writing.

    Existing outputs may use records=None; supplied records must exactly equal
    their stored counterparts. A newly created directory is never cleaned up or
    repaired on failure: an incomplete directory must not look like a valid hit.
    """
    identity = _parse(_canonical(identity))
    validate_identity(identity)
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        existing = load_cache(output, identity, expected_manifest_sha256=expected_manifest_sha256)
        if records is not None and _entries(records, existing.identity_sha256) != dict(existing.entries):
            raise ValueError("Supplied scores differ from the immutable existing cache")
        return existing
    if expected_manifest_sha256 is not None:
        raise ValueError("A pinned existing manifest cannot be used to create a missing cache")
    if records is None:
        raise ValueError("New caches require caller-supplied scores; this module does not compute them")
    identity_hash = identity_sha256(identity)
    entries = _entries(records, identity_hash)
    identity_bytes = _canonical(identity) + b"\n"
    scores_bytes = b"".join(_canonical(entry.to_dict()) + b"\n" for entry in entries.values())
    validate_identity(identity)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, payload in (("identity.json", identity_bytes), ("scores.jsonl", scores_bytes)):
        with (output / name).open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        artifacts[name] = {"sha256": _digest(payload), "size_bytes": len(payload)}
    validate_identity(identity)
    manifest = {"format": FORMAT, "schema_version": 1, "status": "cache_complete", "identity_sha256": identity_hash,
                "record_count": len(entries), "artifacts": artifacts, "score_origin": "caller_supplied"}
    with (output / "manifest.json").open("xb") as stream:
        stream.write(_canonical(manifest) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    result = load_cache(output, identity)
    return ReferenceCache(result.path, result.identity_sha256, result.manifest_sha256,
                          result.entries, False, result._identity_json)
