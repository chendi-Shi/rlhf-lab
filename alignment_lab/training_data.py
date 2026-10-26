"""Verified CPU loading of accepted v3 training partitions.

Only training JSONL is deserialized. All audit artifacts (including holdouts)
and raw inputs are streamed as bytes for integrity verification, never used as
training examples. No audit is rerun and no model/device is loaded.

The manifest SHA must come from the caller's accepted experiment identity.
The loader validates that anchored audit, not its scientific methodology or GPU
capacity. It always encodes with a fresh local-only tokenizer from the manifest;
an optional caller codec is compared with that real tokenizer, never trusted as
an implementation. Returned rows contain immutable token tuples, not tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import unicodedata
from typing import Any

from . import chat_codec
from .chat_codec import Qwen3ChatCodec, SERIALIZATION_VERSION, TERMINAL_EOS_TOKEN_ID

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = frozenset({"sft_train", "preference_train", "ppo_prompts"})
ARTIFACTS = frozenset({
    "preference_train.jsonl", "eligible_train_pool.jsonl", "sft_train.jsonl",
    "ppo_prompts.jsonl", "tuning_dev.jsonl", "diagnostic_holdout.jsonl",
    "final_holdout.jsonl", "ties.jsonl", "exclusions.jsonl",
    "assigned_train_lengths.jsonl", "audit.json",
})
SOURCES = frozenset({
    "alignment_lab/data_audit.py", "alignment_lab/__init__.py",
    "alignment_lab/chat_codec.py", "scripts/audit_alignlab_data.py",
})
PPO_FIELDS = (
    "record_id", "prompt_id", "cluster_id", "context", "domain", "language",
    "source_split", "source_index",
)


@dataclass(frozen=True)
class TrainingRow:
    record_id: str
    cluster_id: str
    prompt_id: str
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...] | None = None
    rejected_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TrainingPartition:
    rows: tuple[TrainingRow, ...]
    partition: str
    manifest_sha256: str
    identity_sha256: str
    serialization_version: str
    _identity_json: bytes = field(repr=False)

    @property
    def dataset_identity(self) -> dict[str, Any]:
        """Detached copy; mutation cannot alter the partition's stored identity."""
        return json.loads(self._identity_json)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA-256")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(payload: bytes):
    return json.loads(payload, object_pairs_hook=_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular, non-symlink file: {path}")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        result = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError(f"File changed while hashing: {path}")
    return result


def _check_hash(path: Path, expected: str):
    if _file_hash(path) != _sha(expected):
        raise ValueError(f"File hash differs from accepted audit: {path}")


def _relative(directory: Path, name: str) -> Path:
    if (not isinstance(name, str) or "\\" in name or ":" in name
            or name.startswith("/") or not name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Unsafe relative artifact path")
    path = directory.joinpath(*PurePosixPath(name).parts)
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Artifact escapes its declared directory")
    return path


def _absolute(value: Any) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Manifest paths must be absolute")
    return Path(value)


def _verified_bytes(path: Path, expected: str) -> bytes:
    _check_hash(path, expected)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"File changed before parsing: {path}")
    return payload


def _ready(summary, name):
    if (not isinstance(summary, dict) or summary.get("data_ready") is not True
            or summary.get("readiness_blockers") != []
            or summary.get("training_started") is not False):
        raise ValueError(f"{name} must be accepted and ready with no blockers")


def _verify(audit_dir: Path, expected_manifest_sha256: str):
    if audit_dir.is_symlink() or not audit_dir.is_dir():
        raise ValueError("Audit must be an existing real directory")
    manifest = _parse(_verified_bytes(audit_dir / "manifest.json", expected_manifest_sha256))
    _ready(manifest, "Manifest")
    if manifest.get("status") != "audit_complete":
        raise ValueError("Manifest must be audit_complete")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or _digest(identity) != _sha(manifest.get("identity_sha256")):
        raise ValueError("Audit identity self-hash mismatch")
    if (identity.get("version") != "alignlab-data-audit-v3-native-nfc"
            or identity.get("serialization_version") != SERIALIZATION_VERSION):
        raise ValueError("Unsupported audit/serialization version")
    if any(not isinstance(identity.get(k), str) or not re.fullmatch(r"[0-9a-f]{40}", identity[k])
           for k in ("source_revision", "model_tokenizer_revision")):
        raise ValueError("Pinned upstream revisions required")
    inputs = identity.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"train", "validation"}:
        raise ValueError("Exact raw input inventory required")
    for record in inputs.values():
        _check_hash(_absolute(record["path"]), record["sha256"])
    sources = identity.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != SOURCES:
        raise ValueError("Exact audit source inventory required")
    for name, sha in sources.items():
        _check_hash(_relative(ROOT, name), sha)
    # Also bind the codec actually imported by this process, not just a path.
    _check_hash(Path(chat_codec.__file__), sources["alignment_lab/chat_codec.py"])
    _check_hash(ROOT / "docs/ALIGNLAB_V2.md", identity["protocol_sha256"])
    libraries = identity.get("tokenizer_library_versions")
    if not isinstance(libraries, dict) or set(libraries) != {"transformers", "tokenizers", "jinja2"}:
        raise ValueError("Exact tokenizer library versions required")
    for name, version in libraries.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Tokenizer library version drift: {name}")
    tokenizer = identity.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("files"), dict):
        raise ValueError("Pinned local tokenizer inventory required")
    directory = _absolute(tokenizer["path"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Tokenizer must be a real local directory")
    inventory = tokenizer["files"]
    if not {"tokenizer.json", "tokenizer_config.json", "config.json"} <= set(inventory):
        raise ValueError("Incomplete tokenizer inventory")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and not any(s.startswith(".") for s in p.relative_to(directory).parts)}
    if actual != set(inventory):
        raise ValueError("Tokenizer inventory changed")
    for name, sha in inventory.items():
        path = _relative(directory, name)
        if path.suffix in {".bin", ".safetensors", ".pt", ".pth"}:
            raise ValueError("Tokenizer directory must not contain model weights")
        _check_hash(path, sha)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != ARTIFACTS:
        raise ValueError("Exact complete audit artifact inventory required")
    if {p.name for p in audit_dir.iterdir()} != ARTIFACTS | {"manifest.json"}:
        raise ValueError("Incomplete or untracked audit artifacts")
    for name, sha in artifacts.items():
        _check_hash(audit_dir / name, sha)
    audit = _parse(_verified_bytes(audit_dir / "audit.json", artifacts["audit.json"]))
    _ready(audit, "Hashed audit report")
    return manifest, identity, audit


def _tokenizer_state(tokenizer) -> bytes:
    # Backend serialization includes vocabulary, merges, normalizer, decoder,
    # added tokens and backend truncation/padding. The wrapper supplies template
    # and special-token behavior not represented by that backend JSON.
    return _canonical({
        "backend": json.loads(tokenizer.backend_tokenizer.to_str()),
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": tokenizer.special_tokens_map,
        "special_ids": tokenizer.all_special_ids,
        "split_special_tokens": tokenizer.split_special_tokens,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "add_prefix_space": getattr(tokenizer, "add_prefix_space", None),
    })


def _local_codec(identity, supplied):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(identity["tokenizer"]["path"],
                                              local_files_only=True, trust_remote_code=False)
    fresh = Qwen3ChatCodec(tokenizer)
    if supplied is not None:
        if (type(supplied) is not Qwen3ChatCodec
                or type(supplied.tokenizer) is not type(tokenizer)
                or _tokenizer_state(supplied.tokenizer) != _tokenizer_state(tokenizer)
                or supplied.eos_token_id != fresh.eos_token_id
                or supplied.eos_token != fresh.eos_token
                or supplied.special_token_ids != fresh.special_token_ids):
            raise ValueError("Supplied codec differs from the actual pinned local tokenizer")
    return fresh


def _rows(audit_dir, manifest, partition):
    payload = _verified_bytes(audit_dir / f"{partition}.jsonl", manifest["artifacts"][f"{partition}.jsonl"])
    lines = payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Training partition must contain nonempty JSONL records")
    result = [_parse(line) for line in lines]
    if any(not isinstance(row, dict) for row in result):
        raise ValueError("Training rows must be JSON objects")
    return result


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_preference(row, config):
    for key in ("record_id", "cluster_id", "prompt_id"):
        _sha(row.get(key))
    if row.get("source_split") != "train" or row.get("local_split") != "train":
        raise ValueError("Only source train / local train rows may be used for training")
    if type(row.get("source_index")) is not int or row["source_index"] < 0:
        raise ValueError("Invalid training source index")
    score = row.get("overall_preference")
    if type(score) is not int or score == 0 or not -3 <= score <= 3:
        raise ValueError("Training requires non-tie official preference labels")
    context = chat_codec.validate_context(row.get("context"))
    a, b = chat_codec.validate_response(row.get("response1")), chat_codec.validate_response(row.get("response2"))
    if (row.get("chosen"), row.get("rejected")) != ((a, b) if score < 0 else (b, a)):
        raise ValueError("Chosen/rejected disagrees with official preference sign")
    payload = {k: row.get(k) for k in ("context", "response1", "response2", "overall_preference",
                                      "domain", "language", "individual_scores")}
    if _digest(payload) != row["record_id"]:
        raise ValueError("record_id does not match raw preference content")
    norm = [{"role": m["role"], "content": " ".join(unicodedata.normalize("NFKC", m["content"]).casefold().split())}
            for m in context]
    if _digest(norm) != row["prompt_id"]:
        raise ValueError("prompt_id does not match normalized context")
    tokens = row.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError("Audited token lengths required")
    for key in ("prompt", "response1", "response2"):
        length = _positive_int(tokens.get(key), "cached token count")
        limit = config["prompt_limit"] if key == "prompt" else config["response_limit"]
        if length > limit:
            raise ValueError("Training row exceeds accepted length limits")


def _summary(rows, audit, name):
    summary = audit.get("partition_statistics", {}).get(name, {})
    if (summary.get("rows") != len(rows)
            or summary.get("independent_prompt_clusters") != len({r["cluster_id"] for r in rows})
            or summary.get("unique_normalized_contexts") != len({r["prompt_id"] for r in rows})):
        raise ValueError("Training partition differs from audited counts")
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training record_id")


def load_training_partition(audit_dir, partition, expected_manifest_sha256, codec=None) -> TrainingPartition:
    """Verify an accepted audit and encode one allowed partition on CPU.

    ``partition`` is a bare name, never a filename/path. Optional ``codec`` is
    identity-checked only; a fresh pinned local codec performs all actual work.
    No truncation, resampling, splitting or batching is performed here.
    """
    if partition not in PARTITIONS:
        raise ValueError("Only sft_train, preference_train, or ppo_prompts are training partitions")
    audit_dir = Path(audit_dir).absolute()
    expected_manifest_sha256 = _sha(expected_manifest_sha256)
    manifest, identity, audit = _verify(audit_dir, expected_manifest_sha256)
    config = identity["config"]
    for key in ("prompt_limit", "response_limit", "minimum_pairs", "minimum_clusters", "minimum_sft"):
        _positive_int(config.get(key), key)
    local_codec = _local_codec(identity, codec)
    preferences = _rows(audit_dir, manifest, "preference_train")
    for row in preferences:
        _validate_preference(row, config)
    _summary(preferences, audit, "preference_train")
    if (len(preferences) < config["minimum_pairs"]
            or len({r["cluster_id"] for r in preferences}) < config["minimum_clusters"]):
        raise ValueError("Accepted training data is below its declared minimum scale")
    membership = {row["record_id"]: row for row in preferences}
    rows = preferences if partition == "preference_train" else _rows(audit_dir, manifest, partition)
    _summary(rows, audit, partition)
    if partition == "sft_train":
        if len(rows) < config["minimum_sft"] or len({r["cluster_id"] for r in rows}) != len(rows):
            raise ValueError("SFT requires its minimum count and one row per cluster")
    if partition == "ppo_prompts" and len({r["prompt_id"] for r in rows}) != len(rows):
        raise ValueError("PPO prompt partition contains duplicate prompt identities")
    encoded = []
    for row in rows:
        parent = membership.get(row["record_id"])
        expected = {key: parent[key] for key in PPO_FIELDS} if parent and partition == "ppo_prompts" else parent
        if parent is None or row != expected:
            raise ValueError("Training row is not an exact member/projection of preference_train")
        if partition == "sft_train" and abs(parent["overall_preference"]) < 2:
            raise ValueError("SFT row does not satisfy the strong-preference rule")
        prompt = local_codec.encode_prompt(parent["context"])
        if len(prompt.input_ids) != parent["tokens"]["prompt"]:
            raise ValueError("Prompt token length drift from audited count")
        chosen = rejected = None
        if partition != "ppo_prompts":
            completions = {}
            for key in ("response1", "response2"):
                ids = local_codec.encode_completion(prompt, parent[key]).completion_ids
                if (not ids or ids[-1] != TERMINAL_EOS_TOKEN_ID
                        or ids.count(TERMINAL_EOS_TOKEN_ID) != 1):
                    raise ValueError("Completion must contain exactly one terminal assistant EOS")
                if len(ids) != parent["tokens"][key]:
                    raise ValueError("Response token length drift from audited count")
                completions[key] = ids
            a, b = ("response1", "response2") if parent["overall_preference"] < 0 else ("response2", "response1")
            chosen = completions[a]
            rejected = completions[b] if partition == "preference_train" else None
        encoded.append(TrainingRow(row["record_id"], row["cluster_id"], row["prompt_id"],
                                   prompt.input_ids, chosen, rejected))
    # A long encoding pass must never publish a mixture of input identities.
    second, _, _ = _verify(audit_dir, expected_manifest_sha256)
    if second != manifest:
        raise ValueError("Accepted audit changed while encoding")
    return TrainingPartition(tuple(encoded), partition, expected_manifest_sha256,
                             manifest["identity_sha256"], SERIALIZATION_VERSION, _canonical(identity))
