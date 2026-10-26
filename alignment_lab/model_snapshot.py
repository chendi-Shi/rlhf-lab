"""Pinned public Qwen3 snapshot acquisition/verification; standard library only.

No model code, tensors, CUDA, pickle, or installed environment is loaded here.
The manifest is the last published file. Existing destinations and stale locks
are never overwritten or guessed to be resumable. Failed attempts stay intact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
API_URL = f"https://huggingface.co/api/models/{REPO_ID}/revision/{REVISION}?blobs=true"
DEFAULT_OUTPUT = ROOT / "models/qwen3-1.7b-70d244c"
TOKENIZER_DIRECTORY = ROOT / "models/qwen3-1.7b-tokenizer-70d244c"
TOKENIZER_FILES = ("config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "merges.txt", "vocab.json")
MANIFEST_NAME = "snapshot-manifest.json"
METADATA_NAME = "upstream-metadata.json"
WEIGHT_FILES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX_NAME = "model.safetensors.index.json"
BUFFER_BYTES = 1024 * 1024
RANGE_BYTES = 64 * 1024 * 1024
METADATA_LIMIT = 2 * 1024 * 1024
# Observed directly from the official immutable revision API, 2026-10-20.
# Git blob IDs of LFS files identify their canonical pointer, not weight bytes.
PINNED_FILES = {
    ".gitattributes": (1570, "52373fe24473b1aa44333d318f578ae6bf04b49b", None),
    "LICENSE": (11343, "6634c8cc3133b3848ec74b9f275acaaa1ea618ab", None),
    "README.md": (13963, "88aaef81cfe31bcdfe47203649a69ff12042826a", None),
    "config.json": (726, "044a86ecf7cb32238f3fae4184e55d354787edec", None),
    "generation_config.json": (239, "20a8a9156fc8c3f25295ca067f61fdf120d517c5", None),
    "merges.txt": (1671853, "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7", None),
    WEIGHT_FILES[0]: (3441185608, "1414f1bda72e17b63551d847c05c68e3396ba717", "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"),
    WEIGHT_FILES[1]: (622329984, "4aab3b30a4b8c6c1d0437a2576e0bea08507f12c", "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"),
    INDEX_NAME: (25605, "986d7db875b47d21f68530f6baac038f1b297b39", None),
    "tokenizer.json": (11422654, "cd71f61a15a522601badb3dc960d800d9cb3766c", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9732, "417d038a63fa3de29cfde265caedae14d1a58d92", None),
    "vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569", None),
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(payload):
    return json.loads(payload, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _hex(value, length):
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ValueError("Invalid lowercase digest")
    return value


def _plain(path, *, directory=False):
    path = Path(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"Symlinks and reparse points are forbidden: {path}")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError(f"Expected a regular {'directory' if directory else 'file'}: {path}")
    return info


def _parents(path):
    for parent in (Path(path), *Path(path).parents):
        if parent.exists() or parent.is_symlink():
            _plain(parent, directory=True)


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _hash_file(path):
    """Stream both local SHA256 and the actual non-LFS Git blob digest."""
    before = _plain(path)
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {before.st_size}\0".encode("ascii"))
    with Path(path).open("rb") as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError("File changed before hashing")
        for block in iter(lambda: stream.read(BUFFER_BYTES), b""):
            sha.update(block)
            blob.update(block)
        after = os.fstat(stream.fileno())
    if _signature(before) != _signature(after) or _signature(after) != _signature(_plain(path)):
        raise ValueError("File changed while hashing")
    return sha.hexdigest(), blob.hexdigest()


def _pointer(sha, size):
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode("ascii")


def _blob(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _validate_metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get("id") != REPO_ID or metadata.get("sha") != REVISION:
        raise ValueError("Official repository/revision identity mismatch")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list) or any(not isinstance(x, dict) for x in siblings):
        raise ValueError("Missing official file inventory")
    by_name = {item.get("rfilename"): item for item in siblings}
    if len(by_name) != len(siblings) or set(by_name) != set(PINNED_FILES):
        raise ValueError("Official file inventory changed")
    normalized = []
    for name, (size, blob, lfs) in sorted(PINNED_FILES.items()):
        item = by_name[name]
        if type(item.get("size")) is not int or item["size"] != size or item.get("blobId") != blob:
            raise ValueError(f"Official file size/Git blob changed: {name}")
        record = {"rfilename": name, "size": size, "blobId": blob}
        if lfs is not None:
            pointer = _pointer(lfs, size)
            expected_lfs = {"sha256": lfs, "size": size, "pointerSize": len(pointer)}
            if item.get("lfs") != expected_lfs or _blob(pointer) != blob:
                raise ValueError(f"Official LFS checksum/pointer changed: {name}")
            record["lfs"] = expected_lfs
        elif item.get("lfs") is not None:
            raise ValueError(f"Unexpected LFS metadata: {name}")
        normalized.append(record)
    return {"id": REPO_ID, "sha": REVISION, "siblings": normalized}


def _timeout(timeout):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
        raise ValueError("HTTP timeout must be 1..300 seconds")


def _request(url, *, timeout, opener, byte_range=None):
    headers = {"User-Agent": "alignlab-pinned-snapshot/1", "Accept-Encoding": "identity"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return opener(urllib.request.Request(url, headers=headers), timeout=timeout)


def fetch_metadata(*, timeout=30, opener=None):
    """Only the small official metadata API is contacted; no weights."""
    _timeout(timeout)
    with _request(API_URL, timeout=timeout, opener=opener or urllib.request.urlopen) as response:
        if response.status != 200:
            raise ValueError("Unexpected metadata HTTP status")
        payload = response.read(METADATA_LIMIT + 1)
    if len(payload) > METADATA_LIMIT:
        raise ValueError("Oversized metadata response")
    return _validate_metadata(_parse(payload))


def _space(parent, total):
    current = Path(parent)
    while not current.exists():
        current = current.parent
    free = shutil.disk_usage(current).free
    required = total + max(256 * 1024 * 1024, total // 20)
    return {"free_bytes": free, "required_free_bytes": required, "sufficient": free >= required}


def plan_snapshot(output=DEFAULT_OUTPUT, *, timeout=30, opener=None):
    """Read-only plan, including pinned metadata and required free disk space."""
    output = Path(output).absolute()
    _parents(output.parent)
    metadata = fetch_metadata(timeout=timeout, opener=opener)
    tokenizer_hashes = _verify_tokenizer()
    total = sum(info[0] for info in PINNED_FILES.values())
    return {
        "mode": "plan", "repo_id": REPO_ID, "revision": REVISION, "output": str(output),
        "destination_exists": output.exists() or output.is_symlink(),
        "total_bytes": total, "weights_bytes": sum(PINNED_FILES[n][0] for n in WEIGHT_FILES),
        "space": _space(output.parent, total),
        "metadata_url": API_URL, "metadata": metadata,
        "existing_tokenizer_sha256": tokenizer_hashes,
        "large_weights_downloaded": False,
    }


def _verify_file(path, name, expected_sha=None):
    size, blob, lfs = PINNED_FILES[name]
    if _plain(path).st_size != size:
        raise ValueError(f"Actual size mismatch: {name}")
    sha, actual_blob = _hash_file(path)
    if expected_sha is not None and sha != _hex(expected_sha, 64):
        raise ValueError(f"Actual SHA256 mismatch: {name}")
    if lfs is not None:
        if sha != lfs or _blob(_pointer(sha, size)) != blob:
            raise ValueError(f"Actual LFS/Git pointer mismatch: {name}")
    elif actual_blob != blob:
        raise ValueError(f"Actual Git blob mismatch: {name}")
    return {"size_bytes": size, "sha256": sha, "upstream_git_blob": blob,
            "upstream_lfs_sha256": lfs}


def _verify_tokenizer(snapshot=None):
    """Same revision must match the already accepted separate tokenizer bytes."""
    _parents(TOKENIZER_DIRECTORY)
    hashes = {}
    for name in TOKENIZER_FILES:
        local = _verify_file(TOKENIZER_DIRECTORY / name, name)
        hashes[name] = local["sha256"]
        if snapshot is not None and _hash_file(snapshot / name)[0] != local["sha256"]:
            raise ValueError(f"Existing fixed tokenizer conflicts with snapshot: {name}")
    return hashes


def _model_structure(path):
    config = _parse((path / "config.json").read_bytes())
    if not isinstance(config, dict) or config.get("model_type") != "qwen3":
        raise ValueError("Actual config.model_type must be qwen3")
    index = _parse((path / INDEX_NAME).read_bytes())
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if (not isinstance(mapping, dict) or not mapping
            or any(not isinstance(k, str) or not k or v not in WEIGHT_FILES for k, v in mapping.items())
            or set(mapping.values()) != set(WEIGHT_FILES)):
        raise ValueError("Actual safetensors index has missing or unexpected shards")
    return config, index


def verify_snapshot(path, expected_manifest_sha256, *, verify_hashes=True):
    """Offline read-only verification of every byte; returns the JSON manifest.

    Caller must pin the manifest digest. Metadata-only acceptance is deliberately
    unavailable: verify_hashes=False raises rather than returning a trusted model.
    """
    if verify_hashes is not True:
        raise ValueError("Full actual file hashes are mandatory; verify_hashes must be True")
    expected_manifest_sha256 = _hex(expected_manifest_sha256, 64)
    path = Path(path).absolute()
    _parents(path)
    expected_names = set(PINNED_FILES) | {MANIFEST_NAME, METADATA_NAME}
    if {p.name for p in path.iterdir()} != expected_names:
        raise ValueError("Snapshot is partial or contains untracked artifacts")
    manifest_path = path / MANIFEST_NAME
    manifest_sha, _ = _hash_file(manifest_path)
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("Pinned snapshot manifest SHA256 mismatch")
    manifest_payload = manifest_path.read_bytes()
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_sha:
        raise ValueError("Manifest changed before parsing")
    manifest = _parse(manifest_payload)
    required = {"schema", "repo_id", "revision", "model_type", "format", "files",
                "weight_files", "source_metadata", "created_utc"}
    if (not isinstance(manifest, dict) or set(manifest) != required
            or type(manifest["schema"]) is not int or manifest["schema"] != 1
            or manifest["repo_id"] != REPO_ID or manifest["revision"] != REVISION
            or manifest["model_type"] != "qwen3" or manifest["format"] != "safetensors"
            or manifest["weight_files"] != list(WEIGHT_FILES)
            or not isinstance(manifest["created_utc"], str)):
        raise ValueError("Snapshot manifest identity/schema mismatch")
    metadata = manifest["source_metadata"]
    if (not isinstance(metadata, dict) or set(metadata) != {"path", "sha256", "url"}
            or metadata["path"] != METADATA_NAME or metadata["url"] != API_URL):
        raise ValueError("Snapshot source metadata identity mismatch")
    metadata_path = path / METADATA_NAME
    if _plain(metadata_path).st_size > METADATA_LIMIT:
        raise ValueError("Oversized cached upstream metadata")
    metadata_payload = metadata_path.read_bytes()
    if hashlib.sha256(metadata_payload).hexdigest() != _hex(metadata["sha256"], 64):
        raise ValueError("Actual upstream metadata hash mismatch")
    _validate_metadata(_parse(metadata_payload))
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(PINNED_FILES):
        raise ValueError("Snapshot manifest file inventory mismatch")
    before = {name: _signature(_plain(path / name)) for name in expected_names}
    for name in sorted(PINNED_FILES):
        record = files[name]
        if not isinstance(record, dict) or set(record) != {"size_bytes", "sha256", "upstream_git_blob", "upstream_lfs_sha256"}:
            raise ValueError(f"Invalid file manifest entry: {name}")
        actual = _verify_file(path / name, name, record["sha256"])
        if actual != record:
            raise ValueError(f"File manifest disagrees with pinned upstream: {name}")
    _model_structure(path)
    _verify_tokenizer(path)
    if {p.name for p in path.iterdir()} != expected_names or any(
            _signature(_plain(path / name)) != signature for name, signature in before.items()):
        raise ValueError("Snapshot changed during verification")
    return manifest


def _write_new(path, payload):
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _download_file(path, name, *, timeout, opener, event):
    size = PINNED_FILES[name][0]
    url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{name}"
    temporary = path.with_name(path.name + ".part")
    with temporary.open("xb") as output:
        # Bounded HTTP ranges for large files, bounded read buffers for all files.
        for start in range(0, size, RANGE_BYTES):
            end = min(start + RANGE_BYTES, size) - 1
            ranged = size > RANGE_BYTES
            with _request(url, timeout=timeout, opener=opener,
                          byte_range=(start, end) if ranged else None) as response:
                if ranged:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError(f"Unexpected HTTP byte range: {name}")
                elif response.status != 200:
                    raise ValueError(f"Unexpected HTTP status: {name}")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected compressed transfer")
                count = end - start + 1
                header = response.headers.get("Content-Length")
                if header is not None and header != str(count):
                    raise ValueError(f"HTTP length mismatch: {name}")
                remaining = count
                while remaining:
                    block = response.read(min(remaining, BUFFER_BYTES))
                    if not block:
                        raise ValueError(f"Truncated HTTP body: {name}")
                    if len(block) > remaining:
                        raise ValueError(f"Oversized HTTP body: {name}")
                    output.write(block)
                    remaining -= len(block)
                if response.read(1):
                    raise ValueError(f"Oversized HTTP body: {name}")
            output.flush()
            event("range_complete", file=name, completed_bytes=end + 1, total_bytes=size)
        os.fsync(output.fileno())
    record = _verify_file(temporary, name)
    # New private destination, no overwrites: rename fails if an entry appeared.
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary.rename(path)
    return record


def download_snapshot(output=DEFAULT_OUTPUT, *, timeout=30, opener=None):
    """Explicit acquisition; existing/failed/locked destinations are rejected.

    A sibling O_EXCL lock and append-only attempt log are created. Failures retain
    the lock, partial file and log; no automatic stale-lock removal or resume.
    Completed snapshots are immutable. Use verify_snapshot to reuse them.
    """
    _timeout(timeout)
    output = Path(output).absolute()
    _parents(output.parent)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Destination already exists; verify complete snapshots or inspect partial evidence")
    # Plan/network metadata verification precedes any local mutation.
    plan = plan_snapshot(output, timeout=timeout, opener=opener)
    if not plan["space"]["sufficient"]:
        raise OSError("Insufficient free space for pinned snapshot plus safety reserve")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.download.lock"
    attempt = output.parent / f".{output.name}.attempt-{uuid.uuid4().hex}.jsonl"
    lock_data = {"pid": os.getpid(), "started_utc": _utc(), "output": str(output),
                 "repo_id": REPO_ID, "revision": REVISION, "attempt_log": str(attempt)}
    _write_new(lock, _json_bytes(lock_data))

    def event(status, **details):
        with attempt.open("ab") as stream:
            stream.write(_json_bytes({"status": status, "utc": _utc(), **details}))
            stream.flush()
            os.fsync(stream.fileno())

    try:
        event("started", **lock_data, required_free_bytes=plan["space"]["required_free_bytes"])
        output.mkdir(exist_ok=False)
        _write_new(output / METADATA_NAME, _json_bytes(plan["metadata"]))
        records = {}
        opener = opener or urllib.request.urlopen
        for name in sorted(PINNED_FILES):
            records[name] = _download_file(output / name, name, timeout=timeout, opener=opener, event=event)
            event("file_verified", file=name, **records[name])
        _model_structure(output)
        manifest = {
            "schema": 1, "repo_id": REPO_ID, "revision": REVISION,
            "model_type": "qwen3", "format": "safetensors", "files": records,
            "weight_files": list(WEIGHT_FILES), "created_utc": _utc(),
            "source_metadata": {"path": METADATA_NAME, "url": API_URL,
                                "sha256": hashlib.sha256((output / METADATA_NAME).read_bytes()).hexdigest()},
        }
        payload = _json_bytes(manifest)
        # Manifest-last: no final manifest exists during partial acquisition.
        temporary_manifest = output / (MANIFEST_NAME + ".part")
        _write_new(temporary_manifest, payload)
        temporary_manifest.rename(output / MANIFEST_NAME)
        manifest_sha = hashlib.sha256(payload).hexdigest()
        verified = verify_snapshot(output, manifest_sha)
        event("completed", manifest_sha256=manifest_sha)
        lock.unlink()
        return {"path": str(output), "manifest_sha256": manifest_sha, "manifest": verified}
    except BaseException as error:
        event("failed", error_type=type(error).__name__, error=str(error),
              manifest_published=(output / MANIFEST_NAME).exists(),
              lock_retained=True, partial_files_retained=True)
        raise
