"""Fetch revision-pinned public data/tokenizer assets; never model weights or CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    {
        "kind": "datasets", "repo": "nvidia/HelpSteer3",
        "revision": "f6d145777bcbde96137596340fab89793acd1031",
        "destination": "data/raw/helpsteer3-preference-f6d1457",
        "files": ["README.md", "preference/train.jsonl.gz", "preference/validation.jsonl.gz"],
    },
    {
        "kind": "models", "repo": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "destination": "models/qwen3-1.7b-tokenizer-70d244c",
        "files": ["LICENSE", "README.md", "config.json", "generation_config.json",
                  "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"],
    },
)


def emit(**message):
    print(json.dumps(message, ensure_ascii=True), flush=True)


def hashes(path: Path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    return {"size": size, "sha256": sha256.hexdigest(), "git_blob_sha1": git_blob.hexdigest()}


def verify(path: Path, entry: dict):
    result = hashes(path)
    if result["size"] != entry["size"]:
        raise ValueError(f"Size mismatch: {path}")
    lfs = entry.get("lfs")
    if lfs and result["sha256"] != lfs["sha256"]:
        raise ValueError(f"LFS SHA256 mismatch: {path}")
    if not lfs and result["git_blob_sha1"] != entry["blobId"]:
        raise ValueError(f"Git blob mismatch: {path}")
    return result


def fetch_source(spec: dict):
    destination = ROOT / spec["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "source-manifest.json"
    api_url = f"https://huggingface.co/api/{spec['kind']}/{spec['repo']}/revision/{spec['revision']}?blobs=true"
    with urllib.request.urlopen(api_url, timeout=45) as response:
        metadata = json.load(response)
    if metadata["sha"] != spec["revision"]:
        raise ValueError("Upstream revision mismatch")
    entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    existing = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["spec"] != spec:
            raise ValueError(f"Existing manifest has another identity: {destination}")
    prefix = "datasets/" if spec["kind"] == "datasets" else ""
    artifacts = {}
    for filename in spec["files"]:
        if filename.endswith((".safetensors", ".bin", ".pt")):
            raise ValueError("Model weights are forbidden in this CPU asset fetcher")
        entry = entries[filename]
        target = destination / filename
        url = f"https://huggingface.co/{prefix}{spec['repo']}/resolve/{spec['revision']}/{filename}"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part." + uuid.uuid4().hex)
            emit(event="download_start", file=filename, bytes=entry["size"])
            downloaded = 0
            last = time.monotonic()
            request = urllib.request.Request(url, headers={"User-Agent": "AlignLab-public-assets/1"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    downloaded += len(block)
                    if time.monotonic() - last >= 15:
                        emit(event="download_progress", file=filename, bytes=downloaded)
                        last = time.monotonic()
            verify(partial, entry)
            if target.exists():
                raise FileExistsError(f"Target appeared during fetch; retaining partial: {partial}")
            os.rename(partial, target)
        result = verify(target, entry)
        result["url"] = url
        result["upstream"] = entry
        artifacts[filename] = result
        emit(event="verified", file=str(target.relative_to(ROOT)), bytes=result["size"])
    manifest = {"schema_version": 1, "spec": spec, "upstream_metadata_url": api_url,
                "artifacts": artifacts, "contains_model_weights": False,
                "training_started": False}
    if existing:
        if existing != manifest:
            raise ValueError("Manifest changed; refusing to overwrite")
    else:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return str(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        emit(sources=SOURCES, scope="CPU audit assets only; pass --execute to fetch")
        return
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    lock = reports / ".alignlab-assets.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at_epoch": time.time()}, handle)
    try:
        manifests = [fetch_source(spec) for spec in SOURCES]
        emit(status="completed", manifests=manifests, training_started=False)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
