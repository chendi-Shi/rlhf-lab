"""Download the pinned official GPT-2 safetensors and tokenizer, with hashes.

Standard-library-only, CPU/network operation. No pickle loading or model code
execution. Interrupted range blocks remain reusable under the ignored model
directory; the final weight is published only after official LFS SHA-256 passes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHT_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
WEIGHT_SIZE = 548105171
FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
         "tokenizer.json", "vocab.json", "merges.txt", "README.md", "model.safetensors")
API = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git_blob_sha(path):
    path = Path(path)
    hasher = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fetch(url, *, start=None, end=None, total=None):
    headers = {"User-Agent": "rlhf-lab-pinned-download/1"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if start is not None:
            expected_range = f"bytes {start}-{end}/{total}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server returned an unexpected byte range")
            limit = end - start + 1
            content = response.read(limit + 1)
            if len(content) != limit:
                raise ValueError("Truncated model download block")
        else:
            # Metadata and tokenizer files are deliberately bounded.
            content = response.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError("Unexpected large metadata/tokenizer response")
        return content


def retry(action, description):
    last = None
    for attempt in range(5):
        try:
            return action()
        except (OSError, ValueError, HTTPException) as error:
            last = error
            if attempt < 4:
                print(f"Retry {attempt + 1}/4: {description} ({type(error).__name__})", flush=True)
                time.sleep(1 + attempt)
    raise RuntimeError(f"Failed: {description}") from last


def verify_file(path, metadata):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        return False
    if "lfs" in metadata:
        return digest(path) == metadata["lfs"]["sha256"]
    return git_blob_sha(path) == metadata["blobId"]


def download_weight(path, metadata, workers=4):
    if verify_file(path, metadata):
        print("Reusing verified model.safetensors", flush=True)
        return
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/model.safetensors"
    block_size = 4 * 1024 * 1024
    starts = list(range(0, metadata["size"], block_size))
    chunks = path.parent / ".download-parts" / WEIGHT_SHA256
    chunks.mkdir(parents=True, exist_ok=True)

    def block(start):
        end = min(start + block_size, metadata["size"]) - 1
        chunk = chunks / f"{start:012d}.chunk"
        if chunk.is_file() and chunk.stat().st_size == end - start + 1:
            return
        content = retry(lambda: fetch(url, start=start, end=end, total=metadata["size"]), f"weight bytes {start}-{end}")
        temporary = chunk.with_suffix(".part")
        temporary.write_bytes(content)
        os.replace(temporary, chunk)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(block, start) for start in starts]
        for completed, job in enumerate(as_completed(pending), 1):
            job.result()
            if completed % 8 == 0 or completed == len(starts):
                print(f"GPT-2 blocks complete: {completed}/{len(starts)}", flush=True)
    temporary = path.with_suffix(".safetensors.part")
    with temporary.open("wb") as output:
        for start in starts:
            with (chunks / f"{start:012d}.chunk").open("rb") as source:
                shutil.copyfileobj(source, output)
    if not verify_file(temporary, metadata):
        raise ValueError("Assembled model failed the official LFS SHA-256; inspect cached blocks")
    os.replace(temporary, path)


def download_model(output=ROOT / "models" / "gpt2", *, workers=4, verify_only=False):
    output = Path(output).resolve()
    if workers < 1 or workers > 8:
        raise ValueError("workers must be 1..8")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "upstream-files.json"
    if verify_only:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = json.loads(retry(lambda: fetch(API), "pinned Hugging Face metadata"))
        metadata = {"id": metadata["id"], "sha": metadata["sha"],
                    "siblings": [item for item in metadata["siblings"] if item["rfilename"] in FILES]}
    if metadata["id"] != MODEL_ID or metadata["sha"] != REVISION:
        raise ValueError("Model revision identity mismatch")
    by_name = {item["rfilename"]: item for item in metadata["siblings"]}
    if set(by_name) != set(FILES):
        raise ValueError("Pinned revision does not contain all expected files")
    weights = by_name["model.safetensors"]
    if weights["size"] != WEIGHT_SIZE or weights.get("lfs", {}).get("sha256") != WEIGHT_SHA256:
        raise ValueError("Official GPT-2 weight identity differs from the pinned value")
    if not verify_only:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for name in FILES:
        destination, meta = output / name, by_name[name]
        if verify_only:
            if not verify_file(destination, meta):
                raise ValueError(f"Missing or altered model artifact: {name}")
            continue
        if name == "model.safetensors":
            download_weight(destination, meta, workers)
        elif not verify_file(destination, meta):
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
            content = retry(lambda: fetch(url), name)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(content)
            if not verify_file(temporary, meta):
                raise ValueError(f"Official Git blob checksum mismatch: {name}")
            os.replace(temporary, destination)
        print(f"Verified {name} ({meta['size']:,} bytes)", flush=True)
    manifest = {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
                "source_metadata_url": API, "format": "safetensors; no pickle weights downloaded",
                "files": {name: {"bytes": by_name[name]["size"], "sha256": digest(output / name),
                                  "upstream_git_blob": by_name[name]["blobId"],
                                  "upstream_lfs_sha256": by_name[name].get("lfs", {}).get("sha256")}
                          for name in FILES}}
    if not verify_only:
        (output / "local-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "directory": str(output), "model_id": MODEL_ID,
                      "revision": REVISION, "weight_sha256": WEIGHT_SHA256}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "gpt2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    download_model(args.output, workers=args.workers, verify_only=args.verify_only)
