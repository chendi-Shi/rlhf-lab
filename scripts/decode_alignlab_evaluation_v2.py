"""CPU-only exact token decoding for the isolated official scorer.

The caller pins this script, tokenizer bytes and the tokenizers distribution.
No Transformers, Torch, model, remote resource or training state is imported.
Input and output are single UTF-8 JSON documents on stdin/stdout.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def decode_request(request):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU decoder requires CUDA_VISIBLE_DEVICES=''")
    if set(request) != {"schema_version", "tokenizer_path", "tokenizer_sha256",
                       "tokenizers_version", "sequences"} or request["schema_version"] != 1:
        raise ValueError("Unexpected decoder request")
    path = Path(request["tokenizer_path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("Tokenizer must be a regular local file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != request["tokenizer_sha256"]:
        raise ValueError("Tokenizer SHA256 mismatch")
    version = importlib.metadata.version("tokenizers")
    if version != request["tokenizers_version"]:
        raise ValueError("Tokenizers distribution version mismatch")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    sequences = request["sequences"]
    if not isinstance(sequences, list):
        raise ValueError("Explicit sequence list required")
    vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)
    texts = []
    for ids in sequences:
        if (not isinstance(ids, list) or len(ids) > 2048
                or any(type(token) is not int or not 0 <= token < vocabulary_size
                       for token in ids)):
            raise ValueError("Invalid original generated token sequence")
        texts.append(tokenizer.decode(ids, skip_special_tokens=False))
    if path.read_bytes() != raw:
        raise ValueError("Tokenizer changed during decoding")
    if "torch" in sys.modules:
        raise RuntimeError("The CPU decoder must not import Torch")
    return {"schema_version": 1, "kind": "alignlab-cpu-decoded-actions-v1",
            "tokenizer_sha256": request["tokenizer_sha256"],
            "tokenizers_version": version,
            "sequence_count": len(sequences), "texts": texts,
            "torch_imported": False, "gpu_operations_performed": False}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    result = decode_request(json.loads(sys.stdin.buffer.read()))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
