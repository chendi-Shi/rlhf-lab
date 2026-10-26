"""Isolated, offline tokenizer-only re-encoding of three selected raw inputs.

No model weights, GPU, training state, remote code or output file is accessed.
The caller pins this script, codec, interpreter and official tokenizer bytes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/encode_alignlab_evaluation_capacity.py", "alignment_lab/chat_codec.py",
           "alignment_lab/__init__.py")
TOKEN_FILES = {"config.json", "generation_config.json", "tokenizer_config.json",
               "tokenizer.json", "merges.txt", "vocab.json"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(path, sha):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Regular absolute pinned file required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Encoder file SHA mismatch: " + str(path))


def encode_request(request):
    if any(os.environ.get(k) != v for k, v in
           {"CUDA_VISIBLE_DEVICES": "", "USE_TORCH": "0", "USE_TF": "0",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items()):
        raise ValueError("Offline isolated tokenizer-only environment required")
    fields = {"schema_version", "kind", "python_path", "python_sha256", "source_sha256",
              "tokenizer_dir", "tokenizer_files", "tokenizers_version", "transformers_version", "cases"}
    if (set(request) != fields or type(request["schema_version"]) is not int or request["schema_version"] != 1
            or request["kind"] != "alignlab-capacity-input-encoding-request-v1"
            or set(request["source_sha256"]) != set(SOURCES)
            or set(request["tokenizer_files"]) != TOKEN_FILES
            or set(request["cases"]) != {"ifeval", "helpsteer_rm", "rm_bench"}
            or request["transformers_version"] != "5.8.1"):
        raise ValueError("Exact pinned three-case encoding schema required")
    python = ROOT / ".runtime/alignlab-v2/Scripts/python.exe"
    if Path(request["python_path"]).resolve() != python.resolve() or Path(sys.executable).resolve() != python.resolve():
        raise ValueError("Use the fixed independent training interpreter")
    directory = Path(request["tokenizer_dir"])
    allowed = TOKEN_FILES | {"README.md", "LICENSE", "source-manifest.json"}
    if ({p.name for p in directory.iterdir()} - allowed
            or any(not p.is_file() or p.is_symlink() for p in directory.iterdir())):
        raise ValueError("Unpinned tokenizer side files are forbidden")

    def files():
        pin(python, request["python_sha256"])
        for name, sha in request["source_sha256"].items():
            pin(ROOT / name, sha)
        for name, sha in request["tokenizer_files"].items():
            pin(directory / name, sha)

    files()
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")}
    if any(versions[name] != request[name + "_version"] for name in versions):
        raise ValueError("Pinned tokenizer package version differs")
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported before independent CPU encoding")
    # Transformers 5.8.1 discovers installed tensor backends even when the
    # historical USE_TORCH switch is zero. Hide only those optional backends in
    # this short-lived tokenizer process, and independently deny real imports.
    # The installed library, tokenizer and official Jinja renderer are unchanged.
    blocked = {"torch", "torchvision", "torchaudio", "tensorflow", "jax", "flax"}
    original_find_spec = importlib.util.find_spec

    def tokenizer_find_spec(name, package=None):
        return None if name.split(".", 1)[0] in blocked else original_find_spec(name, package)

    def forbid_tensor_import(event, arguments):
        if event == "import" and arguments[0].split(".", 1)[0] in blocked:
            raise RuntimeError("Tensor backend import forbidden in independent input encoder")

    importlib.util.find_spec = tokenizer_find_spec
    sys.addaudithook(forbid_tensor_import)
    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer
    from alignment_lab.chat_codec import Qwen3ChatCodec
    codec = Qwen3ChatCodec(AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                                        trust_remote_code=False))
    encoded = {}
    for task, case in request["cases"].items():
        if set(case) != {"context", "response"} or (case["response"] is None) != (task == "ifeval"):
            raise ValueError("Raw prompt and one exact selected response required")
        prompt = codec.encode_prompt(case["context"])
        response = () if task == "ifeval" else codec.encode_completion(prompt, case["response"]).completion_ids
        pair = [list(prompt.input_ids), list(response)]
        encoded[task] = {"prompt_ids": pair[0], "completion_ids": pair[1],
                         "prompt_tokens": len(pair[0]), "completion_tokens": len(pair[1]),
                         "tokens_sha256": digest(pair)}
    files()
    if "torch" in sys.modules:
        raise RuntimeError("Independent input encoder must never import Torch")
    command = ("Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(os.getpid()) +
        "' | ForEach-Object { [pscustomobject]@{ pid=$_.ProcessId; parent_pid=$_.ParentProcessId; "
        "created_at_utc=$_.CreationDate.ToUniversalTime().ToString('o'); command_line=$_.CommandLine } } "
        "| ConvertTo-Json -Compress")
    process = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], check=True,
                            capture_output=True, text=True, timeout=20,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"schema_version": 1, "kind": "alignlab-capacity-input-encoding-result-v1",
            "input_sha256": digest(request), "source_sha256": request["source_sha256"],
            "tokenizer_files": request["tokenizer_files"], "python_sha256": request["python_sha256"],
            "tokenizers_version": versions["tokenizers"], "transformers_version": versions["transformers"],
            "encoded_inputs": encoded, "process_identity": json.loads(process.stdout),
            "torch_imported": False, "gpu_operations_performed": False}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    result = encode_request(json.loads(sys.stdin.buffer.read()))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
