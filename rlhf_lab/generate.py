"""Real model generation, imported only when explicitly requested by the CLI."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def _read_prompts(path: Path, limit: int, reference_field: str = "reference") -> list[dict]:
    from .data import normalize_prompt

    rows = []
    seen = set()
    seen_prompts = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from exc
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("id", "prompt")):
                raise ValueError(f"{path}:{line_number}: expected string id and prompt")
            if reference_field in row and not isinstance(row[reference_field], str):
                raise ValueError(f"{path}:{line_number}: reference must be a string")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate prompt id")
            seen.add(row["id"])
            prompt_key = normalize_prompt(row["prompt"])
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            if len(rows) < limit:
                prompt = {"id": row["id"], "prompt": row["prompt"]}
                if reference_field in row:
                    prompt["reference"] = row[reference_field]
                rows.append(prompt)
    if not rows:
        raise ValueError("evaluation prompts are empty")
    return rows


def generate(
    config: dict,
    model_path: str,
    label: str,
    output_path: str | Path,
    limit: int | None = None,
) -> dict:
    """Load a base or saved trained policy and generate real response JSONL.

    This never runs during preparation/tests. It requires the GPU/RAM readiness
    guard before importing torch or loading weights. Existing output is preserved.
    """
    from .runtime import assert_dependencies, assert_resources, sha256_file, utc_now
    from .tokenization import encode_prompt
    from .data import validate_dataset

    if not isinstance(label, str) or not label.strip():
        raise ValueError("model label must be nonempty")
    destination = Path(output_path)
    metadata_path = Path(str(destination) + ".manifest.json")
    for output in (destination, metadata_path):
        if output.exists():
            raise FileExistsError(output)
    actual_limit = config.get("eval_limit", 32) if limit is None else limit
    if type(actual_limit) is not int or actual_limit < 1:
        raise ValueError("generation limit must be a positive integer")
    split = config.get("eval_split", "valid")
    if split not in ("valid", "test"):
        raise ValueError("eval_split must be valid or test")
    validate_dataset(config["data_dir"])
    dataset_manifest_sha256 = sha256_file(Path(config["data_dir"]) / "manifest.json")
    prompt_file = Path(config["data_dir"]) / ("preferences_valid.jsonl" if split == "valid" else "eval_prompts.jsonl")
    prompts = _read_prompts(prompt_file, actual_limit, "chosen" if split == "valid" else "reference")
    local_source = Path(model_path)
    source_manifest_sha256 = None
    if local_source.is_dir():
        source = str(local_source.resolve())
        source_manifest_path = local_source / "run_manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("local model is missing its completed run_manifest.json")
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("status") != "completed" or manifest.get("stage") not in ("sft", "ppo"):
            raise ValueError("generation requires a completed SFT or PPO policy checkpoint")
        if manifest.get("model_id") != config["model_id"] or manifest.get("model_revision") != config.get("model_revision"):
            raise ValueError("checkpoint base model identity does not match evaluation config")
        if manifest.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("checkpoint dataset manifest does not match evaluation data")
        trained_config = manifest.get("config", {})
        if not isinstance(trained_config, dict):
            raise ValueError("checkpoint is missing its training config")
        for key in ("max_prompt_tokens", "max_response_tokens", "seed"):
            if trained_config.get(key) != config.get(key):
                raise ValueError(f"checkpoint {key} does not match evaluation config")
        weight_names = ("model.safetensors", "model.safetensors.index.json")
        if not any((local_source / name).is_file() for name in weight_names):
            raise ValueError("local checkpoint contains no supported policy weights")
        source_manifest_sha256 = sha256_file(source_manifest_path)
        load_options = {"local_files_only": True}
    elif model_path == config["model_id"]:
        source = model_path
        revision = config.get("model_revision")
        if not revision:
            raise ValueError("base model generation requires model_revision")
        load_options = {"revision": revision}
    else:
        raise ValueError("model_path must be an existing checkpoint directory or configured model_id")
    max_prompt = config["max_prompt_tokens"]
    max_response = config["max_response_tokens"]
    if any(type(value) is not int or value < 1 for value in (max_prompt, max_response)):
        raise ValueError("prompt/response token limits must be positive integers")
    hardware_before = assert_resources(config, require_cuda=True)
    dependencies = assert_dependencies()

    # Match the training entry point's physical GPU selection before CUDA starts.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_index", 0))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Deliberately lazy: all offline evaluation commands work without these imports.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    started_at_utc = utc_now()
    torch.manual_seed(config.get("seed", 42))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("generation requires a CUDA device with bf16 support")
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **load_options)
    tokenizer.truncation_side = "left"  # Preserve the summary instruction at the end.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if local_source.is_dir():
            raise ValueError("trained checkpoint tokenizer is missing its PAD token")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # Resize in float32, as in SFT, then convert all compared policies to the same
    # inference dtype. This preserves the base PAD initialization convention.
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False, use_safetensors=True, **load_options,
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        if local_source.is_dir():
            raise ValueError("trained checkpoint tokenizer and model vocabulary sizes differ")
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_positions, int) and max_prompt + max_response > max_positions:
        raise ValueError("prompt plus response budget exceeds model context length")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Mark ownership before writing any responses. An interrupted generation must
    # not look like an external JSONL with no provenance during blind export.
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump({"status": "running", "started_at_utc": started_at_utc,
                   "output_path": str(destination), "model_path": source,
                   "model_label": label, "config": config}, handle,
                  ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    # A failed generation can leave a partial file: retain it for diagnosis and use
    # a new filename for a rerun instead of silently overwriting measured outputs.
    with destination.open("x", encoding="utf-8") as handle:
        for prompt in prompts:
            seed_material = json.dumps([config.get("seed", 42), prompt["id"]], ensure_ascii=False)
            prompt_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            torch.manual_seed(prompt_seed)
            torch.cuda.manual_seed_all(prompt_seed)
            prompt_ids = encode_prompt(tokenizer, prompt["prompt"], config)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_response,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response_ids = generated[0, inputs["input_ids"].shape[1]:]
            row = {
                "id": prompt["id"],
                "prompt": prompt["prompt"],
                "response": tokenizer.decode(response_ids, skip_special_tokens=True).strip(),
                "model_label": label,
                "model_path": source,
                "seed": prompt_seed,
            }
            if "reference" in prompt:
                row["reference"] = prompt["reference"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    result = {
        "status": "completed",
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
        "manifest_path": str(metadata_path),
        "examples": len(prompts),
        "eval_split": split,
        "prompts_sha256": sha256_file(prompt_file),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "prompt_selection": "first source row per normalized article, then limit",
        "model_label": label,
        "model_path": source,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["model_revision"] if source_manifest_sha256 is None else "saved_with_checkpoint",
        "local_tokenizer_files_sha256": {
            path.name: sha256_file(path)
            for path in local_source.glob("*token*.json")
            if path.is_file()
        } if local_source.is_dir() else {},
        "source_manifest_sha256": source_manifest_sha256,
        "config": config,
        "hardware_before": hardware_before,
        "dependencies": dependencies,
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now(),
        "decoding": "greedy",
        "inference_dtype": "bfloat16",
        "elapsed_seconds": time.perf_counter() - started,
    }
    # This replaces only the running manifest created by this invocation; existing
    # sidecars were rejected before any generation began.
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result
