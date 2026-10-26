"""Evaluate a trained reward model on held-out human comparisons; imports stay light."""

import hashlib
import json
import math
import os
from pathlib import Path
import time

from . import data
from .runtime import assert_dependencies, assert_resources, sha256_file, utc_now
from .tokenization import encode_pair
from .training import checkpoint_ready


def _aggregate(rows):
    """Pure aggregation of actual score records; also used by synthetic unit tests."""
    if not rows:
        raise ValueError("No distinguishable comparison pairs remain for reward evaluation")
    grouped = {}
    correct, ties, loss_sum = 0, 0, 0.0
    for row in rows:
        margin = row["chosen_score"] - row["rejected_score"]
        if not math.isfinite(margin):
            raise ValueError("Reward model produced a non-finite score margin")
        # Stable -log(sigmoid(margin)); no overflow for very large negative margins.
        loss = max(0.0, -margin) + math.log1p(math.exp(-abs(margin)))
        win, tie = int(margin > 0), int(margin == 0)
        correct += win
        ties += tie
        loss_sum += loss
        values = grouped.setdefault(row["prompt_id"], {"correct": 0, "loss": 0.0, "pairs": 0})
        values["correct"] += win
        values["loss"] += loss
        values["pairs"] += 1
    return {
        "pairs": len(rows), "unique_prompts": len(grouped), "ties": ties,
        "pairwise_accuracy": correct / len(rows), "pairwise_loss": loss_sum / len(rows),
        "prompt_macro_accuracy": sum(item["correct"] / item["pairs"] for item in grouped.values()) / len(grouped),
        "prompt_macro_loss": sum(item["loss"] / item["pairs"] for item in grouped.values()) / len(grouped),
    }


def score_reward(config, model_path, split="valid", output_path=None, limit=None):
    """Run explicit GPU inference; validation is default, test must be selected explicitly."""
    if split not in {"valid", "test"}:
        raise ValueError("Reward evaluation split must be valid or test; training data is forbidden")
    limit = config["eval_limit"] if limit is None else limit
    if type(limit) is not int or limit <= 0:
        raise ValueError("Reward evaluation limit must be a positive integer")
    output = Path(output_path) if output_path is not None else (
        Path(config["output_dir"]) / "evaluations" / f"reward_{split}.json")
    if output.exists():
        raise FileExistsError(f"Will not overwrite reward evaluation: {output}")
    hardware = assert_resources(config)
    dependencies = assert_dependencies()
    dataset_dir = Path(config["data_dir"])
    validated = data.validate_dataset(dataset_dir)
    source = validated["source"]
    if source.get("human_preferences") is not True or source.get("synthetic") is True:
        raise ValueError("Reward evaluation requires real human-preference provenance, not fixtures")
    dataset_hash = sha256_file(dataset_dir / "manifest.json")
    model_path = Path(model_path)
    checkpoint_ready(model_path, config, "reward", dataset_hash)
    model_manifest_hash = sha256_file(model_path / "run_manifest.json")
    pairs_path = dataset_dir / f"preferences_{split}.jsonl"
    raw_pairs = data.read_jsonl(pairs_path)
    selected = raw_pairs[:limit]

    # No CUDA initialization or heavyweight import occurs before all readiness checks.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_index", 0))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Reward evaluation requires the configured CUDA bf16 environment")
    started_at = utc_now()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager", trust_remote_code=False,
        use_safetensors=True).to("cuda")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    rows, dropped = [], 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pair in selected:
            encoded = encode_pair(tokenizer, pair, config)
            if encoded is None:
                dropped += 1
                continue
            scores = []
            for name in ("chosen_ids", "rejected_ids"):
                tokens = torch.tensor([encoded[name]], dtype=torch.long, device="cuda")
                score = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits.float().flatten()
                if score.numel() != 1:
                    raise ValueError("The reward checkpoint must have exactly one scalar output")
                scores.append(score.item())
            rows.append({"id": pair["id"], "prompt_id": hashlib.sha256(
                data.normalize_prompt(pair["prompt"]).encode("utf-8")).hexdigest(),
                "chosen_score": scores[0], "rejected_score": scores[1]})
    metrics = _aggregate(rows)
    report = {
        "schema_version": 1, "status": "completed", "stage": "reward_evaluation", "split": split,
        "metrics": metrics, "config": config, "model_path": str(model_path.resolve()),
        "model_manifest_sha256": model_manifest_hash, "dataset_manifest_sha256": dataset_hash,
        "comparison_file_sha256": sha256_file(pairs_path), "dependencies": dependencies,
        "hardware_before": hardware, "started_at_utc": started_at, "finished_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - start, "available_pairs": len(raw_pairs),
        "requested_limit": limit, "selected_pairs": len(selected),
        "dropped_identical_after_truncation": dropped,
        "sampling": "First limit pairs in the deterministic prepared split; not an independent random sample.",
        "tie_policy": "Exact equal scores count as incorrect for strict pairwise and prompt-macro accuracy.",
        "prompt_macro_definition": "Compute accuracy/loss within each normalized article, then equally average articles.",
        "note": "This evaluates reward ranking on human comparisons, not generated-summary quality or PPO improvement.",
        "scores": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also protects against another process writing after the initial check.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report
