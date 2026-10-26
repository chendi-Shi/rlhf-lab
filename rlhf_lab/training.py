"""Real SFT -> learned human-preference reward -> PPO training.

All heavyweight imports occur inside execute(). This module is safe to inspect on a busy machine.
"""

import hashlib
import json
import os
from pathlib import Path
import time

from . import data
from .runtime import (ROOT, assert_dependencies, assert_resources, sha256_file,
                      utc_now, write_json)
from .tokenization import encode_pair, encode_prompt, encode_sft


def checkpoint_ready(path, config, expected_stage, dataset_hash):
    path = Path(path)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("stage") != expected_stage:
        raise ValueError(f"{path} is not a completed {expected_stage} run")
    if manifest.get("dataset_manifest_sha256") != dataset_hash:
        raise ValueError(f"{path} was trained on a different prepared dataset")
    for key in ("model_id", "model_revision", "max_prompt_tokens", "max_response_tokens", "seed"):
        if manifest["config"][key] != config[key]:
            raise ValueError(f"Checkpoint mismatch for {key}: {path}")
    if not manifest.get("weights_changed") or not (path / "model.safetensors").is_file():
        raise ValueError(f"{path} lacks evidence of completed parameter updates")
    return manifest


def weight_fingerprint(model):
    digest = hashlib.sha256()
    # Small deterministic probes of every trainable parameter, not a full checkpoint hash.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().reshape(-1)[:1024].float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _metadata(config, stage, hardware, dependencies, dataset_hash):
    return {"status": "running", "stage": stage, "model_id": config["model_id"],
            "model_revision": config["model_revision"], "config": config,
            "dataset_manifest_sha256": dataset_hash, "dependencies": dependencies,
            "started_at_utc": utc_now(), "hardware_before": hardware,
            "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p)
                            for p in (ROOT / "rlhf_lab").glob("*.py")}}


def execute(config, stage):
    if stage not in {"sft", "reward", "ppo"}:
        raise ValueError("Unknown training stage")
    hardware = assert_resources(config)  # Before torch import or a single tensor allocation.
    dependencies = assert_dependencies()
    dataset_dir = Path(config["data_dir"])
    data.validate_dataset(dataset_dir)
    dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    source = dataset_manifest.get("source", {})
    if source.get("human_preferences") is not True or source.get("synthetic") is True:
        raise ValueError("Real human-preference provenance must be declared; fixtures are not training data")
    dataset_hash = sha256_file(dataset_dir / "manifest.json")
    run_root = Path(config["output_dir"])
    if stage in {"reward", "ppo"}:
        checkpoint_ready(run_root / "sft", config, "sft", dataset_hash)
    if stage == "ppo":
        checkpoint_ready(run_root / "reward", config, "reward", dataset_hash)
    output = run_root / stage
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}; use a new output_dir to run another experiment")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_index", 0))
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import set_seed

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable; CPU fallback is disabled")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires CUDA bf16 support; revise and test the profile explicitly")
    set_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=False)
    manifest = _metadata(config, stage, hardware, dependencies, dataset_hash)
    write_json(output / "run_manifest.json", manifest)
    start = time.perf_counter()
    try:
        if stage == "ppo":
            result = _train_ppo(config, output)
        else:
            result = _train_supervised(config, output, stage)
        manifest.update(result)
        if not result["weights_changed"]:
            raise RuntimeError("No change detected in trainable parameter probes; run not marked successful")
        manifest.update(status="completed", finished_at_utc=utc_now(),
                        elapsed_seconds=time.perf_counter() - start,
                        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
                        peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2)
        write_json(output / "run_manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", finished_at_utc=utc_now(),
                        elapsed_seconds=time.perf_counter() - start,
                        error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_manifest.json", manifest)
        raise


def _training_args(config, output, stage):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(output), max_steps=config[f"{stage}_max_steps"],
        learning_rate=config[f"{stage}_learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradient_accumulation_steps"],
        bf16=True, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="steps", save_steps=50, save_total_limit=2,
        eval_strategy="no", report_to="none", push_to_hub=False,
        seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0,
        dataloader_pin_memory=False, remove_unused_columns=False,
        optim="adamw_torch", max_grad_norm=1.0, warmup_ratio=0.05,
    )


def _trace_callback(path):
    from transformers import TrainerCallback
    class Trace(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                row = {"time_utc": utc_now(), "step": state.global_step,
                       **{k: float(v) if hasattr(v, "item") else v for k, v in logs.items()}}
                with Path(path).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return Trace()


def _pad_sequences(sequences, pad_token_id):
    import torch
    width = max(map(len, sequences))
    return {"input_ids": torch.tensor([row + [pad_token_id] * (width - len(row)) for row in sequences]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in sequences])}


def _train_supervised(config, output, stage):
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, Trainer

    dataset_dir = Path(config["data_dir"])
    source_path = config["model_id"] if stage == "sft" else str(Path(config["output_dir"]) / "sft")
    remote = {"revision": config["model_revision"]} if stage == "sft" else {}
    tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=False, **remote)
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    cls = AutoModelForCausalLM if stage == "sft" else AutoModelForSequenceClassification
    kwargs = {} if stage == "sft" else {"num_labels": 1}
    model = cls.from_pretrained(source_path, dtype=torch.float32, attn_implementation="eager",
                               trust_remote_code=False, use_safetensors=True, **remote, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    before = weight_fingerprint(model)
    callback = _trace_callback(output / "metrics.jsonl")
    args = _training_args(config, output, stage)

    if stage == "sft":
        def collate(rows):
            batch = _pad_sequences([row["input_ids"] for row in rows], tokenizer.pad_token_id)
            width = batch["input_ids"].shape[1]
            batch["labels"] = torch.tensor([row["labels"] + [-100] * (width - len(row["labels"])) for row in rows])
            return batch
        train_rows = [encode_sft(tokenizer, row, config) for row in data.read_jsonl(dataset_dir / "sft_train.jsonl")]
        valid_rows = [encode_sft(tokenizer, row, config) for row in data.read_jsonl(dataset_dir / "sft_valid.jsonl")]
        trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                          train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(valid_rows),
                          data_collator=collate, callbacks=[callback])
        training = trainer.train()
        metrics = {"train": training.metrics, "validation": trainer.evaluate()}
        token_counts = {"train_examples": len(train_rows), "valid_examples": len(valid_rows)}
    else:
        def pair_rows(filename):
            raw = data.read_jsonl(dataset_dir / filename)
            encoded = [encode_pair(tokenizer, row, config) for row in raw]
            return [row for row in encoded if row is not None], sum(row is None for row in encoded)
        train_rows, train_dropped = pair_rows("preferences_train.jsonl")
        valid_rows, valid_dropped = pair_rows("preferences_valid.jsonl")
        if not train_rows or not valid_rows:
            raise ValueError("No distinguishable preference pairs remain after tokenization")
        def collate(rows):
            return _pad_sequences([row["chosen_ids"] for row in rows] + [row["rejected_ids"] for row in rows], tokenizer.pad_token_id)
        class PairwiseTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                outputs = model(**inputs)
                chosen, rejected = outputs.logits.squeeze(-1).chunk(2)
                loss = -F.logsigmoid(chosen - rejected).mean()
                # Center scores to reduce unconstrained drift; does not manufacture preferences.
                loss = loss + 0.01 * ((chosen + rejected) / 2).square().mean()
                return (loss, outputs) if return_outputs else loss
        trainer = PairwiseTrainer(model=model, args=args, processing_class=tokenizer,
                                  train_dataset=Dataset.from_list(train_rows),
                                  data_collator=collate, callbacks=[callback])
        training = trainer.train()
        model.eval()
        correct, ties, total_loss = 0, 0, 0.0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in valid_rows:
                batch = {key: value.to(model.device) for key, value in collate([row]).items()}
                scores = model(**batch).logits.float().flatten()
                margin = (scores[0] - scores[1]).item()
                correct += margin > 0
                ties += margin == 0
                total_loss += F.softplus(-(scores[0] - scores[1])).item()
        metrics = {"train": training.metrics, "validation": {
            "pairwise_accuracy": correct / len(valid_rows), "ties": ties,
            "pairs": len(valid_rows), "pairwise_loss": total_loss / len(valid_rows)},
            "test_set": "untouched; reserve for final evaluation"}
        token_counts = {"train_pairs": len(train_rows), "validation_pairs": len(valid_rows),
                        "train_pairs_dropped_identical_after_truncation": train_dropped,
                        "valid_pairs_dropped_identical_after_truncation": valid_dropped}

    after = weight_fingerprint(model)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    trainer.save_state()
    write_json(output / "evaluation.json", metrics)
    return {"weight_fingerprint_before": before, "weight_fingerprint_after": after,
            "weights_changed": before != after, "metrics": metrics, "tokenization": token_counts,
            "completed_optimizer_steps": trainer.state.global_step,
            "note": "Parameter probes establish updates, not improved model quality."}


def _train_ppo(config, output):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from trl import PPOConfig, PPOTrainer

    run_root = Path(config["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(run_root / "sft", padding_side="left", trust_remote_code=False)
    common = {"attn_implementation": "eager", "trust_remote_code": False, "use_safetensors": True}
    policy = AutoModelForCausalLM.from_pretrained(run_root / "sft", dtype=torch.float32, **common)
    value = AutoModelForSequenceClassification.from_pretrained(run_root / "reward", dtype=torch.float32, **common)
    reference = AutoModelForCausalLM.from_pretrained(run_root / "sft", dtype=torch.bfloat16, **common)
    reward = AutoModelForSequenceClassification.from_pretrained(run_root / "reward", dtype=torch.bfloat16, **common)
    for model in (policy, value, reference, reward):
        model.config.pad_token_id = tokenizer.pad_token_id
    for model in (reference, reward):
        model.requires_grad_(False)
        model.eval()
    before = weight_fingerprint(policy)
    critic_before = weight_fingerprint(value)
    def prompts(filename):
        return Dataset.from_list([{"input_ids": encode_prompt(tokenizer, row["prompt"], config)}
                                  for row in data.read_jsonl(Path(config["data_dir"]) / filename)])
    train_data = prompts("ppo_train.jsonl")
    # Validation only. Never feed the held-out test prompts into training/checkpoint selection.
    valid_data = prompts("preferences_valid.jsonl")
    batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    if len(train_data) < batch:
        raise ValueError(f"PPO needs at least {batch} training prompts for a complete rollout batch")
    args = PPOConfig(
        output_dir=str(output), sft_model_path=str(run_root / "sft"), reward_model_path=str(run_root / "reward"),
        learning_rate=config["ppo_learning_rate"], total_episodes=config["ppo_total_episodes"],
        per_device_train_batch_size=config["per_device_train_batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_mini_batches=1, num_ppo_epochs=config["ppo_num_epochs"],
        local_rollout_forward_batch_size=1, response_length=config["max_response_tokens"],
        temperature=0.7, kl_coef=config["kl_coef"], kl_estimator="k1", whiten_rewards=False,
        stop_token="eos", missing_eos_penalty=None,
        bf16=True, fp16=False, num_sample_generations=0, report_to="none", push_to_hub=False,
        seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0,
        logging_steps=1, save_steps=16, save_total_limit=2, optim="adamw_torch",
        # Short sequences are used first; checkpointing is off for the custom policy/value wrapper.
        gradient_checkpointing=False,
    )
    trainer = PPOTrainer(args=args, processing_class=tokenizer, model=policy, ref_model=reference,
                         reward_model=reward, value_model=value, train_dataset=train_data,
                         eval_dataset=valid_data, callbacks=[_trace_callback(output / "metrics.jsonl")])
    trainer.train()
    after = weight_fingerprint(policy)
    critic_after = weight_fingerprint(value)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    trainer.save_state()
    # Upstream saves policy only. Preserve final critic separately for inspection (no PPO resume promise).
    value.save_pretrained(output / "critic", safe_serialization=True)
    return {"weight_fingerprint_before": before, "weight_fingerprint_after": after,
            "weights_changed": before != after, "critic_weights_changed": critic_before != critic_after,
            "completed_rollout_batches": trainer.state.global_step,
            "completed_optimizer_steps": trainer.state.global_step * args.num_ppo_epochs * args.num_mini_batches,
            "metrics": {"history": trainer.state.log_history,
                        "episodes": getattr(trainer.state, "episode", None)},
            "note": "Learned reward/KL are optimization diagnostics; blind held-out evaluation is still required."}
