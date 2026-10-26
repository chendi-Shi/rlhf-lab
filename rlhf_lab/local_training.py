"""Local-only, inspectable SFT / reward / DPO / clipped PPO training.

No TRL dependency, remote model loading, silent CPU fallback, or online-quality
claim. PPO's value head shares the policy backbone. Rollouts and optimizer/RNG
state are checkpointed together so a resumed epoch retains its old policy.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

from . import data
from .runtime import sha256_file, utc_now
from .tokenization import encode_pair, encode_prompt, encode_sft


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_state(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(directory):
    path = directory / ".local-training.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another writer or unreconciled interrupted process owns {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_at_utc": utc_now()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def _configuration(config, stage):
    c = json.loads(json.dumps(config, allow_nan=False))
    pause_after = c.pop("max_updates_this_run", None)
    if stage not in {"sft", "reward", "dpo", "ppo"}:
        raise ValueError("stage must be sft, reward, dpo or ppo")
    for key in ("model_path", "data_dir", "output_dir", "device"):
        if key not in c:
            raise ValueError(f"Explicit {key} is required")
    defaults = {"seed": 42, "threads": 2, "max_prompt_tokens": 192,
                "max_response_tokens": 64, "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1, "sft_max_steps": 100,
                "reward_max_steps": 100, "dpo_max_steps": 100,
                "sft_learning_rate": 2e-5, "reward_learning_rate": 1e-5,
                "dpo_learning_rate": 1e-6, "ppo_learning_rate": 1e-6,
                "weight_decay": 0., "max_grad_norm": 1., "dpo_beta": .1,
                "reward_centering_coefficient": .01, "kl_coef": .05,
                "ppo_rollout_batch_size": 4, "ppo_minibatch_size": 1,
                "ppo_cliprange": .2, "ppo_value_cliprange": .2,
                "ppo_value_coef": .5, "ppo_entropy_coef": 0.,
                "ppo_gamma": 1., "ppo_lam": .95, "ppo_whiten_advantages": True,
                "generation_temperature": 1., "checkpoint_every": 10,
                "gradient_checkpointing": True, "deterministic": True,
                "software_fixture": False, "disable_dropout": True}
    c = {**defaults, **c}
    if "num_threads" in config and "threads" not in config:
        c["threads"] = c["num_threads"]
    c.setdefault("ppo_response_length", c["max_response_tokens"])
    c.setdefault("ppo_epochs", c.get("ppo_num_epochs", 2))
    if "ppo_updates" not in c:
        total = c.get("ppo_total_episodes", 4)
        if total % c["ppo_rollout_batch_size"]:
            raise ValueError("ppo_total_episodes must divide into complete rollout batches")
        c["ppo_updates"] = total // c["ppo_rollout_batch_size"]
    for key in ("threads", "max_prompt_tokens", "max_response_tokens", "per_device_train_batch_size",
                "gradient_accumulation_steps", "sft_max_steps", "reward_max_steps", "dpo_max_steps",
                "ppo_updates", "ppo_rollout_batch_size", "ppo_minibatch_size", "ppo_response_length",
                "ppo_epochs", "checkpoint_every"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if pause_after is not None and (type(pause_after) is not int or pause_after < 1):
        raise ValueError("max_updates_this_run must be a positive optimizer-step limit")
    if type(c["seed"]) is not int or not 0 <= c["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if c["device"] not in {"cpu", "cuda"}:
        raise ValueError("explicit cpu or cuda device required")
    if c["ppo_rollout_batch_size"] < 4 or c["ppo_rollout_batch_size"] % c["ppo_minibatch_size"]:
        raise ValueError("PPO needs at least four rollouts and complete minibatches")
    if c["max_response_tokens"] < 2 or c["ppo_response_length"] > c["max_response_tokens"]:
        raise ValueError("invalid response budget")
    for key in ("sft_learning_rate", "reward_learning_rate", "dpo_learning_rate", "ppo_learning_rate",
                "dpo_beta", "max_grad_norm", "generation_temperature"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "kl_coef", "reward_centering_coefficient", "ppo_value_coef", "ppo_entropy_coef",
                "ppo_cliprange", "ppo_value_cliprange"):
        if not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= c["ppo_gamma"] <= 1 or not 0 <= c["ppo_lam"] <= 1:
        raise ValueError("GAE gamma and lambda must lie in [0, 1]")
    if c["disable_dropout"] is not True:
        raise ValueError("This matched local protocol disables dropout, including during PPO updates")
    for key in ("software_fixture", "deterministic", "gradient_checkpointing", "ppo_whiten_advantages"):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("model_path", "data_dir", "output_dir"):
        c[key] = str(Path(c[key]).resolve())
    return c, pause_after


def _source_hashes():
    root = Path(__file__).parent
    return {name: sha256_file(root / name)
            for name in ("local_training.py", "objectives.py", "data.py", "tokenization.py")}


def _files_hashes(directory, exclude=()):
    directory = Path(directory)
    return {str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
            and path.name not in exclude and not any(part.startswith(".") for part in path.relative_to(directory).parts)
            and ".tmp-" not in path.name}


def _dataset_identity(config):
    """Hash every split for integrity, decode only training/validation records."""
    root = Path(config["data_dir"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    source = manifest.get("source", {})
    if config["software_fixture"]:
        if source.get("synthetic") is not True:
            raise ValueError("software_fixture requires an explicitly synthetic dataset")
    elif source.get("human_preferences") is not True or source.get("synthetic") is True:
        raise ValueError("Real training requires human-preference provenance, not software fixtures")
    for name, info in manifest.get("files", {}).items():
        if Path(name).name != name or sha256_file(root / name) != info["sha256"]:
            raise ValueError(f"Prepared dataset checksum mismatch: {name}")
    expected = {"sft_train.jsonl", "sft_valid.jsonl", "preferences_train.jsonl", "preferences_valid.jsonl",
                "ppo_train.jsonl", "preferences_test.jsonl", "eval_prompts.jsonl"}
    if not expected <= set(manifest.get("files", {})):
        raise ValueError("Missing prepared training/held-out file identities")
    train = data.read_jsonl(root / "preferences_train.jsonl")
    valid = data.read_jsonl(root / "preferences_valid.jsonl")
    if not train or not valid:
        raise ValueError("Training and validation preferences must be nonempty")
    train_prompts = {data.normalize_prompt(row["prompt"]) for row in train}
    valid_prompts = {data.normalize_prompt(row["prompt"]) for row in valid}
    if train_prompts & valid_prompts:
        raise ValueError("Training/validation prompt overlap")
    for split, rows in (("train", train), ("valid", valid)):
        expected_sft = {(data.normalize_prompt(row["prompt"]), data.normalize_prompt(row["chosen"])) for row in rows}
        actual_sft = {(data.normalize_prompt(row["prompt"]), data.normalize_prompt(row["response"]))
                      for row in data.read_jsonl(root / f"sft_{split}.jsonl")}
        if expected_sft != actual_sft:
            raise ValueError("SFT rows do not match the corresponding human preferences")
    if {data.normalize_prompt(row["prompt"]) for row in data.read_jsonl(root / "ppo_train.jsonl")} != train_prompts:
        raise ValueError("PPO prompts differ from training preference prompts")
    return sha256_file(root / "manifest.json"), manifest


def checkpoint_ready_local(path, config, expected_stage, dataset_hash):
    path = Path(path)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("stage") != expected_stage:
        raise ValueError(f"Parent must be a completed {expected_stage} run: {path}")
    if manifest.get("dataset_manifest_sha256") != dataset_hash or not manifest.get("weights_changed"):
        raise ValueError("Parent data/update evidence mismatch")
    for key in ("model_path", "seed", "max_prompt_tokens", "max_response_tokens", "software_fixture", "completion_eos_policy"):
        if manifest["config"].get(key) != config.get(key):
            raise ValueError(f"Parent configuration mismatch: {key}")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("Parent was created with different training source")
    for name, digest in manifest.get("artifacts", {}).items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or sha256_file(target) != digest:
            raise ValueError(f"Parent artifact mismatch: {name}")
    if not manifest.get("artifacts") or "model.safetensors" not in manifest["artifacts"]:
        raise ValueError("Parent lacks a complete local model artifact")
    return manifest


def _state_hash(module):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rng_state(generator, cuda):
    import torch
    return {"python": random.getstate(), "numpy_generator": generator.bit_generator.state,
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if cuda else None}


def _restore_rng(state, generator, cuda):
    import torch
    random.setstate(state["python"])
    generator.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch"].cpu())
    if cuda:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def _autocast(config):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if config["device"] == "cuda" else nullcontext()


def _disable_dropout(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    for name in ("resid_pdrop", "embd_pdrop", "attn_pdrop"):
        if hasattr(model.config, name):
            setattr(model.config, name, 0.)
    model.config.use_cache = False


def _load_model(path, config, reward=False, frozen=False):
    import torch
    from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification
    cls = GPT2ForSequenceClassification if reward else GPT2LMHeadModel
    kwargs = {"num_labels": 1} if reward else {}
    model = cls.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
                                use_safetensors=True, dtype=torch.float32,
                                attn_implementation="eager", **kwargs)
    _disable_dropout(model)
    if config["max_prompt_tokens"] + config["max_response_tokens"] > model.config.n_positions:
        raise ValueError("Configured prompt+response exceeds model context")
    model.to(config["device"])
    if frozen:
        model.requires_grad_(False).eval()
        if config["device"] == "cuda":
            model.to(dtype=torch.bfloat16)
    elif config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def pad_batch(sequences, pad_id, response_starts=None, device="cpu"):
    import torch
    if not sequences or any(not row for row in sequences):
        raise ValueError("Cannot pad empty token sequences")
    width = max(map(len, sequences))
    ids = torch.tensor([list(row) + [pad_id] * (width-len(row)) for row in sequences], device=device)
    attention = torch.tensor([[1]*len(row)+[0]*(width-len(row)) for row in sequences], device=device)
    response = torch.zeros_like(ids, dtype=torch.bool)
    if response_starts is not None:
        for index, (row, start) in enumerate(zip(sequences, response_starts)):
            if not 0 < start < len(row):
                raise ValueError("Every response needs a prompt predecessor and a response token")
            response[index, start:len(row)] = True
    return {"input_ids": ids, "attention_mask": attention, "response_mask": response}


def reward_scores(model, batch):
    """Pool the last attended token, even when EOS is also the padding ID."""
    import torch
    hidden = model.transformer(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               use_cache=False, return_dict=True).last_hidden_state
    scores = model.score(hidden).float().squeeze(-1)
    positions = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)
    last = positions.masked_fill(~batch["attention_mask"].bool(), -1).max(1).values
    if bool((last < 0).any()):
        raise ValueError("Reward scoring requires attended tokens")
    return scores.gather(1, last[:, None]).squeeze(1)


def _policy_outputs(policy, value_head, batch, temperature=1.):
    outputs = policy(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     use_cache=False, return_dict=True, output_hidden_states=value_head is not None)
    values = value_head(outputs.hidden_states[-1]).float().squeeze(-1)[:, :-1] if value_head is not None else None
    return outputs.logits / temperature, values


def _encoded_rows(tokenizer, config, stage):
    root = Path(config["data_dir"])
    if stage == "sft":
        raw = data.read_jsonl(root / "sft_train.jsonl")
        rows = [encode_sft(tokenizer, row, config) for row in raw]
        for row in rows:
            row["response_start"] = next(i for i, label in enumerate(row["labels"]) if label != -100)
    elif stage in {"reward", "dpo"}:
        raw = data.read_jsonl(root / "preferences_train.jsonl")
        rows = []
        for row in raw:
            pair = encode_pair(tokenizer, row, config)
            if pair is not None:
                pair["response_start"] = len(encode_prompt(tokenizer, row["prompt"], config))
                rows.append(pair)
    else:
        raw = data.read_jsonl(root / "ppo_train.jsonl")
        rows = [{"input_ids": encode_prompt(tokenizer, row["prompt"], config), "id": row["id"]} for row in raw]
    if not rows:
        raise ValueError("No usable training examples after tokenization")
    if stage == "ppo" and len(rows) < config["ppo_rollout_batch_size"]:
        raise ValueError("Not enough distinct training prompts for a PPO rollout batch")
    return rows, {"raw_rows": len(raw), "usable_rows": len(rows), "dropped_after_tokenization": len(raw)-len(rows)}


def _supervised_loss(model, reference, tokenizer, rows, config, stage):
    import torch
    from .objectives import sequence_logprobs, reward_model_loss, dpo_loss, response_token_mask, completion_logprobs
    if stage == "sft":
        batch = pad_batch([r["input_ids"] for r in rows], tokenizer.pad_token_id,
                          [r["response_start"] for r in rows], config["device"])
        logits, _ = _policy_outputs(model, None, batch)
        logps = completion_logprobs(logits, batch["input_ids"], batch["response_mask"], batch["attention_mask"])
        mask = response_token_mask(batch["input_ids"], batch["response_mask"], batch["attention_mask"])
        loss = -logps.sum() / mask.sum()
        return loss, {"response_nll": float(loss.detach()), "response_tokens": int(mask.sum())}
    batch = pad_batch([r["chosen_ids"] for r in rows]+[r["rejected_ids"] for r in rows], tokenizer.pad_token_id,
                      [r["response_start"] for r in rows]*2, config["device"])
    if stage == "reward":
        scores = reward_scores(model, batch)
        return reward_model_loss(*scores.chunk(2), centering_coefficient=config["reward_centering_coefficient"])
    logits, _ = _policy_outputs(model, None, batch)
    policy_logps = sequence_logprobs(logits, batch["input_ids"], batch["response_mask"], batch["attention_mask"])
    with torch.no_grad():
        ref_logits, _ = _policy_outputs(reference, None, batch)
        ref_logps = sequence_logprobs(ref_logits, batch["input_ids"], batch["response_mask"], batch["attention_mask"])
    chosen, rejected = policy_logps.chunk(2)
    ref_chosen, ref_rejected = ref_logps.chunk(2)
    return dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=config["dpo_beta"])


def _generate_response(policy, prompt, tokenizer, config):
    import torch
    from torch.nn import functional as F
    ids = torch.tensor([prompt], dtype=torch.long, device=config["device"])
    attention = torch.ones_like(ids)
    response, cache = [], None
    with torch.no_grad(), _autocast(config):
        for _ in range(config["ppo_response_length"]):
            outputs = policy(input_ids=ids if cache is None else ids[:, -1:], attention_mask=attention,
                             past_key_values=cache, use_cache=True, return_dict=True)
            probabilities = F.softmax(outputs.logits[:, -1].float()/config["generation_temperature"], dim=-1)
            token = torch.multinomial(probabilities, 1)
            token_id = int(token.item())
            response.append(token_id)
            ids = torch.cat([ids, token], dim=1)
            attention = torch.ones_like(ids)
            cache = outputs.past_key_values
            if token_id == tokenizer.eos_token_id:
                break
    return list(prompt)+response, len(prompt), response[-1] == tokenizer.eos_token_id


def _rollout(policy, value_head, reference, reward, tokenizer, rows, generator, config):
    import torch
    from .objectives import completion_logprobs, response_token_mask, sampled_kl_rewards, generalized_advantage_estimate
    policy.eval()
    selected = generator.choice(len(rows), size=config["ppo_rollout_batch_size"], replace=False).tolist()
    sequences, starts, eos = [], [], []
    for index in selected:
        sequence, start, ended = _generate_response(policy, rows[index]["input_ids"], tokenizer, config)
        sequences.append(sequence)
        starts.append(start)
        eos.append(ended)
    batch = pad_batch(sequences, tokenizer.pad_token_id, starts, config["device"])
    # Score one sequence at a time to bound logits/activation memory on an 8 GiB GPU.
    old_logps, old_values, ref_logps, scores = [], [], [], []
    with torch.no_grad(), _autocast(config):
        for i in range(len(selected)):
            item = {key: value[i:i+1] for key, value in batch.items()}
            logits, values = _policy_outputs(policy, value_head, item, config["generation_temperature"])
            old_logps.append(completion_logprobs(logits, item["input_ids"], item["response_mask"], item["attention_mask"]))
            old_values.append(values)
            ref_logits, _ = _policy_outputs(reference, None, item, config["generation_temperature"])
            ref_logps.append(completion_logprobs(ref_logits, item["input_ids"], item["response_mask"], item["attention_mask"]))
            scores.append(reward_scores(reward, item))
        old_logps, old_values = torch.cat(old_logps), torch.cat(old_values)
        ref_logps, scores = torch.cat(ref_logps), torch.cat(scores)
        mask = response_token_mask(batch["input_ids"], batch["response_mask"], batch["attention_mask"])
        rewards, stats = sampled_kl_rewards(old_logps, ref_logps, scores, mask, config["kl_coef"])
        advantages, returns = generalized_advantage_estimate(rewards, old_values, mask,
                                      gamma=config["ppo_gamma"], lam=config["ppo_lam"],
                                      whiten=config["ppo_whiten_advantages"])
    plan = []
    for _ in range(config["ppo_epochs"]):
        order = generator.permutation(len(selected)).tolist()
        plan.extend([order[i:i+config["ppo_minibatch_size"]] for i in range(0, len(order), config["ppo_minibatch_size"])])
    tensors = {**batch, "old_logprobs": old_logps, "old_values": old_values,
               "reference_logprobs": ref_logps, "scores": scores, "token_mask": mask,
               "advantages": advantages, "returns": returns, "rewards": rewards}
    return {"tensors": {key: value.detach().cpu() for key, value in tensors.items()},
            "minibatches": plan, "cursor": 0, "prompt_indices": selected,
            "diagnostics": {"reward_mean": float(scores.mean()),
                            "sampled_kl_mean": float((old_logps-ref_logps)[mask.bool()].mean()),
                            "response_tokens_mean": float(mask.float().sum(1).mean()),
                            "eos_fraction": sum(eos)/len(eos),
                            "terminal_horizon": "EOS or fixed generation horizon; zero terminal bootstrap; no missing-EOS penalty"}}


def _ppo_loss(policy, value_head, pending, config):
    import torch
    from torch.nn import functional as F
    from .objectives import completion_logprobs, ppo_loss
    indices = pending["minibatches"][pending["cursor"]]
    batch = {key: value[indices].to(config["device"]) for key, value in pending["tensors"].items()}
    policy.eval()  # Gradients remain enabled; dropout must not change the old/new policy ratio.
    logits, values = _policy_outputs(policy, value_head, batch, config["generation_temperature"])
    logps = completion_logprobs(logits, batch["input_ids"], batch["response_mask"], batch["attention_mask"])
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    entropy = -(lp.exp()*lp).sum(-1)
    loss, metrics = ppo_loss(logps, batch["old_logprobs"], batch["advantages"], values,
                              batch["old_values"], batch["returns"], batch["token_mask"],
                              clip_range=config["ppo_cliprange"], value_clip_range=config["ppo_value_cliprange"],
                              value_coefficient=config["ppo_value_coef"], entropy=entropy,
                              entropy_coefficient=config["ppo_entropy_coef"])
    mask = batch["token_mask"].bool()
    target = batch["returns"][mask]
    variance = target.var(unbiased=False)
    metrics["value_explained_variance"] = (float(1-(target-values[mask].detach()).var(unbiased=False)/variance)
                                             if float(variance) > 1e-12 else None)
    metrics.update(pending["diagnostics"])
    return loss, metrics


def _validation(model, reference, tokenizer, config, stage):
    import torch
    from .objectives import dpo_loss
    if stage == "ppo":
        return {"scope": "No held-out generation evaluation here; evaluate blind test separately."}
    filename = "sft_valid.jsonl" if stage == "sft" else "preferences_valid.jsonl"
    raw = data.read_jsonl(Path(config["data_dir"])/filename)
    limit = config.get("eval_limit" if stage == "sft" else "eval_preference_limit", len(raw))
    raw = raw[:limit]
    rows = []
    for row in raw:
        item = encode_sft(tokenizer, row, config) if stage == "sft" else encode_pair(tokenizer, row, config)
        if item is not None:
            item["response_start"] = len(encode_prompt(tokenizer, row["prompt"], config))
            rows.append(item)
    losses, accuracies = [], []
    model.eval()
    with torch.no_grad(), _autocast(config):
        for row in rows:
            loss, metrics = _supervised_loss(model, reference, tokenizer, [row], config, stage)
            losses.append(float(loss))
            if stage == "reward":
                batch = pad_batch([row["chosen_ids"], row["rejected_ids"]], tokenizer.pad_token_id, device=config["device"])
                values = reward_scores(model, batch)
                accuracies.append(float(values[0] > values[1]))
    if not rows:
        raise ValueError("No usable validation examples")
    return {"split": "valid", "examples": len(rows), "loss": sum(losses)/len(losses),
            "pairwise_accuracy": sum(accuracies)/len(accuracies) if accuracies else None,
            "selection": "fixed budget; this diagnostic does not select a checkpoint", "test_labels_decoded": False}


def execute_local(config, stage, resume=False):
    """Train one local stage; max_updates_this_run pauses after optimizer steps."""
    c, pause_after = _configuration(config, stage)
    from .local_runtime import assert_local_resources
    hardware = assert_local_resources(c)  # Read-only preflight before torch import/model allocation.
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    dataset_hash, dataset_manifest = _dataset_identity(c)
    root = Path(c["output_dir"])
    parents = {}
    if stage != "sft":
        checkpoint_ready_local(root/"sft", c, "sft", dataset_hash)
        parents["sft"] = sha256_file(root/"sft"/"run_manifest.json")
    if stage == "ppo":
        checkpoint_ready_local(root/"reward", c, "reward", dataset_hash)
        parents["reward"] = sha256_file(root/"reward"/"run_manifest.json")
    model_path = Path(c["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError("Base model must already exist locally")
    base_files = _files_hashes(model_path)
    dependencies = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors")}
    identity = {"config": c, "stage": stage, "dataset_manifest_sha256": dataset_hash,
                "source_sha256": _source_hashes(), "parent_sha256": parents,
                "base_model_sha256": base_files, "dependencies": dependencies}
    out = root/stage
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out):
        manifest_path, state_path = out/"run_manifest.json", out/"latest.pt"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not resume or previous.get("identity") != identity:
                raise ValueError("Existing run requires explicit resume with identical source/config/data/parents")
            if previous.get("status") == "completed":
                checkpoint_ready_local(out, c, stage, dataset_hash)
                return previous
        elif resume:
            raise FileNotFoundError("Cannot resume without a run manifest")
        elif any(p.is_file() and not p.name.startswith(".") for p in out.iterdir()):
            raise FileExistsError("Refusing nonempty unregistered stage directory")
        manifest = {"schema_version": 1, "engine": "local_torch_autograd", "status": "running",
                    **identity, "identity": identity, "started_at_utc": utc_now(), "hardware_before": hardware,
                    "software_fixture": c["software_fixture"], "test_labels_decoded": False,
                    "scope": "software fixture, not human-data evidence" if c["software_fixture"] else "human-preference training; optimization diagnostics do not establish improved summaries"}
        _atomic_json(manifest_path, manifest)
        try:
            torch.set_num_threads(c["threads"])
            if c["device"] == "cuda":
                if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Explicit CUDA bf16 training is unavailable; no CPU fallback")
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.cuda.set_device(c.get("gpu_index", 0))
            random.seed(c["seed"])
            torch.manual_seed(c["seed"])
            generator = np.random.default_rng(c["seed"])
            torch.use_deterministic_algorithms(c["deterministic"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            if c["device"] == "cuda":
                torch.cuda.manual_seed_all(c["seed"])
                torch.cuda.reset_peak_memory_stats()
            source = model_path if stage == "sft" else root/"sft"
            tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
            if tokenizer.eos_token_id is None:
                raise ValueError("GPT2 tokenizer requires an EOS token")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model = _load_model(source, c, reward=stage == "reward")
            model.config.pad_token_id = tokenizer.pad_token_id
            value_head = None
            if stage == "ppo":
                value_head = torch.nn.Linear(model.config.n_embd, 1).to(c["device"])
                torch.nn.init.zeros_(value_head.weight)
                torch.nn.init.zeros_(value_head.bias)
            parameters = list(model.parameters()) + ([] if value_head is None else list(value_head.parameters()))
            optimizer = torch.optim.AdamW(parameters, lr=c[f"{stage}_learning_rate"], weight_decay=c["weight_decay"])
            initial_hash = _state_hash(model)
            initial_value_hash = _state_hash(value_head) if value_head is not None else None
            reference = _load_model(root/"sft", c, frozen=True) if stage in {"dpo", "ppo"} else None
            reward = _load_model(root/"reward", c, reward=True, frozen=True) if stage == "ppo" else None
            rows, tokenization = _encoded_rows(tokenizer, c, stage)
            order = generator.permutation(len(rows)).tolist()
            position, steps, rollout_updates, pending, history, elapsed_before = 0, 0, 0, None, [], 0.
            if resume and state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                if state["identity"] != identity:
                    raise ValueError("Resume state identity mismatch")
                model.load_state_dict(state["model"])
                if value_head is not None:
                    value_head.load_state_dict(state["value_head"])
                optimizer.load_state_dict(state["optimizer"])
                _restore_rng(state["rng"], generator, c["device"] == "cuda")
                steps, rollout_updates = state["steps"], state["rollout_updates"]
                pending, history = state["pending"], state["history"]
                order, position = state["order"], state["position"]
                initial_hash, initial_value_hash = state["initial_hash"], state["initial_value_hash"]
                elapsed_before = state["elapsed_seconds"]
                del state
            started, initial_steps = time.perf_counter(), steps

            def elapsed():
                return elapsed_before+time.perf_counter()-started

            def peak():
                return torch.cuda.max_memory_allocated() if c["device"] == "cuda" else None

            def write_metrics():
                temporary = out/"metrics.jsonl.tmp"
                with temporary.open("w", encoding="utf-8") as stream:
                    for entry in history:
                        stream.write(json.dumps(entry, allow_nan=False)+"\n")
                os.replace(temporary, out/"metrics.jsonl")

            def commit():
                _atomic_state(state_path, {"schema_version": 1, "identity": identity,
                    "model": model.state_dict(), "value_head": value_head.state_dict() if value_head is not None else None,
                    "optimizer": optimizer.state_dict(), "rng": _rng_state(generator, c["device"] == "cuda"),
                    "steps": steps, "rollout_updates": rollout_updates, "pending": pending, "history": history,
                    "order": order, "position": position, "initial_hash": initial_hash,
                    "initial_value_hash": initial_value_hash, "elapsed_seconds": elapsed()})
                write_metrics()

            def sample_batch():
                nonlocal order, position
                selected = []
                for _ in range(c["per_device_train_batch_size"]):
                    if position == len(order):
                        order, position = generator.permutation(len(rows)).tolist(), 0
                    selected.append(rows[order[position]])
                    position += 1
                return selected

            if not state_path.exists():
                commit()  # Even a crash before the first update has an exact restart point.
            target = c["ppo_updates"] if stage == "ppo" else c[f"{stage}_max_steps"]
            while (rollout_updates if stage == "ppo" else steps) < target:
                step_started = time.perf_counter()
                if stage == "ppo" and pending is None:
                    pending = _rollout(model, value_head, reference, reward, tokenizer, rows, generator, c)
                    commit()  # Persist behavior-policy quantities before any PPO epoch changes the policy.
                optimizer.zero_grad(set_to_none=True)
                metrics, loss_value = {}, 0.
                accumulation = 1 if stage == "ppo" else c["gradient_accumulation_steps"]
                for _ in range(accumulation):
                    model.eval() if stage in {"ppo", "dpo"} else model.train()
                    with _autocast(c):
                        loss, current = (_ppo_loss(model, value_head, pending, c) if stage == "ppo" else
                                          _supervised_loss(model, reference, tokenizer, sample_batch(), c, stage))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite training objective")
                    (loss/accumulation).backward()
                    loss_value += float(loss.detach())/accumulation
                    for key, value in current.items():
                        if value is None or isinstance(value, str):
                            metrics[key] = value
                        else:
                            metrics[key] = metrics.get(key, 0.)+float(value)/accumulation
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, c["max_grad_norm"]))
                if not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm; optimizer step rejected")
                optimizer.step()
                steps += 1
                if stage == "ppo":
                    pending["cursor"] += 1
                    if pending["cursor"] == len(pending["minibatches"]):
                        pending = None
                        rollout_updates += 1
                if c["device"] == "cuda":
                    torch.cuda.synchronize()
                entry = {"stage": stage, "optimizer_step": steps, "rollout_updates": rollout_updates,
                         "loss": loss_value, "grad_norm": norm, "elapsed_seconds": elapsed(),
                         "step_seconds": time.perf_counter()-step_started, "peak_allocated_bytes": peak(), **metrics}
                history.append(entry)
                print(json.dumps(entry, allow_nan=False), flush=True)
                write_metrics()
                paused = pause_after is not None and steps-initial_steps >= pause_after
                complete = (rollout_updates if stage == "ppo" else steps) >= target
                if steps % c["checkpoint_every"] == 0 or paused or complete or (stage == "ppo" and pending is None):
                    commit()
                if paused and not complete:
                    break
            complete = (rollout_updates if stage == "ppo" else steps) >= target
            final_hash = _state_hash(model)
            manifest.update(status="completed" if complete else "paused", completed_optimizer_steps=steps,
                            completed_rollout_batches=rollout_updates, elapsed_seconds=elapsed(),
                            weight_fingerprint_before=initial_hash, weight_fingerprint_after=final_hash,
                            fingerprint_scope="SHA256 of every model state tensor", weights_changed=initial_hash != final_hash,
                            peak_allocated_bytes=peak(), tokenization=tokenization,
                            pending_ppo_minibatch_cursor=None if pending is None else pending["cursor"],
                            finished_at_utc=utc_now(), dependencies=dependencies)
            if value_head is not None:
                manifest["value_head_weights_changed"] = initial_value_hash != _state_hash(value_head)
            if complete:
                if not manifest["weights_changed"]:
                    raise RuntimeError("No parameter updates detected; not marking this run completed")
                manifest["validation"] = _validation(model, reference, tokenizer, c, stage)
                model.save_pretrained(str(out), safe_serialization=True)
                tokenizer.save_pretrained(str(out))
                if value_head is not None:
                    save_file({k: v.detach().cpu().contiguous() for k, v in value_head.state_dict().items()}, str(out/"value_head.safetensors"))
                manifest["artifacts"] = _files_hashes(out, exclude=("run_manifest.json",))
            _atomic_json(manifest_path, manifest)
            return manifest
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=utc_now())
            _atomic_json(manifest_path, manifest)
            raise
