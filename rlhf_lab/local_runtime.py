"""Local CPU/CUDA profiles; inventory never imports torch or initializes CUDA."""
import json
import math
import re
from pathlib import Path

from .runtime import doctor, resolve_path

STAGES = ("sft", "reward", "dpo", "ppo")


def load_local_config(path):
    path = resolve_path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_path", "data_dir", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing path: {key}")
        config[key] = str(resolve_path(config[key]))
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must explicitly be cpu or cuda")
    if config.get("model_id") != "openai-community/gpt2":
        raise ValueError("This local profile is implemented for GPT-2")
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("model_revision", "")):
        raise ValueError("Pin model_revision to a full commit SHA")
    for key in ("seed", "gpu_index"):
        if type(config.get(key, 0)) is not int or config.get(key, 0) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("max_prompt_tokens", "max_response_tokens", "gradient_accumulation_steps",
                "per_device_train_batch_size", "sft_max_steps", "reward_max_steps", "dpo_max_steps",
                "ppo_updates", "ppo_rollout_batch_size", "ppo_response_length", "ppo_epochs",
                "ppo_minibatch_size", "num_threads", "eval_limit", "eval_preference_limit"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_response_tokens"] < 2 or config["max_prompt_tokens"] + max(config["max_response_tokens"], config["ppo_response_length"]) > 1024:
        raise ValueError("Invalid GPT-2 response/context length")
    if config["ppo_rollout_batch_size"] < 4 or config["ppo_minibatch_size"] > config["ppo_rollout_batch_size"]:
        raise ValueError("PPO requires >=4 rollout sequences and minibatch <= rollout batch")
    for key in ("sft_learning_rate", "reward_learning_rate", "dpo_learning_rate", "ppo_learning_rate",
                "dpo_beta", "ppo_cliprange", "ppo_value_coef", "min_free_ram_gb", "min_free_gpu_mb"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= config.get("kl_coef", -1) or not 0 <= config.get("max_gpu_utilization", -1) <= 100:
        raise ValueError("Invalid KL coefficient or GPU utilization threshold")
    if config.get("precision") != ("float32" if config["device"] == "cpu" else "bf16"):
        raise ValueError("Use float32 on CPU, bf16 autocast on CUDA")
    output = Path(config["output_dir"])
    for key in ("data_dir", "model_path"):
        protected = Path(config[key])
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("output_dir must be separate from data/model directories")
    config["config_path"] = str(path)
    return config


def local_blockers(info, config):
    blockers = []
    ram = info.get("available_ram_gb")
    if ram is None or ram < config.get("min_free_ram_gb", 6):
        blockers.append(f"Available RAM {ram!r} GB; require {config.get('min_free_ram_gb', 6)} GB")
    if config.get("device") == "cpu":
        return blockers
    gpu = next((g for g in info.get("gpus", []) if g["index"] == config.get("gpu_index", 0)), None)
    if gpu is None:
        blockers.append("Selected NVIDIA GPU cannot be inspected")
    else:
        if gpu["free_mb"] < config.get("min_free_gpu_mb", 6144):
            blockers.append(f"Free VRAM {gpu['free_mb']} MiB; require {config.get('min_free_gpu_mb', 6144)} MiB")
        if gpu["utilization_percent"] > config.get("max_gpu_utilization", 20):
            blockers.append(f"GPU utilization {gpu['utilization_percent']}%; require <= {config.get('max_gpu_utilization', 20)}%")
    return blockers


def local_doctor(config):
    info = doctor(None)
    info.update(device=config["device"], blockers=local_blockers(info, config),
                profile="local-pytorch", note="TRL/datasets are not required. CUDA operator compatibility is checked by the engine after the resource guard.")
    return info


def assert_local_resources(config):
    info = local_doctor(config)
    if info["blockers"]:
        raise RuntimeError("Local run not started: " + "; ".join(info["blockers"]))
    return info


def local_plan(config, stage=None):
    return {"status": "plan_only", "training_started": False, "cuda_initialized": False,
            "device": config["device"], "output_dir": config["output_dir"],
            "stages": [stage] if stage and stage != "all" else list(STAGES),
            "branches": {"dpo": "base -> SFT -> DPO", "ppo": "base -> SFT -> reward model -> PPO"},
            "scope": config.get("experiment_scope", "pilot"),
            "step_budget": {s: config.get(s + "_max_steps", config.get("ppo_updates")) for s in STAGES}}
