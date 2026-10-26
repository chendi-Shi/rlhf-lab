"""Read-only resource checks and configuration. No torch import here."""

import csv
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PINS = {"torch": "2.8.0", "transformers": "4.56.2", "trl": "0.25.1",
        "accelerate": "1.10.1", "datasets": "4.3.0", "numpy": "2.2.6", "psutil": "7.1.0"}


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_config(path):
    config_path = resolve_path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = ("seed", "model_id", "model_revision", "data_dir", "output_dir",
                "max_prompt_tokens", "max_response_tokens", "gradient_accumulation_steps",
                "per_device_train_batch_size", "sft_max_steps", "reward_max_steps",
                "ppo_total_episodes", "ppo_num_epochs", "min_free_gpu_mb", "min_free_ram_gb",
                "max_gpu_utilization", "eval_limit", "sft_learning_rate", "reward_learning_rate",
                "ppo_learning_rate", "kl_coef", "train_limit", "valid_limit", "test_limit")
    missing = set(required) - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    for key in ("max_prompt_tokens", "max_response_tokens", "gradient_accumulation_steps",
                "per_device_train_batch_size", "sft_max_steps", "reward_max_steps",
                "ppo_total_episodes", "ppo_num_epochs", "eval_limit", "train_limit", "valid_limit", "test_limit"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_prompt_tokens"] + config["max_response_tokens"] > 1024:
        raise ValueError("GPT-2 context must not exceed 1024 tokens")
    if config["max_response_tokens"] < 2:
        raise ValueError("max_response_tokens must include content and EOS")
    if config["model_id"] != "openai-community/gpt2":
        raise ValueError("This initial implementation is validated statically for GPT-2 only")
    if not isinstance(config["model_revision"], str) or len(config["model_revision"]) != 40:
        raise ValueError("Pin model_revision to a 40-character Hugging Face commit")
    if config.get("precision", "bf16") != "bf16":
        raise ValueError("This initial CUDA profile uses bf16 autocast with float32 trainable weights")
    for key in ("sft_learning_rate", "reward_learning_rate", "ppo_learning_rate", "min_free_gpu_mb", "min_free_ram_gb"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["kl_coef"] < 0 or not 0 <= config["max_gpu_utilization"] <= 100:
        raise ValueError("Invalid KL coefficient or GPU utilization threshold")
    for key in ("data_dir", "output_dir"):
        config[key] = str(resolve_path(config[key]))
    config["config_path"] = str(config_path)
    return config


def available_ram_gb():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.available / 1024**3
    try:
        entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(entries["MemAvailable"].split()[0]) / 1024**2
    except (OSError, KeyError, ValueError):
        return None


def gpu_inventory():
    command = ["nvidia-smi", "--query-gpu=index,name,driver_version,utilization.gpu,memory.total,memory.free", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        rows = []
        for fields in csv.reader(io.StringIO(result.stdout)):
            if not fields:
                continue
            idx, name, driver, util, total, free = [value.strip() for value in fields]
            rows.append({"index": int(idx), "name": name, "driver": driver,
                         "utilization_percent": float(util), "total_mb": float(total), "free_mb": float(free)})
        return {"gpus": rows, "error": None}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"gpus": [], "error": str(exc)}


def dependency_versions():
    versions = {}
    for package in PINS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def doctor(config=None):
    info = {"checked_at_utc": utc_now(), "python": sys.version.split()[0],
            "executable": sys.executable, "platform": platform.platform(),
            "available_ram_gb": available_ram_gb(), "dependencies": dependency_versions(),
            **gpu_inventory(), "cuda_initialized": False, "training_started": False}
    if config is not None:
        info["blockers"] = resource_blockers(info, config)
    return info


def resource_blockers(info, config):
    problems = []
    ram = info.get("available_ram_gb")
    if ram is None or ram < config["min_free_ram_gb"]:
        problems.append(f"Available system RAM {ram!r} GB; require {config['min_free_ram_gb']} GB")
    gpu = next((x for x in info.get("gpus", []) if x["index"] == config.get("gpu_index", 0)), None)
    if gpu is None:
        problems.append("Selected NVIDIA GPU could not be inspected")
        return problems
    if gpu["free_mb"] < config["min_free_gpu_mb"]:
        problems.append(f"Free VRAM {gpu['free_mb']} MiB; require {config['min_free_gpu_mb']} MiB")
    if gpu["utilization_percent"] > config["max_gpu_utilization"]:
        problems.append(f"GPU utilization {gpu['utilization_percent']}%; require <= {config['max_gpu_utilization']}%")
    try:
        driver = tuple(int(n) for n in gpu["driver"].split("."))
        # Conservative CUDA 12.6 native-driver floor, not a promise that every CUDA operation succeeds.
        floor = (560, 76) if os.name == "nt" else (560, 28, 3)
        if driver < floor:
            problems.append(f"Driver {gpu['driver']} is below the conservative CUDA 12.6 profile floor {'.'.join(map(str, floor))}")
    except ValueError:
        problems.append("Driver version could not be parsed")
    return problems


def assert_resources(config, require_cuda=True):
    if not require_cuda:
        raise ValueError("CPU training fallback is intentionally unsupported in this prepared GPU profile")
    info = doctor(config)
    if info["blockers"]:
        raise RuntimeError("Training/generation not started: " + "; ".join(info["blockers"]))
    return info


def assert_dependencies():
    found = dependency_versions()
    bad = [f"{name}=={version} (found {found[name]})" for name, version in PINS.items()
           if (found[name] or "").split("+")[0] != version]
    if bad:
        raise RuntimeError("Use the isolated training environment. Missing/mismatched candidate pins: " + ", ".join(bad))
    return found


def plan(config, stage=None):
    return {"status": "plan_only", "training_started": False, "gpu_initialized": False,
            "model": config["model_id"], "model_revision": config["model_revision"],
            "stage": stage or "sft -> reward -> ppo", "data_dir": config["data_dir"],
            "output_dir": config["output_dir"], "prompt_token_limit": config["max_prompt_tokens"],
            "response_token_limit": config["max_response_tokens"],
            "sft_steps": config["sft_max_steps"], "reward_steps": config["reward_max_steps"],
            "ppo_episodes": config["ppo_total_episodes"],
            "effective_batch_size": config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
            "requirements": "Prepared real data; matching isolated dependencies; free GPU/RAM; --execute",
            "warning": "VRAM fit, throughput and model improvement remain unmeasured."}
