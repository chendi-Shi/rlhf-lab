"""A bounded initial NF4 SFT training/recovery probe, never full capacity."""
from __future__ import annotations
import copy
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

from rlhf_lab.runtime import available_ram_gb, gpu_inventory

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-nf4-sft-initial-probe-v1"
AUDIT_SHA = "84d059ba111c37d8f89d70cec6db6bd6c4fd7aef8d4cf933343d67556a204e0b"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def admission_blockers(gpus, ram, processes, *, min_free_mb=7000, min_ram_gb=8, max_util=20):
    blockers = []
    gpu = next((g for g in gpus if g["index"] == 0), None)
    if gpu is None:
        blockers.append("GPU 0 unavailable")
    elif gpu["free_mb"] < min_free_mb or gpu["utilization_percent"] > max_util:
        blockers.append(f"GPU resources busy: {gpu}")
    if ram is None or ram < min_ram_gb:
        blockers.append(f"Available RAM insufficient: {ram}")
    for process in processes:
        command = (process.get("CommandLine") or "").lower()
        if "run_alignlab_probe.py" in command:
            continue  # Exclusive project lock handles duplicate probe processes.
        if not command:
            blockers.append(f"Cannot inspect Python PID {process.get('ProcessId')}")
        elif any(marker in command for marker in ("run_campaign.py", "local-train", "local-evaluate",
                      "windows.train", "train_sft", "train_dpo", "train_ppo", "run_training", "serve_adapter")):
            blockers.append(f"Other training/evaluation/inference process: {process}")
    return blockers


def resources():
    inventory = gpu_inventory()
    command = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } "
               "| Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine) | ConvertTo-Json -Depth 4 -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, timeout=20, check=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    processes = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(processes, dict):
        processes = [processes]
    ram = available_ram_gb()
    return {"checked_at_utc": utc(), **inventory, "available_ram_gb": ram, "python_processes": processes,
            "blockers": admission_blockers(inventory["gpus"], ram, processes)}


def select_examples(rows, *, seed, count, prompt_limit, response_limit):
    eligible = [row for row in rows if len(row.prompt_ids) <= prompt_limit
                and len(row.chosen_ids) <= response_limit]
    random.Random(seed).shuffle(eligible)
    if len(eligible) < count:
        raise ValueError("Insufficient complete examples for the registered compatibility probe")
    return eligible[:count], len(eligible)


def require_active_boundary(deadline_epoch, stop_requested, *, clock=time.time):
    """A cooperative boundary; an in-flight CUDA operation may finish later."""
    if stop_requested():
        raise InterruptedError("Probe STOP requested at an optimizer boundary")
    if clock() >= deadline_epoch:
        raise TimeoutError("Probe scheduling budget exhausted")


def validate_plan(plan):
    if plan.get("kind") != KIND or plan.get("formal_training") is not False:
        raise ValueError("Not an initial compatibility-only plan")
    expected = {"seed": 42, "samples": 4, "effective_batch": 2, "microbatch": 1,
                "max_original_updates": 2, "max_replay_updates": 1, "prompt_limit": 512,
                "response_limit": 128, "wall_seconds": 1800, "learning_rate": 1e-4}
    if plan.get("work") != expected or plan.get("audit_manifest_sha256") != AUDIT_SHA:
        raise ValueError("Compatibility work/time/data identity changed")
    output = Path(plan["output"]).resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        raise ValueError("Probe output must be a fresh dedicated runs directory")
    sources = plan.get("source_sha256")
    required = {str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "alignment_lab").glob("*.py")}
    required |= {"scripts/run_alignlab_probe.py", "rlhf_lab/runtime.py"}
    if not isinstance(sources, dict) or set(sources) != required:
        raise ValueError("Probe must freeze the complete implementation inventory")
    if any(sha(ROOT / name) != digest for name, digest in sources.items()):
        raise ValueError("Frozen probe source changed")
    report = Path(plan["cpu_check_report"])
    if sha(report) != plan["cpu_check_report_sha256"]:
        raise ValueError("CPU check report identity changed")
    checks = json.loads(report.read_text(encoding="utf-8"))
    if checks.get("status") != "cpu_software_checks_passed" or not checks.get("source_unchanged"):
        raise ValueError("CPU correctness gate is incomplete")
    for name, digest in checks["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Source/test differs from the accepted CPU checks")
    return output


def run_probe(plan):
    import torch
    from .model_loader import ModelLoadConfig, load_quantized_stage
    from .training_data import load_training_partition
    from .stage_engine import SupervisedConfig, parameter_mapping, run_supervised
    from .checkpointing import build_identity, save_checkpoint, load_checkpoint

    output = validate_plan(plan)
    if output.exists():
        raise FileExistsError("Probe output exists; no overwrite, budget reset, or implicit resume")
    before = resources()
    if before["blockers"]:
        return {"status": "waiting_resources", "resources": before, "training_started": False}
    lock = ROOT / "reports/.alignlab-gpu.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created_at_utc": utc(), "output": str(output)}, stream)
    started = time.time()
    status = {"kind": KIND, "status": "starting", "started_at_utc": utc(),
              "started_at_epoch": started, "deadline_epoch": started + plan["work"]["wall_seconds"],
              "pid": os.getpid(), "plan": plan, "resources_before": before,
              "training_started": False, "formal_training": False, "capacity_verified": False,
              "actual_optimizer_updates": 0, "phases": [], "gpu_samples": []}

    def write():
        (output / "probe.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def emit(stage, **values):
        event = {"time_utc": utc(), "phase": stage, **values}
        status["phases"].append(event)
        status["status"] = stage
        write()
        print(json.dumps(event), flush=True)

    def record_gpu():
        torch.cuda.synchronize(0)
        status["gpu_samples"].append({"time_utc": utc(), **gpu_inventory(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(0)})

    try:
        output.mkdir(parents=True, exist_ok=False)
        write()
        emit("loading_verified_training_data")
        partition = load_training_partition(plan["audit_dir"], "sft_train", AUDIT_SHA)
        rows, eligible = select_examples(partition.rows, seed=42, count=4, prompt_limit=512, response_limit=128)
        status["selected_record_ids"] = [r.record_id for r in rows]
        status["eligible_short_examples"] = eligible
        status["selected_lengths"] = [{"prompt": len(r.prompt_ids), "response": len(r.chosen_ids)} for r in rows]
        require_active_boundary(status["deadline_epoch"], lambda: (output / "STOP").exists())
        latest_resources = resources()
        status["resources_before_cuda"] = latest_resources
        if latest_resources["blockers"]:
            raise RuntimeError(f"Resources changed before CUDA admission: {latest_resources['blockers']}")
        emit("loading_nf4_qwen3")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        loaded = load_quantized_stage(plan["model_snapshot"], plan["model_manifest_sha256"],
                                     plan["tokenizer_dir"], ModelLoadConfig("sft"))
        status["model"] = loaded.metadata
        status["cuda_device"] = str(loaded.device)
        record_gpu()
        require_active_boundary(status["deadline_epoch"], lambda: (output / "STOP").exists())
        backend = loaded.backend
        mapping = parameter_mapping(backend, "sft")
        optimizer = torch.optim.AdamW(mapping.values(), lr=plan["work"]["learning_rate"])
        config = SupervisedConfig(kind="sft", seed=42, effective_batch=2, microbatch=1,
            max_updates=2, deadline_epoch=status["deadline_epoch"], pad_id=151643, eos_id=151645,
            max_grad_norm=1.0, chunk_tokens=16)
        snapshot_manifest = json.loads((Path(plan["model_snapshot"]) / "snapshot-manifest.json").read_text())
        files = {name: ROOT / name for name in plan["source_sha256"]}
        files.update({f"base/{name}": Path(plan["model_snapshot"]) / name for name in snapshot_manifest["files"]})
        files.update({"data/manifest": Path(plan["audit_dir"]) / "manifest.json",
                      "data/sft": Path(plan["audit_dir"]) / "sft_train.jsonl",
                      "data/protocol": ROOT / "docs/ALIGNLAB_V2.md"})
        identity = build_identity(files, {"purpose": KIND, "stage": asdict(config), "model": loaded.metadata,
                                         "record_ids": status["selected_record_ids"]})
        initial = {k: p.detach().cpu().clone() for k, p in mapping.items()}
        saved = []
        generator = torch.Generator(device=loaded.device).manual_seed(42)

        def checkpoint(cursor, extra):
            status["actual_optimizer_updates"] += 1
            if status["actual_optimizer_updates"] > 2:
                raise RuntimeError("Original update budget exceeded")
            receipt = save_checkpoint(output / f"update-{cursor['optimizer_updates']:06d}",
                identity=identity, parameters=mapping, optimizer=optimizer, cursor=cursor,
                generators={"sampling": generator}, extra_state=extra)
            saved.append(receipt)
            status["training_started"] = True
            record_gpu()
            emit("sft_gpu_update", optimizer_updates=cursor["optimizer_updates"],
                 checkpoint_manifest_sha256=receipt["manifest_sha256"])

        emit("training_sft_nf4")
        with loaded.autocast():
            result = run_supervised(backend, "sft", optimizer, rows, config, device=loaded.device,
                checkpoint_callback=checkpoint, stop_requested=lambda: (output / "STOP").exists())
        status["original_stop_reason"] = result.stop_reason
        status["original_metrics"] = list(result.metrics)
        if result.stop_reason != "data_exhausted" or result.cursor["optimizer_updates"] != 2:
            raise RuntimeError(f"Initial probe incomplete: {result.stop_reason}")
        final_parameters = {k: p.detach().cpu().clone() for k, p in mapping.items()}
        status["adapter_weights_changed"] = any(not torch.equal(initial[k], final_parameters[k]) for k in mapping)
        if not status["adapter_weights_changed"]:
            raise RuntimeError("GPU updates did not change adapter weights")
        require_active_boundary(status["deadline_epoch"], lambda: (output / "STOP").exists())
        emit("verifying_disk_recovery")
        restored = load_checkpoint(saved[0]["checkpoint_path"], expected_identity=identity,
            parameters=mapping, optimizer=optimizer, generators={"sampling": generator},
            expected_manifest_sha256=saved[0]["manifest_sha256"])
        with loaded.autocast():
            replay = run_supervised(backend, "sft", optimizer, rows, config, device=loaded.device,
                                   state=restored["cursor"], stop_requested=lambda: (output / "STOP").exists())
        status["actual_optimizer_updates"] += len(replay.metrics)
        if replay.stop_reason != "data_exhausted" or len(replay.metrics) != 1 or status["actual_optimizer_updates"] != 3:
            raise RuntimeError("Recovery failed or exceeded the one-update replay budget")
        require_active_boundary(status["deadline_epoch"], lambda: (output / "STOP").exists())
        maximum_error = 0.0
        for name, parameter in mapping.items():
            actual = parameter.detach().cpu()
            maximum_error = max(maximum_error, float((actual - final_parameters[name]).abs().max()))
            torch.testing.assert_close(actual, final_parameters[name], rtol=2e-5, atol=2e-6)
        status["recovery_max_absolute_parameter_error"] = maximum_error
        status["recovery_verified"] = True
        record_gpu()
        status["scope_limitations"] = ["Four short SFT examples only, not full-length capacity",
            "No DPO20/PPO8 or longest-evaluation verification; full capacity remains false",
            "No independent quality metric or formal study result",
            "nvidia-smi usage is sampled, torch allocated/reserved peaks measured",
            "30-minute cooperative scheduling budget; an in-flight CUDA operation may finish after the deadline"]
        require_active_boundary(status["deadline_epoch"], lambda: (output / "STOP").exists())
        emit("initial_gpu_training_probe_passed")
        return status
    except BaseException as exc:
        status["status"] = ("budget_exhausted" if isinstance(exc, TimeoutError) else
                            "paused" if isinstance(exc, InterruptedError) else "failed")
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
        if output.exists():
            write()
        raise
    finally:
        status["finished_at_utc"] = utc()
        status["elapsed_seconds"] = time.time() - started
        if output.exists():
            write()
        # Only our own exclusive lock, not another run's lock.
        if lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid():
            lock.unlink()
