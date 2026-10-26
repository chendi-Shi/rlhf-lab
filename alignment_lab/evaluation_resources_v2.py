"""Explicit service-preserving admission for the independent NF4 evaluation.

This module changes no historical training guard. Importing it never imports
Torch. The allocator ceiling is configured only after live admission, before
the first model allocation. Health/process/GPU observations fail closed.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
import re
import time
import urllib.request

from .evaluation_protocol_v2 import parse, verified_bytes, _finite, digest

MIB = 1024 * 1024
KIND = "alignlab-evaluation-resource-profile-v2"
LIMITS = {"gpu_index": 0, "gpu_total_mib": 8192, "allocator_limit_mib": 2560,
          "non_allocator_allowance_mib": 256, "minimum_global_free_mib": 1024,
          "min_available_ram_gib": 8, "max_start_utilization_percent": 20,
          "health_timeout_seconds": 3, "poll_seconds": 10}
FIELDS = {"schema_version", "kind", "services", "background_processes", "restoration_report", *LIMITS}
IDENTITY_FIELDS = {"pid", "parent_pid", "created_at_utc", "command_line"}


def signature(process):
    if (not isinstance(process, dict) or type(process.get("pid")) is not int
            or process["pid"] <= 0 or type(process.get("parent_pid")) is not int
            or not isinstance(process.get("command_line"), str) or not process["command_line"]):
        raise ValueError("Complete process identity required")
    stamp = process["created_at_utc"]
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    fraction = re.search(r"T\d\d:\d\d:\d\d(?:\.(\d{1,9}))?(?:Z|[+-]\d\d:\d\d)$", stamp)
    if value.tzinfo is None or fraction is None:
        raise ValueError("Precise zoned process creation timestamp required")
    created = (value.astimezone(timezone.utc).replace(microsecond=0),
               int((fraction.group(1) or "").ljust(9, "0")))
    return (process["pid"], process["parent_pid"], created, process["command_line"])


def validate_profile(profile):
    if (not isinstance(profile, dict) or set(profile) != FIELDS
            or type(profile["schema_version"]) is not int or profile["schema_version"] != 1
            or profile["kind"] != KIND
            or any(type(profile[k]) is not int or profile[k] != v for k, v in LIMITS.items())):
        raise ValueError("Exact independent evaluation resource ceilings required")
    services = profile["services"]
    if (not isinstance(services, list) or len(services) != 2
            or {s.get("name") for s in services} != {"evidoc", "shopassist"}):
        raise ValueError("Both restored services must be frozen")
    seen = set()
    if not isinstance(profile["background_processes"], list):
        raise ValueError("Explicit background process identities required")
    for process in profile["background_processes"]:
        if set(process) != IDENTITY_FIELDS or process["pid"] in seen:
            raise ValueError("Exact unique background process required")
        signature(process)
        seen.add(process["pid"])
    for service in services:
        if set(service) != {"name", "url", "processes", "json_equals"}:
            raise ValueError("Exact service binding required")
        port = 8083 if service["name"] == "evidoc" else 8765
        if (not isinstance(service["url"], str)
                or not service["url"].startswith(f"http://127.0.0.1:{port}/")
                or "?" in service["url"] or "#" in service["url"]):
            raise ValueError("Only the restored service's loopback health URL is allowed")
        if not isinstance(service["processes"], list) or not service["processes"]:
            raise ValueError("Service process identities required")
        for process in service["processes"]:
            if set(process) != IDENTITY_FIELDS or process["pid"] in seen:
                raise ValueError("Unique exact service process identities required")
            signature(process)
            seen.add(process["pid"])
        equal = service["json_equals"]
        if not isinstance(equal, dict) or not equal or any(not isinstance(k, str) or not k for k in equal):
            raise ValueError("Pinned health fields required")
        if service["name"] == "evidoc" and any(equal.get(k) != v for k, v in
                {"ready": True, "device": "cuda:0", "dtype": "bf16", "quantization": "none"}.items()):
            raise ValueError("EviDoc CUDA BF16 ready identity must be preserved")
        if service["name"] == "shopassist" and equal != {
                "status": "ok", "mode": "offline", "version": "0.2.0", "gpu_required": False}:
            raise ValueError("Restored ShopAssist offline 0.2.0 identity required")
    return profile


def load_profile(binding):
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not Path(binding["path"]).is_absolute()):
        raise ValueError("Externally pinned absolute resource profile required")
    profile = validate_profile(parse(verified_bytes(Path(binding["path"]), binding["sha256"])))
    restoration = profile["restoration_report"]
    if (not isinstance(restoration, dict) or set(restoration) != {"path", "sha256"}
            or not Path(restoration["path"]).is_absolute()):
        raise ValueError("Pinned verified restoration report required")
    evidence = parse(verified_bytes(Path(restoration["path"]), restoration["sha256"]))
    if evidence.get("status") != "restored_and_verified" or evidence.get("restoration_complete") is not True:
        raise ValueError("Both services must already have completed restoration")
    for service in profile["services"]:
        saved = evidence[service["name"]]
        if {signature(p) for p in service["processes"]} != {signature(p) for p in saved["processes"]}:
            raise ValueError("Profile service process differs from restoration evidence")
        expected_health = {k: v for k, v in saved["health"].items() if k != "busy"}
        if service["json_equals"] != expected_health:
            raise ValueError("All stable restored health fields must remain pinned")
    return profile


def _health(url, timeout):
    # Disable environment proxies: these are fixed local service observations.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError("Service health did not return HTTP 200")
        payload = response.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("Unexpected health payload size")
    value = parse(payload)
    if not isinstance(value, dict):
        raise ValueError("Object health response required")
    return value


def _dotted(value, key):
    for part in key.split("."):
        value = value[part]
    return value


def required_free_mib(profile, expected_peak_reserved_bytes=None):
    validate_profile(profile)
    ceiling = profile["allocator_limit_mib"]
    if expected_peak_reserved_bytes is not None:
        if (type(expected_peak_reserved_bytes) is not int or expected_peak_reserved_bytes <= 0
                or expected_peak_reserved_bytes > ceiling * MIB):
            raise ValueError("Capacity peak must be positive and within the frozen allocator ceiling")
        ceiling = math.ceil(expected_peak_reserved_bytes / MIB)
    return ceiling + profile["non_allocator_allowance_mib"] + profile["minimum_global_free_mib"]


def collect_resources(profile, *, expected_peak_reserved_bytes=None, active=False,
                      allowed_processes=(), inventory_reader=None, ram_reader=None,
                      process_reader=None, health_reader=None, allocator_receipt=None):
    """Observe actual resources; injected readers exist only for CPU unit tests."""
    validate_profile(profile)
    from rlhf_lab.runtime import available_ram_gb, gpu_inventory
    from .probe_dispatch import _System
    inventory = (inventory_reader or gpu_inventory)()
    ram = (ram_reader or available_ram_gb)()
    live = (process_reader or _System().processes)()
    read_health = health_reader or _health
    blockers, health = [], {}
    expected = {}
    for process in profile["background_processes"]:
        expected[process["pid"]] = process
    for service in profile["services"]:
        for process in service["processes"]:
            expected[process["pid"]] = process
    for process in allowed_processes:
        signature(process)
        if process["pid"] in expected and signature(expected[process["pid"]]) != signature(process):
            raise ValueError("Conflicting allowed process identity")
        expected[process["pid"]] = process
    seen = set()
    for process in live:
        try:
            actual = signature(process)
            pid = process["pid"]
            if pid in seen:
                raise ValueError("Duplicate process PID")
            seen.add(pid)
            if pid in expected:
                if actual != signature(expected[pid]):
                    blockers.append(f"Frozen process identity changed: {pid}")
            else:
                # The live reader enumerates only Python processes. A familiar
                # command or parent cannot grant a new PID a frozen identity.
                blockers.append(f"Unknown Python process outside exact frozen identities: {pid}")
        except (ValueError, KeyError, TypeError) as error:
            blockers.append(f"Uninspectable Python process: {error}")
    for pid in expected:
        if pid not in seen:
            blockers.append(f"Frozen service/evaluator process disappeared: {pid}")
    for service in profile["services"]:
        try:
            value = read_health(service["url"], profile["health_timeout_seconds"])
            for key, wanted in service["json_equals"].items():
                actual = _dotted(value, key)
                if type(actual) is not type(wanted) or actual != wanted:
                    raise ValueError(f"Frozen health field changed: {key}")
            if not active and service["name"] == "evidoc" and value.get("busy") is not False:
                raise ValueError("EviDoc must be idle before evaluation allocation")
            health[service["name"]] = {"url": service["url"], "observed": value, "matched": True}
        except Exception as error:
            blockers.append(f"{service['name']} health unavailable: {type(error).__name__}: {error}")
    free_required = profile["minimum_global_free_mib"] if active else required_free_mib(profile, expected_peak_reserved_bytes)
    if allocator_receipt is not None:
        verify_allocator_receipt(allocator_receipt, profile)
        if not active:
            # The existing CUDA context is already included in observed used
            # device memory. Do not charge its startup allowance a second time.
            free_required -= profile["non_allocator_allowance_mib"]
    gpu = next((g for g in inventory.get("gpus", []) if g.get("index") == 0), None)
    if inventory.get("error") or gpu is None:
        blockers.append("GPU observation unavailable")
    else:
        try:
            total = _finite(gpu["total_mb"], "GPU total")
            free = _finite(gpu["free_mb"], "GPU free")
            util = _finite(gpu["utilization_percent"], "GPU utilization")
            if total != profile["gpu_total_mib"] or free < free_required or free > total or not 0 <= util <= 100:
                blockers.append(f"GPU memory outside frozen profile: {gpu}; required_free_mib={free_required}")
            if not active and util > profile["max_start_utilization_percent"]:
                blockers.append(f"GPU busy before load: {util}%")
        except (KeyError, TypeError, ValueError) as error:
            blockers.append(f"Invalid GPU observation: {error}")
    if ram is None or not math.isfinite(ram) or ram < profile["min_available_ram_gib"]:
        blockers.append(f"Available RAM insufficient: {ram}")
    return {"kind": "alignlab-evaluation-resource-observation-v2", "checked_at_epoch": time.time(),
            **inventory, "available_ram_gb": ram, "processes": live, "service_health": health,
            "active_evaluation": active, "required_free_mib": free_required,
            "cuda_context_already_initialized": allocator_receipt is not None,
            "allocator_limit_mib": profile["allocator_limit_mib"], "blockers": blockers}


def configure_cuda_allocator(profile):
    """Real PyTorch allocator limit, set before loading the NF4 model."""
    validate_profile(profile)
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    if not (profile["gpu_total_mib"] - 1) * MIB <= total <= profile["gpu_total_mib"] * MIB:
        raise RuntimeError("Actual CUDA capacity differs from the frozen device")
    fraction = profile["allocator_limit_mib"] * MIB / total
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    actual = torch.cuda.get_per_process_memory_fraction(0)
    if abs(actual - fraction) > 1e-12:
        raise RuntimeError("CUDA allocator ceiling was not applied")
    return {"kind": "alignlab-cuda-allocator-ceiling-v2", "device": "cuda:0",
            "total_memory_bytes": total, "limit_bytes": profile["allocator_limit_mib"] * MIB,
            "fraction": actual, "configured_before_model_load": True}


def verify_allocator_receipt(receipt, profile):
    validate_profile(profile)
    if not isinstance(receipt, dict):
        raise ValueError("Actual allocator receipt required")
    total = receipt.get("total_memory_bytes")
    if (type(total) is not int
            or not (profile["gpu_total_mib"] - 1) * MIB <= total <= profile["gpu_total_mib"] * MIB):
        raise ValueError("CUDA-reported device capacity differs from frozen GPU")
    expected = {"kind": "alignlab-cuda-allocator-ceiling-v2", "device": "cuda:0",
                "total_memory_bytes": total,
                "limit_bytes": profile["allocator_limit_mib"] * MIB,
                "fraction": profile["allocator_limit_mib"] * MIB / total,
                "configured_before_model_load": True}
    if receipt != expected:
        raise ValueError("Actual pre-model CUDA allocator ceiling evidence required")


class RuntimeResourceGuard:
    """Per-action deadline plus bounded live service/memory checks every ten seconds."""
    def __init__(self, profile, *, system, deadline_epoch, allowed_processes,
                 stop_requested=lambda: False, collector=None, memory_reader=None):
        self.profile, self.system, self.deadline = validate_profile(profile), system, deadline_epoch
        self.allowed, self.stop = tuple(allowed_processes), stop_requested
        self.collector, self.memory_reader = collector or collect_resources, memory_reader
        self.last = float("-inf")
        self.observations = []

    def __call__(self, *, force=False):
        if self.stop():
            raise InterruptedError("Evaluation STOP requested")
        if self.system.now() >= self.deadline:
            raise TimeoutError("Original independent evaluation deadline exhausted")
        if force or self.system.now() - self.last >= self.profile["poll_seconds"]:
            result = self.collector(self.profile, active=True, allowed_processes=self.allowed)
            if self.memory_reader is None:
                import torch
                torch.cuda.synchronize(0)
                memory = {"reserved_bytes": torch.cuda.memory_reserved(0),
                          "max_reserved_bytes": torch.cuda.max_memory_reserved(0)}
            else:
                memory = self.memory_reader()
            if any(type(memory.get(k)) is not int or not 0 <= memory[k] <= self.profile["allocator_limit_mib"] * MIB
                   for k in ("reserved_bytes", "max_reserved_bytes")):
                raise RuntimeError("Evaluation allocator ceiling exceeded or unobservable")
            self.observations.append({"checked_at_epoch": self.system.now(), "gpus": result["gpus"],
                "available_ram_gb": result["available_ram_gb"], "blockers": result["blockers"],
                "service_health_sha256": digest(result["service_health"]), "memory": memory})
            self.last = self.system.now()
            if result["blockers"]:
                raise RuntimeError("Service-preserving evaluation guard: " + "; ".join(result["blockers"]))
            if self.system.now() >= self.deadline:
                raise TimeoutError("Independent deadline expired during resource observation")
        return False
