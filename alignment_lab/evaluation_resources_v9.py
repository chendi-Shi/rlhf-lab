"""Admission after a verified service restoration and continued authorized pause.

This module changes no historical training guard. Importing it never imports
Torch. The allocator ceiling is configured only after live admission, before
the first model allocation. Health/process/GPU observations fail closed.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from .evaluation_protocol_v2 import parse, verified_bytes, _finite, digest, canonical

MIB = 1024 * 1024
KIND = "alignlab-evaluation-resource-profile-v9"
BASE_KIND = "alignlab-evaluation-resource-profile-v2"
PAUSE_KIND = "alignlab-evidoc-pause-continuation-authorization-v1"
PAUSE_QUOTE = "允许临时暂停 EviDoc，推进评测（推荐）"
STARTED = 1792895411
DEADLINE = 1793673011
WALL_SECONDS = 777600
LIMITS = {"gpu_index": 0, "gpu_total_mib": 8192, "allocator_limit_mib": 2560,
          "non_allocator_allowance_mib": 256, "minimum_global_free_mib": 1024,
          "min_available_ram_gib": 8, "max_start_utilization_percent": 20,
          "health_timeout_seconds": 3, "poll_seconds": 10}
BASE_FIELDS = {"schema_version", "kind", "services", "background_processes", "restoration_report", *LIMITS}
FIELDS = BASE_FIELDS | {"authorized_pause", "service_identity_succession"}
IDENTITY_FIELDS = {"pid", "parent_pid", "created_at_utc", "command_line"}
V8_BASELINE_SHA256 = "1464807fd737b1062ebe965a76732bd4ad9d3a8e49103f51f9c4d33a2c94b94f"
V8_PLAN_SHA256 = "47ff74f67bff413f46208a02b168725d089e88a4e37bc4ebecad268c8a2a3e17"
SUCCESSION_SHA256 = {
    "restoration_report": "909f3470f8bf6026ad533f1e3341ef5bc2093eacb91035c3c84039b05ed26d19",
    "restoration_manifest": "4bf0d7d1cb5c41eba1803abbb0c2161209a53873f278e29c6820ccaf2ef070a7",
    "prelaunch_closure": "a66fef19a9d50771ff5f0bd2241da593a53b394e41c4d84c63a1ec662c329340",
}
PREPARE_EVIDENCE = "static-prepare-accepted-20261029T0440465895427Z.json"
ADMISSION_EVIDENCE = "admission-20261029T0630008242764Z.json"
DIRECT_ATTEMPT_EVIDENCE = "direct-user-single-launch-attempt-20261029T0629032319123Z.json"
CLOSURE_EVIDENCE = frozenset({"plan.json", PREPARE_EVIDENCE, ADMISSION_EVIDENCE,
    "launch-failed.json", "guardian/RESTORE", "guardian/restore-launch-attempt.json",
    "guardian/status.json", "restoration-complete.json", "restoration-transaction.json",
    "restoration-manifest.json", "pause-complete.json", "evidoc-pause-authorization.json",
    "resource-profile.json", DIRECT_ATTEMPT_EVIDENCE})
INHERITED_SLOTS = frozenset("ifeval/" + name for name in
    ("base", "sft", "dpo42", "ppo42", "dpo43", "ppo43", "dpo44"))


def signature(process):
    if (not isinstance(process, dict) or type(process.get("pid")) is not int
            or process["pid"] <= 0 or type(process.get("parent_pid")) is not int
            or not isinstance(process.get("command_line"), str) or not process["command_line"]
            or not isinstance(process.get("created_at_utc"), str)):
        raise ValueError("Complete process identity required")
    stamp = process["created_at_utc"]
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    fraction = re.search(r"T\d\d:\d\d:\d\d(?:\.(\d{1,9}))?(?:Z|[+-]\d\d:\d\d)$", stamp)
    if value.tzinfo is None or fraction is None:
        raise ValueError("Precise zoned process creation timestamp required")
    created = (value.astimezone(timezone.utc).replace(microsecond=0),
               int((fraction.group(1) or "").ljust(9, "0")))
    return (process["pid"], process["parent_pid"], created, process["command_line"])


def _created_ticks(process):
    """Compute Windows UTC ticks exactly; never round through a floating value."""
    _, _, (second, nanoseconds), _ = signature(process)
    if nanoseconds % 100:
        raise ValueError("Process creation time cannot be represented in Windows ticks")
    elapsed = second - datetime(1, 1, 1, tzinfo=timezone.utc)
    return str((elapsed.days * 86400 + elapsed.seconds) * 10_000_000 + nanoseconds // 100)


def _identity_evidence(process):
    """Preserve the observed identity even when a field cannot be inspected."""
    if not isinstance(process, dict):
        return {"observed_type": type(process).__name__, "identity_complete": False,
                "created_utc_ticks": None, "error": "Process identity is not an object"}
    evidence = {key: process.get(key) for key in sorted(IDENTITY_FIELDS)}
    for key in ("name", "executable", "created_utc_ticks", "handle_start_utc_ticks"):
        if key in process:
            # Existing readers may supply integer ticks. Persist only exact text.
            evidence[key] = (str(process[key]) if key.endswith("ticks")
                             and type(process[key]) is int else process[key])
    try:
        evidence["created_utc_ticks"] = _created_ticks(process)
        evidence["identity_complete"] = True
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        evidence.update(identity_complete=False, created_utc_ticks=None,
                        error=f"{type(error).__name__}: {error}")
    return evidence


def _validate_base(profile):
    if (not isinstance(profile, dict) or set(profile) != BASE_FIELDS
            or type(profile["schema_version"]) is not int or profile["schema_version"] != 1
            or profile["kind"] != BASE_KIND
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


def _binding(binding):
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute()
            or not isinstance(binding["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is None):
        raise ValueError("Externally pinned absolute evidence binding required")
    return binding


def _read(binding):
    _binding(binding)
    return parse(verified_bytes(Path(binding["path"]), binding["sha256"]))


def _base(profile):
    base = {key: value for key, value in profile.items()
            if key not in {"authorized_pause", "service_identity_succession"}}
    base.update(kind=BASE_KIND, schema_version=1)
    return base


def validate_profile(profile):
    if (not isinstance(profile, dict) or set(profile) != FIELDS
            or type(profile["schema_version"]) is not int or profile["schema_version"] != 7
            or profile["kind"] != KIND):
        raise ValueError("Explicit V9 authorized-pause resource profile required")
    _validate_base(_base(profile))
    pause = profile["authorized_pause"]
    if (not isinstance(pause, dict) or set(pause) != {
            "service", "mode", "baseline_profile", "authorization", "deadline_epoch"}
            or pause["service"] != "evidoc" or pause["mode"] != "temporarily_paused"
            or type(pause["deadline_epoch"]) is not int or pause["deadline_epoch"] != DEADLINE):
        raise ValueError("Only the finite explicitly authorized EviDoc pause is allowed")
    _binding(pause["baseline_profile"])
    _binding(pause["authorization"])
    succession = profile["service_identity_succession"]
    if not isinstance(succession, dict) or set(succession) != set(SUCCESSION_SHA256):
        raise ValueError("Pinned V8 restoration, manifest and prelaunch closure required for identity succession")
    for binding in succession.values():
        _binding(binding)
    evidoc = next(service for service in profile["services"] if service["name"] == "evidoc")
    shop = next(service for service in profile["services"] if service["name"] == "shopassist")
    if (len(profile["background_processes"]) != 3 or len(shop["processes"]) != 2
            or len(evidoc["processes"]) != 1
            or evidoc["url"] != "http://127.0.0.1:8083/health"
            or "scripts.serve_adapter" not in evidoc["processes"][0]["command_line"]):
        raise ValueError("Only the restored adapter may pause; three background and two ShopAssist identities remain")
    return profile


def active_services(profile):
    validate_profile(profile)
    return tuple(service for service in profile["services"] if service["name"] != "evidoc")


def active_service_processes(profile):
    return tuple(process for service in active_services(profile) for process in service["processes"])


def _expect(value, expected, message):
    if not isinstance(value, dict) or any(type(value.get(k)) is not type(v)
            or value.get(k) != v for k, v in expected.items()):
        raise ValueError(message)


def _exact_identity(value):
    identity = signature(value)
    ticks = value.get("created_utc_ticks")
    if type(ticks) not in (int, str) or str(ticks) != _created_ticks(value):
        raise ValueError("Exact CIM creation ticks required; handle ticks cannot substitute")
    return identity


def _verify_prelaunch_closure(profile, baseline, receipt, manifest):
    """Replay the failed V8 admission and its one completed restoration, read-only.

    This proves historical closure, never resource recovery or new live admission.
    All referenced files stay in their original allocation and are rehashed.
    """
    succession, prior_pause = profile["service_identity_succession"], baseline["authorized_pause"]
    closure = _read(succession["prelaunch_closure"])
    _expect(closure, {"kind": "alignlab-v8-prelaunch-admission-failure-closure-v1",
        "status": "v8_prelaunch_admission_failed_closed_services_restored",
        "failure_reason": "available_ram_below_frozen_admission_floor",
        "required_admission_ram_gib": 12.5, "failed_admission_blockers": ["admission_ram_below_12_5_gib"],
        "static_prepare_passed": True, "cpu_tests_passed": 121, "new_gpu_run_started": False,
        "evaluation_output_created": False, "evaluation_process_created": False,
        "execution_launch_created": False, "second_admission_attempted": False,
        "failed_plan_must_not_be_redispatched": True, "retry_authorized": False,
        "restore_pending": False, "restoration_complete": True, "restored_identity_matches": True,
        "port_8083_listener_owners": [60380], "health_fields_checked": 28, "all_28_fields_match": True,
        "unknown_python_processes": [], "guardian_status": "complete",
        "guardian_exact_identity_exited": True, "guardian_pid_current_occupant": None,
        "restoration_launch_attempt_count": 1, "accepted_policy_score_count": 7,
        "accepted_policy_generated_records": 3787, "excluded_ppo44_partial_records": 303,
        "deadline_epoch": DEADLINE, "no_other_service_operations": True, "no_frozen_source_changes": True,
        "source_count": 192, "manual_single_attempt": True, "natural_resource_improvement_claimed": False,
        "direct_user_instruction": "那就启动啊", "original_budget_unchanged": True},
        "Exact safely closed V8 first-admission failure required")
    evidence = closure.get("evidence")
    launch = Path(closure["launch_directory"])
    if (not launch.is_absolute() or not isinstance(evidence, dict) or set(evidence) != CLOSURE_EVIDENCE
            or Path(succession["prelaunch_closure"]["path"]).parent.resolve() != launch.resolve()):
        raise ValueError("All fourteen original closure evidence pins must remain in their allocation")
    for name, pin in evidence.items():
        _binding(pin)
        if Path(pin["path"]).resolve() != (launch / name).resolve():
            raise ValueError("Closure evidence path differs: " + name)
        verified_bytes(Path(pin["path"]), pin["sha256"])
    expected_pins = {"resource-profile.json": profile["authorized_pause"]["baseline_profile"],
        "restoration-complete.json": succession["restoration_report"],
        "restoration-manifest.json": succession["restoration_manifest"],
        "evidoc-pause-authorization.json": prior_pause["authorization"], "plan.json": closure.get("plan")}
    if (any(evidence[name] != pin for name, pin in expected_pins.items())
            or evidence["plan.json"]["sha256"] != V8_PLAN_SHA256
            or evidence["restoration-transaction.json"]["sha256"] != succession["restoration_report"]["sha256"]):
        raise ValueError("Closure must bind the frozen V8 plan/profile and identical restoration transaction")
    plan = _read(evidence["plan.json"])
    _expect(plan, {"kind": "alignlab-primary-evaluation-campaign-v2", "started_at_epoch": STARTED,
        "deadline_epoch": DEADLINE, "wall_seconds": WALL_SECONDS,
        "resource_profile": profile["authorized_pause"]["baseline_profile"]}, "V8 original plan and budget changed")
    output = Path(plan["output"])
    if (not output.is_absolute() or str(output) != closure["intended_output_root"]
            or output.exists() or any((launch / name).exists() for name in
                ("process.json", "execution-launch.json"))
            or len(plan.get("source_sha256", {})) != 192
            or set(plan.get("inherited_inference", {})) != INHERITED_SLOTS
            or plan.get("inherited_scores") != plan["inherited_inference"]):
        raise ValueError("Failed V8 allocation must remain unused with 192 sources and seven original slots")
    prepared = _read(evidence[PREPARE_EVIDENCE])
    _expect(prepared, {"kind": "alignlab-v8-static-prepare-accepted",
        "status": "static_validation_passed_waiting_for_resource_change", "tool_exit_code": 0,
        "tool_session_collected_and_closed": True, "ready_for_launch": False,
        "plan": evidence["plan.json"], "source_count": 192, "prior_source_count": 173, "new_source_count": 19,
        "current_source_sha256_all_match": True, "actual_full_static_validation_passed": True,
        "prior_sources_unchanged": True, "output_created": False, "model_loaded": False,
        "cuda_used": False, "service_pause_performed": False, "model_acceptance_repeated": False,
        "inherited_result_chains_verified": 7, "inherited_complete_capacity_models_verified": 9,
        "inherited_complete_capacity_audits_verified": 18, "inherited_results_accepted": 3787,
        "ppo44_partial_records_excluded": 303, "cpu_tests_passed": 121,
        "cpu_errors": 0, "cpu_failures": 0, "cpu_skips": 0, "deadline_epoch": DEADLINE},
        "V8 static preparation evidence must precede its failed first admission")
    if (set(prepared.get("inherited_inference_slots", [])) != INHERITED_SLOTS
            or set(prepared.get("inherited_score_slots", [])) != INHERITED_SLOTS
            or prepared.get("fresh_output") != plan["output"]
            or prepared.get("cpu_proof") != plan.get("cpu_check_report")
            or prepared.get("capacity_binding") != plan.get("inherited_capacity")
            or not isinstance(prepared.get("candidate"), dict)
            or prepared["candidate"].get("sha256") != evidence["plan.json"]["sha256"]
            or Path(prepared["candidate"].get("path", "")).resolve() != launch / "plan-candidate.json"):
        raise ValueError("V8 static proof must bind all seven slots, output and CPU proof")
    _binding(prepared["candidate"])
    verified_bytes(Path(prepared["candidate"]["path"]), prepared["candidate"]["sha256"])
    _binding(prepared["prepare_log"])
    verified_bytes(Path(prepared["prepare_log"]["path"]), prepared["prepare_log"]["sha256"])
    cpu = _read(prepared["cpu_proof"])
    _expect(cpu, {"kind": "alignlab-independent-evaluation-v8-cpu-proof",
        "status": "cpu_software_checks_passed", "tests_run": 121, "failures": 0, "errors": 0,
        "skipped": 0, "source_unchanged": True, "device": "cpu", "torch_imported": False,
        "cuda_initialized": False, "pretrained_model_weights_used": False, "training_started": False,
        "model_acceptance_repeated": False, "capacity_verified": False}, "Original V8 CPU proof differs")
    # The pinned CPU report audits 348 files; the plan freezes 192 required
    # files. Additional audited files are valid, but every required pin must
    # be present and identical. The entire report is still hash-verified above.
    audited, required = cpu.get("source_sha256"), plan["source_sha256"]
    if (not isinstance(audited, dict) or not isinstance(required, dict)
            or any(name not in audited or type(audited[name]) is not str
                   or audited[name] != sha256 for name, sha256 in required.items())):
        raise ValueError("Original V8 CPU proof must audit every required plan source with its exact SHA")
    admission, failed = _read(evidence[ADMISSION_EVIDENCE]), _read(evidence["launch-failed.json"])
    _expect(admission, {"kind": "alignlab-evaluation-v8-prestart-admission",
        "status": "waiting_resources_before_start", "plan": evidence["plan.json"],
        "presence": {"process": False, "output": False, "campaign": False,
                     "execution_launch": False, "plan_stop": False},
        "admission_ram_headroom_gib": 12.5, "runtime_ram_floor_gib": 8,
        "sample_conditions_passed": False, "first_admission": None, "sample_separation_seconds": None,
        "required_sample_separation_seconds": 300, "unknown_python": [], "locks": [],
        "blockers": ["admission_ram_below_12_5_gib"], "deadline_epoch": DEADLINE,
        "actual_capacity_started": False, "actual_inference_started": False, "model_or_python_started": False},
        "Only the original failed first RAM admission may supply this service succession")
    ram = _finite(admission.get("available_ram_gib"), "Failed admission RAM")
    if (not 0 <= ram < 12.5 or closure["failed_admission_available_ram_gib"] != ram
            or admission.get("hashes") != {"plan": evidence["plan.json"]["sha256"],
                "protocol": plan["protocol"]["sha256"], "profile": expected_pins["resource-profile.json"]["sha256"],
                "authorization": plan["authorization"]["sha256"]}):
        raise ValueError("Actual first RAM sample and exact frozen admission bindings required")
    _expect(failed, {"status": "prelaunch_admission_failed", "plan": evidence["plan.json"],
        "admission": evidence[ADMISSION_EVIDENCE], "available_ram_gib": ram, "required_ram_gib": 12.5,
        "blockers": admission["blockers"], "restore_required": True, "gpu_started": False,
        "retry_forbidden": True, "direct_user_attempt": evidence[DIRECT_ATTEMPT_EVIDENCE]},
        "Original failed admission and restore signal disagree")
    attempt_authority = _read(evidence[DIRECT_ATTEMPT_EVIDENCE])
    _expect(attempt_authority, {"kind": "alignlab-v8-direct-user-single-launch-attempt",
        "direct_user_instruction": "那就启动啊", "preceding_direct_user_instruction": "那继续跑啊",
        "natural_resource_improvement_claimed": False, "automatic_retry_authorized": False,
        "maximum_attempts": 1, "plan": evidence["plan.json"], "static_acceptance": evidence[PREPARE_EVIDENCE],
        "source_count_verified": 192, "helper_files_verified": 9,
        "limits_unchanged": {"gpu_free_mib": 3840, "sample_gap_max_seconds": 900,
            "sample_gap_min_seconds": 300, "admission_ram_gib": 12.5,
            "deadline_epoch": DEADLINE, "gpu_utilization_max": 20},
        "pause_scope": "Only exact EviDoc 8083 adapter PID60124 with frozen full identity",
        "failure_action": "Immediately signal frozen guardian to restore original configuration and verify all28 fields; close this allocation permanently",
        "other_services_or_apps_operations_authorized": False, "frozen_plan_source_or_proofs_modified": False,
        "gpu_execution_requires_two_passed_actual_admissions": True,
        "historical_unknown_pid58104_origin_resolved": False},
        "The V8 single human-directed attempt must not become a retry or resource-recovery claim")
    restored = receipt["processes"][0]
    _expect(restored, {"pid": 60380, "parent_pid": 61972,
        "created_at_utc": "2026-10-29T06:30:17.9429660Z"}, "Exact restored V8 successor required")
    if (_created_ticks(restored) != "639288522179429660"
            or any(_exact_identity(closure[name]) != _exact_identity(restored)
                   for name in ("restored_identity", "restored_live_identity"))):
        raise ValueError("Closure must retain the exact 60380 CIM identity")
    health = receipt["health"]
    if len(health) != 28 or closure.get("health") != health or manifest["service"]["json_equals"] != health:
        raise ValueError("All 28 restoration health fields must agree")
    preserved = baseline["background_processes"] + [p for s in baseline["services"]
        if s["name"] != "evidoc" for p in s["processes"]]
    signatures = {signature(p) for p in preserved}
    checks = closure.get("preserved_process_checks", [])
    if (len(checks) != 5 or {signature(c["expected"]) for c in checks} != signatures
            or any(c.get("matches") is not True or _exact_identity(c["live"]) != signature(c["expected"])
                   for c in checks)
            or len(closure.get("python_processes", [])) != 6
            or {_exact_identity(p) for p in closure["python_processes"]} != signatures | {signature(restored)}
            or len(admission.get("identities", [])) != 5
            or {signature(c["expected"]) for c in admission["identities"]} != signatures
            or any(c.get("matches") is not True for c in admission["identities"])):
        raise ValueError("All five peers and only the restored adapter must remain accounted for")
    guardian, attempt, paused = (_read(evidence[name]) for name in
        ("guardian/status.json", "guardian/restore-launch-attempt.json", "pause-complete.json"))
    common = {"manifest_sha256": succession["restoration_manifest"]["sha256"],
              "pause_authorization_sha256": prior_pause["authorization"]["sha256"]}
    _expect(guardian, {**common, "kind": "alignlab-evidoc-restore-guardian-v1", "status": "complete",
        "reason": "explicit_restore_requested", "deadline_epoch": DEADLINE, "supervisor": None,
        "plan": None, "restoration_evidence": succession["restoration_report"]["path"]},
        "Prior guardian must have completed before any supervisor binding")
    _expect(attempt, {**common, "max_restore_launches": 1}, "Exactly one original restore launch required")
    _expect(paused, {**common, "kind": "alignlab-evidoc-pause-complete-v1", "status": "paused",
        "port_8083_absent": True, "restore_pending": True, "restoration_complete": False,
        "deadline_epoch": DEADLINE}, "Original pause transaction differs")
    if (signature(paused["target"]) != signature(manifest["target"])
            or {signature(p) for p in paused.get("preserved_processes", [])} != signatures
            or guardian["guardian"]["pid"] != restored["parent_pid"]
            or _exact_identity(guardian["guardian"]) != _exact_identity(attempt["guardian"])
            or _exact_identity(guardian["guardian"]) != _exact_identity(paused["guardian"])
            or any(attempt.get(k) != manifest.get("restore", {}).get(k)
                   for k in ("executable", "arguments", "working_directory"))):
        raise ValueError("Original guardian, pause and single unchanged-config restore launch differ")
    return closure


def load_profile(binding, *, now_fn=None):
    """Verify the complete prior pause/restoration chain without changing any live process."""
    from .evaluation_resources_v8 import load_profile as load_prior_profile
    profile = validate_profile(_read(binding))
    pause = profile["authorized_pause"]
    if (pause["baseline_profile"]["sha256"] != V8_BASELINE_SHA256
            or any(profile["service_identity_succession"][key]["sha256"] != digest
                   for key, digest in SUCCESSION_SHA256.items())):
        raise ValueError("Only the exact closed V8 allocation may supply V9 service succession")
    baseline = load_prior_profile(pause["baseline_profile"], now_fn=now_fn)
    succession = profile["service_identity_succession"]
    receipt = _read(succession["restoration_report"])
    manifest = _read(succession["restoration_manifest"])
    prior_pause = baseline["authorized_pause"]
    prior_authority = _read(prior_pause["authorization"])
    old_service = next(service for service in baseline["services"] if service["name"] == "evidoc")
    current_service = next(service for service in profile["services"] if service["name"] == "evidoc")
    preserved = baseline["background_processes"] + [
        process for service in baseline["services"] if service["name"] != "evidoc"
        for process in service["processes"]]
    def same_identities(actual, expected):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and {signature(p) for p in actual} == {signature(p) for p in expected})
    if (receipt.get("kind") != "alignlab-evidoc-restoration-complete-v1"
            or receipt.get("status") != "restored_and_verified"
            or receipt.get("restoration_complete") is not True
            or receipt.get("restore_pending") is not False
            or receipt.get("all_28_fields_match") is not True
            or type(receipt.get("deadline_epoch")) is not int or receipt["deadline_epoch"] != DEADLINE
            or receipt.get("manifest_sha256") != succession["restoration_manifest"]["sha256"]
            or receipt.get("pause_authorization_sha256") != prior_pause["authorization"]["sha256"]
            or not same_identities(receipt.get("preserved_processes"), preserved)):
        raise ValueError("A completed exact earlier restoration and all five preserved identities are required")
    if (manifest.get("kind") != "alignlab-evidoc-restoration-manifest-v2"
            or type(manifest.get("deadline_epoch")) is not int or manifest["deadline_epoch"] != DEADLINE
            or manifest.get("baseline_profile") != prior_pause["baseline_profile"]
            or manifest.get("original_restoration") != baseline["restoration_report"]
            or manifest.get("service_identity_succession") != baseline["service_identity_succession"]
            or manifest.get("inherited_authorization") != prior_authority["inherited_authorization"]
            or signature(manifest["target"]) != signature(old_service["processes"][0])
            or not same_identities(manifest.get("preserved_processes"), preserved)
            or manifest.get("preserved_services") != [s for s in baseline["services"] if s["name"] != "evidoc"]
            or manifest.get("service", {}).get("name") != "evidoc"
            or manifest["service"].get("port") != 8083
            or manifest["service"].get("url") != old_service["url"]):
        raise ValueError("Restoration manifest must bind the original exact adapter and unchanged peers")
    stable = lambda health: {key: value for key, value in health.items() if key != "busy"}
    if (stable(receipt.get("health", {})) != old_service["json_equals"]
            or receipt.get("health", {}).get("busy") is not False
            or stable(manifest["service"].get("json_equals", {})) != old_service["json_equals"]):
        raise ValueError("All stable restored health fields must remain pinned")
    restored = receipt.get("processes")
    if not same_identities(restored, current_service["processes"]):
        raise ValueError("Current adapter must be the exact identity from the completed restoration receipt")
    for process in restored:
        ticks = process.get("created_utc_ticks")
        if (type(ticks) not in (int, str) or str(ticks) != _created_ticks(process)):
            raise ValueError("Completed restoration must retain exact integer/string UTC creation ticks")
    old_identity, new_identity = signature(old_service["processes"][0]), signature(restored[0])
    if (new_identity[0] == old_identity[0] or new_identity[2] <= old_identity[2]
            or new_identity[3] != old_identity[3]):
        raise ValueError("Successor must be a later restored process with the unchanged full command")
    # Restore the original identity solely for equality comparison: no other
    # limit, health, service, background identity or early receipt may change.
    import copy
    expected = _base(baseline)
    current = copy.deepcopy(_base(profile))
    next(s for s in current["services"] if s["name"] == "evidoc")["processes"] = old_service["processes"]
    if current != expected:
        raise ValueError("All baseline limits, services, health and background identities must remain unchanged")
    restoration = profile["restoration_report"]
    authority = _read(pause["authorization"])
    required = {"kind": PAUSE_KIND, "status": "authorized", "authorized": True,
                "user_authorization_quote": PAUSE_QUOTE, "service": "evidoc", "port": 8083,
                "baseline_profile": pause["baseline_profile"], "restoration_report": restoration,
                "deadline_epoch": DEADLINE, "restore_required": True,
                "inherited_authorization": prior_pause["authorization"],
                "service_identity_succession": succession}
    if not isinstance(authority, dict) or any(
            type(authority.get(key)) is not type(value) or authority.get(key) != value
            for key, value in required.items()):
        raise ValueError("Explicit user authorization and exact earlier EviDoc baseline binding required")
    evaluation_authority = _read(authority.get("evaluation_authorization"))
    prior_authority = _read(prior_pause["authorization"])
    if authority["evaluation_authorization"]["sha256"] != prior_authority["evaluation_authorization"]["sha256"]:
        raise ValueError("The original evaluation authorization bytes cannot change across restoration")
    budget = {"kind": "alignlab-independent-quality-evaluation-authorization-v2",
              "started_at_epoch": STARTED, "deadline_epoch": DEADLINE, "wall_seconds": WALL_SECONDS,
              "automatic_extension": False, "reuses_old_budget": False}
    if not isinstance(evaluation_authority, dict) or any(
            type(evaluation_authority.get(key)) is not type(value) or evaluation_authority.get(key) != value
            for key, value in budget.items()):
        raise ValueError("Original independent evaluation authorization and absolute budget must be unchanged")
    now = _finite((now_fn or time.time)(), "Authorization observation time")
    if not STARTED <= now < DEADLINE:
        raise ValueError("EviDoc pause authorization is outside the original evaluation window")
    _verify_prelaunch_closure(profile, baseline, receipt, manifest)
    return profile


def validate_authorized_profile_relation(prior_binding, new_binding):
    profile = load_profile(new_binding)
    if profile["authorized_pause"]["baseline_profile"] != prior_binding:
        raise ValueError("Authorized pause must bind the prior plan's exact resource profile")
    return profile


def _port_8083_listeners():
    """All TCP listeners, including non-Python replacements; read-only hidden PowerShell."""
    if os.name != "nt":
        raise RuntimeError("Live paused-service port verification requires Windows")
    command = (
        "$ErrorActionPreference='Stop'; @(Get-NetTCPConnection -State Listen | "
        "Where-Object { $_.LocalPort -eq 8083 } | ForEach-Object { "
        "[pscustomobject]@{ local_address=$_.LocalAddress; local_port=[int]$_.LocalPort; "
        "pid=[int]$_.OwningProcess } }) | ConvertTo-Json -Depth 4 -Compress"
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], check=True,
                            capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    value = parse(result.stdout.encode("utf-8")) if result.stdout.strip() else []
    return [value] if isinstance(value, dict) else value


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
                      process_reader=None, health_reader=None, allocator_receipt=None,
                      port_reader=None, now_fn=None):
    """Observe actual resources; injected readers exist only for CPU unit tests."""
    validate_profile(profile)
    from rlhf_lab.runtime import available_ram_gb, gpu_inventory
    from .probe_dispatch import _System
    inventory = (inventory_reader or gpu_inventory)()
    ram = (ram_reader or available_ram_gb)()
    live = (process_reader or _System().processes)()
    read_health = health_reader or _health
    blockers, health, diagnostics = [], {}, []
    now = _finite((now_fn or time.time)(), "Resource observation time")
    def record_identity_failure(reason, process, *, expected_process=None, error=None):
        detail = {"checked_at_epoch": now, "reason": reason,
                  "observed": _identity_evidence(process)}
        if expected_process is not None:
            detail["expected"] = _identity_evidence(expected_process)
        if error is not None:
            detail["error"] = f"{type(error).__name__}: {error}"
        diagnostics.append(detail)
        return canonical(detail).decode("utf-8")
    if not STARTED <= now < profile["authorized_pause"]["deadline_epoch"]:
        blockers.append("EviDoc pause authorization expired or not yet applicable")
    paused = next(service for service in profile["services"] if service["name"] == "evidoc")
    paused_pid = paused["processes"][0]["pid"]
    listeners = None
    try:
        listeners = (port_reader or _port_8083_listeners)()
        if not isinstance(listeners, list):
            raise ValueError("Explicit TCP listener list required")
        if listeners:
            blockers.append(f"Paused EviDoc port 8083 has a listener: {listeners}")
    except Exception as error:
        blockers.append(f"Paused EviDoc port absence unobservable: {type(error).__name__}: {error}")
    expected = {}
    for process in profile["background_processes"]:
        expected[process["pid"]] = process
    for service in active_services(profile):
        for process in service["processes"]:
            expected[process["pid"]] = process
    for process in allowed_processes:
        signature(process)
        if process["pid"] == paused_pid:
            raise ValueError("Paused EviDoc PID cannot be granted evaluator ownership")
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
            if pid == paused_pid:
                detail = record_identity_failure("paused_service_present", process,
                    expected_process=paused["processes"][0])
                blockers.append(f"Authorized-paused EviDoc process must be absent: {pid}; identity={detail}")
            if pid in expected:
                if actual != signature(expected[pid]):
                    detail = record_identity_failure("frozen_identity_changed", process,
                        expected_process=expected[pid])
                    blockers.append(f"Frozen process identity changed: {pid}; identity={detail}")
            else:
                # The live reader enumerates only Python processes. A familiar
                # command or parent cannot grant a new PID a frozen identity.
                detail = record_identity_failure("unknown_python", process)
                blockers.append(f"Unknown Python process outside exact frozen identities: {pid}; identity={detail}")
        except (ValueError, KeyError, TypeError) as error:
            detail = record_identity_failure("uninspectable_python", process, error=error)
            blockers.append(f"Uninspectable Python process: {error}; identity={detail}")
    for pid in expected:
        if pid not in seen:
            blockers.append(f"Frozen service/evaluator process disappeared: {pid}")
    health["evidoc"] = {"url": paused["url"], "mode": "authorized_temporarily_paused",
                        "process_absent": paused_pid not in seen,
                        "port_8083_absent": listeners == [],
                        "authorization": profile["authorized_pause"]["authorization"]}
    for service in active_services(profile):
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
    return {"kind": "alignlab-evaluation-resource-observation-v9", "checked_at_epoch": now,
            **inventory, "available_ram_gb": ram, "processes": live, "service_health": health,
            "process_identity_diagnostics": diagnostics,
            "paused_service_listeners": listeners,
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
            observation = {"checked_at_epoch": self.system.now(), "gpus": result["gpus"],
                "available_ram_gb": result["available_ram_gb"], "blockers": result["blockers"],
                "service_health_sha256": digest(result["service_health"]),
                "process_identity_diagnostics": result.get("process_identity_diagnostics", [])}
            self.observations.append(observation)
            self.last = self.system.now()
            # Persist the exact offending snapshot before raising. An unknown
            # process cannot gain a grace period or reach another CUDA operation.
            if result["blockers"]:
                raise RuntimeError("Service-preserving evaluation guard: " + "; ".join(result["blockers"]))
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
            observation["memory"] = memory
            observation["checked_at_epoch"] = self.system.now()
            self.last = self.system.now()
            if self.system.now() >= self.deadline:
                raise TimeoutError("Independent deadline expired during resource observation")
        return False
