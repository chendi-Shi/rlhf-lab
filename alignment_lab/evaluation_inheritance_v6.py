"""Read-only, explicit inheritance of the accepted V5 base inference only.

Saved text is used solely for the same structural replay as V5's supervisor.
It is never an independent token-decoder receipt or an official quality score.
No model/tensor imports, launches, output copying, or process termination occur.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from .evaluation_protocol_v2 import parse, verified_bytes, _relative, _finite
from .evaluation_resources_v5 import signature
from . import evaluation_execution_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
BINDING_FIELDS = frozenset({"prior_plan", "campaign", "closure", "request", "manifest"})
V5_SOURCE_COUNT = 129
V6_EXECUTION = "alignment_lab/evaluation_execution_v6.py"
POWERSHELL7 = ("C:/Users/Intern1/.cache/codex-runtimes/codex-primary-runtime/"
               "dependencies/native/powershell/pwsh.exe")
COUNTS = {"records": 541, "prompts": 541, "instructions": 834}


def _read(binding):
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute()
            or any(p.is_symlink() for p in (Path(binding["path"]), *Path(binding["path"]).parents))):
        raise ValueError("Exact absolute path/SHA256 binding without symlinks required")
    return parse(verified_bytes(Path(binding["path"]), binding["sha256"]))


def _v5_sources(sources):
    from .evaluation_analysis_v5 import REQUIRED_SOURCES
    from .evaluation_acceptance_v2 import SOURCES
    from .evaluation_supervisor_v5 import NEW_SOURCES
    required = REQUIRED_SOURCES | set(SOURCES) | NEW_SOURCES | {"scripts/verify_alignlab_cpu.py"}
    if (not isinstance(sources, dict) or len(sources) != V5_SOURCE_COUNT
            or not required <= sources.keys() or V6_EXECUTION in sources):
        raise ValueError("Complete original 129-file V5 source inventory required")
    for name, sha in sources.items():
        verified_bytes(_relative(ROOT, name), sha)


def verify_versioned_evaluation_result(output, *, expected_manifest_sha256,
                                      expected_request=None, decode_tokens=None, codec=None):
    """Route only pinned V6 inference or the original complete V5 base artifact.

    This validates inference, not the separate inheritance authorization. The
    supervisor/analysis must also call verify_inherited_inference for V5 reuse.
    """
    manifest = _read({"path": str(Path(output).resolve() / "manifest.json"),
                      "sha256": expected_manifest_sha256})
    request = manifest.get("request", {})
    if expected_request is not None and request != expected_request:
        raise ValueError("Version routing request differs from external request")
    sources = request.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("Pinned versioned execution inventory required")
    if V6_EXECUTION in sources:
        verified_bytes(_relative(ROOT, V6_EXECUTION), sources[V6_EXECUTION])
        from .evaluation_execution_v6 import verify_evaluation_result
    else:
        if request.get("task") != "ifeval" or request.get("model_id") != "base":
            raise ValueError("Only original V5 ifeval/base inference may be inherited")
        _v5_sources(sources)
        verify_evaluation_result = v5.verify_evaluation_result
    return verify_evaluation_result(output, expected_manifest_sha256=expected_manifest_sha256,
        expected_request=expected_request, decode_tokens=decode_tokens, codec=codec)


def _identity(value):
    result = signature(value)
    whole, nanos = result[2]
    epoch = datetime(1, 1, 1, tzinfo=timezone.utc)
    delta = whole - epoch
    ticks = (delta.days * 86400 + delta.seconds) * 10_000_000 + nanos // 100
    if nanos % 100 or ("created_utc_ticks" in value and value["created_utc_ticks"] != ticks):
        raise ValueError("Process creation ticks differ from exact UTC creation time")
    return result


class _LiveSystem:
    def processes(self):
        if os.name != "nt":
            raise RuntimeError("Windows exact read-only process inspection required")
        command = (
            "@(Get-CimInstance Win32_Process | ForEach-Object { [pscustomobject]@{ "
            "pid=$_.ProcessId; parent_pid=$_.ParentProcessId; "
            "created_at_utc=$_.CreationDate.ToUniversalTime().ToString('o'); "
            "created_utc_ticks=$_.CreationDate.ToUniversalTime().Ticks; "
            "command_line=$_.CommandLine } }) | ConvertTo-Json -Depth 4 -Compress"
        )
        result = subprocess.run([POWERSHELL7, "-NoProfile", "-Command", command],
            check=True, capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW)
        values = parse(result.stdout.encode("utf-8"))
        return [values] if isinstance(values, dict) else values or []


def _exited(expected, system):
    live = system.processes()
    if not isinstance(live, list):
        raise ValueError("Explicit live process list required")
    seen = set()
    for prior in expected:
        sig = _identity(prior)
        if sig in seen:
            continue
        seen.add(sig)
        for current in live:
            if current.get("pid") != prior["pid"]:
                continue
            # Different creation time means a different process, even if the
            # operating system has reused this PID. Never operate on that PID.
            # Windows may hide the command of an unrelated reused PID. Its
            # different creation time is sufficient to distinguish that reuse.
            now = _identity({**current, "command_line": current.get("command_line") or "<unavailable>"})
            if now[2] == sig[2]:
                raise ValueError("An exact old process is still alive or its identity drifted")
    return len(seen)


def _closed(binding, prior, campaign, closure):
    flags = {"kind": "alignlab-quality-v5-failure-closure-v1",
        "status": "failed_closed_services_restored", "campaign_status": "failed",
        "failure_phase": "ifeval-base-score", "failure_exit_code": 1,
        "formal_generations_completed": 541, "formal_generations_expected": 4328,
        "base_inference_accepted": True, "base_inference_exit_code": 0,
        "official_scoring_complete": False, "primary_evaluation_complete": False,
        "research_complete": False, "capacity_verified": True, "capacity_models_completed": 9,
        "all_evaluation_processes_exited": True, "guardian_status": "complete",
        "awake_status": "released", "restoration_complete": True, "restore_pending": False,
        "health_field_count": 28, "all_28_fields_match": True,
        "preserved_services_verified": True, "relaunch_permitted": False,
        "recovery_requires_fresh_launch_and_output": True}
    if (any(type(closure.get(k)) is not type(v) or closure.get(k) != v for k, v in flags.items())
            or closure.get("plan") != binding["prior_plan"]
            or closure.get("original_deadline_epoch") != prior["deadline_epoch"]
            or closure.get("unknown_python") != []):
        raise ValueError("Exact closed V5 score failure and restored services required")
    restoration = _read(closure["restoration"])
    profile = _read(prior["resource_profile"])
    expected_health = next(s["json_equals"] for s in profile["services"] if s["name"] == "evidoc")
    expected_health = {**expected_health, "busy": False}
    health_checks = closure.get("health_checks")
    if (restoration.get("kind") != "alignlab-evidoc-restoration-complete-v1"
            or restoration.get("restoration_complete") is not True
            or restoration.get("restore_pending") is not False
            or restoration.get("all_28_fields_match") is not True
            or restoration.get("deadline_epoch") != prior["deadline_epoch"]
            or restoration.get("pause_authorization_sha256") != profile["authorized_pause"]["authorization"]["sha256"]
            or restoration.get("health") != expected_health or closure.get("health") != expected_health
            or len(expected_health) != 28 or not isinstance(health_checks, list)
            or len(health_checks) != 28 or {v.get("name") for v in health_checks} != set(expected_health)
            or any(v.get("matches") is not True for v in health_checks)):
        raise ValueError("Pinned restoration must prove all 28 original health fields")
    restored = restoration.get("processes", [])
    if (len(restored) != 1 or len(closure.get("restored_processes", [])) != 1
            or _identity(restored[0]) != _identity(closure["restored_processes"][0])):
        raise ValueError("Exact restored process receipt required")
    owners = closure.get("restored_listener_owners")
    owners = [owners] if type(owners) is int else owners
    if owners != [restored[0]["pid"]]:
        raise ValueError("Restored 8083 listener must belong to the restored process")
    preserved = profile["background_processes"] + [p for s in profile["services"]
        if s["name"] != "evidoc" for p in s["processes"]]
    for records in (closure.get("preserved_processes"), restoration.get("preserved_processes")):
        if (not isinstance(records, list) or len(records) != 5
                or {_identity(p) for p in records} != {_identity(p) for p in preserved}):
            raise ValueError("All five original preserved process identities are required")
    observed = _read(closure["latest_monitor"])
    if observed.get("plan") != binding["prior_plan"]:
        raise ValueError("Helper identity monitor belongs to another plan")
    guardian = observed["guardian"]["guardian"]
    awake = observed["awake"]
    helper = {"pid": awake["helper_pid"], "parent_pid": awake["helper_parent_pid"],
        "created_at_utc": awake["helper_created_utc"], "command_line": awake["helper_command"]}
    known = list(campaign["supervisor_processes"])
    for child in campaign["children"]:
        if child.get("active_processes") != [] or not child.get("known_processes"):
            raise ValueError("Every old child must have observed identities and no active processes")
        known.extend(child["known_processes"])
    signatures = {_identity(p) for p in known}
    if len(signatures) != 8 or len({p[0] for p in signatures}) != 8:
        raise ValueError("Exact eight-process V5 execution history required")
    for field, identities in (("evaluation_processes", known), ("helper_processes", [guardian, helper])):
        records = closure.get(field)
        if (not isinstance(records, list) or len(records) != len({_identity(p) for p in identities})
                or {p.get("pid") for p in records} != {p["pid"] for p in identities}
                or any(p.get("present") is not False for p in records)):
            raise ValueError("Closure must explicitly record every old process as exited")
    return known + [guardian, helper]


def verify_inherited_inference(binding, plan, *, system=None):
    """Validate immutable V5 base 541/834 evidence for a fresh V6 campaign.

    The five pins are required in the new frozen plan. All original source
    hashes remain mandatory; no inherited result is copied or rewritten.
    ``system.processes()`` is the only injected seam, for CPU lifecycle tests.
    """
    if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS:
        raise ValueError("Exactly five explicit inheritance evidence bindings required")
    documents = {key: _read(value) for key, value in binding.items()}
    prior, campaign, closure, request, manifest = (
        documents[k] for k in ("prior_plan", "campaign", "closure", "request", "manifest"))
    if not isinstance(plan, dict):
        raise ValueError("Frozen destination plan required")
    for key in ("authorization", "protocol", "roster", "decoder", "started_at_epoch",
                "deadline_epoch", "wall_seconds", "model_python_sha256", "scoring_python_sha256"):
        if key not in prior or plan.get(key) != prior[key]:
            raise ValueError("Inherited scientific identity, decoder or original budget changed: " + key)
    start, deadline = (_finite(prior[k], k) for k in ("started_at_epoch", "deadline_epoch"))
    if deadline - start != prior["wall_seconds"] or start <= 0:
        raise ValueError("Original absolute budget is inconsistent")
    authority = _read(prior["authorization"])
    if (authority.get("kind") != "alignlab-independent-quality-evaluation-authorization-v2"
            or any(authority.get(k) != prior[k] for k in ("started_at_epoch", "deadline_epoch", "wall_seconds"))
            or authority.get("automatic_extension") is not False
            or authority.get("reuses_old_budget") is not False or authority.get("no_new_training") is not True):
        raise ValueError("Original independent authorization required")
    for name in ("protocol", "roster"):
        _read(prior[name])
    decoder = prior["decoder"]
    for name, path in (("python", decoder["python"]), ("script", decoder["script"]),
                       ("tokenizer", decoder["tokenizer_path"])):
        verified_bytes(Path(path), decoder[name + "_sha256"])
    _v5_sources(prior["source_sha256"])
    sources = plan.get("source_sha256")
    if (not isinstance(sources, dict) or V6_EXECUTION not in sources
            or any(sources.get(k) != v for k, v in prior["source_sha256"].items())):
        raise ValueError("Fresh V6 sources must include every unchanged original V5 source pin")
    for name, sha in sources.items():
        verified_bytes(_relative(ROOT, name), sha)
    old_output, new_output = Path(prior["output"]), Path(plan["output"])
    if (not old_output.is_absolute() or not new_output.is_absolute()
            or old_output.resolve().parent != (ROOT / "runs").resolve()
            or new_output.resolve().parent != (ROOT / "runs").resolve()
            or old_output.resolve() == new_output.resolve()
            or Path(binding["campaign"]["path"]).resolve() != old_output.resolve() / "campaign.json"
            or Path(binding["request"]["path"]).resolve() != old_output.resolve() / "ifeval-base.request.json"
            or Path(binding["manifest"]["path"]).resolve() != old_output.resolve() / "ifeval-base/manifest.json"):
        raise ValueError("Inheritance must reference the original base output and a distinct new campaign")
    if (campaign.get("kind") != "alignlab-primary-evaluation-campaign-v2"
            or campaign.get("status") != "failed" or campaign.get("capacity_verified") is not True
            or {"path": campaign.get("plan_path"), "sha256": campaign.get("plan_sha256")} != binding["prior_plan"]
            or campaign.get("started_at_epoch") != start or campaign.get("deadline_epoch") != deadline
            or campaign.get("primary_evaluation_complete") is not False or campaign.get("scores") != {}):
        raise ValueError("Original campaign must fail only after accepted base inference")
    children = campaign.get("children", [])
    if (len(children) != 3 or [(c.get("name"), c.get("phase")) for c in children] !=
            [("capacity", "capacity"), ("ifeval-base", "inference"), ("ifeval-base-score", "score")]):
        raise ValueError("Only the base-only V5 campaign followed by score failure may be inherited")
    capacity, accepted, failed = children
    if (any(c.get("status") != "accepted" or type(c.get("exit_code")) is not int
            or c["exit_code"] != 0 for c in (capacity, accepted))
            or type(failed.get("exit_code")) is not int or failed["exit_code"] != 1
            or failed.get("status") == "accepted" or failed.get("result") is not None
            or accepted.get("request") != request or accepted.get("result") != binding["manifest"]
            or {"path": accepted.get("request_path"), "sha256": accepted.get("request_sha256")} != binding["request"]
            or accepted.get("output") != request.get("output")):
        raise ValueError("Only accepted exit-zero original request and manifest can be inherited")
    if (request.get("task") != "ifeval" or request.get("model_id") != "base"
            or Path(request["output"]).resolve() != old_output.resolve() / "ifeval-base"
            or request.get("deadline_epoch") != deadline
            or request.get("source_sha256") != prior["source_sha256"]
            or request.get("resource_profile") != prior["resource_profile"]
            or request.get("capacity_manifest") != capacity.get("result")
            or {"path": request.get("protocol_path"), "sha256": request.get("protocol_sha256")} != prior["protocol"]
            or {"path": request.get("roster_path"), "sha256": request.get("roster_sha256")} != prior["roster"]
            or manifest.get("request") != request or manifest.get("counts") != COUNTS):
        raise ValueError("Mixed request/model/protocol/roster/source/capacity is forbidden")
    score_request = _read({"path": failed.get("request_path"), "sha256": failed.get("request_sha256")})
    expected_score = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-request-v1",
        "inference_manifest": binding["manifest"]["path"],
        "inference_manifest_sha256": binding["manifest"]["sha256"],
        "source_sha256": prior["source_sha256"], "decoder": prior["decoder"], "deadline_epoch": deadline}
    if score_request != expected_score or failed.get("request") != score_request:
        raise ValueError("The failed scoring request must reference this exact accepted base result")
    _read(capacity["result"])
    expected_processes = _closed(binding, prior, campaign, closure)
    accepted_identities = {_identity(p) for p in accepted["known_processes"]}
    manifest_identities = {_identity(p) for p in manifest.get("processes", [])}
    if (not accepted_identities <= manifest_identities
            or not manifest_identities <= {_identity(p) for p in expected_processes}
            or manifest.get("pid") not in {p["pid"] for p in accepted["known_processes"]}):
        raise ValueError("Inference manifest must belong to its observed accepted child")
    system = system or _LiveSystem()
    exited_count = _exited(expected_processes, system)
    raw = verified_bytes(Path(request["output"]) / "outputs.jsonl", manifest["files"]["outputs.jsonl"]["sha256"])
    mapping = {}
    for row in (parse(line) for line in raw.split(b"\n") if line.strip()):
        ids = row["generated_token_ids"]
        key = tuple(ids[:-1] if ids and ids[-1] == 151645 else ids)
        if key in mapping and mapping[key] != row["response"]:
            raise ValueError("Stored identical actions have inconsistent text")
        mapping[key] = row["response"]
    result = v5.verify_evaluation_result(request["output"],
        expected_manifest_sha256=binding["manifest"]["sha256"], expected_request=request,
        decode_tokens=mapping.__getitem__)
    if result["counts"] != COUNTS:
        raise ValueError("Complete original 541 prompts and 834 instructions required")
    # Close both byte and process races after the complete saved-output audit.
    for key, pin in binding.items():
        if _read(pin) != documents[key]:
            raise ValueError("Inheritance evidence changed during verification")
    for name, sha in sources.items():
        verified_bytes(_relative(ROOT, name), sha)
    for name in ("authorization", "protocol", "roster"):
        _read(prior[name])
    _exited(expected_processes, system)
    return {"slot": "ifeval/base", "request": copy.deepcopy(request),
        "manifest_binding": copy.deepcopy(binding["manifest"]),
        "accepted_record": copy.deepcopy(accepted),
        "source_campaign_binding": copy.deepcopy(binding["campaign"]),
        "verification": {"kind": "alignlab-explicit-base-inference-inheritance-v6",
            "counts": dict(COUNTS), "prior_sources_verified": V5_SOURCE_COUNT,
            "exact_old_processes_exited": exited_count, "frozen_output_reverified": True,
            "original_budget_unchanged": True, "restoration_verified": True,
            "independent_decoding_performed": False, "official_scoring_complete": False,
            "requires_independent_decoder_at_scoring": True, "outputs_copied_or_rewritten": False}}
