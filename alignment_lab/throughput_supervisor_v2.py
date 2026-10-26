"""V2 dispatch for the strict metadata-container compatibility fix only.

Original R3 input validators, strict process guard, DPO/PPO engines and full
artifact acceptance remain binding. The PPO42 bridge is explicitly audited.
All V1 sources remain pinned; later fresh stages retain their V1 implementation.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from . import alignment_supervisor_r3 as legacy
from . import alignment_process_guard_r3 as process_guard
from . import throughput_stage_v2 as resume
from .probe_dispatch import _System, _stamp, _parse
from .capacity_supervisor import sha, _pin
from .model_snapshot import verify_snapshot, REPO_ID, REVISION

ROOT = legacy.ROOT
KIND = "alignlab-throughput-campaign-v2"
STAGE_KIND = legacy.STAGE_KIND
SEQUENCE = legacy.SEQUENCE
JsonWriter = legacy.JsonWriter
_command = legacy._command
_same_identity = legacy._same_identity
_waitable_peer_decision = legacy._waitable_peer_decision
_peer_decision = legacy._peer_decision
_parents_gone = legacy._parents_gone
_cache = legacy._cache
workload = legacy.workload
NEW_SOURCES = {"alignment_lab/throughput_supervisor_v1.py",
    "scripts/run_alignlab_throughput_campaign_v1.py", "tests_throughput_v1/test_supervisor_v1.py",
    "scripts/run_alignlab_throughput_cpu_checks_v1.py", "alignment_lab/throughput_validation_v1.py",
    "scripts/run_alignlab_throughput_validation_v1.py", "tests_throughput_v1/test_validation_v1.py",
    "alignment_lab/throughput_fresh_v1.py", "scripts/run_alignlab_throughput_fresh_v1.py",
    "tests_throughput_v1/test_fresh_v1.py", "tests_throughput_v1/test_sampling_v1.py",
    "tests_throughput_v1/test_telemetry_v1.py", "tests_throughput_v1/__init__.py"}
NEW_SOURCES |= {"alignment_lab/throughput_supervisor_v2.py",
    "scripts/run_alignlab_throughput_campaign_v2.py", "tests_throughput_v2/test_supervisor_v2.py",
    "alignment_lab/throughput_stage_v2.py", "scripts/run_alignlab_throughput_stage_v2.py",
    "tests_throughput_v2/test_stage_v2.py", "tests_throughput_v2/__init__.py",
    "scripts/run_alignlab_throughput_cpu_checks_v2.py",
    "alignment_lab/throughput_stage_v1.py", "scripts/run_alignlab_throughput_stage_v1.py",
    "tests_throughput_v1/test_stage_v1.py", "alignment_lab/throughput_telemetry_v1.py",
    "alignment_lab/throughput_sampling_v1.py"}


def _pin_record(record):
    path = resume.pin(record).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Campaign evidence must be project-local")
    return path


def _cpu_proof(plan):
    report = _parse(_pin_record(plan["cpu_report"]))
    required_modules = {name[:-3].replace("/", ".") for name in plan["source_sha256"]
        if name.startswith(("tests_throughput_v1/test_", "tests_throughput_v2/test_")) and name.endswith(".py")}
    if (report.get("kind") != "alignlab-throughput-cpu-validation-v1"
            or report.get("source_before") != plan["source_sha256"]
            or report.get("source_after") != plan["source_sha256"]
            or report.get("cuda_initialized") is not False
            or report.get("device") != "cpu" or report.get("tests_run", 0) < 1
            or any(type(report.get(k)) is not int or report[k] != 0 for k in ("errors", "failures", "skipped"))
            or report.get("status") != "passed" or not required_modules
            or not required_modules <= set(report.get("modules", []))):
        raise ValueError("Exact source-bound CPU proof with no failures or CUDA required")


def _resume_request(plan, plan_path, plan_sha, own, waited):
    return {"schema_version": 1, "kind": resume.KIND, "stage": "ppo",
        "output": str(Path(plan["output"]) / "ppo42"),
        "original_started_at_epoch": resume.STARTED, "original_deadline_epoch": resume.DEADLINE,
        "original_request": plan["original_request"], "history": plan["history"],
        "source_sha256": plan["source_sha256"], "authorization": plan["authorization"],
        "validation_report": plan["validation_report"], "accepted_dpo42": plan["accepted_dpo42"],
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "supervisor_processes": list(own.values()), "selected_backend": plan["selected_backend"],
        "process_wait_spent_seconds": waited}


def _authorized_failed_prefix(plan, campaign):
    """The concrete 609/630 incident is an explicit exception, never a retry policy."""
    pin = plan["history"].get("rollback_authorization")
    if pin is None:
        return False
    authorization = _parse(_pin_record(pin))
    incident = _parse(_pin_record(authorization["incident"]))
    if (authorization.get("kind") != "alignlab-explicit-rollback-replay-authorization-v1"
            or authorization.get("source") != "direct_user_reply_in_this_chat"
            or authorization.get("user_selected_answer") != "允许从609恢复并重算21步，验证后继续（推荐）"
            or authorization.get("original_plan") != plan["original_plan"]
            or authorization.get("throughput_authorization") != plan["authorization"]
            or authorization.get("original_started_at_epoch") != resume.STARTED
            or authorization.get("original_deadline_epoch") != resume.DEADLINE
            or authorization.get("restore_optimizer_updates") != 609
            or authorization.get("prior_observed_optimizer_updates") != 630
            or authorization.get("replay_updates") != 21
            or incident.get("pins", {}).get("campaign.json") != plan["prior_campaign"]
            or incident.get("pins", {}).get("ppo42/progress.json") != plan["history"]["prior_progress"]
            or incident.get("observed_optimizer_updates") != 630
            or incident.get("last_saved_optimizer_updates") != 609
            or incident.get("checkpoint", {}).get("manifest_sha256") != plan["history"]["checkpoint"]["manifest_sha256"]
            or campaign.get("status") != "failed"
            or campaign.get("error") != "RuntimeError: RuntimeError: Known competing training or process identity changed"):
        raise ValueError("Failure continuation requires the exact incident and direct 21-step replay authorization")
    return True


def validate_plan(path, expected_sha):
    path = Path(path).resolve()
    _pin(path, expected_sha)
    plan = _parse(path)
    keys = {"schema_version", "kind", "output", "original_plan", "prior_campaign",
        "original_started_at_epoch", "original_deadline_epoch", "original_request", "history",
        "source_sha256", "authorization", "validation_report", "cpu_report", "accepted_dpo42",
        "selected_backend", "process_wait_spent_seconds", "future_ppo_backend"}
    if (type(plan) is not dict or set(plan) != keys or plan["schema_version"] != 1 or plan["kind"] != KIND
            or plan["original_started_at_epoch"] != resume.STARTED
            or plan["original_deadline_epoch"] != resume.DEADLINE
            or plan["selected_backend"] not in resume.BACKENDS
            or plan["future_ppo_backend"] != plan["selected_backend"]):
        raise ValueError("Exact authorized same-budget continuation plan required")
    old_path = _pin_record(plan["original_plan"])
    old = legacy.validate_plan(old_path, plan["original_plan"]["sha256"])
    output = Path(plan["output"])
    if (not output.is_absolute() or output.resolve().parent != (ROOT / "runs").resolve()
            or output.resolve() == Path(old["output"]).resolve()
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("A separate fresh direct-child run output is required")
    resume._source_pins(plan["source_sha256"], old["source_sha256"])
    if not NEW_SOURCES <= plan["source_sha256"].keys():
        raise ValueError("New supervisor and its tests must be pinned")
    _cpu_proof(plan)
    original_request = _parse(_pin_record(plan["original_request"]))
    if (Path(plan["original_request"]["path"]).resolve() != Path(old["output"]).resolve() / "ppo42.request.json"
            or original_request["alignment_plan_sha256"] != plan["original_plan"]["sha256"]):
        raise ValueError("The resumed request must belong to the exact original R3 campaign")
    campaign = _parse(_pin_record(plan["prior_campaign"]))
    failed_prefix_authorized = _authorized_failed_prefix(plan, campaign)
    if (Path(plan["prior_campaign"]["path"]).resolve() != Path(old["output"]).resolve() / "campaign.json"
            or campaign.get("kind") != legacy.KIND
            or not (campaign.get("status") == "paused" or failed_prefix_authorized)
            or campaign.get("plan_sha256") != plan["original_plan"]["sha256"]
            or campaign.get("started_at_epoch") != resume.STARTED or campaign.get("deadline_epoch") != resume.DEADLINE
            or campaign.get("child_may_still_be_running")
            or [s.get("name") for s in campaign.get("stages", [])] != ["dpo42", "ppo42"]):
        raise ValueError("Only the explicitly stopped original two-stage prefix can continue")
    accepted, interrupted = campaign["stages"]
    if (accepted.get("status") != "accepted" or accepted.get("exit_code") != 0
            or accepted.get("active_processes") or interrupted.get("active_processes")
            or interrupted.get("exit_code") is None or not interrupted.get("known_processes")
            or interrupted.get("request") != original_request
            or interrupted.get("monitoring_error") and not failed_prefix_authorized):
        raise ValueError("Original accepted predecessor / stopped child evidence is incomplete")
    prior = _parse(_pin_record(plan["history"]["prior_progress"]))
    waited = plan["process_wait_spent_seconds"]
    if (type(waited) not in (int, float) or not math.isfinite(waited)
            or waited != prior.get("process_guard", {}).get("total_waited_seconds")
            or not campaign["total_process_wait_seconds"] <= waited <= 259200):
        raise ValueError("Shared waiting ledger cannot be reset or invented")
    dpo_request = _parse(_pin_record(plan["accepted_dpo42"]["request"]))
    if (accepted["request"] != dpo_request
            or plan["accepted_dpo42"].get("campaign") != plan["prior_campaign"]
            or accepted["progress_sha256"] != plan["accepted_dpo42"]["progress"]["sha256"]
            or accepted["artifact"]["manifest_sha256"] != plan["accepted_dpo42"]["artifact_manifest"]["sha256"]):
        raise ValueError("Inherited DPO acceptance differs from the original campaign")
    legacy._verify_stage(Path(dpo_request["output"]), dpo_request,
        {p["pid"]: p for p in accepted["known_processes"]}, 0)
    request = _resume_request(plan, path, expected_sha,
        {p["pid"]: p for p in original_request["supervisor_processes"]}, waited)
    resume.validate_request(request)
    # The run loop retains original common parent/model/data fields. No child
    # receives this view as an original R3 plan or silently rewrites its pins.
    view = {**copy.deepcopy(old), **copy.deepcopy(plan)}
    view["_original"] = old
    view["_authorized"] = plan
    view["_prior_campaign"] = campaign
    view["continuation"] = {**old["continuation"], "process_wait_spent_seconds": waited}
    _pin(path, expected_sha)
    return view


def _prior_processes_gone(plan, processes):
    known = plan["_prior_campaign"]["supervisor_processes"][:]
    for stage in plan["_prior_campaign"]["stages"]:
        known.extend(stage["known_processes"])
    for actual in processes:
        for old in known:
            if (actual["pid"] == old["pid"] and actual["created_at_utc"] == old["created_at_utc"]):
                raise RuntimeError("Original R3 process is still alive; parallel continuation is forbidden")


class _LiveIdentity:
    @classmethod
    def from_plan(cls, path, expected_sha, plan):
        original = plan["_original"]
        sentinel = legacy._LiveIdentity.from_plan(plan["original_plan"]["path"],
            plan["original_plan"]["sha256"], original)
        sentinel.add_file(path, expected_sha, always_hash=True)
        for name, digest in plan["source_sha256"].items():
            sentinel.add_file(ROOT / name, digest, always_hash=True)
        def visit(value):
            if type(value) is dict:
                if set(value) == {"path", "sha256"}:
                    sentinel.add_file(value["path"], value["sha256"], always_hash=True)
                else:
                    for item in value.values():
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(plan["_authorized"])
        sentinel.add_tree(plan["history"]["checkpoint"]["path"])
        sentinel.add_tree(plan["_prior_campaign"]["stages"][0]["reference_cache"]["path"])
        return sentinel


def _own_processes(processes, plan_path):
    by_pid = {p["pid"]: p for p in processes}
    current, own = by_pid.get(os.getpid()), {}
    expected_sha = sha(plan_path)
    while current and _command(current, "run_alignlab_throughput_campaign_v2.py", plan_path):
        if current["pid"] in own or expected_sha not in current["command_line"]:
            raise RuntimeError("Supervisor command/ancestry mismatch")
        _stamp(current["created_at_utc"])
        own[current["pid"]] = current
        current = by_pid.get(current.get("parent_pid"))
    if not own:
        raise RuntimeError("Cannot identify exact new supervisor chain")
    return own


def _build_request(plan, plan_path, expected_sha, own, waited, stage, seed, cache_pin):
    if (stage, seed) == ("ppo", 42):
        return _resume_request(plan, plan_path, expected_sha, own, waited), "run_alignlab_throughput_stage_v2.py"
    old = plan["_original"]
    name = f"{stage}{seed}"
    keys = ("source_sha256", "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
        "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path",
        "parent_plan_sha256", "parent_acceptance_path", "parent_acceptance_sha256", "allowed_background_processes",
        "process_wait_policy", "continuation", "repair_attempt")
    inner = {k: copy.deepcopy(old[k]) for k in keys}
    inner.update(schema_version=1, kind=legacy.STAGE_KIND, stage=stage, seed=seed,
        output=str(Path(plan["output"]) / name), deadline_epoch=resume.DEADLINE,
        alignment_plan_path=plan["original_plan"]["path"], alignment_plan_sha256=plan["original_plan"]["sha256"],
        supervisor_processes=list(own.values()), process_wait_spent_seconds=waited,
        parents={k: old["parents"][k] for k in (("sft",) if stage == "dpo" else ("sft", "reward"))},
        workload=workload(stage, old["ppo_rows"]), reference_cache={"mode": "reuse", **cache_pin} if stage == "dpo" else None)
    from .throughput_fresh_v1 import make_request
    return make_request(inner, plan["_authorized"], {"path": str(plan_path), "sha256": expected_sha}), "run_alignlab_throughput_fresh_v1.py"


def _verify_stage(output, request, known, code):
    if request["kind"] == resume.KIND:
        return resume.verify_stage(output, request, known, code)
    if request["kind"] == legacy.STAGE_KIND:
        return legacy._verify_stage(output, request, known, code)
    from .throughput_fresh_v1 import verify_stage
    return verify_stage(output, request, known, code)


def run_alignment(plan_path, expected_plan_sha256, *, execute=False, system=None):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(plan_path, expected_plan_sha256)
    flags = {"primary_alignment_verified": False, "parents_verified": True,
             "capacity_verified": True, "formal_training": False, "research_complete": False}
    if not execute:
        return {"status": "plan_only", "plan": plan, **flags}
    live_identity = _LiveIdentity.from_plan(plan_path, expected_plan_sha256, plan)
    # Bracket the initial full validation with metadata observations rather
    # than silently blessing a file replacement that occurred while hashing.
    validate_plan(plan_path, expected_plan_sha256)
    live_identity.check()
    system, output = system or _System(), Path(plan["output"]).resolve()
    stopped_on_disk = lambda: (output / "STOP").exists() or Path(str(plan_path) + ".stop").exists()
    _status = JsonWriter(plan["continuation"]["deadline_epoch"], stopped_on_disk, clock=system.now, sleep=system.sleep)
    def _new(path, value, *, terminal=False):
        return _status(path, value, exclusive=True, terminal=terminal)
    lock = ROOT / "reports/.alignlab-alignment.lock"
    foreign_locks = [ROOT / ("reports/" + name) for name in
                     (".alignlab-gpu.lock", ".alignlab-parents.lock", ".alignlab-capacity.lock")]
    if output.exists() or lock.exists() or any(p.exists() for p in foreign_locks):
        raise FileExistsError("Existing output or lock; never reuse outputs or delete unknown locks")
    if system.now() >= plan["continuation"]["deadline_epoch"]:
        return {"status": "budget_exhausted", "started_at_epoch": plan["continuation"]["started_at_epoch"],
                "deadline_epoch": plan["continuation"]["deadline_epoch"], **flags}
    processes = system.processes()
    _parents_gone(plan, processes)
    _prior_processes_gone(plan, processes)
    own = _own_processes(processes, plan_path)
    initial_guard = process_guard.strict_admission(system.processes,
        owned_identities=list(own.values()), allowed_identities=plan["allowed_background_processes"],
        deadline_epoch=plan["continuation"]["deadline_epoch"], clock=system.now, sleep=system.sleep)
    if not initial_guard["caller_admission_allowed"]:
        return {"status": "blocked_before_start", "process_guard": initial_guard,
                "started_at_epoch": None, **flags}
    admission = system.resources()
    if admission["blockers"]:
        return {"status": "blocked_before_start", "resources": admission, "started_at_epoch": None, **flags}
    if system.now() >= plan["continuation"]["deadline_epoch"]:
        return {"status": "budget_exhausted", **flags}
    if Path(str(plan_path) + ".stop").exists():
        return {"status": "stopped_before_start", **flags}
    token = {"owner": KIND, "pid": os.getpid(), "plan_sha256": expected_plan_sha256, "token": uuid.uuid4().hex}
    _new(lock, token)
    started = plan["continuation"]["started_at_epoch"]
    attempt_started = system.now()
    state = {"kind": KIND, "status": "starting", **flags, "formal_training": True,
             "scope": "primary_six_alignment_branches_only", "plan_path": str(plan_path),
             "plan_sha256": expected_plan_sha256, "started_at_epoch": started,
             "deadline_epoch": plan["continuation"]["deadline_epoch"], "attempt_started_at_epoch": attempt_started,
             "continuation": plan["continuation"], "repair_attempt": plan["repair_attempt"],
             "supervisor_processes": list(own.values()),
              "resources_before": admission, "initial_process_guard": initial_guard,
              "total_process_wait_seconds": plan["continuation"]["process_wait_spent_seconds"], "process_wait_policy": plan["process_wait_policy"], "stages": [copy.deepcopy(plan["_prior_campaign"]["stages"][0])]}
    child, child_finished = None, True

    def write(status=None, **values):
        if status:
            state["status"] = status
        state.update(values, updated_at_epoch=system.now())
        terminal = state["status"] in {"failed", "paused", "budget_exhausted", "stopping"}
        _status(output / "campaign.json", state, terminal=terminal)

    def stop_reason():
        if (output / "STOP").exists() or Path(str(plan_path) + ".stop").exists():
            return "paused"
        return "budget_exhausted" if system.now() >= state["deadline_epoch"] else None

    def signal(reason):
        if not (output / "STOP").exists():
            _new(output / "STOP", {"owner": KIND, "plan_sha256": expected_plan_sha256, "reason": reason}, terminal=True)

    def frozen_identity():
        try:
            live_identity.check()
            return True
        except Exception as exc:
            state["identity_guard_error"] = f"{type(exc).__name__}: {exc}"
            return False

    def complete_identity():
        live_identity.check()
        validate_plan(plan_path, expected_plan_sha256)
        live_identity.check()

    def stage_admission(name):
        """No child/GPU work is active here; all waiting keeps the same ledger."""
        def event(value):
            with (output / "supervisor-process-guard.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage_name": name, **value}, allow_nan=False) + "\n")
            if value.get("event") == "wait_start":
                write("waiting_process_identity", waiting_stage=name)

        session = process_guard.GuardSession(system.processes, owned_identities=list(own.values()),
            allowed_identities=plan["allowed_background_processes"], deadline_epoch=state["deadline_epoch"],
            policy=plan["process_wait_policy"], spent_seconds=state["total_process_wait_seconds"],
            stop_requested=lambda: stop_reason() == "paused", identity_unchanged=frozen_identity,
            clock=system.now, sleep=system.sleep, on_event=event)
        receipt = lambda: {"complete": True, "optimizer_updates": 0, "kind": "before_cuda",
            "path": str(output / "plan.snapshot.json"), "sha256": sha(output / "plan.snapshot.json")}
        decision = session.check(boundary="supervisor_before_" + name, optimizer_updates=0,
                                 save_boundary=receipt)
        state["total_process_wait_seconds"] = decision["total_waited_seconds"]
        state["last_process_guard"] = decision
        if not decision["admission_allowed"]:
            reason = decision["reason"]
            status = "budget_exhausted" if reason == "deadline_exhausted" else (
                "paused" if reason == "stop_requested" else "failed")
            write(status, process_guard_failure=reason)
            return False
        return True

    try:
        output.mkdir(exist_ok=False)
        _new(output / "plan.snapshot.json", plan["_authorized"])
        live_identity.add_file(output / "plan.snapshot.json", sha(output / "plan.snapshot.json"), always_hash=True)
        _status(output / "live-identity-sentinel.json", live_identity.record())
        write()
        snapshot = verify_snapshot(plan["model_snapshot"], plan["model_manifest_sha256"])
        if snapshot["repo_id"] != REPO_ID or snapshot["revision"] != REVISION:
            raise ValueError("Actual model snapshot differs from registered revision")
        cache_pin = {k: state["stages"][0]["reference_cache"][k] for k in ("path", "manifest_sha256", "identity_sha256")}
        for stage, seed in SEQUENCE[1:]:
            name = f"{stage}{seed}"
            while True:
                live_identity.check()
                reason = stop_reason()
                if reason:
                    write(reason)
                    return state
                if not stage_admission(name):
                    return state
                if any(p.exists() for p in foreign_locks):
                    raise FileExistsError("Foreign or retained training/GPU lock")
                admission = system.resources()
                if not admission["blockers"]:
                    break
                write("waiting_resources", resources=admission)
                remaining = state["deadline_epoch"] - system.now()
                if remaining > 0:
                    system.sleep(min(10, remaining))
            request, child_script = _build_request(plan, plan_path, expected_plan_sha256,
                own, state["total_process_wait_seconds"], stage, seed, cache_pin)
            request_waited = state["total_process_wait_seconds"]
            path = output / f"{name}.request.json"
            _new(path, request)
            live_identity.add_file(path, sha(path), always_hash=True)
            _status(output / "live-identity-sentinel.json", live_identity.record())
            args = [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B", "-u",
                    "scripts/" + child_script, "--request", str(path),
                    "--request-sha256", sha(path), "--execute"]
            record = {"name": name, "stage": stage, "seed": seed, "request": request,
                      "request_sha256": sha(path), "command": args, "launch_epoch": system.now(),
                      "known_processes": [], "status": "dispatching"}
            state["stages"].append(record)
            write("dispatching")
            with (output / f"{name}.stdout.log").open("xb") as stdout, (output / f"{name}.stderr.log").open("xb") as stderr:
                complete_identity()
                reason = stop_reason()
                if reason:
                    write(reason)
                    return state
                # The child independently guards its before-CUDA boundary. An
                # unknown process appearing after stage_admission must use that
                # bounded wait, not force a fresh request/ledger or early STOP.
                child = system.spawn(args, stdout, stderr)
                child_finished = False
                record["wrapper_pid"] = child.pid
                known, monitor_error, stopped = {}, None, None
                snapshot_uncertain_since = None
                while True:
                    try:
                        live_identity.check()
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    stopped = stopped or stop_reason()
                    if stopped or monitor_error:
                        signal(stopped or "identity_failure")
                    try:
                        live = system.processes()
                    except Exception as exc:
                        # Unknown whole-system observations are handled by the
                        # live child's bounded safe-boundary wait. A supervisor
                        # query failure cannot itself prove child exit.
                        if snapshot_uncertain_since is None:
                            snapshot_uncertain_since = system.now()
                        observation_deadline = state["deadline_epoch"]
                        record.update(pending_boundary_wait=True,
                            process_snapshot_error=f"{type(exc).__name__}: {exc}",
                            process_snapshot_uncertain_since=snapshot_uncertain_since,
                            process_snapshot_observation_deadline=observation_deadline,
                            exit_code=child.poll())
                        if system.now() >= observation_deadline:
                            failed_status = stop_reason() or "failed"
                            signal(stop_reason() or "process_snapshot_unavailable")
                            # Keep the lock and make the remaining uncertainty
                            # explicit; a wrapper exit alone is not child exit.
                            state["child_may_still_be_running"] = True
                            record["exit_identity_verified"] = False
                            write(failed_status,
                                  process_guard_failure="process_snapshot_unavailable_exit_unverified")
                            return state
                        write("stopping" if stopped or monitor_error else "pending_boundary_wait")
                        system.sleep(max(0, min(10, observation_deadline - system.now())))
                        continue
                    snapshot_uncertain_since = None
                    by_pid = {p["pid"]: p for p in live}
                    for _ in range(len(live) + 1):
                        added = False
                        for p in live:
                            if p["pid"] not in known and (p["pid"] == child.pid or p.get("parent_pid") in known):
                                if (not _command(p, child_script, path)
                                        or "--request-sha256" not in p["command_line"]
                                        or record["request_sha256"] not in p["command_line"]
                                        or _stamp(p["created_at_utc"]).timestamp() < record["launch_epoch"]):
                                    monitor_error = "Child command/creation/ancestry mismatch"
                                else:
                                    known[p["pid"]] = p
                                    added = True
                        if not added:
                            break
                    try:
                        _parents_gone(plan, live)
                        peer_guard = _peer_decision(live, own, known, plan["allowed_background_processes"])
                        record["process_guard"] = peer_guard
                        pending = not peer_guard["caller_admission_allowed"]
                        record["pending_boundary_wait"] = pending and _waitable_peer_decision(peer_guard)
                        if pending and not record["pending_boundary_wait"]:
                            raise RuntimeError("Known competing training or process identity changed")
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    if monitor_error:
                        signal("identity_failure")
                    active = [p for pid, p in known.items() if pid in by_pid and _same_identity(by_pid[pid], p)]
                    code = child.poll()
                    record.update(known_processes=list(known.values()), active_processes=active,
                                  monitoring_error=monitor_error, stop_reason=stopped, exit_code=code)
                    write("stopping" if stopped or monitor_error else (
                        "pending_boundary_wait" if record.get("pending_boundary_wait") else "running"))
                    if code is not None and not active:
                        unverified = [p for p in live if
                            p["pid"] not in known and (p["pid"] == child.pid or p.get("parent_pid") in known)
                            or p["pid"] in known
                            and process_guard.strict.ProcessIdentity.from_record(p).instance
                                == process_guard.strict.ProcessIdentity.from_record(known[p["pid"]]).instance
                            and not _same_identity(p, known[p["pid"]])]
                        if not known or unverified:
                            state["child_may_still_be_running"] = True
                            record["exit_identity_verified"] = False
                            record["unverified_descendants"] = unverified
                            raise RuntimeError("Child exit identity is unverified; wrapper exit is insufficient")
                        child_finished = True
                        if monitor_error:
                            raise RuntimeError(monitor_error)
                        if stopped:
                            write(stopped)
                            return state
                        if any(p.exists() for p in foreign_locks):
                            raise FileExistsError("Exited child retained a lock; do not advance")
                        complete_identity()
                        result = _verify_stage(output / name, request, known, code)
                        verified_wait = result["process_guard"]["total_waited_seconds"]
                        if (request_waited != state["total_process_wait_seconds"]
                                or not state["total_process_wait_seconds"] <= verified_wait <= plan["wall_seconds"]):
                            raise ValueError("Child reset or exceeded the shared campaign waiting ledger")
                        state["total_process_wait_seconds"] = verified_wait
                        if stage == "dpo":
                            produced = {k: result["reference_cache"][k]
                                        for k in ("path", "manifest_sha256", "identity_sha256")}
                            if cache_pin is None:
                                if Path(produced["path"]).resolve() != output / name / "reference-cache":
                                    raise ValueError("Initial DPO cache escaped its owning stage")
                                cache_pin = produced
                                # Cache becomes immutable only after full
                                # completed-stage acceptance. Later branches
                                # reuse its exact files; no live rebaselining.
                                live_identity.add_tree(cache_pin["path"])
                            elif produced != cache_pin:
                                raise ValueError("Later seeds did not reuse the original frozen reference cache")
                        record.update(status="accepted", progress_path=str(output / name / "progress.json"),
                            progress_sha256=sha(output / name / "progress.json"), artifact=result["artifact"],
                            reference_cache=result.get("reference_cache"),
                            inherited_process_wait_seconds=request_waited,
                            total_process_wait_seconds=verified_wait, finished_at_epoch=system.now())
                        write("stage_accepted")
                        break
                    system.sleep(10)
        complete_identity()
        if not stage_admission("final_publication"):
            return state
        if any(p.exists() for p in foreign_locks):
            raise FileExistsError("A training/GPU lock remains after child exit")
        reason = stop_reason()
        if reason:
            write(reason)
            return state
        write("primary_alignment_passed", primary_alignment_verified=True, research_complete=False)
        return state
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        if output.exists():
            if child is not None and not child_finished:
                signal("supervisor_error")
                state["child_may_still_be_running"] = True
            write("failed")
        raise
    finally:
        try:
            if output.exists():
                state.update(finished_at_epoch=system.now(), elapsed_seconds=system.now() - started,
                             attempt_elapsed_seconds=system.now() - attempt_started)
                _status(output / "campaign.json", state, terminal=True)
        finally:
            if child_finished and lock.exists() and _parse(lock) == token:
                lock.unlink()
