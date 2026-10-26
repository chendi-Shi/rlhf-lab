"""One-shot, manifest-verified download -> initial CUDA probe dispatch.

Waiting is bounded by one absolute deadline; it does not consume/reset the
probe's separate 1800-second cooperative update budget. No retries, stale-lock
cleanup, process termination, campaign launch, or full-capacity claim.
Only the child probe is launched. Windows process creation time and command
line are observed, not inferred from PID. Tests inject a CPU-only system.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .gpu_probe import resources, validate_plan
from .model_snapshot import REPO_ID, REVISION, MANIFEST_NAME, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-download-probe-dispatch-v1"
SOURCE_FILES = ("alignment_lab/probe_dispatch.py", "scripts/dispatch_alignlab_probe.py",
                "tests_v2/test_probe_dispatch.py")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _parse(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _status(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    _new(temporary, value)
    os.replace(temporary, path)


def _stamp(value):
    if not isinstance(value, str):
        raise ValueError("Explicit process creation timestamp required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Process creation timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _same_process(actual, expected):
    return (actual["pid"] == expected["pid"]
            and _stamp(actual["created_at_utc"]) == _stamp(expected["created_at_utc"])
            and actual["command_line"] == expected["command_line"])


def _within(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent) or path == parent:
        raise ValueError(f"Path must stay inside {parent}")
    return path


def _request(path, expected_sha):
    if not isinstance(expected_sha, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha):
        raise ValueError("An external request SHA-256 pin is required")
    if _sha(path) != expected_sha:
        raise ValueError("Dispatch request SHA changed")
    request = _parse(path)
    required = {"schema_version", "kind", "report_dir", "plan_template", "model", "download_processes",
                "prerequisites", "max_wait_seconds", "poll_seconds", "dispatcher_source_sha256"}
    if (not isinstance(request, dict) or set(request) != required or request["schema_version"] != 1
            or request["kind"] != KIND or request["model"] != {"repo": REPO_ID, "revision": REVISION}):
        raise ValueError("Unsupported dispatch request/schema/model identity")
    if (type(request["max_wait_seconds"]) is not int or not 1 <= request["max_wait_seconds"] <= 7200
            or type(request["poll_seconds"]) is not int or not 1 <= request["poll_seconds"] <= 60):
        raise ValueError("Dispatch wait must be <= 7200 seconds with a bounded polling interval")
    _within(request["report_dir"], ROOT / "reports")
    plan = request["plan_template"]
    if not isinstance(plan, dict) or "model_manifest_sha256" in plan:
        raise ValueError("Manifest digest is bound only after pinned snapshot verification")
    validate_plan(plan)
    _within(plan["model_snapshot"], ROOT / "models")
    _within(plan["tokenizer_dir"], ROOT / "models")
    if _sha(Path(plan["audit_dir"]) / "manifest.json") != plan["audit_manifest_sha256"]:
        raise ValueError("Accepted data manifest SHA changed")
    sources = request["dispatcher_source_sha256"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES):
        raise ValueError("Dispatcher source/test inventory must be exact")
    if any(_sha(ROOT / name) != digest for name, digest in sources.items()):
        raise ValueError("Frozen dispatcher source/test changed")
    downloads = request["download_processes"]
    if (not isinstance(downloads, list) or not downloads
            or len({p.get("pid") for p in downloads if isinstance(p, dict)}) != len(downloads)):
        raise ValueError("Explicit unique downloader process identities required")
    for process in downloads:
        if (not isinstance(process, dict) or set(process) != {"pid", "created_at_utc", "command_line"}
                or type(process["pid"]) is not int or process["pid"] <= 0
                or not isinstance(process["command_line"], str)
                or "fetch_alignlab_model.py" not in process["command_line"]
                or "--execute" not in process["command_line"]):
            raise ValueError("Malformed downloader identity")
        _stamp(process["created_at_utc"])
    if not isinstance(request["prerequisites"], list):
        raise ValueError("Prerequisites must be explicit, including an empty list if none")
    for item in request["prerequisites"]:
        if (not isinstance(item, dict) or set(item) != {"path", "sha256", "json_equals"}
                or not isinstance(item["json_equals"], dict) or _sha(item["path"]) != item["sha256"]):
            raise ValueError("Prerequisite file identity changed")
        if item["json_equals"]:
            content = _parse(item["path"])
            for dotted, expected in item["json_equals"].items():
                actual = content
                for key in dotted.split("."):
                    actual = actual[key]
                if actual != expected or type(actual) is not type(expected):
                    raise ValueError(f"Prerequisite state mismatch: {dotted}")
    if _sha(path) != expected_sha:
        raise ValueError("Request changed during validation")
    return request


class _System:
    def now(self):
        return time.time()

    def sleep(self, seconds):
        time.sleep(seconds)

    def processes(self):
        if os.name != "nt":
            raise RuntimeError("Live dispatcher requires Windows process identity checks")
        command = (
            "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
            "ForEach-Object { [pscustomobject]@{ pid=$_.ProcessId; parent_pid=$_.ParentProcessId; "
            "created_at_utc=$_.CreationDate.ToUniversalTime().ToString('o'); command_line=$_.CommandLine } }) "
            "| ConvertTo-Json -Depth 4 -Compress"
        )
        result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], check=True,
                                capture_output=True, text=True, timeout=20,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        processes = json.loads(result.stdout)
        return [processes] if isinstance(processes, dict) else processes or []

    def resources(self):
        return resources()

    def spawn(self, args, stdout, stderr):
        if os.name != "nt":
            raise RuntimeError("Only a hidden Windows probe child may be launched")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        return subprocess.Popen(args, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                creationflags=subprocess.CREATE_NO_WINDOW)


def _downloads(request, processes):
    by_pid = {process["pid"]: process for process in processes}
    expected_ids = {process["pid"] for process in request["download_processes"]}
    for actual in processes:
        if ("fetch_alignlab_model.py" in (actual.get("command_line") or "")
                and actual["pid"] not in expected_ids):
            raise RuntimeError("Unexpected downloader process; refusing to guess ownership")
    active = []
    for expected in request["download_processes"]:
        actual = by_pid.get(expected["pid"])
        if actual is not None:
            if not _same_process(actual, expected):
                raise RuntimeError("Downloader PID/create-time/command identity mismatch")
            active.append(actual)
    return active


def _no_probe(request, processes):
    if Path(request["plan_template"]["output"]).exists():
        raise FileExistsError("Probe output already exists; no overwrite or implicit resume")
    if (ROOT / "reports/.alignlab-gpu.lock").exists():
        raise FileExistsError("Existing probe lock; never delete a foreign/stale lock")
    if any("run_alignlab_probe.py" in (process.get("command_line") or "") for process in processes):
        raise RuntimeError("Another probe process is already active")


def _stop_requested(request_path, report):
    return (report / "STOP").exists() or Path(str(request_path) + ".stop").exists()


def _signal_stop(output, request_sha, status):
    if not output.is_dir():
        return False
    marker = output / "STOP"
    if marker.exists():
        return True  # Preserve user/other owner's marker.
    try:
        _new(marker, {"owner": KIND, "request_sha256": request_sha, "dispatcher_pid": os.getpid()})
    except FileExistsError:
        pass
    status["probe_stop_marker"] = str(marker)
    return True


def _probe_result(output, plan, known, returncode):
    if returncode != 0:
        raise RuntimeError(f"Probe child exited {returncode}; inspect stdout/stderr; no retry")
    result = _parse(output / "probe.json")
    if (result.get("status") != "initial_gpu_training_probe_passed"
            or result.get("capacity_verified") is not False or result.get("formal_training") is not False
            or result.get("training_started") is not True or result.get("recovery_verified") is not True
            or result.get("actual_optimizer_updates") != 3 or result.get("plan") != plan
            or result.get("pid") not in known or not result.get("finished_at_utc")):
        raise RuntimeError("Probe exited without complete verified short-probe evidence")
    return result


def run_dispatch(request_path, expected_request_sha256, *, execute=False, system=None):
    """No process launched unless execute=True; injected system is for CPU tests."""
    request_path = Path(request_path).resolve()
    request = _request(request_path, expected_request_sha256)
    if not execute:
        return {"status": "plan_only", "request": request, "capacity_verified": False}
    system = system or _System()
    report = Path(request["report_dir"]).resolve()
    if report.exists():
        raise FileExistsError("Dispatcher report exists; waiting budgets and dispatch attempts cannot restart")
    lock = ROOT / "reports/.alignlab-probe-dispatch.lock"
    started = system.now()
    token = {"owner": KIND, "pid": os.getpid(), "request_sha256": expected_request_sha256,
             "started_at_epoch": started, "token": uuid.uuid4().hex}
    _new(lock, token)
    child, child_finished = None, False
    status = {"kind": KIND, "status": "starting", "request_sha256": expected_request_sha256,
              "request_path": str(request_path), "report_dir": str(report), "started_at_epoch": started,
              "dispatch_deadline_epoch": started + request["max_wait_seconds"],
              "probe_budget_seconds": 1800, "probe_dispatched": False, "capacity_verified": False,
              "formal_training": False, "working_directory": str(ROOT), "events": []}

    def write(state=None, **values):
        if state:
            status["status"] = state
        status.update(values)
        status["updated_at_epoch"] = system.now()
        _status(report / "dispatch.json", status)

    def stop():
        return _stop_requested(request_path, report)

    def wait():
        system.sleep(min(request["poll_seconds"], max(0, status["dispatch_deadline_epoch"] - system.now())))

    try:
        report.mkdir(parents=True, exist_ok=False)
        _new(report / "request.snapshot.json", request)
        write()
        initial_processes = system.processes()
        own = next((p for p in initial_processes if p["pid"] == os.getpid()), None)
        if own is None:
            raise RuntimeError("Cannot observe dispatcher's actual process identity")
        status["dispatcher_process"] = own
        manifest_sha = None
        while True:
            _request(request_path, expected_request_sha256)
            if stop():
                write("stopped_before_dispatch")
                return status
            if system.now() >= status["dispatch_deadline_epoch"]:
                write("dispatch_wait_budget_exhausted")
                return status
            processes = system.processes()
            _no_probe(request, processes)
            active = _downloads(request, processes)
            if active:
                write("waiting_download", downloader_processes=active)
                wait()
                continue
            model_path = Path(request["plan_template"]["model_snapshot"])
            manifest_path = model_path / MANIFEST_NAME
            if not manifest_path.is_file():
                raise RuntimeError("Downloader processes exited without a complete snapshot manifest")
            if (model_path.parent / f".{model_path.name}.download.lock").exists():
                raise RuntimeError("Downloader retained its failure lock; do not dispatch")
            if manifest_sha is None:
                candidate_sha = _sha(manifest_path)
                verified = verify_snapshot(model_path, candidate_sha)
                if verified["repo_id"] != REPO_ID or verified["revision"] != REVISION:
                    raise ValueError("Verified snapshot differs from registered official repository/revision")
                manifest_sha = candidate_sha
                status["verified_model_manifest_sha256"] = manifest_sha
            elif _sha(manifest_path) != manifest_sha:
                raise RuntimeError("Previously verified model manifest changed")
            current_resources = system.resources()
            if current_resources["blockers"]:
                write("waiting_resources", resources=current_resources)
                wait()
                continue
            # Full verification immediately before dispatch; the child performs
            # another independent full check before loading the model.
            verify_snapshot(model_path, manifest_sha)
            _request(request_path, expected_request_sha256)
            if stop() or system.now() >= status["dispatch_deadline_epoch"]:
                continue
            processes = system.processes()
            if _downloads(request, processes):
                raise RuntimeError("Downloader became active again after completion")
            _no_probe(request, processes)
            current_resources = system.resources()
            if current_resources["blockers"]:
                write("waiting_resources", resources=current_resources)
                wait()
                continue
            plan = copy.deepcopy(request["plan_template"])
            plan["model_manifest_sha256"] = manifest_sha
            validate_plan(plan)
            if stop() or system.now() >= status["dispatch_deadline_epoch"]:
                continue
            plan_path = report / "probe-plan.json"
            _new(plan_path, plan)
            args = [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B", "-u",
                    "scripts/run_alignlab_probe.py", "--plan", str(plan_path), "--execute"]
            write("dispatching_probe", probe_dispatched=True, probe_plan_path=str(plan_path),
                  probe_plan_sha256=_sha(plan_path), probe_command=args, resources=current_resources,
                  probe_stdout=str(report / "probe.stdout.log"), probe_stderr=str(report / "probe.stderr.log"),
                  probe_launch_epoch=system.now())
            # Persist the one-shot intent before starting: exceptions never retry.
            with (report / "probe.stdout.log").open("xb") as stdout, (report / "probe.stderr.log").open("xb") as stderr:
                if stop() or system.now() >= status["dispatch_deadline_epoch"]:
                    write("stopped_before_child_launch")
                    return status
                child = system.spawn(args, stdout, stderr)
                known, monitor_error, user_stopped = {}, None, False
                write("probe_running", probe_wrapper_pid=child.pid)
                while True:
                    try:
                        _request(request_path, expected_request_sha256)
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    user_stopped = user_stopped or stop()
                    if user_stopped or monitor_error:
                        _signal_stop(Path(plan["output"]), expected_request_sha256, status)
                    live = system.processes()
                    by_pid = {p["pid"]: p for p in live}
                    wrapper = by_pid.get(child.pid)
                    if wrapper and child.pid not in known:
                        if ("run_alignlab_probe.py" in (wrapper.get("command_line") or "")
                                and str(plan_path).lower() in wrapper["command_line"].lower()
                                and _stamp(wrapper["created_at_utc"]).timestamp() >= status["probe_launch_epoch"]):
                            known[child.pid] = wrapper
                        else:
                            monitor_error = "Cannot bind spawned probe PID to its actual command/create-time"
                    parents = {child.pid} | set(known)
                    for process in live:
                        if (process.get("parent_pid") in parents
                                and "run_alignlab_probe.py" in (process.get("command_line") or "")
                                and str(plan_path).lower() in process["command_line"].lower()
                                and _stamp(process["created_at_utc"]).timestamp() >= status["probe_launch_epoch"]):
                            if process["pid"] not in known:
                                known[process["pid"]] = process
                    active_owned = []
                    for pid, expected in known.items():
                        actual = by_pid.get(pid)
                        if actual:
                            if not _same_process(actual, expected):
                                monitor_error = "Probe PID was reused or its command identity changed"
                            else:
                                active_owned.append(actual)
                    code = child.poll()
                    write("probe_stopping" if user_stopped or monitor_error else "probe_running",
                          probe_processes=list(known.values()), active_probe_processes=active_owned,
                          dispatcher_wait_deadline_reached=system.now() >= status["dispatch_deadline_epoch"],
                          monitoring_error=monitor_error, user_stop_requested=user_stopped)
                    if code is not None and not active_owned:
                        child_finished = True
                        if monitor_error:
                            raise RuntimeError(monitor_error)
                        if user_stopped:
                            write("stopped_after_dispatch", probe_exit_code=code)
                            return status
                        result = _probe_result(Path(plan["output"]), plan, known, code)
                        write("initial_probe_completed", probe_exit_code=code, probe_result=result,
                              scope="Initial short SFT/recovery probe only; full capacity remains unverified")
                        return status
                    # Dispatch wait budget never kills an already-running child
                    # and never resets the child's independent probe budget.
                    system.sleep(request["poll_seconds"])
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        if child is not None and not child_finished:
            _signal_stop(Path(request["plan_template"]["output"]), expected_request_sha256, status)
            status["child_may_still_be_running"] = True
        if report.exists():
            write("failed")
        raise
    finally:
        status["finished_at_epoch"] = system.now()
        status["elapsed_seconds"] = system.now() - started
        if report.exists():
            _status(report / "dispatch.json", status)
        # Retain our lock if ownership/liveness became uncertain after launch.
        if child is None or child_finished:
            if lock.exists() and _parse(lock) == token:
                lock.unlink()
