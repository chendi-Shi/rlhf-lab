"""One-shot formal SFT/RM parents; no retry, resume, evaluation or process kill.

Every registered source stays pinned. Later unregistered modules do not change
an already frozen dependency inventory. The capacity run remains historical.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import uuid

from .probe_dispatch import _System, _same_process, _stamp, _parse, _new, _status
from .capacity_supervisor import sha, _pin, _canonical, _prerequisites, _bundle, NF4
from .model_snapshot import verify_snapshot, REPO_ID, REVISION

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-formal-parents-campaign-v1"
STAGE_KIND = "alignlab-formal-parent-stage-v1"
SCOPE = "full_accepted_real_training_partition"
STAGES = ("sft", "reward")
COUNTS = {"sft": 4000, "reward": 13139}
UPDATES = {"sft": 250, "reward": 822}
PARTITIONS = {"sft": "sft_train", "reward": "preference_train"}
PARENTS = {"sft": (), "reward": ("sft",)}
NEW_SOURCE_FILES = {
    "alignment_lab/study_supervisor.py", "alignment_lab/study_stage.py",
    "scripts/run_alignlab_parents.py", "scripts/run_alignlab_study_stage.py",
    "tests_v2/test_study_supervisor.py", "tests_v2/test_study_stage.py",
}
BUSY = ("run_alignlab_probe.py", "dispatch_alignlab_probe.py", "run_alignlab_capacity",
        "run_alignlab_parents.py", "run_alignlab_study_stage.py", "run_campaign.py",
        "local-train", "local-evaluate", "windows.train", "train_sft", "train_dpo",
        "train_ppo", "run_training", "serve_adapter", "bootstrap_run.py",
        "fetch_alignlab_model.py")


def _relative_source(name):
    if (not isinstance(name, str) or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or not name.endswith(".py")):
        raise ValueError("Source inventory requires canonical project-relative Python paths")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen source escaped the project")
    return path


def _capacity_acceptance(plan):
    _pin(plan["capacity_acceptance_path"], plan["capacity_acceptance_sha256"])
    accepted = _parse(plan["capacity_acceptance_path"])
    if (accepted.get("status") != "accepted_registered_full_capacity_workload_only"
            or accepted.get("all_checks_passed") is not True
            or accepted.get("capacity_verified") is not True
            or accepted.get("formal_training") is not False
            or accepted.get("model_snapshot", {}).get("sha256") != plan["model_manifest_sha256"]
            or accepted.get("data", {}).get("manifest", {}).get("sha256") != plan["audit_manifest_sha256"]):
        raise ValueError("Independent complete capacity acceptance for these model/data pins required")
    snapshot = accepted["source_snapshot"]
    _pin(snapshot["path"], snapshot["sha256"])
    manifest = _parse(snapshot["path"])
    if (manifest.get("kind") != "alignlab-frozen-capacity-source-snapshot-v1"
            or manifest.get("file_count") != 52 or len(manifest.get("files", {})) != 52):
        raise ValueError("The exact 52 historical capacity source files must remain archived")
    archived = {}
    for name, item in manifest["files"].items():
        _relative_source(name)
        _pin(item["snapshot_path"], item["sha256"])
        if Path(item["snapshot_path"]).stat().st_size != item["size_bytes"]:
            raise ValueError("Archived capacity source size changed")
        archived[name] = item["sha256"]
    return archived


def validate_plan(path, expected_sha):
    """Read pinned inputs only; no CUDA or directory-glob dependency identity."""
    _pin(path, expected_sha)
    plan = _parse(path)
    fields = {"schema_version", "kind", "output", "wall_seconds", "poll_seconds",
              "source_sha256", "model_snapshot", "model_manifest_sha256", "tokenizer_dir",
              "audit_dir", "audit_manifest_sha256", "cpu_check_report", "cpu_check_report_sha256",
              "capacity_acceptance_path", "capacity_acceptance_sha256", "prerequisites"}
    if (set(plan) != fields or type(plan["schema_version"]) is not int
            or plan["schema_version"] != 1 or plan["kind"] != KIND):
        raise ValueError("Exact versioned formal-parent plan required")
    if (type(plan["wall_seconds"]) is not int or plan["wall_seconds"] != 43200
            or type(plan["poll_seconds"]) is not int or plan["poll_seconds"] != 10):
        raise ValueError("Parents share one 43200-second budget with ten-second monitoring")
    output = Path(plan["output"])
    if (not output.is_absolute() or output.is_symlink()
            or output.resolve().parent != (ROOT / "runs").resolve()):
        raise ValueError("Use a fresh direct child of project runs")
    for directory, filename, key in ((plan["model_snapshot"], "snapshot-manifest.json", "model_manifest_sha256"),
                                    (plan["audit_dir"], "manifest.json", "audit_manifest_sha256")):
        if not Path(directory).is_absolute():
            raise ValueError("Input directories must be absolute")
        _pin(Path(directory) / filename, plan[key])
    if not Path(plan["tokenizer_dir"]).is_absolute():
        raise ValueError("Tokenizer directory must be absolute")
    archived = _capacity_acceptance(plan)
    sources = plan["source_sha256"]
    if (not isinstance(sources, dict) or not (set(archived) | NEW_SOURCE_FILES) <= set(sources)
            or any(sources.get(name) != digest for name, digest in archived.items())):
        raise ValueError("Frozen sources must include unchanged historical dependencies and formal implementation/tests")
    for name, digest in sources.items():
        _pin(_relative_source(name), digest)
    _pin(plan["cpu_check_report"], plan["cpu_check_report_sha256"])
    cpu = _parse(plan["cpu_check_report"])
    if (cpu.get("status") != "cpu_software_checks_passed"
            or cpu.get("source_unchanged") is not True or cpu.get("cuda_initialized") is not False
            or cpu.get("device") != "cpu" or type(cpu.get("tests_run")) is not int
            or cpu["tests_run"] < 253
            or any(type(cpu.get(k)) is not int or cpu[k] != 0 for k in ("errors", "failures", "skipped"))):
        raise ValueError("Fresh complete CPU checks must pass without CUDA/failures/skips")
    checked = {k.replace("\\", "/"): v for k, v in cpu.get("source_before", cpu.get("source_sha256", {})).items()}
    required = {k for k in sources if k.startswith(("alignment_lab/", "tests_v2/"))}
    required.add("scripts/verify_alignlab_cpu.py")
    if not required <= set(checked) or any(checked[k] != sources[k] for k in required):
        raise ValueError("CPU source-before hashes differ from frozen parent dependencies")
    _prerequisites(plan["prerequisites"])
    _pin(path, expected_sha)
    return plan


def _command(record, script, path):
    text = (record.get("command_line") or "").lower().replace("\\", "/")
    return script in text and str(path).lower().replace("\\", "/") in text and "--execute" in text


def _own_processes(processes, plan_path):
    by_pid = {p["pid"]: p for p in processes}
    own = by_pid.get(os.getpid())
    if own is None or not _command(own, "run_alignlab_parents.py", plan_path):
        raise RuntimeError("Cannot identify supervisor PID/create-time/command")
    result = {}
    current = own
    while current and _command(current, "run_alignlab_parents.py", plan_path):
        if current["pid"] in result:
            raise RuntimeError("Cyclic supervisor process ancestry")
        _stamp(current["created_at_utc"])
        result[current["pid"]] = current
        current = by_pid.get(current.get("parent_pid"))
    return result


def _peers(processes, own, known=()):
    allowed = dict(own)
    allowed.update(dict(known))
    for p in processes:
        if p["pid"] in allowed:
            if not _same_process(p, allowed[p["pid"]]):
                raise RuntimeError("Known process PID reused or command identity changed")
        elif not p.get("command_line") or any(marker in p["command_line"].lower() for marker in BUSY):
            raise RuntimeError(f"Other training/probe/campaign process active: {p}")


def _formal_completion(manifest, stage):
    completion = manifest.get("completion", {})
    if (manifest.get("stage") != stage or completion.get("scope") != SCOPE
            or completion.get("synthetic_rows") != 0
            or completion.get("full_accepted_dataset_pass") is not True
            or completion.get("protocol_complete") is not True
            or completion.get("data_pass_complete") is not True
            or completion.get("stop_reason") != "data_exhausted"
            or completion.get("rows_processed") != COUNTS[stage]
            or completion.get("optimizer_updates") != UPDATES[stage]):
        raise ValueError("Capacity or incomplete parent cannot initialize formal training")
    return completion


def _verify_stage(output, request, known, code):
    """Verify exited formal child and actual pinned files; never accept CPU stubs."""
    output = Path(output).resolve()
    if code != 0:
        raise RuntimeError(f"Formal parent child exited {code}; no retry")
    result = _parse(output / "progress.json")
    stage = request["stage"]
    count, updates = COUNTS[stage], UPDATES[stage]
    if (result.get("status") != "formal_parent_stage_passed"
            or result.get("formal_training") is not True
            or result.get("research_complete") is not False
            or result.get("stage") != stage or result.get("request") != request
            or result.get("pid") not in known
            or type(result.get("started_at_epoch")) not in (int, float)
            or type(result.get("finished_at_epoch")) not in (int, float)
            or not result["started_at_epoch"] <= result["finished_at_epoch"] <= request["deadline_epoch"]
            or result.get("deadline_epoch") != request["deadline_epoch"]
            or result.get("actual_optimizer_updates") != updates):
        raise RuntimeError("Child lacks exact completed formal request/process/update/deadline evidence")
    if _parse(output / "request.json") != request:
        raise ValueError("Saved child request changed")
    model = result.get("model", {})
    if (model.get("device") != "cuda:0" or model.get("quantization") != NF4
            or model.get("quantized_linear_count") != 196
            or type(model.get("parameters_selected")) is not int or model["parameters_selected"] <= 0
            or model.get("snapshot_manifest_sha256") != request["model_manifest_sha256"]):
        raise RuntimeError("Missing actual CUDA NF4 model evidence")
    samples = result.get("gpu_samples")
    if (not isinstance(samples, list) or not samples
            or any(not s.get("gpus") or not any(g.get("index") == 0 for g in s["gpus"])
                   or not all(type(s.get(k)) in (int, float) and math.isfinite(s[k]) and s[k] > 0
                              for k in ("max_allocated_bytes", "max_reserved_bytes")) for s in samples)):
        raise RuntimeError("Missing finite positive measured GPU memory evidence")
    selection = result["selection"]
    if (selection.get("scope") != SCOPE or selection.get("partition") != PARTITIONS[stage]
            or selection.get("configured_rows") != count
            or selection.get("selected_rows") != count or selection.get("real_rows") != count
            or selection.get("synthetic_rows") != 0
            or selection.get("accepted_manifest_sha256") != request["audit_manifest_sha256"]):
        raise RuntimeError("Formal selection must contain the complete real training partition")
    selection_path = Path(selection["selection_manifest_path"]).resolve()
    if selection_path != output / "selection.json":
        raise ValueError("Selection manifest escaped the stage")
    _pin(selection_path, selection["selection_manifest_sha256"])
    selected = _parse(selection_path)
    if selected["summary"] != {k: v for k, v in selection.items()
                               if k not in {"selection_manifest_path", "selection_manifest_sha256"}}:
        raise ValueError("Selection summary differs from its pinned manifest")
    records = selected["records"]
    if (len(records) != count or len({r["record_id"] for r in records}) != count
            or any(not isinstance(r["record_id"], str) or not r["record_id"]
                   or r["record_id"].startswith("synthetic-")
                   or not 1 <= r["prompt_tokens"] <= 1024
                   or not 1 <= r["chosen_tokens"] <= 768
                   or (stage == "reward" and not 1 <= r["rejected_tokens"] <= 768)
                   or (stage == "sft" and r["rejected_tokens"] is not None)
                   or not re.fullmatch("[0-9a-f]{64}", r["token_sha256"]) for r in records)
            or hashlib.sha256(_canonical(records)).hexdigest() != selection["records_sha256"]):
        raise ValueError("Incomplete, duplicate or invalid real selection records")
    data_manifest = _parse(Path(request["audit_dir"]) / "manifest.json")
    partition_path = Path(request["audit_dir"]) / (PARTITIONS[stage] + ".jsonl")
    partition_sha = data_manifest["artifacts"][partition_path.name]
    _pin(partition_path, partition_sha)
    with partition_path.open(encoding="utf-8") as stream:
        accepted_ids = [json.loads(line)["record_id"] for line in stream if line.strip()]
    if [r["record_id"] for r in records] != accepted_ids:
        raise ValueError("Selection must preserve every accepted partition record in order")
    identity = result["identity"]
    if _parse(output / "identity.json") != identity:
        raise ValueError("Saved identity differs from progress")
    config = identity["config"]
    if (config.get("stage") != stage or config.get("purpose") != STAGE_KIND
            or config.get("scope") != SCOPE or config.get("formal_training") is not True
            or config.get("synthetic_rows") != 0 or config.get("expected_rows") != count
            or config.get("expected_updates") != updates):
        raise ValueError("Identity does not declare formal full-partition training")
    if set(identity["parents"]) != set(request["parents"]):
        raise ValueError("Formal parent identities differ")
    for name, parent in request["parents"].items():
        expected_path = Path(parent["artifact_dir"]).resolve() / "manifest.json"
        item = identity["parents"][name]
        if Path(item["path"]).resolve() != expected_path or item["sha256"] != parent["manifest_sha256"]:
            raise ValueError("Stage parent identity not anchored to admitted formal artifact")
        _formal_completion(_bundle(expected_path.parent, parent["manifest_sha256"],
                                   {"weights.pt"}, "alignlab-completed-stage-v1"), name)
    inputs = identity["inputs"]
    required = {"source/" + name: (ROOT / name, digest)
                for name, digest in request["source_sha256"].items()}
    required.update({
        "parent_plan": (request["parent_plan_path"], request["parent_plan_sha256"]),
        "capacity_acceptance": (request["capacity_acceptance_path"], request["capacity_acceptance_sha256"]),
        "data/manifest": (Path(request["audit_dir"]) / "manifest.json", request["audit_manifest_sha256"]),
        "data/partition": (partition_path, partition_sha),
        "data/selection": (selection_path, selection["selection_manifest_sha256"]),
    })
    snapshot = _parse(Path(request["model_snapshot"]) / "snapshot-manifest.json")
    required.update({"base/" + name: (Path(request["model_snapshot"]) / name, item["sha256"])
                     for name, item in snapshot["files"].items()})
    tokenizer_dir = Path(request["tokenizer_dir"])
    token_manifest = _parse(tokenizer_dir / "source-manifest.json")
    required.update({"tokenizer/" + name: (tokenizer_dir / name, item["sha256"])
                     for name, item in token_manifest["artifacts"].items()})
    required["tokenizer/source-manifest.json"] = (tokenizer_dir / "source-manifest.json",
                                                 sha(tokenizer_dir / "source-manifest.json"))
    for name, (path, digest) in required.items():
        item = inputs.get(name, {})
        if item.get("sha256") != digest or Path(item.get("path", "")).resolve() != Path(path).resolve():
            raise ValueError("Stage input/source identity mismatch: " + name)
    for name, item in inputs.items():
        if not name.startswith("base/"):
            _pin(item["path"], item["sha256"])
            if Path(item["path"]).stat().st_size != item["size_bytes"]:
                raise ValueError("Stage input size changed")
    cursor = result["cursor"]
    if (result.get("outcome_stop_reason") != "data_exhausted" or cursor["offset"] != count
            or cursor["optimizer_updates"] != updates or cursor["active_rollout"]
            or cursor["completed_rollouts"] != 0
            or result.get("adapter_weights_changed") is not True
            or result.get("scalar_heads_changed") != ({"reward": True} if stage == "reward" else {})):
        raise RuntimeError("Incomplete real formal optimizer work")
    metrics = [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
               for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    objectives = [m for m in metrics if m.get("event") == "objective_after_engine_return"]
    boundaries = [m for m in metrics if m.get("event") == "optimizer_boundary"]
    if (result.get("objective_logging") != "after_engine_returns"
            or result.get("objective_records") != updates
            or [m["cursor"]["optimizer_updates"] for m in boundaries] != list(range(1, updates + 1))
            or [m.get("update") for m in objectives] != list(range(1, updates + 1))
            or sum(m.get("rows", 0) for m in objectives) != count
            or any(m.get("rows") != min(16, count - 16 * i)
                   or not all(type(m.get(k)) in (int, float) and math.isfinite(m[k])
                              for k in ("objective", "gradient_norm"))
                   for i, m in enumerate(objectives))):
        raise RuntimeError("Missing continuous finite full-partition objectives or final partial batch")
    artifact = result["artifact"]
    if (Path(artifact["artifact_dir"]).resolve() != output / "final-artifact"
            or Path(artifact["manifest_path"]).resolve() != output / "final-artifact/manifest.json"):
        raise ValueError("Artifact escaped its owned stage output")
    manifest = _bundle(output / "final-artifact", artifact["manifest_sha256"],
                       {"weights.pt"}, "alignlab-completed-stage-v1")
    _formal_completion(manifest, stage)
    response_tokens = sum(r["chosen_tokens"] + (r["rejected_tokens"] or 0) for r in records)
    if (manifest["identity"] != identity or manifest["completion"] != result["completion"]
            or manifest["completion"].get("response_tokens") != response_tokens
            or cursor.get("response_tokens") != response_tokens):
        raise ValueError("Final artifact identity differs")
    checkpoints = result["checkpoints"]
    names = [Path(c["checkpoint_path"]).name for c in checkpoints]
    required_updates = {1, updates} | set(range(50, updates + 1, 50))
    if (len(names) != len(set(names))
            or not {f"update-{u:06d}" for u in required_updates} <= set(names)):
        raise ValueError("Final safe checkpoint missing or duplicate checkpoint names")
    latest = result["latest_safe_checkpoint"]
    if latest not in checkpoints or latest.get("optimizer_updates") != updates:
        raise ValueError("Latest safe checkpoint is not the completed final update")
    for receipt in checkpoints:
        directory = Path(receipt["checkpoint_path"]).resolve()
        update = receipt["optimizer_updates"]
        if (directory.parent != output or type(update) is not int or not 1 <= update <= updates
                or directory.name != f"update-{update:06d}" or not receipt["reasons"]
                or type(receipt.get("active_elapsed_seconds")) not in (int, float)
                or not 0 <= receipt["active_elapsed_seconds"] <= result["finished_at_epoch"] - result["started_at_epoch"]):
            raise ValueError("Checkpoint escaped its stage")
        saved = _bundle(directory, receipt["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
        if saved["identity"] != identity:
            raise ValueError("Checkpoint identity changed")
    return result


def run_parents(plan_path, expected_plan_sha256, *, execute=False, system=None):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(plan_path, expected_plan_sha256)
    if not execute:
        return {"status": "plan_only", "plan": plan, "parents_verified": False, "capacity_verified": True, "formal_training": False, "research_complete": False}
    system, output = system or _System(), Path(plan["output"]).resolve()
    lock, gpu_lock = ROOT / "reports/.alignlab-parents.lock", ROOT / "reports/.alignlab-gpu.lock"
    if output.exists() or lock.exists() or gpu_lock.exists() or (ROOT / "reports/.alignlab-capacity.lock").exists():
        raise FileExistsError("Existing output or parent/capacity/GPU lock; never reuse or remove unknown locks")
    processes = system.processes(); own = _own_processes(processes, plan_path); _peers(processes, own)
    admission = system.resources()
    if admission["blockers"]:
        return {"status": "blocked_before_start", "resources": admission, "parents_verified": False, "capacity_verified": True,
                "formal_training": False, "research_complete": False, "started_at_epoch": None}
    if Path(str(plan_path) + ".stop").exists():
        return {"status": "stopped_before_start", "parents_verified": False, "capacity_verified": True, "formal_training": False, "research_complete": False}
    token = {"owner": KIND, "pid": os.getpid(), "plan_sha256": expected_plan_sha256, "token": uuid.uuid4().hex}
    _new(lock, token)
    started = system.now()
    state = {"kind": KIND, "status": "starting", "plan_path": str(plan_path), "plan_sha256": expected_plan_sha256,
             "started_at_epoch": started, "deadline_epoch": started + plan["wall_seconds"],
             "parents_verified": False, "capacity_verified": True, "formal_training": True, "research_complete": False,
             "supervisor_processes": list(own.values()), "resources_before": admission, "stages": []}
    child, child_finished = None, True

    def write(status=None, **values):
        if status:
            state["status"] = status
        state.update(values, updated_at_epoch=system.now())
        _status(output / "campaign.json", state)

    def stop_reason():
        if (output / "STOP").exists() or Path(str(plan_path) + ".stop").exists():
            return "paused"
        return "budget_exhausted" if system.now() >= state["deadline_epoch"] else None

    def signal(reason):
        if not (output / "STOP").exists():
            _new(output / "STOP", {"owner": KIND, "plan_sha256": expected_plan_sha256, "reason": reason})

    try:
        output.mkdir(exist_ok=False)
        _new(output / "plan.snapshot.json", plan); write()
        snapshot = verify_snapshot(plan["model_snapshot"], plan["model_manifest_sha256"])
        if snapshot["repo_id"] != REPO_ID or snapshot["revision"] != REVISION:
            raise ValueError("Snapshot differs from registered model")
        accepted = {}
        for stage in STAGES:
            while True:
                validate_plan(plan_path, expected_plan_sha256)
                reason = stop_reason()
                if reason:
                    write(reason); return state
                _peers(system.processes(), own)
                if gpu_lock.exists() or (ROOT / "reports/.alignlab-capacity.lock").exists():
                    raise FileExistsError("Foreign or retained GPU lock")
                admission = system.resources()
                if not admission["blockers"]:
                    break
                write("waiting_resources", resources=admission)
                remaining = state["deadline_epoch"] - system.now()
                if remaining > 0:
                    system.sleep(min(plan["poll_seconds"], remaining))
            parents = {name: {k: accepted[name]["artifact"][k] for k in ("artifact_dir", "manifest_sha256")} for name in PARENTS[stage]}
            for name, parent in parents.items():
                _formal_completion(_bundle(parent["artifact_dir"], parent["manifest_sha256"], {"weights.pt"}, "alignlab-completed-stage-v1"), name)
            request = {k: plan[k] for k in ("source_sha256", "model_snapshot", "model_manifest_sha256",
                                          "tokenizer_dir", "audit_dir", "audit_manifest_sha256",
                                          "capacity_acceptance_path", "capacity_acceptance_sha256")}
            request.update(schema_version=1, kind=STAGE_KIND, stage=stage,
                output=str(output / stage), deadline_epoch=state["deadline_epoch"], parents=parents,
                parent_plan_path=str(plan_path), parent_plan_sha256=expected_plan_sha256,
                expected_rows=COUNTS[stage], expected_updates=UPDATES[stage], seed=42)
            path = output / f"{stage}.request.json"; _new(path, request)
            args = [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B", "-u",
                    "scripts/run_alignlab_study_stage.py", "--request", str(path),
                    "--request-sha256", sha(path), "--execute"]
            record = {"stage": stage, "request": request, "request_sha256": sha(path), "command": args,
                      "launch_epoch": system.now(), "known_processes": [], "status": "dispatching"}
            state["stages"].append(record); write("dispatching")
            with (output / f"{stage}.stdout.log").open("xb") as stdout, (output / f"{stage}.stderr.log").open("xb") as stderr:
                validate_plan(plan_path, expected_plan_sha256)
                if stop_reason():
                    write(stop_reason()); return state
                child = system.spawn(args, stdout, stderr); child_finished = False
                record["wrapper_pid"] = child.pid
                known, monitor_error, stopped = {}, None, None
                while True:
                    try:
                        validate_plan(plan_path, expected_plan_sha256)
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    stopped = stopped or stop_reason()
                    if stopped or monitor_error:
                        signal(stopped or "identity_failure")
                    live = system.processes(); by_pid = {p["pid"]: p for p in live}
                    for _ in range(len(live) + 1):
                        added = False
                        for p in live:
                            if p["pid"] not in known and (p["pid"] == child.pid or p.get("parent_pid") in known):
                                if _command(p, "run_alignlab_study_stage.py", path) and _stamp(p["created_at_utc"]).timestamp() >= record["launch_epoch"]:
                                    known[p["pid"]] = p; added = True
                                else:
                                    monitor_error = "Spawned wrapper/descendant command or creation mismatch"
                                    # Track even an unexpected real descendant
                                    # until natural exit; never call it accepted.
                                    known[p["pid"]] = p; added = True
                        if not added:
                            break
                    try:
                        _peers(live, own, known)
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    if monitor_error:
                        signal("identity_failure")
                    active = [p for pid, p in known.items() if pid in by_pid and _same_process(by_pid[pid], p)]
                    code = child.poll()
                    record.update(known_processes=list(known.values()), active_processes=active,
                                  monitoring_error=monitor_error, stop_reason=stopped, exit_code=code)
                    write("stopping" if stopped or monitor_error else "running")
                    if code is not None and not active:
                        child_finished = True
                        if monitor_error:
                            raise RuntimeError(monitor_error)
                        if stopped:
                            write(stopped); return state
                        if not known:
                            raise RuntimeError("Child exited before any real process identity was observed")
                        if gpu_lock.exists() or (ROOT / "reports/.alignlab-capacity.lock").exists():
                            raise FileExistsError("Exited child retained a GPU lock; do not advance")
                        result = _verify_stage(output / stage, request, known, code)
                        accepted[stage] = result
                        record.update(status="accepted", progress_path=str(output / stage / "progress.json"),
                                      progress_sha256=sha(output / stage / "progress.json"),
                                      artifact=result.get("artifact"), finished_at_epoch=system.now())
                        write("stage_accepted")
                        break
                    system.sleep(plan["poll_seconds"])
        validate_plan(plan_path, expected_plan_sha256)
        _peers(system.processes(), own)
        if gpu_lock.exists() or (ROOT / "reports/.alignlab-capacity.lock").exists():
            raise FileExistsError("GPU lock remains after all child exits")
        if stop_reason():
            write(stop_reason()); return state
        write("parents_passed", parents_verified=True, capacity_verified=True, research_complete=False)
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
        if output.exists():
            state.update(finished_at_epoch=system.now(), elapsed_seconds=system.now() - started)
            _status(output / "campaign.json", state)
        if child_finished and lock.exists() and _parse(lock) == token:
            lock.unlink()
