"""One-shot serial capacity admission/dispatch; no retry, resume or process kill.

Only a fully accepted five-stage workload can set capacity_verified. This does
not authorize a formal training campaign or establish model quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from .probe_dispatch import _System, _same_process, _stamp, _parse, _new, _status
from .model_snapshot import verify_snapshot, REPO_ID, REVISION

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-full-capacity-campaign-v1"
STAGES = ("sft", "reward", "dpo", "ppo", "eval")
UPDATES = dict(sft=1, reward=1, dpo=20, ppo=256, eval=0)
COUNTS = dict(sft=16, reward=16, dpo=320, ppo=128, eval=128)
PARENTS = dict(sft=(), reward=("sft",), dpo=("sft",), ppo=("sft", "reward"), eval=("sft", "reward"))
NF4 = {"method": "bitsandbytes", "bits": 4, "quant_type": "nf4",
       "double_quant": True, "compute_dtype": "bfloat16"}
CLI_FILES = {"scripts/run_alignlab_capacity.py", "scripts/run_alignlab_capacity_stage.py",
             "scripts/verify_alignlab_cpu.py", "rlhf_lab/runtime.py"}
BUSY = ("run_alignlab_probe.py", "dispatch_alignlab_probe.py", "run_alignlab_capacity",
        "run_campaign.py", "local-train", "local-evaluate", "windows.train",
        "train_sft", "train_dpo", "train_ppo", "run_training", "serve_adapter",
        "bootstrap_run.py", "fetch_alignlab_model.py")


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pin(path, expected):
    if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected) or sha(path) != expected:
        raise ValueError(f"Frozen file SHA changed: {path}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def source_inventory():
    return {p.relative_to(ROOT).as_posix() for folder in ("alignment_lab", "tests_v2")
            for p in (ROOT / folder).glob("*.py")} | CLI_FILES


def _prerequisites(items):
    if not isinstance(items, list) or not items:
        raise ValueError("Explicit priority prerequisite files and states required")
    for item in items:
        if set(item) != {"path", "sha256", "json_equals"} or not isinstance(item["json_equals"], dict):
            raise ValueError("Malformed priority prerequisite")
        _pin(item["path"], item["sha256"])
        if item["json_equals"]:
            value = _parse(item["path"])
            for dotted, expected in item["json_equals"].items():
                actual = value
                for key in dotted.split("."):
                    actual = actual[key]
                if type(actual) is not type(expected) or actual != expected:
                    raise ValueError(f"Priority prerequisite state changed: {dotted}")


def validate_plan(path, expected_sha):
    """Read and verify the externally pinned immutable plan without CUDA."""
    _pin(path, expected_sha)
    plan = _parse(path)
    fields = {"schema_version", "kind", "formal_training", "output", "wall_seconds", "poll_seconds",
              "source_sha256", "model_snapshot", "model_manifest_sha256", "tokenizer_dir",
              "audit_dir", "audit_manifest_sha256", "cpu_check_report", "cpu_check_report_sha256",
              "initial_probe_acceptance", "initial_probe_acceptance_sha256", "prerequisites"}
    if set(plan) != fields or plan["schema_version"] != 1 or plan["kind"] != KIND or plan["formal_training"] is not False:
        raise ValueError("Exact versioned capacity-only plan required")
    if (type(plan["wall_seconds"]) is not int or plan["wall_seconds"] != 14335
            or type(plan["poll_seconds"]) is not int or not 1 <= plan["poll_seconds"] <= 60):
        raise ValueError("Capacity has one registered 14335-second remaining budget")
    output = Path(plan["output"])
    if not output.is_absolute() or output.is_symlink() or output.resolve().parent != (ROOT / "runs").resolve():
        raise ValueError("Use a fresh direct child of project runs")
    sources = plan["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != source_inventory():
        raise ValueError("Complete exact implementation/test inventory required")
    for name, digest in sources.items():
        _pin(ROOT / name, digest)
    for directory, filename, key in ((plan["model_snapshot"], "snapshot-manifest.json", "model_manifest_sha256"),
                                      (plan["audit_dir"], "manifest.json", "audit_manifest_sha256")):
        if not Path(directory).is_absolute():
            raise ValueError("Input directories must be absolute")
        _pin(Path(directory) / filename, plan[key])
    if not Path(plan["tokenizer_dir"]).is_absolute():
        raise ValueError("Tokenizer directory must be absolute")
    _pin(plan["cpu_check_report"], plan["cpu_check_report_sha256"])
    cpu = _parse(plan["cpu_check_report"])
    if (cpu.get("status") != "cpu_software_checks_passed" or cpu.get("source_unchanged") is not True
            or cpu.get("cuda_initialized") is not False or cpu.get("device") != "cpu"
            or type(cpu.get("tests_run")) is not int or cpu["tests_run"] < 253
            or any(cpu.get(k) != 0 for k in ("errors", "failures", "skipped"))):
        raise ValueError("Fresh complete CPU checks must pass without CUDA/failures/skips")
    checked = {k.replace("\\", "/"): v for k, v in cpu.get("source_before", cpu.get("source_sha256", {})).items()}
    required = {k for k in sources if k.startswith(("alignment_lab/", "tests_v2/"))} | {"scripts/verify_alignlab_cpu.py"}
    if not required <= set(checked) or any(checked[k] != sources.get(k) for k in checked):
        raise ValueError("CPU source-before hashes differ from frozen campaign sources")
    _pin(plan["initial_probe_acceptance"], plan["initial_probe_acceptance_sha256"])
    prior = _parse(plan["initial_probe_acceptance"])
    if (prior.get("status") != "accepted_initial_short_sft_probe_only"
            or prior.get("all_checks_passed") is not True or prior.get("formal_training") is not False
            or prior.get("capacity_verified") is not False
            or prior.get("model_snapshot", {}).get("manifest_sha256") != plan["model_manifest_sha256"]):
        raise ValueError("Independent initial GPU acceptance required for this snapshot")
    elapsed = prior.get("timing", {}).get("probe_elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or elapsed + plan["wall_seconds"] > 14400:
        raise ValueError("Initial probe time must remain charged against four-hour budget")
    _prerequisites(plan["prerequisites"])
    _pin(path, expected_sha)
    return plan


def _command(record, script, path):
    text = (record.get("command_line") or "").lower().replace("\\", "/")
    return script in text and str(path).lower().replace("\\", "/") in text and "--execute" in text


def _own_processes(processes, plan_path):
    by_pid = {p["pid"]: p for p in processes}
    own = by_pid.get(os.getpid())
    if own is None or not _command(own, "run_alignlab_capacity.py", plan_path):
        raise RuntimeError("Cannot identify supervisor PID/create-time/command")
    result = {own["pid"]: own}
    parent = by_pid.get(own.get("parent_pid"))
    while parent and _command(parent, "run_alignlab_capacity.py", plan_path):
        result[parent["pid"]] = parent
        parent = by_pid.get(parent.get("parent_pid"))
    return result


def _peers(processes, own, known=()):
    allowed = dict(own); allowed.update(dict(known))
    for p in processes:
        if p["pid"] in allowed:
            if not _same_process(p, allowed[p["pid"]]):
                raise RuntimeError("Known process PID reused or command identity changed")
        elif not p.get("command_line") or any(marker in p["command_line"].lower() for marker in BUSY):
            raise RuntimeError(f"Other training/probe/campaign process active: {p}")


def _bundle(directory, expected_sha, payload_files, expected_format):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir() or {p.name for p in directory.iterdir()} != payload_files | {"manifest.json"}:
        raise ValueError("Incomplete or extra artifact/checkpoint files")
    _pin(directory / "manifest.json", expected_sha)
    manifest = _parse(directory / "manifest.json")
    if manifest.get("format") != expected_format or manifest.get("status") not in {"complete", "cache_complete"}:
        raise ValueError("Incomplete artifact/checkpoint manifest")
    if set(manifest.get("artifacts", {})) != payload_files:
        raise ValueError("Artifact inventory mismatch")
    for name, item in manifest["artifacts"].items():
        _pin(directory / name, item["sha256"])
        if (directory / name).stat().st_size != item["size_bytes"]:
            raise ValueError("Artifact size mismatch")
    if "identity" in manifest and hashlib.sha256(_canonical(manifest["identity"])).hexdigest() != manifest.get("identity_sha256"):
        raise ValueError("Artifact identity self-hash mismatch")
    return manifest


def _verify_stage(output, request, known, code):
    if code != 0:
        raise RuntimeError(f"Capacity child exited {code}; no retry")
    result = _parse(output / "progress.json")
    stage, count, updates = request["stage"], COUNTS[request["stage"]], UPDATES[request["stage"]]
    if (result.get("status") != "capacity_stage_passed" or result.get("stage_capacity_passed") is not True
            or result.get("capacity_verified") is not False or result.get("formal_training") is not False
            or result.get("stage") != stage or result.get("request") != request or result.get("pid") not in known
            or type(result.get("finished_at_epoch")) not in (int, float)
            or not result["started_at_epoch"] <= result["finished_at_epoch"] <= request["deadline_epoch"]
            or result.get("deadline_epoch") != request["deadline_epoch"]
            or result.get("actual_optimizer_updates") != updates
            or result.get("recovery_optimizer_updates") != int(stage == "ppo")):
        raise RuntimeError("Child lacks exact completed request/process/update/deadline evidence")
    if _parse(output / "request.json") != request:
        raise ValueError("Saved child request changed")
    model = result.get("model", {})
    if (model.get("device") != "cuda:0" or model.get("quantization") != NF4
            or model.get("quantized_linear_count") != 196
            or type(model.get("parameters_selected")) is not int or model["parameters_selected"] <= 0
            or model.get("snapshot_manifest_sha256") != request["model_manifest_sha256"]):
        raise RuntimeError("Missing actual CUDA NF4 model evidence")
    samples = result.get("gpu_samples")
    if not isinstance(samples, list) or not samples or any(
            not sample.get("gpus") or not all(type(sample.get(k)) in (int, float)
                and math.isfinite(sample[k]) and sample[k] > 0
                for k in ("max_allocated_bytes", "max_reserved_bytes")) for sample in samples):
        raise RuntimeError("Missing finite positive measured GPU memory evidence")
    audits = result.get("frozen_full_audits", [])
    expected_roles = ["reference", "reward"] if stage in {"ppo", "eval"} else ["reference"] if stage == "dpo" else []
    if (not audits or audits[-1].get("roles") != expected_roles
            or audits[-1].get("reward_head_content_verified") is not (stage in {"ppo", "eval"})
            or not result["started_at_epoch"] <= audits[-1]["time_epoch"] <= result["finished_at_epoch"]
            or (stage == "dpo" and audits[-1].get("cache_manifest_sha256") != result.get("reference_cache_manifest_sha256"))):
        raise RuntimeError("Missing final full frozen-reference/reward audit")
    selection = result["selection"]
    synthetic = int(stage in {"sft", "reward", "dpo"})
    if (selection.get("selected_rows") != count or selection.get("configured_rows") != count
            or selection.get("synthetic_rows") != synthetic or selection.get("real_rows") != count - synthetic
            or selection.get("accepted_manifest_sha256") != request["audit_manifest_sha256"]):
        raise RuntimeError("Capacity selection evidence mismatch")
    if synthetic:
        shape = selection["synthetic_stress"]
        if (shape.get("prompt_tokens") != 1024 or shape.get("chosen_tokens") != 768
                or shape.get("padding_tokens") != 0 or shape.get("attention_all_true") is not True
                or (stage != "sft" and shape.get("rejected_tokens") != 768)):
            raise RuntimeError("Missing attended maximum training shape")
    identity = result["identity"]
    if set(identity["parents"]) != set(request["parents"]):
        raise ValueError("Stage parent identities differ")
    for parent, item in request["parents"].items():
        record = identity["parents"][parent]
        if Path(record["path"]).resolve() != Path(item["artifact_dir"]).resolve() / "manifest.json" or record["sha256"] != item["manifest_sha256"]:
            raise ValueError("Stage parent identity not anchored to admitted artifact")
    inputs = identity["inputs"]
    for name, digest in request["source_sha256"].items():
        item = inputs.get("source/" + name, {})
        if item.get("sha256") != digest or Path(item.get("path", "")).resolve() != (ROOT / name).resolve():
            raise ValueError("Stage did not bind the complete frozen source inventory")
    for label, path, digest in (("capacity_plan", request["capacity_plan_path"], request["capacity_plan_sha256"]),
            ("data/manifest", Path(request["audit_dir"]) / "manifest.json", request["audit_manifest_sha256"])):
        item = inputs.get(label, {})
        if item.get("sha256") != digest or Path(item.get("path", "")).resolve() != Path(path).resolve():
            raise ValueError("Stage input manifest identity mismatch")
    snapshot = _parse(Path(request["model_snapshot"]) / "snapshot-manifest.json")
    for name, pinned in snapshot["files"].items():
        item = inputs.get("base/" + name, {})
        if item.get("sha256") != pinned["sha256"] or Path(item.get("path", "")).resolve() != Path(request["model_snapshot"]).resolve() / name:
            raise ValueError("Stage base identity differs from verified snapshot")
    # Base bytes were checked independently by each child; check remaining
    # source/data/tokenizer/parent identities again after it has exited.
    for name, item in inputs.items():
        if not name.startswith("base/"):
            _pin(item["path"], item["sha256"])
            if Path(item["path"]).stat().st_size != item["size_bytes"]:
                raise ValueError("Stage input size changed")
    if stage == "eval":
        from .capacity_stage import verify_generation
        verify_generation(result["generation"]["eos_aware"], forced=False)
        verify_generation(result["generation"]["forced_shape"], forced=True)
    else:
        cursor = result["cursor"]
        if (result.get("outcome_stop_reason") != "data_exhausted" or cursor["offset"] != count
                or cursor["optimizer_updates"] != updates or cursor["active_rollout"]
                or not result.get("changed_parameter_names")):
            raise RuntimeError("Incomplete real optimizer work")
        heads = {"reward": True} if stage == "reward" else {"value": True} if stage == "ppo" else {}
        if result.get("adapter_weights_changed") is not True or result.get("scalar_heads_changed") != heads:
            raise RuntimeError("Actual adapter/scalar-head weights did not change")
        metrics = [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                   for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        objective = [m for m in metrics if m.get("event") == "objective"]
        if [m.get("update") for m in objective] != list(range(1, updates + 1)) or any(
                not all(type(m.get(k)) in (int, float) and math.isfinite(m[k]) for k in ("objective", "gradient_norm")) for m in objective):
            raise RuntimeError("Missing finite measured update objectives")
        artifact = result["artifact"]
        if Path(artifact["artifact_dir"]).resolve() != output / "final-artifact" or Path(artifact["manifest_path"]).resolve() != output / "final-artifact/manifest.json":
            raise ValueError("Artifact escaped its owned stage output")
        manifest = _bundle(output / "final-artifact", artifact["manifest_sha256"], {"weights.pt"}, "alignlab-completed-stage-v1")
        completion = manifest["completion"]
        if (manifest["stage"] != stage or manifest["identity"] != identity
                or completion.get("protocol_complete") is not True or completion.get("data_pass_complete") is not True
                or completion.get("stop_reason") != "data_exhausted"
                or completion.get("optimizer_updates") != updates or completion.get("rows_processed") != count):
            raise ValueError("Final artifact does not attest the completed selected work")
        expected_updates = {1, 2, 256} if stage == "ppo" else {updates}
        checkpoints = result["checkpoints"]
        if {Path(c["checkpoint_path"]).name for c in checkpoints} != {f"update-{u:06d}" for u in expected_updates} or len(checkpoints) != len(expected_updates):
            raise ValueError("Required checkpoint boundaries missing")
        for item in checkpoints:
            directory = Path(item["checkpoint_path"]).resolve()
            if directory.parent != output:
                raise ValueError("Checkpoint escaped its stage")
            saved = _bundle(directory, item["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
            if saved["identity"] != identity:
                raise ValueError("Checkpoint identity changed")
        if stage == "dpo":
            _bundle(output / "reference-cache", result["reference_cache_manifest_sha256"],
                    {"identity.json", "scores.jsonl"}, "alignlab-reference-logprobs-v1")
        if stage == "ppo":
            recovery, stress = result["recovery"], result["synthetic_backward"]
            if (cursor["completed_rollouts"] != 8 or selection["real_unique_prompts"] != 128
                    or recovery.get("verified") is not True or recovery.get("cursor_oldstats_rng_exact") is not True
                    or recovery.get("final_artifact_exported_before_replay") is not True
                    or recovery.get("parameters_optimizer_rtol") != 2e-5
                    or recovery.get("parameters_optimizer_atol") != 2e-6
                    or stress.get("parameters_unchanged") is not True or stress.get("optimizer_updates") != 0
                    or stress.get("prompt_tokens") != 1024 or stress.get("action_tokens") != 256):
                raise RuntimeError("Missing PPO eight-rollout/recovery/maximum-shape evidence")
            checkpoints = {Path(c["checkpoint_path"]).name: c["manifest_sha256"] for c in checkpoints}
            if recovery["resume_checkpoint"] != checkpoints["update-000001"] or recovery["continuous_checkpoint"] != checkpoints["update-000002"]:
                raise ValueError("Recovery checkpoint pins mismatch")
            replay = Path(recovery["replay_checkpoint_path"]).resolve()
            if replay != output / "replayed-update-000002":
                raise ValueError("Recovery checkpoint escaped its stage")
            _bundle(replay, recovery["replay_checkpoint"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
    return result


def run_capacity(plan_path, expected_plan_sha256, *, execute=False, system=None):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(plan_path, expected_plan_sha256)
    if not execute:
        return {"status": "plan_only", "plan": plan, "capacity_verified": False, "formal_training": False}
    system, output = system or _System(), Path(plan["output"]).resolve()
    lock, gpu_lock = ROOT / "reports/.alignlab-capacity.lock", ROOT / "reports/.alignlab-gpu.lock"
    if output.exists() or lock.exists() or gpu_lock.exists():
        raise FileExistsError("Existing output or capacity/GPU lock; never reuse or remove unknown locks")
    processes = system.processes(); own = _own_processes(processes, plan_path); _peers(processes, own)
    admission = system.resources()
    if admission["blockers"]:
        return {"status": "blocked_before_start", "resources": admission, "capacity_verified": False,
                "formal_training": False, "started_at_epoch": None}
    if Path(str(plan_path) + ".stop").exists():
        return {"status": "stopped_before_start", "capacity_verified": False, "formal_training": False}
    token = {"owner": KIND, "pid": os.getpid(), "plan_sha256": expected_plan_sha256, "token": uuid.uuid4().hex}
    _new(lock, token)
    started = system.now()
    state = {"kind": KIND, "status": "starting", "plan_path": str(plan_path), "plan_sha256": expected_plan_sha256,
             "started_at_epoch": started, "deadline_epoch": started + plan["wall_seconds"],
             "capacity_verified": False, "formal_training": False, "supervisor_processes": list(own.values()),
             "resources_before": admission, "stages": [], "prior_probe_seconds": _parse(plan["initial_probe_acceptance"])["timing"]["probe_elapsed_seconds"]}
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
                if gpu_lock.exists():
                    raise FileExistsError("Foreign or retained GPU lock")
                admission = system.resources()
                if not admission["blockers"]:
                    break
                write("waiting_resources", resources=admission)
                remaining = state["deadline_epoch"] - system.now()
                if remaining > 0:
                    system.sleep(min(plan["poll_seconds"], remaining))
            parents = {name: {k: accepted[name]["artifact"][k] for k in ("artifact_dir", "manifest_sha256")} for name in PARENTS[stage]}
            for parent in parents.values():
                _bundle(parent["artifact_dir"], parent["manifest_sha256"], {"weights.pt"}, "alignlab-completed-stage-v1")
            request = {k: plan[k] for k in ("source_sha256", "model_snapshot", "model_manifest_sha256",
                                          "tokenizer_dir", "audit_dir", "audit_manifest_sha256")}
            request.update(schema_version=1, kind="alignlab-full-capacity-stage-v1", stage=stage,
                output=str(output / stage), deadline_epoch=state["deadline_epoch"], parents=parents,
                capacity_plan_path=str(plan_path), capacity_plan_sha256=expected_plan_sha256)
            path = output / f"{stage}.request.json"; _new(path, request)
            args = [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B", "-u",
                    "scripts/run_alignlab_capacity_stage.py", "--request", str(path),
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
                                if _command(p, "run_alignlab_capacity_stage.py", path) and _stamp(p["created_at_utc"]).timestamp() >= record["launch_epoch"]:
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
                        if gpu_lock.exists():
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
        if gpu_lock.exists():
            raise FileExistsError("GPU lock remains after all child exits")
        if stop_reason():
            write(stop_reason()); return state
        write("capacity_passed", capacity_verified=True)
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
