"""One serial primary-evaluation allocation; no automatic resume or retry.

The CPU coordinator has a deliberately distinct ``supervise_`` entrypoint.
Frozen GPU workers retain their original admission checks and GPU lock. This
module never loads a model, deserializes tensors, or computes model scores.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import uuid

from .probe_dispatch import _System, _same_process, _stamp
from .evaluation_protocol_v2 import EvaluationProtocol, canonical, parse, verified_bytes
from .evaluation_resources_v10 import load_profile, collect_resources, active_service_processes
from .evaluation_supervisor_v9 import NEW_SOURCES as PREVIOUS_SOURCES
from .evaluation_inheritance_v10 import SLOTS as INHERITED_SLOTS, BINDING_FIELDS
from .evaluation_atomic_status_v10 import AtomicStatusPublisher

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-primary-evaluation-campaign-v2"
ENTRYPOINT = "scripts/supervise_alignlab_evaluation_v10.py"
INFERENCE_ENTRYPOINT = "scripts/run_alignlab_evaluation_stage_v10.py"
SCORE_ENTRYPOINT = "scripts/score_alignlab_evaluation_v6.py"
ANALYSIS_ENTRYPOINT = "scripts/summarize_alignlab_evaluation_v10.py"
DECODER_ENTRYPOINT = "scripts/decode_alignlab_evaluation_v2.py"
ENCODER_ENTRYPOINT = "scripts/encode_alignlab_evaluation_capacity_v2.py"
MODEL_PYTHON = ".runtime/alignlab-v2/Scripts/python.exe"
SCORE_PYTHON = ".runtime/alignlab-eval-20261020/Scripts/python.exe"
POLICIES = ("base", "sft", "dpo42", "ppo42", "dpo43", "ppo43", "dpo44", "ppo44")
TASKS = tuple(("ifeval", p) for p in POLICIES) + (("helpsteer_rm", "reward"), ("rm_bench", "reward"))
NEW_SOURCES = PREVIOUS_SOURCES | frozenset({
    "alignment_lab/evaluation_supervisor_v10.py", ENTRYPOINT, "tests_v2/test_evaluation_supervisor_v10.py",
    "scripts/encode_alignlab_evaluation_capacity_v2.py",
    "alignment_lab/evaluation_resources_v10.py", "tests_v2/test_evaluation_resources_v10.py",
    "scripts/prepare_alignlab_evaluation_v10.py", "scripts/verify_alignlab_evaluation_v10.py",
    "tests_v2/test_evaluation_prepare_v10.py",
    "alignment_lab/evaluation_model_v5.py", "tests_v2/test_evaluation_model_v5.py",
    "alignment_lab/evaluation_attention_v5.py", "tests_v2/test_evaluation_attention_v5.py",
    "alignment_lab/evaluation_attention_receipt_v5.py",
    "alignment_lab/evaluation_inheritance_v10.py", "tests_v2/test_evaluation_inheritance_v10.py",
    "alignment_lab/evaluation_capacity_inheritance_v10.py", "tests_v2/test_evaluation_capacity_inheritance_v10.py",
    "alignment_lab/official_evaluation_v6.py", "scripts/score_alignlab_evaluation_v6.py",
    "tests_v2/test_official_evaluation_v6.py", "tests_v2/test_scorer_v6.py",
    "tests_v2/test_scorer_v10.py",
    "alignment_lab/evaluation_atomic_status_v10.py", "tests_v2/test_evaluation_atomic_status_v10.py",
    "alignment_lab/evaluation_status_semantics_v10.py", "tests_v2/test_evaluation_status_semantics_v10.py",
})
FIELDS = frozenset({
    "schema_version", "kind", "output", "wall_seconds", "poll_seconds", "source_sha256",
    "cpu_check_report", "protocol", "roster", "model_python_sha256", "scoring_python_sha256",
    "decoder", "analysis_wall_seconds", "prerequisites",
    "authorization", "resource_profile", "started_at_epoch", "deadline_epoch", "inherited_inference", "inherited_scores", "inherited_capacity",
})
FOREIGN_LOCKS = (".alignlab-gpu.lock", ".alignlab-parents.lock", ".alignlab-alignment.lock",
                 ".alignlab-capacity.lock", ".alignlab-probe-dispatch.lock")
BUSY = ("run_alignlab_", "dispatch_alignlab_", "supervise_alignlab_evaluation_v10.py",
        "score_alignlab_evaluation_v6.py", "summarize_alignlab_evaluation_v10.py",
        "decode_alignlab_evaluation_v2.py", "local-train", "local-evaluate", "run_campaign.py",
        "windows.train", "train_sft", "train_dpo", "train_ppo", "run_training",
        "serve_adapter", "bootstrap_run.py", "fetch_alignlab_model.py")


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read(binding):
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute()):
        raise ValueError("An absolute path and external SHA256 are required")
    return parse(verified_bytes(Path(binding["path"]), binding["sha256"]))


def _pin(path):
    return {"path": str(Path(path).resolve()), "sha256": _sha(path)}


def _new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def _secondary_failure(state, primary, phase, error):
    """Keep the first exception authoritative while cleanup still runs."""
    state.setdefault("secondary_errors", []).append(
        {"phase": phase, "error_type": type(error).__name__, "error": str(error)})
    if primary is not None:
        primary.add_note(f"{phase}: {type(error).__name__}: {error}")


def _source(name):
    if (not isinstance(name, str) or "\\" in name or ":" in name
            or ".." in PurePosixPath(name).parts or PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name or not name.endswith(".py")):
        raise ValueError("Canonical project-relative Python source required")
    path = ROOT / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Source may not traverse symlinks")
    return path


def _inheritance_shape(plan):
    inference, scores = plan.get("inherited_inference"), plan.get("inherited_scores")
    if (not isinstance(inference, dict) or not isinstance(scores, dict)
            or set(inference) != INHERITED_SLOTS or scores != inference):
        raise ValueError("Exactly seven identical explicit inference/score inheritance chains required")
    for entry in inference.values():
        if not isinstance(entry, dict) or set(entry) != BINDING_FIELDS:
            raise ValueError("Seven complete historical inference/score pins required")
        for binding in entry.values():
            _read(binding)
    capacity = plan.get("inherited_capacity")
    if not isinstance(capacity, dict) or set(capacity) != {"prior_plan", "campaign", "closure", "bridge"}:
        raise ValueError("The complete V6 to V9 capacity bridge is mandatory")
    for binding in capacity.values():
        _read(binding)
    for entry in inference.values():
        if any(entry[key] != capacity[key] for key in ("prior_plan", "campaign", "closure")):
            raise ValueError("All inherited results must bind the same closed V6 campaign")


def _static(plan_path, expected_sha, plan=None):
    """Small files are fully hashed at every dispatch and monitor iteration."""
    actual = _read({"path": str(Path(plan_path).resolve()), "sha256": expected_sha})
    if plan is not None and actual != plan:
        raise ValueError("Frozen evaluation plan changed")
    if (set(actual) != FIELDS or type(actual["schema_version"]) is not int
            or actual["schema_version"] != 1 or actual["kind"] != KIND):
        raise ValueError("Exact versioned evaluation campaign plan required")
    for name in ("wall_seconds", "analysis_wall_seconds"):
        if type(actual[name]) is not int or actual[name] <= 0:
            raise ValueError("Explicit positive integer budgets are required")
    if (actual["analysis_wall_seconds"] > actual["wall_seconds"]
            or type(actual["poll_seconds"]) is not int or actual["poll_seconds"] != 10):
        raise ValueError("Analysis must fit within the total budget; poll interval is ten seconds")
    start, deadline = actual["started_at_epoch"], actual["deadline_epoch"]
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, deadline))
            or start <= 0 or deadline - start != actual["wall_seconds"]):
        raise ValueError("One finite absolute allocation must include preparation and waits")
    authority = _read(actual["authorization"])
    if (authority.get("kind") != "alignlab-independent-quality-evaluation-authorization-v2"
            or authority.get("started_at_epoch") != start
            or authority.get("deadline_epoch") != deadline
            or authority.get("wall_seconds") != actual["wall_seconds"]
            or authority.get("automatic_extension") is not False
            or authority.get("reuses_old_budget") is not False
            or authority.get("no_new_training") is not True):
        raise ValueError("Evaluation allocation differs from the independent authorization")
    _inheritance_shape(actual)
    _read(actual["resource_profile"])
    load_profile(actual["resource_profile"])
    output = Path(actual["output"])
    if (not output.is_absolute() or output.resolve().parent != (ROOT / "runs").resolve()
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("Output must be a fresh direct project runs directory")
    from .evaluation_analysis_v10 import REQUIRED_SOURCES
    from .evaluation_acceptance_v2 import SOURCES
    required = REQUIRED_SOURCES | set(SOURCES) | NEW_SOURCES | {"scripts/verify_alignlab_cpu.py"}
    sources = actual["source_sha256"]
    if not isinstance(sources, dict) or not required <= sources.keys():
        raise ValueError("Complete static implementation and CPU-test inventory required")
    for name, digest in sources.items():
        verified_bytes(_source(name), digest)
    cpu = _read(actual["cpu_check_report"])
    if (cpu.get("status") != "cpu_software_checks_passed" or cpu.get("source_unchanged") is not True
            or cpu.get("device") != "cpu" or cpu.get("cuda_initialized") is not False
            or type(cpu.get("tests_run")) is not int or cpu["tests_run"] <= 0
            or any(type(cpu.get(k)) is not int or cpu[k] != 0 for k in ("failures", "errors", "skipped"))):
        raise ValueError("Passing CPU-only proof without skipped checks required")
    from scripts.prepare_alignlab_evaluation_v10 import verify_cpu_proof
    verify_cpu_proof(cpu, sources)
    checked = {k.replace("\\", "/"): v for k, v in
               cpu.get("source_before", cpu.get("source_sha256", {})).items()}
    # The recovery proof pins executable entrypoints as well as imported modules.
    required_checks = set(sources)
    if any(checked.get(k) != sources[k] for k in required_checks):
        raise ValueError("CPU report does not cover the frozen implementation/test inventory")
    for binding in (actual["protocol"], actual["roster"]):
        _read(binding)
    verified_bytes(ROOT / MODEL_PYTHON, actual["model_python_sha256"])
    verified_bytes(ROOT / SCORE_PYTHON, actual["scoring_python_sha256"])
    decoder = actual["decoder"]
    fields = {"python", "python_sha256", "script", "script_sha256",
              "tokenizer_path", "tokenizer_sha256", "tokenizers_version"}
    if (not isinstance(decoder, dict) or set(decoder) != fields
            or Path(decoder["python"]).resolve() != (ROOT / MODEL_PYTHON).resolve()
            or decoder["python_sha256"] != actual["model_python_sha256"]
            or Path(decoder["script"]).resolve() != (ROOT / DECODER_ENTRYPOINT).resolve()
            or decoder["script_sha256"] != sources[DECODER_ENTRYPOINT]
            or not isinstance(decoder["tokenizers_version"], str) or not decoder["tokenizers_version"]):
        raise ValueError("The actual pinned CPU token decoder is required")
    for key in ("python", "script", "tokenizer"):
        path = decoder["tokenizer_path"] if key == "tokenizer" else decoder[key]
        verified_bytes(Path(path), decoder[key + "_sha256"])
    from .capacity_supervisor import _prerequisites
    _prerequisites(actual["prerequisites"])
    return actual


def _full_inputs(plan, system=None):
    from .evaluation_acceptance_v2 import verify_evaluation_roster
    from .evaluation_execution_v10 import _roster
    roster = verify_evaluation_roster(plan["roster"]["path"], plan["roster"]["sha256"], system=system)
    protocol = EvaluationProtocol(canonical(_read(plan["protocol"])))
    protocol.verify_bindings()
    _roster(roster, protocol)
    value = protocol.identity
    if (value.get("scorer_seed") != 0 or value.get("primary_metric") != "ifeval_strict_prompt_accuracy"
            or type(value.get("bootstrap_replicates")) is not int or value["bootstrap_replicates"] < 2000
            or value.get("generation", {}).get("max_new_tokens") != 2048
            or value["generation"].get("input_truncation") is not False):
        raise ValueError("Complete preregistered primary evaluation protocol required")
    if Path(plan["decoder"]["tokenizer_path"]).resolve() != (Path(roster["tokenizer_dir"]) / "tokenizer.json").resolve():
        raise ValueError("Decoder differs from the accepted tokenizer")
    return protocol


def validate_plan(plan_path, expected_sha, *, system=None):
    plan = _static(plan_path, expected_sha)
    _full_inputs(plan, system)
    _static(plan_path, expected_sha, plan)
    from .evaluation_inheritance_v10 import verify_inherited_inference
    from .evaluation_capacity_inheritance_v10 import verify_inherited_capacity
    for binding in plan["inherited_inference"].values():
        verify_inherited_inference(binding, plan, system=system)
    verify_inherited_capacity(plan["inherited_capacity"], plan)
    return plan


def _tokens(command):
    return [v[1:-1] if len(v) >= 2 and v[0] == v[-1] == '"' else v
            for v in shlex.split(command, posix=False)]


def _matches(process, args):
    tokens = _tokens(process.get("command_line") or "")
    target = Path(args[4]).name.lower()
    for i, token in enumerate(tokens):
        if Path(token.replace("\\", "/")).name.lower() == target:
            candidate = Path(token)
            candidate = candidate if candidate.is_absolute() else ROOT / candidate
            if candidate.resolve() == (ROOT / args[4]).resolve():
                return tokens[i + 1:] == args[5:]
    return False


def _own(processes, plan_path, plan_sha):
    args = [str(ROOT / MODEL_PYTHON), "-I", "-B", "-u", ENTRYPOINT,
            "--plan", str(plan_path), "--plan-sha256", plan_sha, "--execute"]
    by_pid = {p["pid"]: p for p in processes}
    current, result = by_pid.get(os.getpid()), {}
    if current is None or not _matches(current, args):
        raise RuntimeError("Cannot identify the actual supervisor process and frozen plan command")
    while current is not None and _matches(current, args):
        if current["pid"] in result:
            raise RuntimeError("Cyclic supervisor ancestry")
        _stamp(current["created_at_utc"])
        result[current["pid"]] = current
        current = by_pid.get(current.get("parent_pid"))
    return result


def _peers(processes, own, known=None):
    allowed = {**own, **(known or {})}
    seen, blockers = set(), []
    for process in processes:
        pid = process.get("pid")
        if type(pid) is not int or pid <= 0 or pid in seen or not process.get("command_line"):
            raise RuntimeError("Unknown or duplicate process identity")
        seen.add(pid)
        _stamp(process["created_at_utc"])
        if pid in allowed:
            if not _same_process(process, allowed[pid]):
                raise RuntimeError("Observed PID/creation/command identity changed")
        else:
            blockers.append(f"Unknown Python process identity: {pid}")
    for pid, expected in own.items():
        if pid not in seen:
            raise RuntimeError("Supervisor or wrapper process disappeared")
    return blockers


class EvaluationSystem(_System):
    def spawn(self, args, stdout, stderr):
        if os.name != "nt":
            raise RuntimeError("Production evaluation requires hidden Windows children")
        script = Path(args[4]).name
        cpu = script in {Path(SCORE_ENTRYPOINT).name, Path(ANALYSIS_ENTRYPOINT).name}
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if cpu else "0",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1")
        return subprocess.Popen(args, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                creationflags=subprocess.CREATE_NO_WINDOW)


def _foreign_locks(*, child=None, known=None):
    for name in FOREIGN_LOCKS:
        path = ROOT / "reports" / name
        if not path.exists():
            continue
        if name == ".alignlab-gpu.lock" and child is not None:
            value = parse(path.read_bytes())
            if (value.get("pid") in (known or {}) and value.get("request_sha256") == child["request_sha256"]
                    and Path(value.get("output", "")).resolve() == Path(child["output"]).resolve()):
                continue
        raise FileExistsError("Unknown or retained lock: " + str(path))


def _historical_prerequisites(plan, output):
    directory = output / "historical-prerequisites"
    directory.mkdir()
    result = []
    for index, item in enumerate(plan["prerequisites"]):
        raw = verified_bytes(Path(item["path"]), item["sha256"])
        path = directory / f"{index:04d}.snapshot"
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        result.append({"original": item, "snapshot": _pin(path),
                       "semantics": "historical_admission_fact_not_current_resource_state"})
    return result


def _request(plan, output, deadline, task=None, model=None):
    value = {"schema_version": 1, "kind": "alignlab-evaluation-inference-stage-v2",
        "output": str(output), "deadline_epoch": deadline, "source_sha256": plan["source_sha256"],
        "protocol_path": plan["protocol"]["path"], "protocol_sha256": plan["protocol"]["sha256"],
        "roster_path": plan["roster"]["path"], "roster_sha256": plan["roster"]["sha256"],
        "resource_profile": plan["resource_profile"], "capacity_manifest": None}
    if task is not None:
        value.update(task=task, model_id=model)
    return value


def _verify_inference(output, request, known):
    """Check complete saved tokens now; independent CPU decoding follows in scoring."""
    from .evaluation_inheritance_v10 import verify_versioned_evaluation_result as verify_evaluation_result
    path = output / "manifest.json"
    manifest = _read(_pin(path))
    if manifest.get("pid") not in known:
        raise ValueError("Inference did not run in an observed child")
    raw = verified_bytes(output / "outputs.jsonl", manifest["files"]["outputs.jsonl"]["sha256"])
    rows = [parse(line) for line in raw.split(b"\n") if line.strip()]
    mapping = {}
    if request["task"] == "ifeval":
        for row in rows:
            ids = row["generated_token_ids"]
            key = tuple(ids[:-1] if ids and ids[-1] == 151645 else ids)
            text = row["response"]
            if key in mapping and mapping[key] != text:
                raise ValueError("Stored identical actions have inconsistent text")
            mapping[key] = text
    verify_evaluation_result(output, expected_manifest_sha256=_sha(path),
        expected_request=request, decode_tokens=mapping.__getitem__ if mapping else None)
    return _pin(path)


def _verify_score(output, request, plan, inference_request, *, plan_binding):
    from .evaluation_analysis_v10 import _scored, _normalize_ifeval
    from .evaluation_inheritance_v10 import verify_versioned_evaluation_result as verify_evaluation_result
    from .evaluation_protocol_v2 import load_evaluation_assets
    binding = _pin(output / "scores.json")
    protocol = EvaluationProtocol(canonical(_read(plan["protocol"])))
    shell = {"protocol": plan["protocol"], "roster": plan["roster"], "evaluation_plan": plan_binding,
             "verification_allocation": {"evaluation_deadline_epoch": request["deadline_epoch"]}}
    slot = inference_request["task"] + "/" + inference_request["model_id"]
    result, actual_request, manifest, rows = _scored(binding, slot, shell, protocol)
    if actual_request != request:
        raise ValueError("CPU scoring used a different dispatched request")
    decode = None
    if inference_request["task"] == "ifeval":
        receipt = result["decoder_receipt"]
        if (receipt.get("sequence_count") != 541 or len(receipt.get("texts", [])) != 541
                or receipt.get("torch_imported") is not False or receipt.get("gpu_operations_performed") is not False):
            raise ValueError("All original action sequences require independent CPU decoding")
        mapping = {}
        for row, text in zip(rows, receipt["texts"]):
            ids = row["generated_token_ids"]
            key = tuple(ids[:-1] if ids[-1] == 151645 else ids)
            if not isinstance(text, str) or key in mapping and mapping[key] != text:
                raise ValueError("Invalid independent decoder receipt")
            mapping[key] = text
        decode = mapping.__getitem__
        asset = protocol.identity["asset_manifest"]
        _normalize_ifeval(result["scores"], load_evaluation_assets(asset["path"], asset["sha256"]).ifeval)
    else:
        expected = ({"all": 2017, "non_ties": 1920, "ties": 97}
                    if inference_request["task"] == "helpsteer_rm"
                    else {"records": 1327, "answer_scores": 7962, "comparisons": 11943})
        if result["scores"]["denominators"] != expected:
            raise ValueError("Complete original reward-evaluation denominator required")
    verify_evaluation_result(Path(request["inference_manifest"]).parent,
        expected_manifest_sha256=request["inference_manifest_sha256"],
        expected_request=inference_request, decode_tokens=decode)
    return binding


def run_evaluation(plan_path, expected_plan_sha256, *, execute=False, system=None):
    """A fresh allocation only. An unready initial admission creates no output."""
    plan_path = Path(plan_path).resolve()
    system = system or EvaluationSystem()
    plan = validate_plan(plan_path, expected_plan_sha256, system=system)
    flags = {"primary_evaluation_complete": False, "research_complete": False,
             "human_labels": 0, "human_win_rate": None, "formal_training": False}
    if not execute:
        return {"status": "plan_only", "plan": plan, **flags}
    output, lock = Path(plan["output"]).resolve(), ROOT / "reports/.alignlab-evaluation.lock"
    if output.exists() or lock.exists():
        raise FileExistsError("Existing campaign output/lock; no retry, resume, overwrite or lock deletion")
    _foreign_locks()
    own = _own(system.processes(), plan_path, expected_plan_sha256)
    profile = load_profile(plan["resource_profile"])
    residents = {p["pid"]: p for p in active_service_processes(profile)}
    residents.update({p["pid"]: p for p in profile["background_processes"]})

    def peer_check(live, known=None):
        return _peers(live, own, {**residents, **(known or {})})

    def resources():
        if isinstance(system, EvaluationSystem):
            return collect_resources(profile, allowed_processes=tuple(own.values()))
        return system.resources()

    initial_processes = system.processes()
    blockers = peer_check(initial_processes)
    resource = resources()
    blockers += resource["blockers"]
    if blockers:
        return {"status": "blocked_before_start", "blockers": blockers,
                "observed_python_processes": initial_processes, "started_at_epoch": None, **flags}
    if Path(str(plan_path) + ".stop").exists():
        return {"status": "stopped_before_start", "started_at_epoch": None, **flags}
    started, deadline = plan["started_at_epoch"], plan["deadline_epoch"]
    if system.now() >= deadline:
        return {"status": "budget_exhausted", "started_at_epoch": started, **flags}
    if system.now() < started:
        raise ValueError("Evaluation allocation has not started")
    token = {"owner": KIND, "pid": os.getpid(), "plan_path": str(plan_path),
        "plan_sha256": expected_plan_sha256, "output": str(output), "started_at_epoch": started,
        "deadline_epoch": deadline, "processes": list(own.values()), "token": uuid.uuid4().hex,
        "resource_profile": plan["resource_profile"]}
    state = {"schema_version": 1, "kind": KIND, "status": "starting", **flags,
        "scope": "primary_instruction_following_and_reward_model_evaluation",
        "plan_path": str(plan_path), "plan_sha256": expected_plan_sha256, "started_at_epoch": started,
        "deadline_epoch": deadline, "supervisor_processes": list(own.values()),
        "working_directory": str(ROOT), "resources_before": resource, "children": [],
        "capacity_verified": False, "scores": {}, "summary": None}
    child, child_finished, current, primary_error = None, True, None, None

    def write(status=None, *, terminal=False, **values):
        if status is not None:
            state["status"] = status
        state.update(values, updated_at_epoch=system.now())
        return publisher.publish(state, terminal=terminal)

    def reason():
        if ((output / "STOP").exists() or Path(str(plan_path) + ".stop").exists()
                or current is not None and (Path(current["output"]) / "STOP").exists()):
            return "paused"
        return "budget_exhausted" if system.now() >= deadline else None

    publisher = AtomicStatusPublisher(output / "campaign.json", deadline_epoch=deadline,
        stop_requested=lambda: reason() == "paused", serialize=canonical,
        now_fn=system.now, sleep_fn=system.sleep)

    def terminal_write(status=None, **values):
        nonlocal primary_error
        try:
            write(status, terminal=True, **values)
        except BaseException as publication_error:
            _secondary_failure(state, primary_error, "terminal_status_publication", publication_error)
            state["status_publication_failure"] = publisher.failure
            (primary_error or publication_error).add_note(
                f"Status publication evidence: {publisher.failure!r}")
            if primary_error is None:
                primary_error = publication_error
                raise

    def error_signal(why):
        try:
            signal(why)
        except BaseException as signal_error:
            _secondary_failure(state, primary_error, "cooperative_stop_signal", signal_error)
            if primary_error is None:
                raise

    def boundary():
        stopped = reason()
        if stopped:
            raise InterruptedError(stopped) if stopped == "paused" else TimeoutError(stopped)
        _static(plan_path, expected_plan_sha256, plan)

    def signal(why):
        if not (output / "STOP").exists():
            _new(output / "STOP", {"owner": KIND, "plan_sha256": expected_plan_sha256, "reason": why})

    def ready(gpu):
        while True:
            boundary()
            _foreign_locks()
            blocked = peer_check(system.processes())
            current_resources = resources() if gpu else {"blockers": []}
            blocked += current_resources["blockers"]
            if not blocked:
                return
            write("waiting_resources", blockers=blocked, resources=current_resources)
            system.sleep(min(10, max(0, deadline - system.now())))

    def dispatch(name, phase, request, script, *, gpu):
        nonlocal child, child_finished, current
        ready(gpu)
        _full_inputs(plan, system)
        boundary()
        request_path, destination = output / (name + ".request.json"), output / name
        if destination.exists():
            raise FileExistsError("Child output already exists; no implicit reuse")
        _new(request_path, request)
        args = [str(ROOT / (MODEL_PYTHON if gpu else SCORE_PYTHON)), "-I", "-B", "-u", script,
                "--request", str(request_path), "--request-sha256", _sha(request_path)]
        args += ["--execute"] if gpu else ["--output", str(destination)]
        current = {"name": name, "phase": phase, "status": "dispatching", "request": request,
            "request_path": str(request_path), "request_sha256": _sha(request_path),
            "output": str(destination), "command": args, "working_directory": str(ROOT),
            "stdout_path": str(output / (name + ".stdout.log")),
            "stderr_path": str(output / (name + ".stderr.log")), "known_processes": []}
        state["children"].append(current)
        with Path(current["stdout_path"]).open("xb") as stdout, Path(current["stderr_path"]).open("xb") as stderr:
            # Admission is repeated immediately before spawn, after potentially long hashing.
            ready(gpu)
            boundary()
            current["launch_epoch"] = system.now()
            child = system.spawn(args, stdout, stderr)
            child_finished = False
            current["wrapper_pid"] = child.pid
            known, monitor_error, stopped = {}, None, None
            while True:
                try:
                    _static(plan_path, expected_plan_sha256, plan)
                except Exception as error:
                    monitor_error = f"{type(error).__name__}: {error}"
                stopped = stopped or reason()
                if stopped or monitor_error:
                    signal(stopped or "identity_failure")
                live = system.processes()
                by_pid = {p["pid"]: p for p in live}
                for _ in range(len(live) + 1):
                    added = False
                    for process in live:
                        if process["pid"] in known or not (
                                process["pid"] == child.pid or process.get("parent_pid") in known):
                            continue
                        decoder = (not gpu and _matches(process, [
                            str(ROOT / MODEL_PYTHON), "-I", "-B", "-u", DECODER_ENTRYPOINT]))
                        encoder = (gpu and phase in {"capacity", "inference"} and process.get("parent_pid") in known
                                   and _matches(process, [str(ROOT / MODEL_PYTHON),
                                                         "-I", "-B", "-u", ENCODER_ENTRYPOINT]))
                        if (not (_matches(process, args) or decoder or encoder)
                                or _stamp(process["created_at_utc"]).timestamp() < current["launch_epoch"]):
                            monitor_error = "Child creation/command/ancestry mismatch"
                            continue
                        known[process["pid"]] = process
                        added = True
                    if not added:
                        break
                try:
                    peer_blockers = peer_check(live, known)
                    if peer_blockers:
                        raise RuntimeError("; ".join(peer_blockers))
                    _foreign_locks(child=current if gpu else None, known=known)
                except Exception as error:
                    monitor_error = monitor_error or f"{type(error).__name__}: {error}"
                if monitor_error:
                    signal("monitoring_failure")
                active = [p for pid, p in known.items()
                          if pid in by_pid and _same_process(by_pid[pid], p)]
                code = child.poll()
                current.update(known_processes=list(known.values()), active_processes=active,
                    monitoring_error=monitor_error, stop_reason=stopped, exit_code=code,
                    observed_python_processes=live)
                # STOP/identity failures retain the normal observation and exit
                # loop. Start the one terminal publication budget only after
                # that loop, not once per ten-second drain observation.
                if not (stopped or monitor_error):
                    write("running")
                if code is not None and not active:
                    child_finished = True
                    current["finished_at_epoch"] = system.now()
                    if monitor_error:
                        raise RuntimeError(monitor_error)
                    if stopped:
                        raise InterruptedError(stopped) if stopped == "paused" else TimeoutError(stopped)
                    if code != 0 or not known:
                        raise RuntimeError("Child failed or exited without observed identity; inspect its logs")
                    _foreign_locks()
                    boundary()
                    _full_inputs(plan, system)
                    boundary()
                    return destination, known
                system.sleep(10)

    _new(lock, token)
    try:
        output.mkdir(exist_ok=False)
        _new(output / "plan.snapshot.json", plan)
        state["historical_prerequisites"] = _historical_prerequisites(plan, output)
        write()
        from .evaluation_capacity_inheritance_v10 import verify_inherited_capacity
        receipt = verify_inherited_capacity(plan["inherited_capacity"], plan)
        bridge = {"kind": "alignlab-evaluation-capacity-inheritance-v10",
                  "plan": _pin(plan_path), "inherited_capacity": plan["inherited_capacity"]}
        _new(output / "capacity-inheritance.json", bridge)
        capacity_binding = _pin(output / "capacity-inheritance.json")
        state["inherited_capacity"] = {"bindings": plan["inherited_capacity"], "bridge": capacity_binding}
        state.update(capacity_verified=True, capacity_inherited=True, capacity=receipt, capacity_manifest=capacity_binding)
        write("capacity_accepted")
        for task, model in TASKS:
            name = task + "-" + model
            slot = task + "/" + model
            if slot in plan["inherited_inference"]:
                from .evaluation_inheritance_v10 import verify_inherited_inference
                inherited = verify_inherited_inference(plan["inherited_inference"][slot], plan, system=system)
                state.setdefault("inherited_inference", {})[slot] = inherited
                state.setdefault("inherited_scores", {})[slot] = inherited
                state["scores"][slot] = inherited["score_binding"]
                write("score_inherited")
                continue
            else:
                request = _request(plan, output / name, deadline, task, model)
                request["capacity_manifest"] = capacity_binding
                destination, known = dispatch(name, "inference", request, INFERENCE_ENTRYPOINT, gpu=True)
                inference = _verify_inference(destination, request, known)
                current.update(status="accepted", result=inference)
                write("inference_accepted")
            score = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-request-v1",
                "inference_manifest": inference["path"], "inference_manifest_sha256": inference["sha256"],
                "source_sha256": plan["source_sha256"], "decoder": plan["decoder"] if task == "ifeval" else None,
                "deadline_epoch": deadline}
            destination, _ = dispatch(name + "-score", "score", score, SCORE_ENTRYPOINT, gpu=False)
            binding = _verify_score(destination, score, plan, request, plan_binding=_pin(plan_path))
            current.update(status="accepted", result=binding)
            state["scores"][task + "/" + model] = binding
            write("score_accepted")
        analysis = {"schema_version": 1, "kind": "alignlab-primary-evaluation-analysis-request-v2",
            "protocol": plan["protocol"], "roster": plan["roster"], "scores": state["scores"],
            "evaluation_plan": _pin(plan_path),
            "source_sha256": plan["source_sha256"],
            "verification_allocation": {"kind": "within_independent_evaluation_budget",
                "wall_seconds": plan["analysis_wall_seconds"], "deadline_epoch": deadline,
                "evaluation_deadline_epoch": deadline}}
        destination, _ = dispatch("analysis", "analysis", analysis, ANALYSIS_ENTRYPOINT, gpu=False)
        from .evaluation_analysis_v10 import verify_analysis
        summary = verify_analysis(destination, _sha(destination / "run_manifest.json"))
        if summary.get("primary_evaluation_complete") is not True or summary.get("research_complete") is not False:
            raise ValueError("Only complete primary evaluation, never whole-research success")
        current.update(status="accepted", result=_pin(destination / "run_manifest.json"))
        boundary()
        _full_inputs(plan, system)
        if peer_check(system.processes()):
            raise RuntimeError("Other evaluation/training process is still active")
        _foreign_locks()
        verify_inherited_capacity(plan["inherited_capacity"], plan)
        from .evaluation_inheritance_v10 import verify_inherited_inference
        for binding in plan["inherited_inference"].values():
            verified_score = verify_inherited_inference(binding, plan, system=system)
            if state["scores"][verified_score["slot"]] != verified_score["score_binding"]:
                raise ValueError("Inherited score changed before completion")
        if set(state["scores"]) != {task + "/" + model for task, model in TASKS}:
            raise ValueError("All ten preregistered scoring slots are required")
        for record in state["children"]:
            _read(record["result"])
            if record["exit_code"] != 0 or record["active_processes"] or record["status"] != "accepted":
                raise ValueError("All related children must exit and pass verification")
        boundary()
        write("primary_evaluation_passed", primary_evaluation_complete=True, summary=summary,
              completed_tasks=len(state["scores"]), processes_exited=True)
        return state
    except (InterruptedError, TimeoutError) as error:
        primary_error = error
        if output.exists():
            if child is not None and not child_finished:
                error_signal("paused" if isinstance(error, InterruptedError) else "budget_exhausted")
                state["child_may_still_be_running"] = True
            terminal_write("paused" if isinstance(error, InterruptedError) else "budget_exhausted",
                           error=f"{type(error).__name__}: {error}")
        return state
    except BaseException as error:
        primary_error = error
        if output.exists():
            if child is not None and not child_finished:
                error_signal("supervisor_error")
                state["child_may_still_be_running"] = True
            terminal_write("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if output.exists():
                finished = system.now()
                if state["status"] == "primary_evaluation_passed" and finished >= deadline:
                    state.update(status="budget_exhausted", primary_evaluation_complete=False)
                state.update(finished_at_epoch=finished, elapsed_seconds=finished - started)
                terminal_write()
        finally:
            try:
                if child_finished and lock.exists() and parse(lock.read_bytes()) == token:
                    lock.unlink()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                _secondary_failure(state, primary_error, "exact_owner_lock_cleanup", cleanup_error)
