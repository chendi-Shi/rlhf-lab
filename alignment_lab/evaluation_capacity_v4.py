"""One-shot full-context evaluation capacity; never a quality measurement.

Import and receipt verification use only the standard library. CUDA/model
imports are lazy, after a pinned campaign, live ancestry, resources and locks
have been checked. Factories are CPU-test seams, not CLI execution alternatives.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from . import evaluation_execution_v4 as ex
from . import evaluation_acceptance_v2 as acceptance
from .evaluation_protocol_v2 import canonical, digest, parse, verified_bytes, _finite, _relative
from .evaluation_resources_v4 import (load_profile, collect_resources, configure_cuda_allocator,
                                    verify_allocator_receipt, RuntimeResourceGuard, MIB)

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-evaluation-capacity-stage-v2"
RESULT_KIND = "alignlab-evaluation-capacity-result-v2"
CAMPAIGN_KIND = "alignlab-primary-evaluation-campaign-v2"
SOURCES = ("alignment_lab/evaluation_capacity_v4.py",
           "scripts/run_alignlab_evaluation_capacity_v4.py",
           "tests_v2/test_evaluation_capacity_v4.py",
           "scripts/encode_alignlab_evaluation_capacity_v2.py")
REQUIRED_SOURCES = frozenset(SOURCES) | ex.REQUIRED_SOURCES | frozenset(acceptance.SOURCES)
ENCODER_SOURCE = "scripts/encode_alignlab_evaluation_capacity_v2.py"
ENCODER_SOURCES = (ENCODER_SOURCE, "alignment_lab/chat_codec.py", "alignment_lab/__init__.py")
TOKENIZER_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
                   "tokenizer.json", "merges.txt", "vocab.json")
MODEL_PYTHON = ".runtime/alignlab-v2/Scripts/python.exe"
REQUEST_FIELDS = {"schema_version", "kind", "output", "deadline_epoch",
                  "evaluation_plan_path", "evaluation_plan_sha256",
                  "protocol_path", "protocol_sha256", "roster_path", "roster_sha256",
                  "source_sha256", "resource_profile", "capacity_manifest"}
FILES = {"request.snapshot.json", "protocol.snapshot.json", "roster.snapshot.json", "progress.json"}
MODEL_IDS = tuple(ex.MODEL_STAGES)
EOS, HORIZON = 151645, 2048
BUSY = ("run_alignlab_", "dispatch_alignlab_", "local-train", "local-evaluate",
        "run_campaign.py", "windows.train", "train_sft", "train_dpo", "train_ppo",
        "bootstrap_run.py", "supervise_alignlab_evaluation_v4.py")


def _read(path, sha):
    return parse(verified_bytes(Path(path), sha))


def _sources(sources):
    if not isinstance(sources, dict) or not REQUIRED_SOURCES <= sources.keys():
        raise ValueError("Complete capacity and transitive implementation pins required")
    for name, sha in sources.items():
        verified_bytes(_relative(ROOT, name), sha)


def _request(request):
    if (not isinstance(request, dict) or set(request) != REQUEST_FIELDS
            or type(request["schema_version"]) is not int or request["schema_version"] != 1
            or request["kind"] != KIND):
        raise ValueError("Exact evaluation capacity request schema required")
    _finite(request["deadline_epoch"], "Shared absolute deadline")
    if request["capacity_manifest"] is not None:
        raise ValueError("Fresh capacity cannot reuse a previous capacity receipt")
    for key in ("output", "evaluation_plan_path", "protocol_path", "roster_path"):
        path = Path(request[key])
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Absolute non-symlink identity paths required")
    output = Path(request["output"]).resolve()
    plan = _read(request["evaluation_plan_path"], request["evaluation_plan_sha256"])
    if (plan.get("schema_version") != 1 or plan.get("kind") != CAMPAIGN_KIND
            or output != Path(plan["output"]).resolve() / "capacity"
            or not output.is_relative_to(ROOT / "runs")
            or plan.get("source_sha256") != request["source_sha256"]
            or plan.get("protocol") != {"path": request["protocol_path"], "sha256": request["protocol_sha256"]}
            or plan.get("roster") != {"path": request["roster_path"], "sha256": request["roster_sha256"]}):
        raise ValueError("Capacity must be the pinned campaign's direct child with identical identities")
    wall = _finite(plan.get("wall_seconds"), "Campaign wall budget")
    if wall <= 0 or plan.get("poll_seconds") != 10:
        raise ValueError("Positive frozen campaign budget and ten-second polling required")
    _sources(request["source_sha256"])
    if plan.get("resource_profile") != request["resource_profile"]:
        raise ValueError("Capacity resource profile differs from frozen campaign")
    load_profile(request["resource_profile"])
    return output, plan


def _context(request):
    # Existing protocol validator checks all complete denominators and bindings.
    inference_request = {key: request[key] for key in ex.REQUEST_FIELDS
                         if key not in {"kind", "task", "model_id", "capacity_manifest"}}
    inference_request.update(kind=ex.KIND, task="helpsteer_rm", model_id="reward", capacity_manifest=None)
    return ex._context(inference_request)


def _verify_roster(request, system):
    return acceptance.verify_evaluation_roster(request["roster_path"], request["roster_sha256"], system=system)


def _argv(command):
    if not isinstance(command, str) or not command:
        raise RuntimeError("Unknown process command")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        shell = ctypes.windll.shell32
        shell.CommandLineToArgvW.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
        shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        count = ctypes.c_int()
        pointer = shell.CommandLineToArgvW(command, ctypes.byref(count))
        if not pointer:
            raise RuntimeError("Cannot parse observed Windows command")
        try:
            return [pointer[i] for i in range(count.value)]
        finally:
            ctypes.windll.kernel32.LocalFree(ctypes.cast(pointer, ctypes.c_void_p))
    return shlex.split(command)


def _command(process, script, arguments):
    tokens = _argv(process.get("command_line"))
    matches = [i for i, token in enumerate(tokens) if Path(token).name == script]
    return (len(matches) == 1 and Path(tokens[matches[0]]).resolve() == ROOT / "scripts" / script
            and tokens[matches[0] + 1:] == arguments)


def _signature(process):
    from .probe_dispatch import _stamp
    if type(process.get("pid")) is not int or type(process.get("parent_pid")) is not int:
        raise RuntimeError("Unknown PID or ancestry")
    return (process["pid"], process["parent_pid"], _stamp(process["created_at_utc"]), process["command_line"])


def _admit(request, request_path, request_sha, plan, system, *, expected=None, resources=False, allocator_receipt=None):
    lock_path = ROOT / "reports/.alignlab-evaluation.lock"
    lock = parse(lock_path.read_bytes())
    if (lock.get("owner") != CAMPAIGN_KIND or lock.get("plan_path") != request["evaluation_plan_path"]
            or lock.get("plan_sha256") != request["evaluation_plan_sha256"]
            or lock.get("output") != plan["output"] or not isinstance(lock.get("token"), str)
            or lock.get("resource_profile") != request["resource_profile"]
            or not lock["token"] or lock.get("deadline_epoch") != request["deadline_epoch"]
            or lock["deadline_epoch"] != _finite(lock["started_at_epoch"], "Campaign start") + plan["wall_seconds"]
            or lock["started_at_epoch"] > system.now()):
        raise RuntimeError("Live campaign lock/deadline differs from pinned request")
    live = system.processes()
    by_pid = {p["pid"]: p for p in live}
    if len(by_pid) != len(live) or os.getpid() not in by_pid:
        raise RuntimeError("Unique observed current process required")
    own = by_pid[os.getpid()]
    arguments = ["--request", str(request_path), "--request-sha256", request_sha, "--execute"]
    if not _command(own, "run_alignlab_evaluation_capacity_v4.py", arguments):
        raise RuntimeError("Capacity process arguments differ from external request")
    owned = [own]
    parent = by_pid.get(own["parent_pid"])
    if parent and _command(parent, "run_alignlab_evaluation_capacity_v4.py", arguments):
        owned.append(parent)
        parent = by_pid.get(parent["parent_pid"])
    supervisors = lock.get("processes")
    if (not isinstance(supervisors, list) or not 1 <= len(supervisors) <= 2
            or len({p.get("pid") for p in supervisors}) != len(supervisors)
            or lock.get("pid") not in {p["pid"] for p in supervisors}):
        raise RuntimeError("Exact supervisor and optional venv-wrapper identity required")
    ancestry = set()
    cursor = parent
    while cursor and cursor["pid"] not in ancestry:
        ancestry.add(cursor["pid"])
        cursor = by_pid.get(cursor["parent_pid"])
    for recorded in supervisors:
        actual = by_pid.get(recorded["pid"])
        if (actual is None or _signature(actual) != _signature(recorded)
                or actual["pid"] not in ancestry
                or not _command(actual, "supervise_alignlab_evaluation_v4.py",
                    ["--plan", request["evaluation_plan_path"], "--plan-sha256",
                     request["evaluation_plan_sha256"], "--execute"])):
            raise RuntimeError("Supervisor PID/create/command or ancestral ownership differs")
    if parent is None or parent["pid"] != lock["pid"]:
        raise RuntimeError("Capacity must descend directly from the recorded supervisor")
    observed = owned + supervisors
    allowed = {p["pid"] for p in observed}
    for process in live:
        _signature(process)
        command = (process.get("command_line") or "").lower()
        if not command:
            raise RuntimeError("Unknown Python process identity")
        if process["pid"] not in allowed and any(marker in command for marker in BUSY):
            raise RuntimeError(f"Other training/evaluation process: {process['pid']}")
    if expected is not None and {_signature(p) for p in expected} != {_signature(p) for p in observed}:
        raise RuntimeError("Capacity or supervisor process identity changed")
    resource = collect_resources(load_profile(request["resource_profile"]),
        allowed_processes=observed, process_reader=system.processes,
        allocator_receipt=allocator_receipt) if resources else None
    if resource is not None and resource.get("blockers") != []:
        raise RuntimeError("Evaluation capacity resources unavailable: " + str(resource.get("blockers")))
    return resource, observed, lock


def select_maximum_inputs(context):
    """Stable first-in-official-order ties; only accepted serialized lengths."""
    result = {}
    datasets = {"ifeval": context.assets.ifeval, "helpsteer_rm": context.dataset,
                "rm_bench": context.assets.rm_bench}
    for task, data in datasets.items():
        best = None
        accepted_lengths = data.lengths
        for source in data.rows:
            key = source[data.id_field]
            lengths = accepted_lengths[str(key) if task == "ifeval" else key]
            flat = [lengths] if task == "ifeval" else lengths if task == "helpsteer_rm" else lengths["chosen"] + lengths["rejected"]
            for index, length in enumerate(flat):
                if type(length) is not int or length <= 0:
                    raise ValueError("Positive accepted input lengths required")
                if best is None or length > best["input_tokens"]:
                    best = {"task": task, "record_id": key, "response_index": None if task == "ifeval" else index,
                            "input_tokens": length, "source_record_sha256": digest(source),
                            "dataset_identity": data.identity}
        if best is None:
            raise ValueError("Empty official evaluation dataset")
        result[task] = best
    return {"algorithm": "maximum_accepted_input_tokens_first_official_order_v1",
            "quality_selection_allowed": False, "input_truncation": False,
            "cases": result}


def _encode_selection(context, selection, codec):
    encoded = {}
    for task, data in (("ifeval", context.assets.ifeval), ("helpsteer_rm", context.dataset),
                       ("rm_bench", context.assets.rm_bench)):
        item = selection["cases"][task]
        source = next(row for row in data.rows if row[data.id_field] == item["record_id"])
        prompt, completions = ex._encoded(source, task, codec)
        completion = () if task == "ifeval" else completions[item["response_index"]]
        if len(prompt) + len(completion) != item["input_tokens"]:
            raise ValueError("Full selected input differs from accepted tokenizer lengths")
        encoded[task] = {"prompt_ids": list(prompt), "completion_ids": list(completion),
                         "prompt_tokens": len(prompt), "completion_tokens": len(completion),
                         "tokens_sha256": digest([list(prompt), list(completion)])}
    return encoded


def _encoder_payload(context, request, plan, selection):
    """Bind raw selected source text and all tokenizer-affecting local files."""
    roster = context.roster
    snapshot = _read(Path(roster["model_snapshot"]) / "snapshot-manifest.json", roster["model_manifest_sha256"])
    token_dir = Path(roster["tokenizer_dir"]).resolve()
    allowed = set(TOKENIZER_FILES) | {"README.md", "LICENSE", "source-manifest.json"}
    if ({p.name for p in token_dir.iterdir()} - allowed
            or any(not p.is_file() or p.is_symlink() for p in token_dir.iterdir())):
        raise ValueError("Unexpected or unsafe tokenizer file can affect independent encoding")
    token_files = {name: snapshot["files"][name]["sha256"] for name in TOKENIZER_FILES}
    for name, sha in token_files.items():
        verified_bytes(token_dir / name, sha)
    source_files = {name: request["source_sha256"][name] for name in ENCODER_SOURCES}
    for name, sha in source_files.items():
        verified_bytes(ROOT / name, sha)
    python = ROOT / MODEL_PYTHON
    verified_bytes(python, plan["model_python_sha256"])
    decoder = plan["decoder"]
    if (Path(decoder["tokenizer_path"]).resolve() != token_dir / "tokenizer.json"
            or decoder["tokenizer_sha256"] != token_files["tokenizer.json"]):
        raise ValueError("Independent encoder must use the protocol's exact tokenizer")
    cases = {}
    for task, data in (("ifeval", context.assets.ifeval), ("helpsteer_rm", context.dataset),
                       ("rm_bench", context.assets.rm_bench)):
        selected = selection["cases"][task]
        source = next(row for row in data.rows if row[data.id_field] == selected["record_id"])
        history = source["context"] if task == "helpsteer_rm" else [{"role": "user", "content": source["prompt"]}]
        answers = (None if task == "ifeval" else [source["response1"], source["response2"]]
                   if task == "helpsteer_rm" else source["chosen"] + source["rejected"])
        cases[task] = {"context": history,
                       "response": None if answers is None else answers[selected["response_index"]]}
    return {"schema_version": 1, "kind": "alignlab-capacity-input-encoding-request-v1",
            "python_path": str(python.resolve()), "python_sha256": plan["model_python_sha256"],
            "source_sha256": source_files, "tokenizer_dir": str(token_dir),
            "tokenizer_files": token_files, "tokenizers_version": decoder["tokenizers_version"],
            "transformers_version": "5.8.1", "cases": cases}


def _invoke_encoder(payload, *, deadline_epoch=None):
    """Bounded isolated CPU process; the caller supplies a live campaign deadline."""
    limit = time.time() + 120 if deadline_epoch is None else min(deadline_epoch, time.time() + 120)
    remaining = limit - time.time()
    if remaining <= 0:
        raise TimeoutError("Shared evaluation deadline expired before independent encoding")
    python, script = Path(payload["python_path"]), ROOT / ENCODER_SOURCE
    verified_bytes(python, payload["python_sha256"])
    for name, sha in payload["source_sha256"].items():
        verified_bytes(ROOT / name, sha)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", USE_TORCH="0", USE_TF="0",
                       USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    child = subprocess.run([str(python), "-I", "-B", str(script)], input=canonical(payload),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=remaining, env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if child.returncode:
        raise RuntimeError("Independent CPU input encoder failed: " + child.stderr.decode("utf-8", errors="replace")[-2000:])
    if time.time() >= limit:
        raise TimeoutError("Independent CPU input encoder exceeded its deadline")
    result = parse(child.stdout)
    for name, sha in payload["source_sha256"].items():
        verified_bytes(ROOT / name, sha)
    verified_bytes(python, payload["python_sha256"])
    return result


def _reencode_inputs(context, request, plan, selection, encoded, *, deadline_epoch=None):
    payload = _encoder_payload(context, request, plan, selection)
    receipt = _invoke_encoder(payload, deadline_epoch=deadline_epoch)
    if (receipt.get("schema_version") != 1 or receipt.get("kind") != "alignlab-capacity-input-encoding-result-v1"
            or receipt.get("input_sha256") != digest(payload)
            or receipt.get("source_sha256") != payload["source_sha256"]
            or receipt.get("tokenizer_files") != payload["tokenizer_files"]
            or receipt.get("python_sha256") != payload["python_sha256"]
            or receipt.get("tokenizers_version") != payload["tokenizers_version"]
            or receipt.get("transformers_version") != payload["transformers_version"]
            or receipt.get("torch_imported") is not False or receipt.get("gpu_operations_performed") is not False
            or receipt.get("encoded_inputs") != encoded):
        raise ValueError("Independent original-source token re-encoding differs from capacity inputs")
    process = receipt.get("process_identity", {})
    _signature(process)
    return receipt


class _CudaMemory:
    def __init__(self):
        import torch
        self.torch = torch
        torch.cuda.set_device(0)

    def reset(self):
        self.torch.cuda.synchronize(0)
        self.torch.cuda.reset_peak_memory_stats(0)

    def sample(self):
        t = self.torch
        t.cuda.synchronize(0)
        return {"device": "cuda:0", "cuda_initialized": t.cuda.is_initialized(),
                "device_name": t.cuda.get_device_name(0),
                "total_memory_bytes": t.cuda.get_device_properties(0).total_memory,
                "max_allocated_bytes": t.cuda.max_memory_allocated(0),
                "max_reserved_bytes": t.cuda.max_memory_reserved(0),
                "allocated_bytes": t.cuda.memory_allocated(0),
                "reserved_bytes": t.cuda.memory_reserved(0)}

    def release(self):
        gc.collect()
        self.torch.cuda.synchronize(0)
        self.torch.cuda.empty_cache()


def _forced(model, prompt, request, stop):
    # Deliberately narrow capacity-only bridge; no generic callback/adapter API.
    from .evaluation_model_v3 import EvaluationModel
    from .capacity_generation import greedy_capacity_probe
    if not isinstance(model, EvaluationModel):
        raise TypeError("Forced production capacity requires the actual read-only EvaluationModel")
    with model._autocast():
        return greedy_capacity_probe(model._backend, model.role, tuple(prompt),
            eos_id=EOS, max_new_tokens=HORIZON, ignore_eos=True, device=model.device,
            deadline_epoch=request["deadline_epoch"], stop_requested=stop)


def _memory(record):
    if (record.get("device") != "cuda:0" or record.get("cuda_initialized") is not True
            or not isinstance(record.get("device_name"), str) or not record["device_name"]):
        raise ValueError("Actual initialized CUDA memory observation required")
    for key in ("total_memory_bytes", "max_allocated_bytes", "max_reserved_bytes", "allocated_bytes", "reserved_bytes"):
        if type(record.get(key)) is not int or record[key] <= 0:
            raise ValueError("Positive actual CUDA allocation/peak evidence required")
    if not (record["allocated_bytes"] <= record["max_allocated_bytes"] <= record["max_reserved_bytes"]
            <= record["total_memory_bytes"] and record["reserved_bytes"] <= record["max_reserved_bytes"]):
        raise ValueError("Inconsistent actual CUDA memory measurements")


def _generation(receipt, prompt, *, forced):
    actions = ex._ids(receipt.get("action_ids") if forced else receipt.get("generated_token_ids"))
    if (tuple(receipt.get("prompt_ids", ())) != tuple(prompt)
            or receipt.get("generated_tokens") != len(actions)
            or receipt.get("prompt_tokens") != len(prompt)
            or receipt.get("max_new_tokens") != HORIZON or receipt.get("eos_id") != EOS
            or receipt.get("ignore_eos") is not forced
            or receipt.get("attention_all_true") is not True or receipt.get("synthetic_eos_added") is not False
            or receipt.get("final_action_flushed_to_kv") is not True
            or receipt.get("max_cache_tokens") != len(prompt) + len(actions)
            or receipt.get("decoder_calls") != len(actions) + 1
            or receipt.get("generated_eos_count") != actions.count(EOS)):
        raise ValueError("Incomplete full-input/full-action KV receipt")
    expected_sha = hashlib.sha256(json.dumps([list(prompt), list(actions)], separators=(",", ":")).encode()).hexdigest()
    if receipt.get("tokens_sha256") != expected_sha:
        raise ValueError("Generated token identity differs from KV receipt")
    if forced:
        if (len(actions) != HORIZON or receipt.get("termination") != "forced_max_new_tokens"
                or receipt.get("mode") != "forced_continuation_kv_stress"
                or receipt.get("capacity_only") is not True or receipt.get("quality_evaluation") is not False):
            raise ValueError("Forced stress must actually finish all 2048 tokens")
    elif (receipt.get("complete") is not True or receipt.get("prompt_truncated") is not False
          or len(actions) > HORIZON
          or not ((receipt.get("termination") == "eos" and actions[-1] == EOS and EOS not in actions[:-1])
                  or (receipt.get("termination") == "length" and len(actions) == HORIZON and EOS not in actions))):
        raise ValueError("Natural generation must stop at first EOS or full horizon")


class _Guard:
    def __init__(self, request, path, sha, plan, system, processes, campaign_lock, gpu_owner, resource_guard=None):
        self.request, self.path, self.sha, self.plan = request, path, sha, plan
        self.system, self.processes, self.last = system, processes, float("-inf")
        self.campaign_lock, self.gpu_owner = campaign_lock, gpu_owner
        self.resource_guard = resource_guard

    def __call__(self, *, force=False):
        ex._boundary(self.request, self.system)
        if self.resource_guard is not None:
            self.resource_guard(force=force)
        if force or self.system.now() - self.last >= 10:
            _, _, lock = _admit(self.request, self.path, self.sha, self.plan, self.system, expected=self.processes)
            if (lock != self.campaign_lock
                    or parse((ROOT / "reports/.alignlab-gpu.lock").read_bytes()) != self.gpu_owner):
                raise RuntimeError("Capacity or campaign lock ownership changed")
            verified_bytes(self.path, self.sha)
            verified_bytes(Path(self.request["evaluation_plan_path"]), self.request["evaluation_plan_sha256"])
            verified_bytes(Path(self.request["protocol_path"]), self.request["protocol_sha256"])
            verified_bytes(Path(self.request["roster_path"]), self.request["roster_sha256"])
            _sources(self.request["source_sha256"])
            self.last = self.system.now()
        return False


def run_evaluation_capacity(request_path, request_sha256, *, system=None,
                            model_factory=None, codec_factory=None):
    """Execute exactly once; preserve failure progress, never mint partial success."""
    request_path = Path(request_path).resolve()
    request = _read(request_path, request_sha256)
    output, plan = _request(request)
    system = ex._system() if system is None else system
    ex._boundary(request, system)
    if output.exists():
        raise FileExistsError("Capacity output exists; overwrite/resume/retry forbidden")
    lock_path = ROOT / "reports/.alignlab-gpu.lock"
    if lock_path.exists():
        raise FileExistsError("Unknown existing GPU lock; never remove it")
    resources, processes, campaign_lock = _admit(request, request_path, request_sha256, plan, system, resources=True)
    context = _context(request)
    _verify_roster(request, system)
    ex._boundary(request, system)
    owner = {"kind": KIND, "pid": os.getpid(), "nonce": uuid.uuid4().hex,
             "request_sha256": request_sha256, "output": str(output), "processes": processes}
    ex._new(lock_path, owner)
    status, resource_guard = None, None
    try:
        guard = _Guard(request, request_path, request_sha256, plan, system, processes, campaign_lock, owner)
        guard(force=True)
        output.mkdir(parents=False, exist_ok=False)
        status = {"status": "starting", "started_at_epoch": system.now(),
                  "deadline_epoch": request["deadline_epoch"], "models": {},
                  "capacity_complete": False, "quality_evaluation_complete": False,
                  "research_complete": False, "formal_training": False}
        ex._new(output / "request.snapshot.json", request)
        ex._new(output / "protocol.snapshot.json", context.protocol.identity)
        ex._new(output / "roster.snapshot.json", context.roster)
        ex._progress(output / "progress.json", status)
        selection = select_maximum_inputs(context)
        codec = (codec_factory or ex._codec)(context.roster)
        encoded = _encode_selection(context, selection, codec)
        status.update(selection=selection, encoded_inputs=encoded)
        ex._progress(output / "progress.json", status)
        guard(force=True)
        _admit(request, request_path, request_sha256, plan, system, expected=processes, resources=True)
        profile = load_profile(request["resource_profile"])
        ex._boundary(request, system)
        allocator = configure_cuda_allocator(profile)
        verify_allocator_receipt(allocator, profile)
        resource_guard = RuntimeResourceGuard(profile, system=system, deadline_epoch=request["deadline_epoch"],
            allowed_processes=processes,
            stop_requested=lambda: (output / "STOP").exists() or (output.parent / "STOP").exists())
        guard.resource_guard = resource_guard
        guard(force=True)
        memory = _CudaMemory()
        for model_id in MODEL_IDS:
            guard(force=True)
            _admit(request, request_path, request_sha256, plan, system, expected=processes,
                   resources=True, allocator_receipt=allocator)
            ex._boundary(request, system)
            memory.reset()
            model = (model_factory or ex._model)(context.roster, model_id)
            try:
                runtime = ex._runtime(model, model_id)
                before = ex._audit(model, model_id)
                guard(force=True)
                tests = {}
                if model_id == "reward":
                    for task in ("helpsteer_rm", "rm_bench"):
                        guard(force=True)
                        entry = encoded[task]
                        score = _finite(model.reward_score(tuple(entry["prompt_ids"]),
                                                           tuple(entry["completion_ids"])), "Capacity reward")
                        guard(force=True)
                        tests[task] = {"kind": "full_context_reward_forward", "quality_sample": False,
                            "input_truncated": False, "input_tokens": selection["cases"][task]["input_tokens"],
                            "input_tokens_sha256": entry["tokens_sha256"], "score": score}
                else:
                    prompt = encoded["ifeval"]["prompt_ids"]
                    natural = model.greedy_generate(tuple(prompt), eos_id=EOS, max_new_tokens=HORIZON,
                        deadline_epoch=request["deadline_epoch"], stop_requested=guard)
                    guard(force=True)
                    _generation(natural, prompt, forced=False)
                    tests["natural_eos"] = {"quality_sample": False, "receipt": natural}
                    stress = _forced(model, prompt, request, guard)
                    guard(force=True)
                    _generation(stress, prompt, forced=True)
                    tests["forced_2048"] = {"quality_sample": False, "receipt": stress}
                after = ex._audit(model, model_id)
                if before != after:
                    raise ValueError("Capacity changed model/source/adapter/head contents")
                peak = memory.sample()
                _memory(peak)
                if peak["max_reserved_bytes"] > profile["allocator_limit_mib"] * MIB:
                    raise RuntimeError("Capacity exceeded frozen allocator ceiling")
                runtime["weights_unchanged"] = True
                status["models"][model_id] = {"model_manifest": context.protocol.identity["model_manifests"][model_id],
                    "runtime": runtime, "frozen_audits": [before, after], "gpu_memory": peak, "tests": tests}
                status.update(status="capacity_running", last_completed_model=model_id)
                ex._progress(output / "progress.json", status)
            finally:
                del model
                memory.release()
            guard(force=True)
        _context(request)
        _verify_roster(request, system)
        guard(force=True)
        status.update(status="evaluation_capacity_passed", capacity_complete=True,
                      completed_at_epoch=system.now(), allocator=allocator,
                      resource_observations=list(resource_guard.observations))
        ex._progress(output / "progress.json", status)
        files = {name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                        "bytes": (output / name).stat().st_size} for name in FILES}
        manifest = {"schema_version": 1, "kind": RESULT_KIND, **status,
            "request": {"path": str(request_path), "sha256": request_sha256},
            "protocol": {"path": request["protocol_path"], "sha256": request["protocol_sha256"]},
            "accepted_roster": {"path": request["roster_path"], "sha256": request["roster_sha256"]},
            "model_execution_order": list(MODEL_IDS), "models_serial": True,
            "resources_before": resources, "process_identity": processes,
            "campaign_lock": campaign_lock, "files": files}
        guard(force=True)
        ex._new(output / "manifest.json", manifest)
        return {"status": manifest["status"], "manifest_path": str(output / "manifest.json"),
                "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()}
    except BaseException as error:
        if status is not None:
            status.update(status="paused" if isinstance(error, InterruptedError) else
                          "budget_exhausted" if isinstance(error, TimeoutError) else "failed",
                          capacity_complete=False, error_type=type(error).__name__, error=str(error),
                          stopped_at_epoch=system.now())
            if resource_guard is not None:
                status["resource_observations"] = list(resource_guard.observations)
            ex._progress(output / "progress.json", status)
        raise
    finally:
        if lock_path.is_file() and parse(lock_path.read_bytes()) == owner:
            lock_path.unlink()


def verify_evaluation_capacity(output, *, expected_manifest_sha256, expected_request=None, deadline_epoch=None):
    """Read-only receipt check plus isolated tokenizer-only CPU verification.

    The calling process imports no Torch/Transformers. Live supervision supplies
    its shared deadline; historical read-only reuse gets a separate 120s CPU cap.
    The caller separately verifies the original capacity child's exit and PID.
    """
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError("Shared evaluation deadline expired before capacity verification")
    output = Path(output).resolve()
    result = _read(output / "manifest.json", expected_manifest_sha256)
    if (result.get("schema_version") != 1 or result.get("kind") != RESULT_KIND
            or result.get("status") != "evaluation_capacity_passed" or result.get("capacity_complete") is not True
            or any(result.get(k) is not False for k in ("quality_evaluation_complete", "research_complete", "formal_training"))):
        raise ValueError("Complete capacity-only result required")
    request = _read(result["request"]["path"], result["request"]["sha256"])
    expected_output, plan = _request(request)
    if output != expected_output or expected_request is not None and request != expected_request:
        raise ValueError("Capacity output/request differs from external dispatch")
    if set(result["files"]) != FILES or {p.name for p in output.iterdir()} != FILES | {"manifest.json"}:
        raise ValueError("Partial or unexpected capacity output files")
    contents = {}
    for name, info in result["files"].items():
        raw = verified_bytes(_relative(output, name), info["sha256"])
        if len(raw) != info["bytes"]:
            raise ValueError("Capacity evidence byte count differs")
        contents[name] = parse(raw)
    context = _context(request)
    _verify_roster(request, None)
    if (contents["request.snapshot.json"] != request
            or contents["protocol.snapshot.json"] != context.protocol.identity
            or contents["roster.snapshot.json"] != context.roster
            or result["protocol"] != {"path": request["protocol_path"], "sha256": request["protocol_sha256"]}
            or result["accepted_roster"] != {"path": request["roster_path"], "sha256": request["roster_sha256"]}):
        raise ValueError("Capacity snapshots differ from actual frozen inputs")
    progress = contents["progress.json"]
    if any(result.get(key) != value for key, value in progress.items()):
        raise ValueError("Final progress disagrees with capacity manifest")
    if (result["selection"] != select_maximum_inputs(context)
            or result.get("model_execution_order") != list(MODEL_IDS)
            or result.get("models_serial") is not True or set(result["models"]) != set(MODEL_IDS)):
        raise ValueError("Full nine-model maximum-input capacity coverage required")
    lock = result["campaign_lock"]
    if (lock.get("owner") != CAMPAIGN_KIND or lock.get("plan_sha256") != request["evaluation_plan_sha256"]
            or lock.get("resource_profile") != request["resource_profile"]
            or lock.get("plan_path") != request["evaluation_plan_path"] or lock.get("output") != plan["output"]
            or lock.get("deadline_epoch") != request["deadline_epoch"]
            or lock["started_at_epoch"] + plan["wall_seconds"] != lock["deadline_epoch"]
            or not lock["started_at_epoch"] <= _finite(result["started_at_epoch"], "Start")
            <= _finite(result["completed_at_epoch"], "Completion") < lock["deadline_epoch"]):
        raise ValueError("Capacity exceeded or changed its campaign's absolute budget")
    if not result.get("process_identity") or result["resources_before"].get("blockers") != []:
        raise ValueError("Original admitted process/resource evidence required")
    profile = load_profile(request["resource_profile"])
    verify_allocator_receipt(result.get("allocator"), profile)
    observations = result.get("resource_observations")
    if (not isinstance(observations, list) or not observations
            or any(row.get("blockers") != [] for row in observations)):
        raise ValueError("Complete service-preserving capacity observations required")
    for row in observations:
        gpu = next((g for g in row.get("gpus", []) if g.get("index") == 0), {})
        if (gpu.get("free_mb", 0) < profile["minimum_global_free_mib"]
                or not result["started_at_epoch"] <= row["checked_at_epoch"] <= result["completed_at_epoch"]
                or row["memory"]["max_reserved_bytes"] > profile["allocator_limit_mib"] * MIB):
            raise ValueError("Capacity resource guarantee violated")
    for process in result["process_identity"]:
        _signature(process)
    for task, entry in result["encoded_inputs"].items():
        prompt = ex._ids(entry["prompt_ids"])
        completion = () if task == "ifeval" else ex._ids(entry["completion_ids"], completion=True)
        if (len(prompt) != entry["prompt_tokens"] or len(completion) != entry["completion_tokens"]
                or len(prompt) + len(completion) != result["selection"]["cases"][task]["input_tokens"]
                or entry["tokens_sha256"] != digest([list(prompt), list(completion)])):
            raise ValueError("Full selected token input identity differs")
    if set(result["encoded_inputs"]) != {"ifeval", "helpsteer_rm", "rm_bench"}:
        raise ValueError("All selected maximum contexts required")
    for model_id, item in result["models"].items():
        runtime = item["runtime"]
        ex._runtime(type("Metadata", (), {"metadata": runtime})(), model_id)
        if runtime.get("cuda") is not True or runtime.get("weights_unchanged") is not True:
            raise ValueError("Actual CUDA and unchanged weights required")
        if item["model_manifest"] != context.protocol.identity["model_manifests"][model_id]:
            raise ValueError("Wrong capacity model manifest")
        audits = item["frozen_audits"]
        if len(audits) != 2 or audits[0] != audits[1]:
            raise ValueError("Matching full before/after frozen audits required")
        for audit in audits:
            ex._check_audit(audit, model_id)
        _memory(item["gpu_memory"])
        if item["gpu_memory"]["max_reserved_bytes"] > profile["allocator_limit_mib"] * MIB:
            raise ValueError("Capacity peak exceeds frozen evaluator ceiling")
        tests = item["tests"]
        expected = {"helpsteer_rm", "rm_bench"} if model_id == "reward" else {"natural_eos", "forced_2048"}
        if set(tests) != expected or any(test.get("quality_sample") is not False for test in tests.values()):
            raise ValueError("Capacity tests cannot be missing or used as quality samples")
        if model_id == "reward":
            for task, test in tests.items():
                if (test.get("kind") != "full_context_reward_forward" or test.get("input_truncated") is not False
                        or test.get("input_tokens") != result["selection"]["cases"][task]["input_tokens"]
                        or test.get("input_tokens_sha256") != result["encoded_inputs"][task]["tokens_sha256"]):
                    raise ValueError("Complete longest reward contexts required")
                _finite(test["score"], "Finite full-context reward")
        else:
            prompt = result["encoded_inputs"]["ifeval"]["prompt_ids"]
            _generation(tests["natural_eos"]["receipt"], prompt, forced=False)
            _generation(tests["forced_2048"]["receipt"], prompt, forced=True)
    receipt = _reencode_inputs(context, request, plan, result["selection"], result["encoded_inputs"],
                              deadline_epoch=deadline_epoch)
    return {**result, "independent_input_encoding_verification": receipt}
