"""One-shot complete inference, separate from independent official CPU scoring.

Importing this module loads no tensor/model library. Production model loading
is lazy and occurs only after resource admission and exclusive GPU ownership.
The injected factories/system are CPU test seams, never CLI alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import uuid

from .evaluation_protocol_v2 import (
    EvaluationProtocol, canonical, digest, parse, verified_bytes, _finite,
    _output_identity, _relative, _sha, load_evaluation_assets, load_final_holdout,
    ordered_results, validate_generation_records, SERIALIZATION,
)
from .evaluation_resources_v6 import (load_profile, collect_resources, configure_cuda_allocator,
                                    verify_allocator_receipt, RuntimeResourceGuard, signature, MIB)
from .evaluation_attention_receipt_v5 import validate_attention_execution_receipt

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-evaluation-inference-stage-v2"
RESULT_KIND = "alignlab-evaluation-inference-result-v2"
MODEL_STAGES = {"base": "base", "sft": "sft", "reward": "reward",
                **{f"{s}{seed}": s for seed in (42, 43, 44) for s in ("dpo", "ppo")}}
REQUEST_FIELDS = {"schema_version", "kind", "task", "model_id", "output",
                  "deadline_epoch", "protocol_path", "protocol_sha256",
                  "roster_path", "roster_sha256", "source_sha256", "resource_profile", "capacity_manifest"}
DENOMINATORS = {"ifeval_prompts": 541, "ifeval_instructions": 834,
                "helpsteer3_all": 2017, "helpsteer3_non_ties": 1920, "helpsteer3_ties": 97,
                "rm_bench_records": 1327, "rm_bench_answers": 7962, "rm_bench_comparisons": 11943}
QUANTIZATION = {"method": "bitsandbytes", "bits": 4, "quant_type": "nf4",
                "double_quant": True, "compute_dtype": "bfloat16"}
# Explicit transitive implementation boundary. Future unrelated modules neither
# enter an existing identity nor make historical requests unverifiable.
REQUIRED_SOURCES = frozenset({
    "alignment_lab/__init__.py", "alignment_lab/evaluation_execution_v6.py",
    "alignment_lab/evaluation_model_v5.py", "alignment_lab/evaluation_protocol_v2.py",
    "alignment_lab/evaluation_attention_v5.py", "tests_v2/test_evaluation_attention_v5.py",
    "alignment_lab/evaluation_attention_receipt_v5.py",
    "alignment_lab/official_evaluation.py", "alignment_lab/chat_codec.py",
    "alignment_lab/capacity_generation.py", "alignment_lab/checkpointing.py",
    "alignment_lab/stage_artifacts.py", "alignment_lab/gpu_reference.py",
    "alignment_lab/model_loader.py", "alignment_lab/model_snapshot.py",
    "alignment_lab/model_loader_eval_v3.py", "tests_v2/test_model_loader_eval_v3.py",
    "tests_v2/test_evaluation_model_v5.py",
    "alignment_lab/shared_qwen.py", "alignment_lab/token_logprobs.py",
    "alignment_lab/reference_bridge.py", "alignment_lab/reference_cache.py",
    "alignment_lab/training_steps.py", "alignment_lab/probe_dispatch.py",
    "alignment_lab/gpu_probe.py", "scripts/run_alignlab_evaluation_stage_v6.py",
    "rlhf_lab/__init__.py", "rlhf_lab/runtime.py",
    "alignment_lab/evaluation_resources_v6.py", "tests_v2/test_evaluation_resources_v6.py",
    "tests_v2/test_evaluation_execution_v6.py",
    "alignment_lab/evaluation_capacity_inheritance_v6.py", "tests_v2/test_evaluation_capacity_inheritance_v6.py",
    "alignment_lab/evaluation_acceptance_v2.py", "alignment_lab/evaluation_acceptance.py",
    "alignment_lab/evaluation_capacity_v6.py", "tests_v2/test_evaluation_capacity_v6.py",
})


def _read(path, sha):
    return parse(verified_bytes(Path(path), sha))


def _attention_receipt(receipt, model_id, *, minimum_calls=1, required_contexts=()):
    """Read-only acceptance of actual instance-bound attention execution evidence."""
    validate_attention_execution_receipt(receipt, enabled=model_id == "reward",
                                         minimum_calls=minimum_calls)
    if model_id == "reward":
        if receipt["cpu_fixture"] is not False:
            raise ValueError("CPU attention fixtures cannot prove production CUDA execution")
        observed = {row["query_tokens"] for row in receipt["observations"]}
        if not set(required_contexts) <= observed:
            raise ValueError("Actual RM attention observations omit a required full input context")
    return receipt


def _new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def _progress(path, value):
    # This is explicitly provisional until the final manifest hashes it.
    # In-place writes avoid Windows antivirus sharing failures on rapid rename;
    # a torn provisional JSON never suffices for completion verification.
    with path.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def _sources(sources):
    if not isinstance(sources, dict) or not REQUIRED_SOURCES <= sources.keys():
        raise ValueError("Pin the complete evaluation implementation inventory")
    for name, sha in sources.items():
        verified_bytes(_relative(ROOT, name), sha)


def _request(value):
    if (not isinstance(value, dict) or set(value) != REQUEST_FIELDS
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != KIND):
        raise ValueError("Exact evaluation inference request schema required")
    if value["task"] not in {"ifeval", "helpsteer_rm", "rm_bench"}:
        raise ValueError("Unknown complete evaluation task")
    model = value["model_id"]
    if model not in MODEL_STAGES or (value["task"] == "ifeval") == (model == "reward"):
        raise ValueError("Task is incompatible with the accepted model role")
    _finite(value["deadline_epoch"], "Absolute deadline")
    output = Path(value["output"])
    if (not output.is_absolute() or not output.resolve().is_relative_to(ROOT / "runs")
            or output.resolve() == ROOT / "runs"
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("Fresh output must stay in project runs without symlinks")
    for field in ("protocol_path", "roster_path"):
        if not Path(value[field]).is_absolute():
            raise ValueError("Explicit absolute identity paths required")
    _sources(value["source_sha256"])
    load_profile(value["resource_profile"])
    capacity = value["capacity_manifest"]
    if capacity is not None and (not isinstance(capacity, dict) or set(capacity) != {"path", "sha256"}
                                or not Path(capacity["path"]).is_absolute()):
        raise ValueError("Externally pinned capacity manifest required")
    return output


def _formal(manifest, model_id=None):
    """Shared CPU-only acceptance for the immutable recovered/direct-v4 artifacts."""
    from .evaluation_acceptance_v2 import validate_formal_artifact
    validate_formal_artifact(manifest, model_id=model_id)


def _artifact(record, accepted, protocol_binding, roster):
    """CPU byte evidence for every completed model; actual loader checks tensors."""
    stage = record["stage"]
    if stage == "base":
        if (any(record[k] is not None for k in ("artifact_dir", "manifest_sha256"))
                or any(accepted[k] is not None for k in
                       ("artifact_dir", "manifest_sha256", "progress_path", "progress_sha256"))):
            raise ValueError("Base may not substitute an adapter or stage progress")
        expected = {"path": str(Path(roster["model_snapshot"]) / "snapshot-manifest.json"),
                    "sha256": roster["model_manifest_sha256"]}
        if protocol_binding != expected:
            raise ValueError("Base protocol must bind the pinned snapshot manifest")
        _read(expected["path"], expected["sha256"])
        return
    directory = Path(record["artifact_dir"])
    if (not directory.is_absolute() or directory.is_symlink()
            or record["artifact_dir"] != accepted["artifact_dir"]
            or record["manifest_sha256"] != accepted["manifest_sha256"]
            or {p.name for p in directory.iterdir()} != {"manifest.json", "weights.pt"}):
        raise ValueError("Accepted completed artifact is missing, changed or has extra files")
    binding = {"path": str(directory / "manifest.json"), "sha256": record["manifest_sha256"]}
    if protocol_binding != binding:
        raise ValueError("Model roster differs from protocol artifact identity")
    manifest = _read(binding["path"], binding["sha256"])
    state = manifest["artifacts"]["weights.pt"]
    if len(verified_bytes(directory / "weights.pt", state["sha256"])) != state["size_bytes"]:
        raise ValueError("Artifact tensor file byte size differs")
    config, complete = manifest["identity"]["config"], manifest["completion"]
    if (manifest.get("format") != "alignlab-completed-stage-v1"
            or manifest.get("status") != "complete" or manifest.get("stage") != stage
            or manifest.get("identity_sha256") != digest(manifest["identity"])
            or config.get("formal_training") is not True or config.get("synthetic_rows") != 0
            or complete.get("synthetic_rows") != 0
            or complete.get("stop_reason") != "data_exhausted"
            or complete.get("data_pass_complete") is not True
            or complete.get("protocol_complete") is not True):
        raise ValueError("Partial, capacity or synthetic parents cannot enter final evaluation")
    _formal(manifest, record["model_id"])
    parents = manifest["identity"]["parents"]
    required_parents = {"sft": (), "reward": ("sft",), "dpo": ("sft",),
                        "ppo": ("sft", "reward")}[stage]
    if set(parents) != set(required_parents):
        raise ValueError("Incorrect completed-stage lineage")
    for parent in required_parents:
        if (parents[parent]["sha256"] != roster["models"][parent]["manifest_sha256"]
                or Path(parents[parent]["path"]).resolve()
                != (Path(roster["models"][parent]["artifact_dir"]) / "manifest.json").resolve()):
            raise ValueError("Branch does not descend from the common accepted parents")
    progress = _read(accepted["progress_path"], accepted["progress_sha256"])
    from .evaluation_acceptance_v2 import validate_formal_progress
    validate_formal_progress(progress, manifest, model_id=record["model_id"])
    artifact = progress.get("artifact", {})
    if (progress.get("formal_training") is not True
            or progress.get("research_complete") is not False
            or progress.get("identity") != manifest["identity"]
            or progress.get("completion") != complete
            or progress.get("actual_optimizer_updates") != complete.get("optimizer_updates")
            or artifact.get("manifest_sha256") != binding["sha256"]
            or artifact.get("manifest_path") != binding["path"]
            or artifact.get("artifact_dir") != str(directory)):
        raise ValueError("Completed model progress/manifest evidence differs")


def _roster(roster, protocol):
    fields = {"schema_version", "kind", "model_snapshot", "model_manifest_sha256",
              "tokenizer_dir", "models"}
    if (not isinstance(roster, dict) or set(roster) != fields
            or type(roster["schema_version"]) is not int or roster["schema_version"] != 2
            or roster["kind"] != "alignlab-accepted-evaluation-roster-v2"
            or set(roster["models"]) != set(MODEL_STAGES)
            or set(protocol.identity["model_manifests"]) != set(MODEL_STAGES)):
        raise ValueError("Complete fixed nine-model accepted roster required")
    for field in ("model_snapshot", "tokenizer_dir"):
        if not Path(roster[field]).is_absolute():
            raise ValueError("Explicit model/tokenizer paths required")
    _sha(roster["model_manifest_sha256"])
    for model_id, stage in MODEL_STAGES.items():
        record = roster["models"][model_id]
        if (set(record) != {"model_id", "stage", "artifact_dir", "manifest_sha256",
                           "acceptance_path", "acceptance_sha256"}
                or record["model_id"] != model_id or record["stage"] != stage):
            raise ValueError("Exact accepted-model record required")
        acceptance = _read(record["acceptance_path"], record["acceptance_sha256"])
        if (acceptance.get("kind") != "alignlab-independent-evaluation-model-acceptance-v2"
                or acceptance.get("status") != "accepted_for_evaluation"
                or acceptance.get("all_checks_passed") is not True
                or acceptance.get("processes_exited") is not True
                or acceptance.get("snapshot_manifest_sha256") != roster["model_manifest_sha256"]):
            raise ValueError("Independent model acceptance is incomplete")
        row = acceptance.get("models", {}).get(model_id)
        if (not isinstance(row, dict) or set(row) != {"stage", "artifact_dir",
                "manifest_sha256", "progress_path", "progress_sha256"} or row["stage"] != stage):
            raise ValueError("Acceptance lacks this exact model")
        _artifact(record, row, protocol.identity["model_manifests"][model_id], roster)


@dataclass(frozen=True)
class _Context:
    protocol: EvaluationProtocol
    roster: dict
    assets: object
    dataset: object
    protocol_raw: bytes
    roster_raw: bytes


def _context(request):
    _request(request)
    protocol_raw = verified_bytes(Path(request["protocol_path"]), request["protocol_sha256"])
    protocol = EvaluationProtocol(canonical(parse(protocol_raw)))
    value = protocol.identity
    settings = value.get("generation", {})
    expected = {"do_sample": False, "enable_thinking": False, "max_new_tokens": 2048,
                "eos_token_id": 151645, "pad_token_id": 151643, "vocab_size": 151936,
                "input_truncation": False}
    if (value.get("kind") != "alignlab-independent-evaluation-protocol-v1"
            or value.get("serialization_version") != SERIALIZATION
            or value.get("denominators") != DENOMINATORS
            or any(type(settings.get(k)) is not type(v) or settings[k] != v for k, v in expected.items())
            or type(value.get("scorer_seed")) is not int or value["scorer_seed"] != 0
            or value.get("training_use_allowed") is not False
            or value.get("quality_selection_allowed") is not False):
        raise ValueError("Frozen complete official evaluation protocol required")
    protocol.verify_bindings()
    roster_raw = verified_bytes(Path(request["roster_path"]), request["roster_sha256"])
    roster = parse(roster_raw)
    _roster(roster, protocol)
    asset = value["asset_manifest"]
    assets = load_evaluation_assets(asset["path"], asset["sha256"])
    held = value["holdout_manifest"]
    holdout = load_final_holdout(Path(held["path"]).parent, held["sha256"])
    for dataset in (assets.ifeval, assets.rm_bench, holdout):
        dataset.verify()
        if value["dataset_identities"].get(dataset.name) != dataset.identity:
            raise ValueError("Live complete datasets differ from the protocol")
    data = {"ifeval": assets.ifeval, "helpsteer_rm": holdout, "rm_bench": assets.rm_bench}[request["task"]]
    return _Context(protocol, roster, assets, data, protocol_raw, roster_raw)


def _ids(values, *, vocab=151936, completion=False):
    if (not isinstance(values, (tuple, list)) or not values
            or any(type(i) is not int or not 0 <= i < vocab for i in values)):
        raise ValueError("Nonempty original vocabulary token IDs required")
    if completion and (values[-1] != 151645 or 151645 in values[:-1]):
        raise ValueError("Full existing response must include exactly its terminal assistant EOS")
    return tuple(values)


def _encoded(source, task, codec):
    context = source["context"] if task == "helpsteer_rm" else [{"role": "user", "content": source["prompt"]}]
    prompt = codec.encode_prompt(context)
    prompt_ids = _ids(prompt.input_ids)
    if task == "ifeval":
        return prompt_ids, ()
    texts = ([source["response1"], source["response2"]] if task == "helpsteer_rm"
             else source["chosen"] + source["rejected"])
    return prompt_ids, tuple(_ids(codec.encode_completion(prompt, text).completion_ids,
                                 completion=True) for text in texts)


def _validate_rows(context, request, rows, *, decode_tokens=None, codec=None):
    data, task = context.dataset, request["task"]
    ordered = ordered_results(data, rows)
    if task == "ifeval":
        ordered = validate_generation_records(data, ordered, protocol=context.protocol,
            model_id=request["model_id"], decode_tokens=decode_tokens)
    for source, row in zip(data.rows, ordered):
        _output_identity(row, context.protocol, request["model_id"])
        prompt = _ids(row.get("prompt_token_ids"))
        if task == "ifeval":
            if len(prompt) != row["prompt_tokens"]:
                raise ValueError("Saved prompt IDs disagree with full prompt length")
            completions = ()
        elif task == "helpsteer_rm":
            if row.get("record_id") != source["record_id"] or row.get("source_index") != source["source_index"]:
                raise ValueError("Holdout observation IDs must not be deduplicated or reassigned")
            completions = tuple(_ids(row.get(f"response{i}_token_ids"), completion=True) for i in (1, 2))
            lengths = [len(prompt) + len(c) for c in completions]
            if row.get("input_lengths") != lengths or lengths != data.lengths[source["evaluation_id"]]:
                raise ValueError("Full held-out contexts required; truncation forbidden")
            one, two = (_finite(row[f"response{i}_score"], "Reward score") for i in (1, 2))
            _finite(one - two, "Reward difference")
        else:
            completions = []
            for side in ("chosen", "rejected"):
                tokens, scores = row.get(side + "_token_ids"), row.get("score_" + side)
                if not isinstance(tokens, list) or len(tokens) != 3 or not isinstance(scores, list) or len(scores) != 3:
                    raise ValueError("All six original RM-Bench responses are required")
                full = [_ids(item, completion=True) for item in tokens]
                lengths = [len(prompt) + len(c) for c in full]
                if row.get("input_lengths_" + side) != lengths or lengths != data.lengths[source["id"]][side]:
                    raise ValueError("Full RM-Bench contexts required")
                for score in scores:
                    _finite(score, "Reward score")
                completions.extend(full)
            completions = tuple(completions)
        if codec is not None and (prompt, completions) != _encoded(source, task, codec):
            raise ValueError("Saved token IDs differ from the original full inputs")
    return ordered


def _codec(roster):
    from transformers import AutoTokenizer
    from .chat_codec import Qwen3ChatCodec
    tokenizer = AutoTokenizer.from_pretrained(roster["tokenizer_dir"], local_files_only=True,
                                              trust_remote_code=False)
    return Qwen3ChatCodec(tokenizer)


def _model(roster, model_id):
    from .evaluation_model_v5 import load_evaluation_model
    return load_evaluation_model(roster["models"][model_id],
        model_snapshot=roster["model_snapshot"], model_manifest_sha256=roster["model_manifest_sha256"],
        tokenizer_dir=roster["tokenizer_dir"])


def _check_audit(audit, model_id):
    if (not isinstance(audit, dict) or audit.get("unchanged") is not True or audit.get("full") is not True
            or audit.get("type") != "alignlab-evaluation-frozen-audit-v1"
            or audit.get("model_id") != model_id
            or audit.get("file_contents_rehashed") is not True
            or audit.get("untracked_content_writes_checked") is not True):
        raise ValueError("Complete unchanged-weight audit required")
    for key in ("source_sha256", "adapter_sha256", "base_sha256", "head_sha256"):
        _sha(audit[key])
    return audit


def _audit(model, model_id):
    return _check_audit(model.frozen_audit(full=True), model_id)


def _runtime(model, model_id):
    value = dict(model.metadata)
    if (value.get("device") != "cuda:0" or value.get("cpu_fixture") is not False
            or value.get("inference_only") is not True or value.get("model_id") != model_id
            or value.get("stage") != MODEL_STAGES[model_id]
            or value.get("quantization") != QUANTIZATION or value.get("quantized_linear_count") != 196):
        raise ValueError("Actual read-only CUDA NF4 model metadata required")
    return {**value, "cuda": True}


def _system():
    from .probe_dispatch import _System
    return _System()


def _admit(system, expected_processes=None, *, request, expected_peak_reserved_bytes=None, allocator_receipt=None):
    processes = system.processes()
    by_pid = {p["pid"]: p for p in processes}
    own = by_pid.get(os.getpid())
    if own is None or not own.get("created_at_utc") or not own.get("command_line"):
        raise RuntimeError("Observed evaluator process identity required")
    owned = {os.getpid()}
    # Windows venv launchers may retain a wrapper with the same CLI arguments.
    parent = by_pid.get(own.get("parent_pid"))
    if (parent and "run_alignlab_evaluation_stage_v6.py" in parent.get("command_line", "")
            and "run_alignlab_evaluation_stage_v6.py" in own["command_line"]
            and parent["command_line"].split("run_alignlab_evaluation_stage_v6.py", 1)[1]
            == own["command_line"].split("run_alignlab_evaluation_stage_v6.py", 1)[1]):
        owned.add(parent["pid"])
    observed = [p for p in processes if p["pid"] in owned]
    lock = parse((ROOT / "reports/.alignlab-evaluation.lock").read_bytes())
    if (lock.get("owner") != "alignlab-primary-evaluation-campaign-v2"
            or lock.get("resource_profile") != request["resource_profile"]
            or lock.get("deadline_epoch") != request["deadline_epoch"]):
        raise RuntimeError("Exact v2 campaign resource/deadline ownership required")
    supervisors = lock.get("processes", [])
    if (not supervisors or lock.get("pid") not in {p.get("pid") for p in supervisors}
            or any(p["pid"] not in by_pid or signature(p) != signature(by_pid[p["pid"]])
                   or "supervise_alignlab_evaluation_v6.py" not in p["command_line"] for p in supervisors)):
        raise RuntimeError("Live pinned v2 supervisor process required")
    ancestor = parent
    ancestry = set()
    while ancestor is not None and ancestor["pid"] not in ancestry:
        ancestry.add(ancestor["pid"])
        ancestor = by_pid.get(ancestor.get("parent_pid"))
    if lock["pid"] not in ancestry:
        raise RuntimeError("Evaluator must descend from the pinned supervisor")
    resource = collect_resources(load_profile(request["resource_profile"]),
        expected_peak_reserved_bytes=expected_peak_reserved_bytes, allowed_processes=observed + supervisors,
        process_reader=system.processes, allocator_receipt=allocator_receipt)
    blockers = list(resource["blockers"])
    markers = ("run_alignlab_", "dispatch_alignlab_", "local-train", "local-evaluate",
               "run_campaign.py", "windows.train", "train_sft", "train_dpo", "train_ppo")
    for process in processes:
        command = (process.get("command_line") or "").lower()
        if process["pid"] not in owned and any(s in command for s in markers):
            blockers.append(f"Other active training/probe/evaluation process: {process['pid']}")
    if blockers:
        raise RuntimeError("Evaluation resources unavailable: " + "; ".join(blockers))
    if expected_processes is not None:
        if {signature(p) for p in observed + supervisors} != {signature(p) for p in expected_processes}:
            raise RuntimeError("Evaluator PID/create-time/command identity changed")
    return resource, observed + supervisors


def _boundary(request, system):
    output = Path(request["output"])
    if (output / "STOP").exists() or (output.parent / "STOP").exists():
        raise InterruptedError("Cooperative evaluation STOP requested")
    if system.now() >= request["deadline_epoch"]:
        raise TimeoutError("Shared absolute evaluation deadline exhausted")


def _counts(task, count):
    if task == "ifeval":
        return {"records": count, "prompts": count, "instructions": 834}
    if task == "helpsteer_rm":
        return {"records": count, "answer_scores": count * 2, "non_ties": 1920, "ties": 97}
    return {"records": count, "answer_scores": 6 * count, "comparisons": 9 * count}


def _capacity_peak(request):
    from .evaluation_capacity_v6 import verify_evaluation_capacity
    binding = request["capacity_manifest"]
    if binding is None:
        raise ValueError("Real full nine-model capacity proof required before production inference")
    candidate = _read(binding["path"], binding["sha256"])
    if candidate.get("kind") == "alignlab-evaluation-capacity-inheritance-v6":
        from .evaluation_capacity_inheritance_v6 import verify_capacity_inheritance_for_request
        return verify_capacity_inheritance_for_request(binding, request)
    result = verify_evaluation_capacity(Path(binding["path"]).parent,
        expected_manifest_sha256=binding["sha256"], deadline_epoch=request["deadline_epoch"])
    capacity_request = _read(result["request"]["path"], result["request"]["sha256"])
    for key in ("resource_profile", "deadline_epoch", "protocol_path", "protocol_sha256",
                "roster_path", "roster_sha256", "source_sha256"):
        if capacity_request[key] != request[key]:
            raise ValueError("Capacity identity differs from production evaluation")
    peak = result["models"][request["model_id"]]["gpu_memory"]["max_reserved_bytes"]
    profile = load_profile(request["resource_profile"])
    if not 0 < peak <= profile["allocator_limit_mib"] * MIB:
        raise ValueError("Full-model capacity peak exceeds allocator ceiling")
    return peak


def run_evaluation_stage(request_path, request_sha256, *, system=None,
                         model_factory=None, codec_factory=None):
    """Execute once; exceptions preserve partial evidence, never a success manifest."""
    request_path = Path(request_path)
    request = _read(request_path, request_sha256)
    output = _request(request)
    system = _system() if system is None else system
    _boundary(request, system)
    if output.exists():
        raise FileExistsError("Evaluation output exists; no overwrite or implicit resume")
    lock = ROOT / "reports/.alignlab-gpu.lock"
    if lock.exists():
        raise FileExistsError("Existing GPU lock; refusing to remove unknown ownership")
    peak = _capacity_peak(request)
    resource, processes = _admit(system, request=request, expected_peak_reserved_bytes=peak)
    context = _context(request)
    _boundary(request, system)
    owner = {"kind": KIND, "pid": os.getpid(), "nonce": uuid.uuid4().hex,
             "request_sha256": request_sha256, "output": str(output), "processes": processes}
    _new(lock, owner)
    status, guard = None, None
    try:
        _boundary(request, system)
        _admit(system, processes, request=request, expected_peak_reserved_bytes=peak)
        output.mkdir(parents=False, exist_ok=False)
        started = system.now()
        status = {"status": "starting", "request": request, "pid": os.getpid(),
                  "started_at_epoch": started, "deadline_epoch": request["deadline_epoch"],
                  "completed_records": 0, "inference_complete": False,
                  "scoring_complete": False, "research_complete": False, "formal_training": False}
        _new(output / "request.snapshot.json", request)
        _new(output / "protocol.snapshot.json", parse(context.protocol_raw))
        _new(output / "roster.snapshot.json", parse(context.roster_raw))
        _progress(output / "progress.json", status)
        codec = (codec_factory or _codec)(context.roster)
        _boundary(request, system)
        _admit(system, processes, request=request, expected_peak_reserved_bytes=peak)
        profile = load_profile(request["resource_profile"])
        _boundary(request, system)
        allocator = configure_cuda_allocator(profile)
        verify_allocator_receipt(allocator, profile)
        _admit(system, processes, request=request, expected_peak_reserved_bytes=peak, allocator_receipt=allocator)
        guard = RuntimeResourceGuard(profile, system=system, deadline_epoch=request["deadline_epoch"],
            allowed_processes=processes,
            stop_requested=lambda: (output / "STOP").exists() or (output.parent / "STOP").exists())
        guard(force=True)
        model = (model_factory or _model)(context.roster, request["model_id"])
        runtime = _runtime(model, request["model_id"])
        audits = [_audit(model, request["model_id"])]
        rows = []
        task, protocol, data = request["task"], context.protocol, context.dataset
        decode = lambda ids: codec.tokenizer.decode(list(ids), skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
        with (output / "outputs.jsonl").open("xb") as stream:
            for source in data.rows:
                _boundary(request, system)
                guard()
                prompt, completions = _encoded(source, task, codec)
                expected_lengths = (data.lengths[str(source["key"])] if task == "ifeval"
                                    else data.lengths[source[data.id_field]])
                actual_lengths = [len(prompt) + len(c) for c in completions]
                expected_flat = (expected_lengths if task == "helpsteer_rm" else
                                 expected_lengths["chosen"] + expected_lengths["rejected"] if task == "rm_bench"
                                 else [])
                if ((task == "ifeval" and len(prompt) != expected_lengths)
                        or task != "ifeval" and actual_lengths != expected_flat):
                    raise ValueError("Full serialized input length differs before model execution")
                row = {data.id_field: source[data.id_field], "model_id": request["model_id"],
                       "model_manifest_sha256": protocol.identity["model_manifests"][request["model_id"]]["sha256"],
                       "protocol_sha256": protocol.sha256, "input_truncated": False,
                       "prompt_token_ids": list(prompt)}
                if task == "ifeval":
                    receipt = model.greedy_generate(prompt, eos_id=151645, max_new_tokens=2048,
                        deadline_epoch=request["deadline_epoch"],
                        stop_requested=guard)
                    _boundary(request, system)
                    actions = _ids(receipt.get("generated_token_ids"))
                    if (receipt.get("complete") is not True or receipt.get("prompt_truncated") is not False
                            or tuple(receipt.get("prompt_ids", ())) != prompt
                            or receipt.get("generated_tokens") != len(actions)
                            or receipt.get("ignore_eos") is not False
                            or receipt.get("synthetic_eos_added") is not False
                            or receipt.get("attention_all_true") is not True):
                        raise ValueError("Incomplete or altered original generation receipt")
                    reason = {"eos": "eos", "length": "length"}.get(receipt["termination"])
                    if reason is None:
                        raise ValueError("Interrupted generation is not a completed result")
                    if (len(actions) > 2048
                            or reason == "eos" and (actions[-1] != 151645 or 151645 in actions[:-1])
                            or reason == "length" and (len(actions) != 2048 or 151645 in actions)):
                        raise ValueError("Generation must end at its first true EOS or exact 2048-action horizon")
                    row.update(prompt=source["prompt"], prompt_tokens=len(prompt),
                        generated_token_ids=list(actions), termination=reason,
                        response=decode(actions[:-1] if reason == "eos" else actions),
                        generation_seed=protocol.identity["scorer_seed"])
                else:
                    scores = []
                    for completion in completions:
                        _boundary(request, system)
                        guard()
                        scores.append(_finite(model.reward_score(prompt, completion), "Model reward"))
                        _boundary(request, system)
                        guard()
                    if task == "helpsteer_rm":
                        row.update(record_id=source["record_id"], source_index=source["source_index"],
                            response1_score=scores[0], response2_score=scores[1],
                            input_lengths=actual_lengths, response1_token_ids=list(completions[0]),
                            response2_token_ids=list(completions[1]))
                    else:
                        for side, index in (("chosen", 0), ("rejected", 3)):
                            row["score_" + side] = scores[index:index + 3]
                            row["input_lengths_" + side] = actual_lengths[index:index + 3]
                            row[side + "_token_ids"] = [list(c) for c in completions[index:index + 3]]
                stream.write(canonical(row) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
                status.update(status="inference_running", completed_records=len(rows),
                              last_record_id=source[data.id_field])
                _progress(output / "progress.json", status)
        _boundary(request, system)
        guard(force=True)
        _validate_rows(context, request, rows, decode_tokens=decode, codec=codec)
        audits.append(_audit(model, request["model_id"]))
        if audits[0] != audits[1]:
            raise ValueError("Initial/final complete model identity differs")
        _context(request)  # Rehash live sources, protocols, accepted models and all inputs.
        verified_bytes(request_path, request_sha256)
        _boundary(request, system)
        guard(force=True)
        runtime["weights_unchanged"] = True
        attention = _attention_receipt(model.attention_execution_receipt(), request["model_id"],
            minimum_calls=(DENOMINATORS["helpsteer3_all"] * 2 * 28 if task == "helpsteer_rm" else
                           DENOMINATORS["rm_bench_answers"] * 28 if task == "rm_bench" else 1))
        finished = system.now()
        status.update(status="evaluation_inference_passed", inference_complete=True,
                      finished_at_epoch=finished, allocator=allocator,
                      resource_observations=list(guard.observations), capacity_peak_reserved_bytes=peak,
                      attention_execution=attention)
        _progress(output / "progress.json", status)
        files = {}
        for path in output.iterdir():
            payload = path.read_bytes()
            files[path.name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        manifest = {"schema_version": 1, "kind": RESULT_KIND, **status, "task": task,
            "model_id": request["model_id"],
            "protocol": {"path": request["protocol_path"], "sha256": request["protocol_sha256"],
                         "identity_sha256": protocol.sha256},
            "accepted_roster": {"path": request["roster_path"], "sha256": request["roster_sha256"]},
            "model_manifest": protocol.identity["model_manifests"][request["model_id"]],
            "dataset_identity": data.identity, "counts": _counts(task, len(rows)), "files": files,
            "model_runtime": runtime, "frozen_audits": audits, "resources_before": resource,
            "processes": processes, "quality_evaluation_complete": False}
        _boundary(request, system)
        _new(output / "manifest.json", manifest)
        return {"manifest_path": str(output / "manifest.json"),
                "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
                "status": manifest["status"]}
    except BaseException as error:
        if status is not None:
            status.update(status="paused" if isinstance(error, InterruptedError) else
                          "budget_exhausted" if isinstance(error, TimeoutError) else "failed",
                          inference_complete=False, error_type=type(error).__name__, error=str(error),
                          stopped_at_epoch=system.now())
            if guard is not None:
                status["resource_observations"] = guard.observations
            _progress(output / "progress.json", status)
        raise
    finally:
        if lock.is_file() and parse(lock.read_bytes()) == owner:
            lock.unlink()


def verify_evaluation_result(output, *, expected_manifest_sha256, expected_request=None,
                             decode_tokens=None, codec=None):
    """Read-only validation. IFEval requires a decoder bound to the pinned tokenizer.

    Supplying the same bound codec additionally re-encodes every raw input and
    checks exact input IDs, beyond the stored accepted full-length inventory.
    """
    output = Path(output)
    manifest = _read(output / "manifest.json", expected_manifest_sha256)
    if (manifest.get("kind") != RESULT_KIND or manifest.get("status") != "evaluation_inference_passed"
            or manifest.get("inference_complete") is not True
            or any(manifest.get(k) is not False for k in
                   ("scoring_complete", "quality_evaluation_complete", "research_complete", "formal_training"))):
        raise ValueError("Only a complete inference manifest is acceptable")
    request = manifest["request"]
    if expected_request is not None and request != expected_request:
        raise ValueError("Output request differs from external dispatch pin")
    if Path(request["output"]).resolve() != output.resolve():
        raise ValueError("Result location differs from its request")
    context = _context(request)
    expected = {"request.snapshot.json", "protocol.snapshot.json", "roster.snapshot.json",
                "outputs.jsonl", "progress.json"}
    if set(manifest["files"]) != expected or {p.name for p in output.iterdir()} != expected | {"manifest.json"}:
        raise ValueError("Missing or unexpected inference files")
    files = {}
    for name, record in manifest["files"].items():
        files[name] = verified_bytes(_relative(output, name), record["sha256"])
        if len(files[name]) != record["bytes"]:
            raise ValueError("Output byte size mismatch")
    if (parse(files["request.snapshot.json"]) != request
            or parse(files["protocol.snapshot.json"]) != context.protocol.identity
            or parse(files["roster.snapshot.json"]) != context.roster
            or manifest["protocol"] != {"path": request["protocol_path"], "sha256": request["protocol_sha256"],
                                        "identity_sha256": context.protocol.sha256}
            or manifest["accepted_roster"] != {"path": request["roster_path"], "sha256": request["roster_sha256"]}
            or manifest["model_manifest"] != context.protocol.identity["model_manifests"][request["model_id"]]
            or manifest["dataset_identity"] != context.dataset.identity
            or manifest["model_id"] != request["model_id"] or manifest["task"] != request["task"]):
        raise ValueError("Manifest or snapshots differ from the live accepted identities")
    started, finished, deadline = (_finite(manifest[k], k) for k in
                                   ("started_at_epoch", "finished_at_epoch", "deadline_epoch"))
    if not started <= finished <= deadline or deadline != request["deadline_epoch"]:
        raise ValueError("Incomplete or extended evaluation budget")
    profile = load_profile(request["resource_profile"])
    verify_allocator_receipt(manifest.get("allocator"), profile)
    observations = manifest.get("resource_observations")
    if (not isinstance(observations, list) or not observations
            or any(row.get("blockers") != [] for row in observations)
            or not 0 < manifest.get("capacity_peak_reserved_bytes", 0) <= profile["allocator_limit_mib"] * MIB):
        raise ValueError("Complete live service/memory evidence required")
    for row in observations:
        gpu = next((g for g in row.get("gpus", []) if g.get("index") == 0), {})
        if (gpu.get("free_mb", 0) < profile["minimum_global_free_mib"]
                or not started <= row["checked_at_epoch"] <= finished
                or row["memory"]["max_reserved_bytes"] > profile["allocator_limit_mib"] * MIB):
            raise ValueError("Resource guarantee violated during inference")
    runtime, audits = manifest["model_runtime"], manifest["frozen_audits"]
    _attention_receipt(manifest.get("attention_execution"), request["model_id"],
        minimum_calls=(DENOMINATORS["helpsteer3_all"] * 2 * 28 if request["task"] == "helpsteer_rm" else
                       DENOMINATORS["rm_bench_answers"] * 28 if request["task"] == "rm_bench" else 1))
    if (runtime.get("device") != "cuda:0" or runtime.get("cuda") is not True
            or runtime.get("weights_unchanged") is not True or runtime.get("cpu_fixture") is not False
            or runtime.get("model_id") != request["model_id"]
            or runtime.get("stage") != MODEL_STAGES[request["model_id"]]
            or runtime.get("quantization") != QUANTIZATION or runtime.get("quantized_linear_count") != 196
            or not isinstance(audits, list) or len(audits) != 2 or audits[0] != audits[1]
            or any(a.get("unchanged") is not True or a.get("full") is not True
                   or a.get("model_id") != request["model_id"] for a in audits)):
        raise ValueError("Complete frozen CUDA inference evidence required")
    for audit in audits:
        _check_audit(audit, request["model_id"])
    if (type(manifest.get("pid")) is not int
            or not any(p.get("pid") == manifest["pid"] and p.get("created_at_utc")
                       and p.get("command_line") for p in manifest.get("processes", []))):
        raise ValueError("Observed evaluator process identity is missing")
    rows = [parse(line) for line in files["outputs.jsonl"].split(b"\n") if line.strip()]
    ordered = _validate_rows(context, request, rows, decode_tokens=decode_tokens, codec=codec)
    progress = parse(files["progress.json"])
    if (manifest["counts"] != _counts(request["task"], len(ordered))
            or progress.get("status") != "evaluation_inference_passed"
            or progress.get("completed_records") != len(ordered)
            or progress.get("request") != request or progress.get("inference_complete") is not True):
        raise ValueError("Final coverage or progress evidence differs")
    verified_bytes(output / "manifest.json", expected_manifest_sha256)
    return {"manifest_path": str(output / "manifest.json"), "manifest_sha256": expected_manifest_sha256,
            "status": manifest["status"], "counts": manifest["counts"], "ordered_results": ordered}
