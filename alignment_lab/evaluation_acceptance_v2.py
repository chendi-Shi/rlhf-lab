"""Evaluation roster for the completed, separately authorized direct-V4 campaign.

This is a new acceptance protocol, not an alias for the original campaign.
Historical optimizer/replay validation is reused from externally pinned completed
acceptance receipts. Current final checkpoints, exports, base weights, source and
input evidence are authenticated again. Historical intermediate tensor archives
are not needed to evaluate the accepted final models and are not re-audited.
The reusable verifier is stdlib-only; only the initial audit decodes final tensors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import sys

from . import evaluation_acceptance as legacy

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-independent-evaluation-model-acceptance-v2"
ROSTER_KIND = "alignlab-accepted-evaluation-roster-v2"
MODELS = legacy.MODELS
SOURCES = tuple(legacy.SOURCES) + (
    "alignment_lab/evaluation_acceptance_v2.py",
    "scripts/audit_alignlab_evaluation_models_v2.py",
    "tests_v2/test_evaluation_acceptance_v2.py",
)
SEQUENCE = ["dpo42", "ppo42", "dpo43", "ppo43", "dpo44", "ppo44"]
POLICY = "pinned_historical_full_acceptance_and_current_final_payload_verification"
_TENSORS = {".pt", ".safetensors", ".bin"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_formal_artifact(manifest, *, model_id=None):
    """Pure metadata rules for this fixed nine-model completed study roster."""
    stage = manifest["stage"]
    config, complete = manifest["identity"]["config"], manifest["completion"]
    _require(stage in {"sft", "reward", "dpo", "ppo"}
             and config.get("stage") == stage and config.get("formal_training") is True
             and config.get("synthetic_rows") == 0 and complete.get("synthetic_rows") == 0
             and complete.get("stop_reason") == "data_exhausted"
             and complete.get("data_pass_complete") is True and complete.get("protocol_complete") is True,
             "Only completed pure-real formal artifacts may be evaluated")
    full, subset = "full_accepted_real_training_partition", "registered_unique_real_prompt_subset"
    if stage in {"sft", "reward"}:
        rows, updates = (4000, 250) if stage == "sft" else (13139, 822)
        _require(config.get("purpose") == "alignlab-formal-parent-stage-v1"
                 and config.get("scope") == full and config.get("expected_rows") == rows
                 and config.get("expected_updates") == updates and complete.get("scope") == full
                 and complete.get("full_accepted_dataset_pass") is True, "Incomplete formal shared parent")
    else:
        rows, updates = (13139, 822) if stage == "dpo" else (1024, 2048)
        work = config.get("registered_workload", {})
        scope = full if stage == "dpo" else subset
        purpose = config.get("purpose")
        _require(purpose == "alignlab-formal-alignment-stage-r3-v1"
                 or stage == "ppo" and purpose == "alignlab-ppo-throughput-resume-v2"
                 and config.get("work", {}).get("seed") == 42,
                 "Artifact purpose is outside the accepted R3/throughput study")
        _require(work.get("rows") == rows and work.get("optimizer_updates") == updates
                 and work.get("scope") == scope and complete.get("scope") == scope
                 and complete.get("full_selected_dataset_pass") is True
                 and complete.get("full_accepted_dataset_pass") is (stage == "dpo"), "Registered alignment workload differs")
        _require(config.get("work", {}).get("seed") in (42, 43, 44), "Unknown alignment seed")
    _require(complete.get("rows_processed") == rows and complete.get("optimizer_updates") == updates,
             "Formal completion counts differ")
    if model_id is not None:
        _require(model_id in MODELS and MODELS[model_id] == stage, "Model label stage differs")
        if stage in {"dpo", "ppo"}:
            _require(config["work"]["seed"] == int(model_id[-2:]), "Model label seed differs")


def validate_formal_progress(progress, manifest, *, model_id=None):
    """Pure progress semantics; callers additionally authenticate paths/hashes."""
    validate_formal_artifact(manifest, model_id=model_id)
    stage = manifest["stage"]
    allowed = ({"formal_parent_stage_passed"} if stage == "sft" else
               {"formal_parent_stage_passed", "formal_parent_recovery_passed"} if stage == "reward"
               else {"formal_alignment_stage_passed"})
    _require(progress.get("status") in allowed and progress.get("formal_training") is True
             and progress.get("research_complete") is False
             and progress.get("identity") == manifest["identity"]
             and progress.get("completion") == manifest["completion"]
             and progress.get("actual_optimizer_updates") == manifest["completion"]["optimizer_updates"],
             "Completed model progress differs from final artifact")
    if progress["status"] == "formal_parent_recovery_passed":
        _require(stage == "reward" and isinstance(manifest["identity"]["config"].get("recovery"), dict),
                 "Recovery status requires the accepted reward recovery identity")


def _pin(path, sha256):
    return {"path": str(Path(path).resolve()), "sha256": sha256}


def _stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(result.tzinfo is not None, "Process creation time must include a timezone")
    fraction = re.search(r"T\d\d:\d\d:\d\d(?:\.(\d{1,9}))?(?:Z|[+-]\d\d:\d\d)$", value)
    _require(fraction is not None, "Precise ISO process creation time required")
    return (result.astimezone(timezone.utc).replace(microsecond=0),
            int((fraction.group(1) or "").ljust(9, "0")))


def _process_history(expected, current, protected_paths):
    """Require original instances exited; PID reuse is distinct from cmd drift."""
    observed = {}
    for item in current:
        _require(type(item.get("pid")) is int and item["pid"] > 0
                 and item["pid"] not in observed and item.get("command_line"),
                 "Unknown or duplicate live process identity")
        _stamp(item["created_at_utc"])
        observed[item["pid"]] = item
    result, seen = [], set()
    for item in expected:
        _require(type(item.get("pid")) is int and item["pid"] > 0 and item.get("command_line"),
                 "Incomplete historical process identity")
        created = _stamp(item["created_at_utc"])
        key = (item["pid"], created, item["command_line"])
        if key in seen:
            continue
        seen.add(key)
        actual = observed.get(item["pid"])
        if actual is not None and _stamp(actual["created_at_utc"]) == created:
            _require(actual["command_line"] == item["command_line"]
                     and actual.get("parent_pid") == item.get("parent_pid"),
                     "Historical PID/creation has changed command or parent")
            raise ValueError("A recorded training process is still alive")
        result.append({"expected": item, "observed": actual,
                       "state": "exited" if actual is None else "pid_reused_original_exited"})
    protected = [str(Path(p).resolve()).lower().replace("\\", "/") for p in protected_paths]
    writers = ("run_alignlab_", "train", "bootstrap_run.py", "run_campaign.py")
    for item in current:
        command = item["command_line"].lower().replace("\\", "/")
        _require(not (any(p in command for p in protected) and any(w in command for w in writers)),
                 "A writer still references a protected training output")
    return result


class Evidence:
    """One-pass deduplication only; content is rehashed on every public verify."""
    def __init__(self):
        self.inventory = {}
        self.signatures = {}
        self.documents = {}
        self.expanded = set()
        self.historical_payloads = {}
        self.historical = {}

    def add(self, path, sha256=None):
        path = legacy._regular(path)
        key = str(path)
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key in self.inventory:
            _require(self.signatures[key] == signature, "Evidence changed during acceptance")
            item = self.inventory[key]
            _require(sha256 is None or sha256 == item["sha256"],
                     f"Conflicting evidence identities for {path}: {item['sha256']} != {sha256}")
            return item
        item = legacy._file(path, sha256)
        after = path.stat()
        _require(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 "Evidence changed during hashing")
        self.inventory[key], self.signatures[key] = item, signature
        return item

    def read(self, pin):
        _require(isinstance(pin, dict) and isinstance(pin.get("path"), str)
                 and Path(pin["path"]).is_absolute() and isinstance(pin.get("sha256"), str),
                 "An absolute externally hashed document is required")
        item = self.add(pin["path"], pin["sha256"])
        raw = Path(item["path"]).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == item["sha256"], "JSON changed during acceptance")
        return legacy._parse(raw)

    def source_inventory(self, sources):
        _require(isinstance(sources, dict) and sources, "Missing frozen source inventory")
        for name, digest in sources.items():
            path = (ROOT / name).resolve()
            _require(path.is_relative_to(ROOT) and path.suffix == ".py", "Invalid frozen source path")
            self.add(path, digest)

    def include(self, pin, *, document=None, expand_pins=True):
        """Authenticate a document; expand only explicitly selected evidence.

        Historical receipts bind their bytes and historical assertions. Their
        nested source/status pins must not become claims about today's files.
        Current model identities and source inventories are checked separately.
        """
        path = Path(pin["path"])
        if not path.is_absolute():
            _require(document is not None and ".." not in path.parts
                     and ":" not in str(path) and not path.drive,
                     "Relative evidence requires an unambiguous containing document")
            candidates = {str((base / path).absolute()): (base / path).absolute()
                          for base in (Path(document).parent, ROOT)}
            existing = [candidate for candidate in candidates.values() if candidate.is_file()]
            _require(len(existing) == 1, "Missing or ambiguous relative evidence path")
            path = existing[0]
            _require(path.resolve().is_relative_to(ROOT), "Relative evidence escaped the project")
        key = (str(path.resolve()), pin["sha256"])
        if key in self.historical:
            path = Path(self.historical[key])
        if path.suffix.lower() in _TENSORS:
            self.historical_payloads[str(path.resolve())] = _pin(path, pin["sha256"])
            return
        item = self.add(path, pin["sha256"])
        if path.suffix.lower() == ".json":
            if item["path"] not in self.documents:
                self.documents[item["path"]] = self.read(item)
            if expand_pins and item["path"] not in self.expanded:
                self.expanded.add(item["path"])  # break metadata cycles
                try:
                    self.walk(self.documents[item["path"]], document=path)
                except Exception:
                    self.expanded.discard(item["path"])
                    raise

    def walk(self, value, *, document=None, expand_pins=True):
        if isinstance(value, list):
            for item in value:
                self.walk(item, document=document, expand_pins=expand_pins)
        elif isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                self.include(value, document=document, expand_pins=expand_pins)
            for key, item in value.items():
                if key in {"source_sha256", "auditor_source_sha256", "source_before", "source_after"} and isinstance(item, dict):
                    self.source_inventory(item)
                elif key.endswith("_sha256") and isinstance(item, str):
                    name = key[:-7]
                    filename = value.get(name, value.get(name + "_path"))
                    if isinstance(filename, str) and Path(filename).is_absolute():
                        self.include(_pin(filename, item), document=document, expand_pins=expand_pins)
                if isinstance(item, (dict, list)):
                    self.walk(item, document=document, expand_pins=expand_pins)


def _validate_final_receipt(receipt, plan_pin, manifest_pin, campaign):
    _require(receipt.get("kind") == "alignlab-direct-v4-final-inherited-acceptance"
             and receipt.get("status") == "passed" and receipt.get("plan") == plan_pin
             and receipt.get("execution_manifest") == manifest_pin
             and receipt.get("stage_names") == SEQUENCE[:4]
             and campaign.get("inherited_prefix_reverified") is True,
             "Complete direct-V4 final inherited acceptance is required")
    times = [receipt.get("started_at_epoch"), receipt.get("finished_at_epoch"),
             campaign.get("finished_at_epoch"), campaign.get("deadline_epoch")]
    _require(all(type(t) in (int, float) and math.isfinite(t) for t in times),
             "Final acceptance timestamps must be finite")
    end = max(s["finished_at_epoch"] for s in campaign["stages"][-2:])
    _require(campaign["started_at_epoch"] <= end <= times[0] <= times[1] <= times[2] <= times[3],
             "Inherited acceptance must follow both exited seed44 stages within the original window")


def _manifest(evidence, directory, expected_sha, payload, format_name):
    directory = Path(directory).resolve()
    _require({p.name for p in directory.iterdir()} == {"manifest.json", payload},
             "Final bundle has missing or extra files")
    value = evidence.read(_pin(directory / "manifest.json", expected_sha))
    _require(value.get("format") == format_name and value.get("status") == "complete"
             and set(value.get("artifacts", {})) == {payload}, "Incomplete final bundle")
    item = value["artifacts"][payload]
    actual = evidence.add(directory / payload, item["sha256"])
    _require(actual["size_bytes"] == item["size_bytes"], "Final payload size differs")
    _require(value.get("identity_sha256") == hashlib.sha256(legacy._canonical(value["identity"])).hexdigest(),
             "Final bundle identity digest differs")
    return value


def _final_model(evidence, name, row, expected_parents):
    stage = MODELS[name]
    progress = evidence.read(_pin(row["progress_path"], row["progress_sha256"]))
    _require(progress.get("artifact", {}).get("artifact_dir") == row["artifact_dir"]
             and progress["artifact"].get("manifest_sha256") == row["manifest_sha256"],
             "Accepted model differs from its frozen progress")
    expected_updates = {"sft": 250, "reward": 822, "dpo": 822, "ppo": 2048}[stage]
    completion = progress["completion"]
    _require(completion.get("optimizer_updates") == expected_updates
             and completion.get("protocol_complete") is True
             and completion.get("data_pass_complete") is True
             and completion.get("stop_reason") == "data_exhausted", "Full final workload required")
    checkpoint = progress["latest_safe_checkpoint"]
    _require(checkpoint["optimizer_updates"] == expected_updates
             and any(c["checkpoint_path"] == checkpoint["checkpoint_path"]
                     and c["manifest_sha256"] == checkpoint["manifest_sha256"] for c in progress["checkpoints"]),
             "Final checkpoint must be a recorded completed boundary")
    artifact = _manifest(evidence, row["artifact_dir"], row["manifest_sha256"],
                         "weights.pt", "alignlab-completed-stage-v1")
    saved = _manifest(evidence, checkpoint["checkpoint_path"], checkpoint["manifest_sha256"],
                      "state.pt", "alignlab-optimizer-checkpoint-v1")
    _require(artifact["identity"] == progress["identity"] == saved["identity"]
             and artifact.get("stage") == stage and artifact.get("completion") == completion,
             "Final export/checkpoint/progress identity or completion differs")
    validate_formal_progress(progress, artifact, model_id=name)
    required = set() if stage == "sft" else ({"sft", "reward"} if stage == "ppo" else {"sft"})
    parents = artifact["identity"]["parents"]
    _require(set(parents) == required, "Wrong model parent set")
    for parent in required:
        bound = expected_parents[parent]
        _require(Path(parents[parent]["path"]).resolve() == Path(bound["artifact_dir"]).resolve() / "manifest.json"
                 and parents[parent]["sha256"] == bound["manifest_sha256"], "Model does not share the accepted parents")
    evidence.walk(progress, document=row["progress_path"], expand_pins=False)
    for receipt in progress["checkpoints"]:
        directory = Path(receipt["checkpoint_path"])
        saved_manifest = evidence.read(_pin(directory / "manifest.json", receipt["manifest_sha256"]))
        evidence.historical_payloads[str((directory / "state.pt").resolve())] = _pin(
            directory / "state.pt", saved_manifest["artifacts"]["state.pt"]["sha256"])
        evidence.walk(saved_manifest, document=directory / "manifest.json", expand_pins=False)
    return progress, artifact, saved, Path(checkpoint["checkpoint_path"])


def _verify_final_payload(progress, artifact_manifest, checkpoint_manifest, artifact_dir, checkpoint_dir):
    """Real final-only CPU tensor audit using frozen deserialization/comparison."""
    from .alignment_supervisor import _cpu_payload
    from .parent_acceptance import _final_weights
    import torch
    _require(not torch.cuda.is_initialized(), "CPU-only acceptance required")
    checkpoint = _cpu_payload(Path(checkpoint_dir) / "state.pt", checkpoint_manifest["artifacts"]["state.pt"]["sha256"])
    exported = _cpu_payload(Path(artifact_dir) / "weights.pt", artifact_manifest["artifacts"]["weights.pt"]["sha256"])
    _require(type(checkpoint) is dict and set(checkpoint) == {"parameters", "optimizer", "cursor", "rng", "extra_state"},
             "Final checkpoint payload schema differs")
    # Observed checkpoint cursors may contain an authenticated telemetry pin.
    cursor = dict(checkpoint["cursor"])
    expected = dict(progress["cursor"])
    telemetry = cursor.pop("telemetry", None)
    expected.pop("telemetry", None)
    if telemetry is not None:
        receipt = progress["latest_safe_checkpoint"]
        _require(telemetry == receipt.get("telemetry_sidecar"), "Actual final checkpoint telemetry pin differs")
        sidecar = legacy._read(telemetry["path"], telemetry["sha256"])[0]
        _require(sidecar.get("cursor") == cursor
                 and sidecar.get("identity_sha256") == checkpoint_manifest["identity_sha256"],
                 "Final checkpoint telemetry cursor or identity differs")
    _require(cursor == expected and cursor.get("optimizer_updates") == progress["completion"]["optimizer_updates"],
             "Actual final checkpoint cursor differs from completed progress")
    _require(checkpoint["extra_state"] is None and isinstance(checkpoint["parameters"], dict)
             and checkpoint["parameters"], "Final checkpoint retains unfinished work")
    _require(all(isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.is_floating_point()
                 and bool(torch.isfinite(t).all()) for t in checkpoint["parameters"].values()),
             "Final parameters must be finite CPU tensors")
    count = _final_weights(checkpoint, exported, artifact_manifest["source_role"], artifact_manifest["heads"])
    specs = [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype)} for name, t in exported.items()]
    _require(specs == artifact_manifest["tensor_specs"], "Export tensor specifications differ")
    _require(not torch.cuda.is_initialized(), "Acceptance initialized CUDA")
    return {"actual_cpu_payloads_verified": True, "final_checkpoint_export_equal": True,
            "tensor_count": count, "optimizer_updates": cursor["optimizer_updates"]}


def _context(bindings, *, payload_audit=False, system=None):
    evidence = Evidence()
    plan_pin, campaign_pin, manifest_pin = (bindings[k] for k in ("plan", "campaign", "execution_manifest"))
    plan, campaign, manifest = (evidence.read(p) for p in (plan_pin, campaign_pin, manifest_pin))
    _require(plan.get("kind") == "alignlab-throughput-campaign-v3"
             and campaign.get("kind") == plan["kind"] and campaign.get("status") == "primary_alignment_passed"
             and all(campaign.get(k) is True for k in ("primary_alignment_verified", "parents_verified", "capacity_verified", "formal_training"))
             and campaign.get("research_complete") is False and not campaign.get("error")
             and not campaign.get("child_may_still_be_running")
             and campaign.get("plan_path") == plan_pin["path"] and campaign.get("plan_sha256") == plan_pin["sha256"]
             and Path(campaign_pin["path"]).resolve() == Path(plan["output"]).resolve() / "campaign.json"
             and campaign.get("direct_execution_manifest") == manifest_pin,
             "Exact clean completed direct-V4 campaign required")
    _require(manifest.get("kind") == "alignlab-user-directed-training-entry-v4"
             and manifest.get("plan") == plan_pin and manifest.get("no_new_tests_run") is True
             and manifest.get("defer_inherited_reverification_until_final") is True,
             "Separate direct-entry authority required")
    evidence.source_inventory(manifest["source_sha256"])
    authority = evidence.read(manifest["authorization"])
    _require(authority.get("source") == "direct_user_message_in_this_chat"
             and authority.get("user_message") == "直接进入训练好吗 我最后说一遍", "Direct-entry authorization differs")
    _require([r.get("name") for r in campaign.get("stages", [])] == SEQUENCE, "Exactly all six ordered stages required")
    supervisors = campaign.get("supervisor_processes")
    _require(isinstance(supervisors, list) and supervisors, "Exact completed supervisor identities required")
    for process in supervisors:
        command = process.get("command_line", "").replace("\\", "/")
        _require("throughput_direct_v4/run_alignlab_throughput_campaign_v3.py" in command
                 and plan_pin["path"].replace("\\", "/") in command and plan_pin["sha256"] in command
                 and manifest_pin["path"].replace("\\", "/") in command and manifest_pin["sha256"] in command,
                 "Completed supervisor command is not bound to direct-V4 plan and manifest")
    final_receipt = evidence.read(campaign["inherited_prefix_final_acceptance"])
    _validate_final_receipt(final_receipt, plan_pin, manifest_pin, campaign)
    original = evidence.read(plan["original_plan"])
    parent = evidence.read(_pin(original["parent_acceptance_path"], original["parent_acceptance_sha256"]))
    _require(parent.get("kind") == "alignlab-independent-recovered-parents-r5-acceptance-v1"
             and parent.get("status") == "accepted_full_real_recovered_parents_r5"
             and all(parent.get(k) is True for k in ("all_checks_passed", "parents_verified", "formal_training", "processes_exited", "actual_cpu_payloads_verified"))
             and parent.get("cuda_initialized") is False and parent.get("device") == "cpu"
             and set(parent.get("stages", {})) == {"sft", "reward"}, "Completed real R5 parent acceptance required")
    recovery = evidence.read(parent["recovery_plan"])
    for path, item in recovery.get("historical_prerequisites", {}).items():
        _require(item["original_path"] == path and item["snapshot"]["sha256"] == item["original_sha256"]
                 and item["semantics"] == "historical_admission_fact_not_current_resource_state",
                 "Historical admission mapping differs")
        evidence.historical[(str(Path(path).resolve()), item["original_sha256"])] = item["snapshot"]["path"]
    prior = evidence.read(plan["failed_v2_campaign"])
    _require(prior.get("kind") == "alignlab-throughput-campaign-v2" and prior.get("status") == "failed"
             and prior.get("primary_alignment_verified") is False
             and [r.get("name") for r in prior.get("stages", [])] == SEQUENCE[:4], "Exact failed V2 prefix required")
    supplemental = evidence.read(plan["supplemental_ppo43"])
    _require(supplemental.get("status") == "ppo43_supplemental_accepted"
             and supplemental.get("original_full_acceptance_passed") is True and isinstance(supplemental.get("binding"), dict),
             "PPO43 real supplemental acceptance required")
    for index, record in enumerate(campaign["stages"]):
        _require(record.get("status") == "accepted" and record.get("exit_code") == 0
                 and record.get("active_processes") == [] and not record.get("monitoring_error")
                 and record.get("stop_reason") is None and record.get("known_processes"),
                 "All stages require complete acceptance and observed clean exit")
        _require(record["stage"] == record["name"][:-2] and record["seed"] == int(record["name"][-2:]), "Stage/seed differs")
        if index < 3:
            _require(record == prior["stages"][index], "Inherited accepted stage receipt changed")
        elif index == 3:
            _require(record["request"] == prior["stages"][3]["request"]
                     and record["progress_sha256"] == supplemental["original_progress_sha256"]
                     and record["artifact"] == supplemental["artifact"]
                     and record.get("supplemental_acceptance") == plan["supplemental_ppo43"], "PPO43 supplemental binding differs")
        else:
            _require(record["request"].get("plan") == plan_pin
                     and record["request"].get("execution_window") == plan["execution_window"], "Seed44 request has wrong current authority")
            args = record.get("command", [])
            _require(manifest_pin["sha256"] in args and "--execution-manifest-sha256" in args
                     and any("throughput_direct_v4" in a.replace("\\", "/") for a in args),
                     "Seed44 command lost direct-entry authority")
            for process in record["known_processes"]:
                command = process.get("command_line", "").replace("\\", "/")
                _require("throughput_direct_v4/run_alignlab_throughput_fresh_v3.py" in command
                         and record["request_sha256"] in command and manifest_pin["sha256"] in command,
                         "Seed44 child identity differs from its frozen request and direct manifest")
        output = Path(record["progress_path"]).resolve().parent
        request_path = output.parent / (record["name"] + ".request.json")
        _require(evidence.read(_pin(request_path, record["request_sha256"])) == record["request"], "Recorded request differs from disk")
    # The final direct-V4 receipt reaccepted the inherited four stages, including
    # replay609 and PPO43 reconstruction. Authenticate the historical receipts,
    # not every older assertion recursively embedded in their provenance.
    # In particular, a pre-training diagnostic may legitimately pin an archived
    # prototype source revision. The current plans, R5 auditor and actual final
    # model identities still authenticate their current frozen source and input
    # files. No historical pin is replaced with a current digest.
    evidence.source_inventory(parent["auditor_source_sha256"])
    for key in ("parent_plan", "original_campaign", "recovery_manifest"):
        evidence.include(parent[key], expand_pins=False)
    for value, document in ((plan, plan_pin["path"]), (campaign, campaign_pin["path"]),
                            (manifest, manifest_pin["path"]), (original, plan["original_plan"]["path"]),
                            (prior, plan["failed_v2_campaign"]["path"]),
                            (supplemental, plan["supplemental_ppo43"]["path"])):
        evidence.walk(value, document=document, expand_pins=False)
    models = {"base": {"stage": "base", "artifact_dir": None, "manifest_sha256": None,
                       "progress_path": None, "progress_sha256": None}}
    rows = {name: parent["stages"][name] for name in ("sft", "reward")}
    for record in campaign["stages"]:
        rows[record["name"]] = {**record["artifact"], "progress_path": record["progress_path"], "progress_sha256": record["progress_sha256"]}
    expected = [p["expected"] for p in parent["processes"]]
    protected = set(parent["protected_paths"]) | {plan["output"], original["output"]}
    for document in (campaign, prior, *[v for v in evidence.documents.values() if isinstance(v, dict)]):
        if document.get("supervisor_processes"):
            expected.extend(document["supervisor_processes"])
        for record in document.get("stages", []) if isinstance(document.get("stages"), list) else []:
            expected.extend(record.get("known_processes", []))
            if record.get("progress_path"):
                protected.add(str(Path(record["progress_path"]).parent))
    from .probe_dispatch import _System
    system = system or _System()
    _process_history(expected, system.processes(), protected)
    audits, finals = {}, {}
    for name, row in rows.items():
        if name in ("sft", "reward"):
            _require(all(row[k] == original["parents"][name][k] for k in ("artifact_dir", "manifest_sha256")), "Original shared parent differs")
        progress, artifact, checkpoint, checkpoint_dir = _final_model(evidence, name, row, rows)
        models[name] = {"stage": MODELS[name], **{k: row[k] for k in ("artifact_dir", "manifest_sha256", "progress_path", "progress_sha256")}}
        finals[name] = {"checkpoint_manifest": _pin(checkpoint_dir / "manifest.json", evidence.add(checkpoint_dir / "manifest.json")["sha256"]),
                        "checkpoint_payload": evidence.add(checkpoint_dir / "state.pt"),
                        "export_payload": evidence.add(Path(row["artifact_dir"]) / "weights.pt")}
        if payload_audit:
            audits[name] = _verify_final_payload(progress, artifact, checkpoint, row["artifact_dir"], checkpoint_dir)
    # Base/tokenizer/data need current content integrity, not a historical stat.
    snapshot_dir = Path(original["model_snapshot"])
    snapshot = evidence.read(_pin(snapshot_dir / "snapshot-manifest.json", original["model_manifest_sha256"]))
    evidence.walk(snapshot, document=snapshot_dir / "snapshot-manifest.json", expand_pins=False)
    for name, item in snapshot["files"].items():
        _require((snapshot_dir / name).resolve().is_relative_to(snapshot_dir.resolve()), "Base file escaped snapshot")
        evidence.add(snapshot_dir / name, item["sha256"])
    data_dir = Path(original["audit_dir"])
    data = evidence.read(_pin(data_dir / "manifest.json", original["audit_manifest_sha256"]))
    for name, digest in data["artifacts"].items():
        _require((data_dir / name).resolve().is_relative_to(data_dir.resolve()), "Data file escaped audit")
        evidence.add(data_dir / name, digest)
    for item in data["identity"]["inputs"].values():
        evidence.add(item["path"], item["sha256"])
    evidence.source_inventory(data["identity"]["source_sha256"])
    tokenizer = data["identity"]["tokenizer"]
    _require(Path(tokenizer["path"]).resolve() == Path(original["tokenizer_dir"]).resolve(), "Tokenizer differs")
    for name, digest in tokenizer["files"].items():
        evidence.add(Path(tokenizer["path"]) / name, digest)
    if data["identity"].get("protocol_sha256"):
        evidence.add(ROOT / "docs/ALIGNLAB_V2.md", data["identity"]["protocol_sha256"])
    histories = _process_history(expected, system.processes(), protected)
    for directory in {Path(plan["output"]), *(Path(r["progress_path"]).parent for r in rows.values())}:
        _require(not (directory / "STOP").exists(), "Completed output has a STOP marker")
    _require(not Path(str(plan_pin["path"]) + ".stop").exists(), "Completed plan has a stop marker")
    for name in SOURCES:
        evidence.add(ROOT / name)
    if payload_audit:
        # Reuse the exact frozen metadata/authority validators. They do not
        # deserialize intermediate optimizer archives or launch any computation.
        from .throughput_supervisor_direct_v4 import validate_manifest, _receipt_view
        from .throughput_fresh_direct_v4 import _accepted_receipt
        validate_manifest(manifest_pin, plan_pin)
        view = _receipt_view(plan_pin["path"], plan_pin["sha256"])
        _require(view["_inherited_stages"] == campaign["stages"][:4], "Frozen inherited metadata verifier differs")
        verified_parent = _accepted_receipt(original["parent_acceptance_path"], original["parent_acceptance_sha256"])
        _require(verified_parent == parent, "Frozen R5 acceptance receipt differs")
    return {"models": models, "evidence_files": evidence.inventory,
            "historical_tensor_pins": evidence.historical_payloads, "final_payloads": finals,
            "payload_audits": audits, "processes": histories, "protected_paths": sorted(protected),
            "model_snapshot": original["model_snapshot"], "model_manifest_sha256": original["model_manifest_sha256"],
            "tokenizer_dir": original["tokenizer_dir"]}


def audit_evaluation_models(plan_path, plan_sha256, campaign_path, campaign_sha256,
                            execution_manifest_path, execution_manifest_sha256, *, output_dir, system=None):
    output = Path(output_dir).resolve()
    _require(output.parent == ROOT / "reports" and not output.exists(), "Use a fresh direct reports directory")
    output.mkdir()
    bindings = {"plan": _pin(plan_path, plan_sha256), "campaign": _pin(campaign_path, campaign_sha256),
                "execution_manifest": _pin(execution_manifest_path, execution_manifest_sha256)}
    report = {"schema_version": 2, "kind": KIND, "status": "failed", "models_verified": False,
              "all_checks_passed": False, "bindings": bindings, "acceptance_policy": POLICY,
              "device": "cpu", "human_labels": 0, "human_win_rate": None, "research_complete": False,
              "historical_full_audit_repeated": False, "blockers": []}
    try:
        result = _context(bindings, payload_audit=True, system=system)
        _require(legacy._cuda_uninitialized(), "CUDA initialized during CPU acceptance")
        report.update(result, status="accepted_for_evaluation", models_verified=True,
                      all_checks_passed=True, processes_exited=True, actual_cpu_payloads_verified=True,
                      auditor_source_sha256={name: result["evidence_files"][str((ROOT / name).resolve())]["sha256"] for name in SOURCES},
                      cuda_initialized=False, snapshot_manifest_sha256=result["model_manifest_sha256"])
        accepted = legacy._new(output / "accepted-models.json", report)
        roster = {"schema_version": 2, "kind": ROSTER_KIND,
                  **{k: result[k] for k in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir")},
                  "models": {name: {"model_id": name, **{k: row[k] for k in ("stage", "artifact_dir", "manifest_sha256")},
                                    "acceptance_path": accepted["path"], "acceptance_sha256": accepted["sha256"]}
                             for name, row in result["models"].items()}}
        roster_pin = legacy._new(output / "roster.json", roster)
        result = {"status": "evaluation_models_accepted", "models_verified": True, "blockers": [],
                  "accepted_models": accepted, "roster": roster_pin}
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", blockers=["verification_failed"])
        result = report
    legacy._new(output / "report.json", result)
    return result


def verify_evaluation_roster(roster_path, expected_sha256, *, verify_bound_files=True, system=None):
    """Authenticate exact evidence closure and current finals, without torch load."""
    _require(verify_bound_files is True, "V2 never permits skipping bound-file verification")
    roster, _ = legacy._read(roster_path, expected_sha256)
    _require(set(roster) == {"schema_version", "kind", "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "models"}
             and roster["schema_version"] == 2 and roster["kind"] == ROSTER_KIND
             and set(roster["models"]) == set(MODELS), "Exact nine-model V2 roster required")
    pins = {(r["acceptance_path"], r["acceptance_sha256"]) for r in roster["models"].values()}
    _require(len(pins) == 1, "All models must share one independent acceptance")
    path, sha256 = pins.pop()
    accepted, _ = legacy._read(path, sha256)
    _require(accepted.get("kind") == KIND and accepted.get("schema_version") == 2
             and accepted.get("status") == "accepted_for_evaluation"
             and accepted.get("acceptance_policy") == POLICY
             and all(accepted.get(k) is True for k in ("models_verified", "all_checks_passed", "processes_exited", "actual_cpu_payloads_verified"))
             and accepted.get("cuda_initialized") is False and accepted.get("device") == "cpu"
             and set(accepted.get("auditor_source_sha256", {})) == set(SOURCES), "Complete V2 independent CPU acceptance required")
    for name, digest in accepted["auditor_source_sha256"].items():
        legacy._file(ROOT / name, digest)
    rebuilt = _context(accepted["bindings"], system=system)
    for key in ("models", "evidence_files", "historical_tensor_pins", "final_payloads", "protected_paths",
                "model_snapshot", "model_manifest_sha256", "tokenizer_dir"):
        _require(rebuilt[key] == accepted[key], "Accepted evidence closure changed or was thinned: " + key)
    _require(set(accepted.get("payload_audits", {})) == set(MODELS) - {"base"}
             and all(r.get("actual_cpu_payloads_verified") is True and r.get("final_checkpoint_export_equal") is True
                     for r in accepted["payload_audits"].values()), "Actual final CPU payload proof missing")
    for name, row in accepted["models"].items():
        expected = {"model_id": name, **{k: row[k] for k in ("stage", "artifact_dir", "manifest_sha256")},
                    "acceptance_path": path, "acceptance_sha256": sha256}
        _require(roster["models"][name] == expected, "Roster model differs from independent acceptance")
    for key in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir"):
        _require(roster[key] == accepted[key], "Roster input identity differs")
    return roster
