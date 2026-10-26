"""Explicit one-attempt recovery after supplemental PPO43 acceptance.

The frozen failed V2 campaign is never rewritten. Only the two remaining seed44
branches can be dispatched, under an explicitly replaced deadline and the
unchanged process guard. Every historical budget remains immutable.
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
from . import throughput_supervisor_v2 as prior
from .probe_dispatch import _System, _stamp, _parse
from .capacity_supervisor import sha, _pin
from .model_snapshot import verify_snapshot, REPO_ID, REVISION

ROOT = legacy.ROOT
KIND = "alignlab-throughput-campaign-v3"
STAGE_KIND = legacy.STAGE_KIND
SEQUENCE = legacy.SEQUENCE
REMAINING_SEQUENCE = SEQUENCE[4:]
JsonWriter = legacy.JsonWriter
_command = legacy._command
_same_identity = legacy._same_identity
_waitable_peer_decision = legacy._waitable_peer_decision
_peer_decision = legacy._peer_decision
_parents_gone = legacy._parents_gone
_cache = legacy._cache
workload = legacy.workload
NEW_SOURCES = prior.NEW_SOURCES | {
    "alignment_lab/throughput_supervisor_v3.py", "scripts/run_alignlab_throughput_campaign_v3.py",
    "tests/test_throughput_supervisor_v3.py", "scripts/run_alignlab_throughput_cpu_checks_v3.py",
    "scripts/run_alignlab_throughput_prepare_v3.py",
    "alignment_lab/throughput_fresh_v3.py", "scripts/run_alignlab_throughput_fresh_v3.py",
    "tests/test_throughput_fresh_v3.py", "alignment_lab/ppo43_supplemental_v3.py",
    "tests/test_ppo43_supplemental_v3.py"}
PLAN_KEYS = {"schema_version", "kind", "output", "original_plan", "prior_campaign",
    "original_started_at_epoch", "original_deadline_epoch", "original_request", "history",
    "source_sha256", "authorization", "validation_report", "cpu_report", "accepted_dpo42",
    "selected_backend", "process_wait_spent_seconds", "future_ppo_backend",
    "failed_v2_plan", "failed_v2_campaign", "recovery_authorization", "supplemental_ppo43",
    "budget_authorization", "execution_window"}
FAILURE = "ValueError: Live optimizer observations differ from returned actual objectives/work"


def _pin_record(record):
    return prior._pin_record(record)


def _cpu_proof(plan):
    report = _parse(_pin_record(plan["cpu_report"]))
    required = {name[:-3].replace("/", ".") for name in plan["source_sha256"]
        if name.endswith(".py") and (name.startswith(("tests_throughput_v1/test_", "tests_throughput_v2/test_"))
        or name in NEW_SOURCES and name.startswith("tests/test_"))}
    if (report.get("kind") != "alignlab-throughput-cpu-validation-v3"
            or report.get("source_before") != plan["source_sha256"]
            or report.get("source_after") != plan["source_sha256"]
            or report.get("cuda_initialized") is not False or report.get("device") != "cpu"
            or type(report.get("tests_run")) is not int or report["tests_run"] < 1
            or any(type(report.get(k)) is not int or report[k] != 0 for k in ("errors", "failures", "skipped"))
            or report.get("status") != "passed" or not required
            or not required <= set(report.get("modules", []))):
        raise ValueError("Exact current V3 source-bound CPU proof with complete tests and no CUDA required")


def _recovery_authority(plan):
    """Validate the exact fresh human authorization, never a general retry rule."""
    authorization = _parse(_pin_record(plan["recovery_authorization"]))
    scope = authorization.get("scope", {})
    if (authorization.get("kind") != "alignlab-explicit-ppo43-supplemental-recovery-authorization-v3"
            or authorization.get("source") != "direct_user_reply_in_this_chat"
            or authorization.get("user_selected_answer") != "允许补验收并继续剩余训练"
            or authorization.get("prior_plan") != plan["failed_v2_plan"]
            or authorization.get("prior_campaign") != plan["failed_v2_campaign"]
            or authorization.get("original_started_at_epoch") != resume.STARTED
            or authorization.get("original_deadline_epoch") != resume.DEADLINE
            or authorization.get("wall_seconds") != 259200
            or scope.get("remaining_training") != ["dpo44", "ppo44"]
            or scope.get("retrain_ppo43") is not False
            or scope.get("future_automatic_retry_authorized") is not False
            or scope.get("supplemental_acceptance_existing_ppo43") is not True
            or scope.get("original_sources_outputs_preserved") is not True
            or scope.get("requires_full_independent_acceptance") is not True):
        raise ValueError("Exact user-authorized PPO43 supplemental acceptance and seed44-only recovery required")


def validate_execution_window(plan):
    """The later direct user message replaces only this attempt's deadline."""
    authorization = _parse(_pin_record(plan["budget_authorization"]))
    window = plan["execution_window"]
    expected = {"kind": "authorized_remaining_training_window_v3",
        "started_at_epoch": 1792833376.012, "deadline_epoch": 1793006176.012, "wall_seconds": 172800,
        "scope": ["dpo44", "ppo44"], "completion_stops_early": True,
        "chosen_by": "assistant_under_user_authorization_to_proceed_without_original_deadline",
        "numeric_hours_explicitly_supplied_by_user": False,
        "all_preparation_validation_waiting_training_and_acceptance_count": True,
        "old_deadline_applies_to_historical_runs_only": True,
        "additional_training_or_automatic_retry": False}
    if (type(window) is not dict or window != expected
            or authorization.get("kind") != "alignlab-explicit-deadline-replacement-authorization-v3"
            or authorization.get("source") != "direct_user_message_in_this_chat"
            or authorization.get("user_message") != "不管截止时间 你自己跑就行了 后面两天我都不在电脑前 你要自己保持项目进行"
            or authorization.get("execution_window") != window
            or authorization.get("recovery_authorization") != plan["recovery_authorization"]
            or authorization.get("original_started_at_epoch") != resume.STARTED
            or authorization.get("original_deadline_epoch") != resume.DEADLINE
            or authorization.get("original_wall_seconds") != 259200
            or authorization.get("guard_policy_unchanged") is not True):
        raise ValueError("Exact separately authorized remaining-training execution window required")
    return copy.deepcopy(window)


def _checked_prefix(plan):
    """Pinned metadata is checked here; full artifact acceptance follows separately."""
    if (type(plan) is not dict or set(plan) != PLAN_KEYS
            or type(plan["schema_version"]) is not int or plan["schema_version"] != 1
            or plan["kind"] != KIND or plan["original_started_at_epoch"] != resume.STARTED
            or plan["original_deadline_epoch"] != resume.DEADLINE
            or plan["selected_backend"] != "original" or plan["future_ppo_backend"] != "original"):
        raise ValueError("Exact same-deadline V3 recovery plan required")
    old = _parse(_pin_record(plan["failed_v2_plan"]))
    if old.get("kind") != prior.KIND:
        raise ValueError("Recovery must bind the failed V2 plan")
    for key in set(old) - {"schema_version", "kind", "output", "source_sha256", "cpu_report", "validation_report", "process_wait_spent_seconds"}:
        if plan.get(key) != old[key]:
            raise ValueError("Recovery changed frozen V2 ancestry, workload or authority: " + key)
    resume._source_pins(plan["source_sha256"], old["source_sha256"])
    if not NEW_SOURCES <= plan["source_sha256"].keys():
        raise ValueError("Complete V3 supervisor, serializer and supplemental verifier sources/tests required")
    _cpu_proof(plan)
    _recovery_authority(plan)
    validate_execution_window(plan)
    resume.validate_execution_authority(plan, _parse(_pin_record(plan["original_request"])))
    output = Path(plan["output"])
    if (not output.is_absolute() or output.resolve().parent != (ROOT / "runs").resolve()
            or output.resolve() == Path(old["output"]).resolve()
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("Independent fresh run output required")
    campaign_path = _pin_record(plan["failed_v2_campaign"])
    campaign = _parse(campaign_path)
    if (campaign_path != Path(old["output"]).resolve() / "campaign.json"
            or campaign.get("kind") != prior.KIND or campaign.get("status") != "failed"
            or campaign.get("error") != FAILURE or campaign.get("primary_alignment_verified") is not False
            or campaign.get("plan_sha256") != plan["failed_v2_plan"]["sha256"]
            or campaign.get("started_at_epoch") != resume.STARTED
            or campaign.get("deadline_epoch") != resume.DEADLINE or campaign.get("child_may_still_be_running")
            or [s.get("name") for s in campaign.get("stages", [])] != ["dpo42", "ppo42", "dpo43", "ppo43"]):
        raise ValueError("Only the exact failed four-stage V2 prefix is authorized")
    for index, record in enumerate(campaign["stages"]):
        if (record.get("exit_code") != 0 or record.get("active_processes")
                or not record.get("known_processes") or record.get("monitoring_error")
                or index < 3 and record.get("status") != "accepted"
                or index == 3 and record.get("status") == "accepted"):
            raise ValueError("Prior stage exit/acceptance/identity evidence is incomplete")
    receipt = _parse(_pin_record(plan["supplemental_ppo43"]))
    if (receipt.get("status") != "ppo43_supplemental_accepted"
            or receipt.get("original_full_acceptance_passed") is not True
            or type(receipt.get("binding")) is not dict):
        raise ValueError("Successful independently reconstructed full PPO43 acceptance required")
    request = campaign["stages"][-1]["request"]
    stage_output = Path(request["legacy_request"]["output"]).resolve()
    progress = _parse(stage_output / "progress.json")
    if (receipt.get("original_progress_sha256") != sha(stage_output / "progress.json")
            or receipt.get("artifact") != progress.get("artifact")):
        raise ValueError("Supplemental acceptance is not bound to the exact original PPO43 output")
    spent = plan["process_wait_spent_seconds"]
    if (type(spent) not in (int, float) or not math.isfinite(spent)
            or spent != progress.get("process_guard", {}).get("total_waited_seconds")
            or not campaign["total_process_wait_seconds"] <= spent <= 259200):
        raise ValueError("Recovery cannot reset or invent the shared wait ledger")
    return old, campaign, receipt


def validate_fresh_authority(plan, wrapper, old):
    """Child checks pinned current authority without repeating full prefix loading."""
    _checked_prefix(plan)
    for key in ("recovery_authorization", "supplemental_ppo43", "source_sha256", "budget_authorization", "execution_window"):
        if wrapper.get(key) != plan[key]:
            raise ValueError("Child recovery authorization differs from the V3 plan: " + key)
    if (old.get("stage") != "ppo" or old.get("seed") != 42
            or plan["original_plan"] != {"path": old["alignment_plan_path"], "sha256": old["alignment_plan_sha256"]}):
        raise ValueError("Child lost the original registered R3 lineage")
    return True


def _known_for_stage(record, progress):
    """Reconcile only equivalent UTC spellings; retain all exact identity fields."""
    from .process_guard_v2 import ProcessIdentity
    known = {p["pid"]: copy.deepcopy(p) for p in record["known_processes"]}
    for observed in progress["process_guard"]["owned_identities"]:
        candidate = known.get(observed["pid"])
        if candidate is None or ProcessIdentity.from_record(candidate) != ProcessIdentity.from_record(observed):
            raise ValueError("Inherited child PID/create/command/parent identity mismatch")
        if ({k: v for k, v in candidate.items() if k != "created_at_utc"}
                != {k: v for k, v in observed.items() if k != "created_at_utc"}):
            raise ValueError("Inherited child non-time identity fields changed")
        known[observed["pid"]] = copy.deepcopy(observed)
    if set(known) != {p["pid"] for p in progress["process_guard"]["owned_identities"]}:
        raise ValueError("Inherited child process inventory differs")
    return known


def _metadata_plan(path, expected_sha):
    """Construct the immutable input inventory before full semantic acceptance."""
    path = Path(path).resolve()
    _pin(path, expected_sha)
    plan = _parse(path)
    _, campaign, receipt = _checked_prefix(plan)
    original = legacy.validate_plan(plan["original_plan"]["path"], plan["original_plan"]["sha256"])
    inherited = copy.deepcopy(campaign["stages"])
    final = inherited[-1]
    output = Path(final["request"]["legacy_request"]["output"]).resolve()
    final.update(status="accepted", progress_path=str(output / "progress.json"),
        progress_sha256=receipt["original_progress_sha256"], artifact=receipt["artifact"],
        reference_cache=None, total_process_wait_seconds=plan["process_wait_spent_seconds"],
        supplemental_acceptance=copy.deepcopy(plan["supplemental_ppo43"]),
        original_failed_campaign=copy.deepcopy(plan["failed_v2_campaign"]),
        original_status=campaign["stages"][-1]["status"])
    view = {**copy.deepcopy(original), **copy.deepcopy(plan), "_original": original,
        "_authorized": plan, "_prior_campaign": campaign, "_inherited_stages": inherited}
    view["continuation"] = {**original["continuation"], "process_wait_spent_seconds": plan["process_wait_spent_seconds"],
        "started_at_epoch": plan["execution_window"]["started_at_epoch"],
        "deadline_epoch": plan["execution_window"]["deadline_epoch"]}
    _pin(path, expected_sha)
    return view


def validate_plan(path, expected_sha):
    path = Path(path).resolve()
    _pin(path, expected_sha)
    plan = _parse(path)
    old, campaign, receipt = _checked_prefix(plan)
    # This reproduces all frozen ancestry, original DPO42 acceptance and the
    # explicit 609 replay authorization before accepting any inherited result.
    previous_view = prior.validate_plan(plan["failed_v2_plan"]["path"], plan["failed_v2_plan"]["sha256"])
    inherited = []
    for record in campaign["stages"][:3]:
        request = record["request"]
        stage_output = Path(request.get("legacy_request", request)["output"]).resolve()
        progress_path = stage_output / "progress.json"
        progress = _parse(progress_path)
        if (record["progress_sha256"] != sha(progress_path)
                or Path(record["progress_path"]).resolve() != progress_path):
            raise ValueError("Inherited accepted progress changed")
        result = prior._verify_stage(stage_output, request, _known_for_stage(record, progress), 0)
        if result["artifact"] != record["artifact"]:
            raise ValueError("Inherited accepted artifact changed")
        inherited.append(copy.deepcopy(record))
    from .ppo43_supplemental_v3 import verify_bound
    result, reproduced = verify_bound(receipt["binding"])
    if reproduced != receipt:
        raise ValueError("PPO43 supplemental acceptance is not reproducible from the pinned evidence")
    final = copy.deepcopy(campaign["stages"][-1])
    output = Path(final["request"]["legacy_request"]["output"]).resolve()
    if (result["artifact"] != receipt["artifact"]
            or result["process_guard"]["total_waited_seconds"] != plan["process_wait_spent_seconds"]):
        raise ValueError("Supplemental PPO43 artifact or wait ledger differs")
    final.update(status="accepted", progress_path=str(output / "progress.json"),
        progress_sha256=receipt["original_progress_sha256"], artifact=result["artifact"],
        reference_cache=None, total_process_wait_seconds=plan["process_wait_spent_seconds"],
        supplemental_acceptance=copy.deepcopy(plan["supplemental_ppo43"]),
        original_failed_campaign=copy.deepcopy(plan["failed_v2_campaign"]),
        original_status=campaign["stages"][-1]["status"])
    inherited.append(final)
    view = {**copy.deepcopy(previous_view["_original"]), **copy.deepcopy(plan)}
    view["_original"] = previous_view["_original"]
    view["_authorized"] = plan
    view["_prior_campaign"] = campaign
    view["_inherited_stages"] = inherited
    view["continuation"] = {**view["_original"]["continuation"],
        "process_wait_spent_seconds": plan["process_wait_spent_seconds"],
        "started_at_epoch": plan["execution_window"]["started_at_epoch"],
        "deadline_epoch": plan["execution_window"]["deadline_epoch"]}
    _pin(path, expected_sha)
    return view


def _prior_processes_gone(plan, processes):
    known = plan["_prior_campaign"]["supervisor_processes"][:]
    for stage in plan["_prior_campaign"]["stages"]:
        known.extend(stage["known_processes"])
    from .process_guard_v2 import ProcessIdentity
    for actual in processes:
        for old in known:
            if ProcessIdentity.from_record(actual).instance == ProcessIdentity.from_record(old).instance:
                raise RuntimeError("Original V2 process is still alive; parallel recovery is forbidden")


class _LiveIdentity:
    @classmethod
    def from_plan(cls, path, expected_sha, plan):
        sentinel = prior._LiveIdentity.from_plan(path, expected_sha, plan)
        for record in plan["_inherited_stages"]:
            request = record["request"].get("legacy_request", record["request"])
            sentinel.add_tree(request["output"])
            progress = _parse(Path(request["output"]) / "progress.json")
            for item in progress["identity"]["inputs"].values():
                # Full hashing is captured separately; active ten-second checks
                # preserve the original inexpensive metadata/small-file policy.
                sentinel.add_file(item["path"])
            for checkpoint in progress["checkpoints"]:
                # Inherited PPO42 checkpoints reference sidecars/rollouts in
                # the original failed stage's sibling telemetry directories.
                sentinel.add_tree(Path(checkpoint["checkpoint_path"]).parent)
        _add_document_dependencies(sentinel)
        return sentinel


def _add_document_dependencies(sentinel):
    """Close all pinned JSON authority/evidence references before full hashing."""
    visited = set()

    def include(filename):
        filename = Path(filename).resolve()
        if not filename.is_relative_to(ROOT):
            raise ValueError("Acceptance dependency escaped the project")
        sentinel.add_file(filename)
        key = str(filename)
        if filename.suffix.lower() != ".json" or key in visited:
            return
        visited.add(key)
        visit(_parse(filename), filename.parent)

    def visit(value, document_directory):
        if type(value) is dict:
            if type(value.get("path")) is str and type(value.get("sha256")) is str:
                candidate = Path(value["path"])
                if candidate.is_absolute():
                    include(candidate)
                else:
                    candidates = [document_directory / candidate, ROOT / candidate]
                    existing = [p for p in candidates if p.is_file()]
                    if not existing:
                        raise ValueError("Relative pinned acceptance input disappeared")
                    for item in existing:
                        include(item)
            for name, digest in value.items():
                if name.endswith("_sha256") and type(digest) is str:
                    base = name[:-7]
                    filename = value.get(base, value.get(base + "_path"))
                    if type(filename) is str:
                        candidate = Path(filename)
                        candidate = candidate if candidate.is_absolute() else ROOT / candidate
                        # Some JSON fields hash literal text rather than file
                        # names. Explicit {path,sha256} pins are handled above;
                        # paired legacy filenames are included when files exist.
                        # Missing required inputs still fail full acceptance.
                        if candidate.is_file():
                            include(candidate)
                if type(digest) in (dict, list):
                    visit(digest, document_directory)
        elif type(value) is list:
            for item in value:
                if type(item) in (dict, list):
                    visit(item, document_directory)

    # Snapshot the current list: include() recursively discovers further pinned
    # documents. No training output or mutable handoff is added by convention.
    for filename in list(sentinel.files):
        if Path(filename).suffix.lower() == ".json":
            include(filename)


class _ContentIdentity:
    """Full bytes and inventories bracket one full semantic acceptance.

    Subsequent boundaries rehash every one of these immutable inputs. There is
    no baseline refresh and no reliance on stat equality to prove content.
    """
    def __init__(self, live):
        self.live = live
        live.check()
        self.members = copy.deepcopy(live.trees)
        self.sha256 = {path: sha(path) for path in sorted(live.files)}
        live.check()

    def check(self):
        self.live.check()
        for directory, expected in self.members.items():
            if self.live._members(directory) != expected:
                raise ValueError("Full acceptance input inventory changed")
        for path, expected in self.sha256.items():
            if sha(path) != expected:
                raise ValueError("Full acceptance input content changed: " + path)
        self.live.check()
        return True

    def record(self):
        return {"kind": "alignlab-v3-full-content-acceptance-binding", "full_content_sha256": self.sha256,
            "immutable_tree_members": self.members, "stat_only_acceptance": False,
            "recheck_boundaries": ["after_initial_full_semantic_acceptance", "before_each_dispatch",
                                  "after_each_child_exit", "final_publication"]}


def _own_processes(processes, plan_path):
    by_pid = {p["pid"]: p for p in processes}
    current, own = by_pid.get(os.getpid()), {}
    expected_sha = sha(plan_path)
    while current and _command(current, "run_alignlab_throughput_campaign_v3.py", plan_path):
        if current["pid"] in own or expected_sha not in current["command_line"]:
            raise RuntimeError("Supervisor command/ancestry mismatch")
        _stamp(current["created_at_utc"])
        own[current["pid"]] = current
        current = by_pid.get(current.get("parent_pid"))
    if not own:
        raise RuntimeError("Cannot identify exact V3 supervisor chain")
    return own


def _build_request(plan, plan_path, expected_sha, own, waited, stage, seed, cache_pin):
    if (stage, seed) not in REMAINING_SEQUENCE:
        raise ValueError("Only remaining DPO44/PPO44 may be dispatched")
    old = plan["_original"]
    keys = ("source_sha256", "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
        "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path",
        "parent_plan_sha256", "parent_acceptance_path", "parent_acceptance_sha256", "allowed_background_processes",
        "process_wait_policy", "continuation", "repair_attempt")
    inner = {k: copy.deepcopy(old[k]) for k in keys}
    inner.update(schema_version=1, kind=legacy.STAGE_KIND, stage=stage, seed=seed,
        output=str(Path(plan["output"]) / f"{stage}{seed}"), deadline_epoch=resume.DEADLINE,
        alignment_plan_path=plan["original_plan"]["path"], alignment_plan_sha256=plan["original_plan"]["sha256"],
        supervisor_processes=list(own.values()), process_wait_spent_seconds=waited,
        parents={k: old["parents"][k] for k in (("sft",) if stage == "dpo" else ("sft", "reward"))},
        workload=workload(stage, old["ppo_rows"]),
        reference_cache={"mode": "reuse", **cache_pin} if stage == "dpo" else None)
    from .throughput_fresh_v3 import make_request
    return make_request(inner, plan["_authorized"], {"path": str(plan_path), "sha256": expected_sha}), "run_alignlab_throughput_fresh_v3.py"


def _verify_stage(output, request, known, code):
    from .throughput_fresh_v3 import verify_stage, WRAPPER_KIND
    if request.get("kind") != WRAPPER_KIND:
        raise ValueError("V3 only accepts its exact fresh seed44 child wrapper")
    return verify_stage(output, request, known, code)


def run_alignment(plan_path, expected_plan_sha256, *, execute=False, system=None):
    plan_path = Path(plan_path).resolve()
    metadata_plan = _metadata_plan(plan_path, expected_plan_sha256)
    live_identity = _LiveIdentity.from_plan(plan_path, expected_plan_sha256, metadata_plan)
    content_identity = _ContentIdentity(live_identity)
    # The initial full independent acceptance executes exactly once and is
    # bracketed by complete file inventories and cryptographic content hashes.
    plan = validate_plan(plan_path, expected_plan_sha256)
    if plan != metadata_plan:
        raise ValueError("Full acceptance changed the frozen metadata view")
    content_identity.check()
    flags = {"primary_alignment_verified": False, "parents_verified": True,
             "capacity_verified": True, "formal_training": False, "research_complete": False}
    if not execute:
        return {"status": "plan_only", "plan": plan, **flags}
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
             "original_started_at_epoch": resume.STARTED, "original_deadline_epoch": resume.DEADLINE,
             "budget_authorization": plan["budget_authorization"], "execution_window": plan["execution_window"],
             "continuation": plan["continuation"], "repair_attempt": plan["repair_attempt"],
             "supervisor_processes": list(own.values()),
              "resources_before": admission, "initial_process_guard": initial_guard,
              "total_process_wait_seconds": plan["continuation"]["process_wait_spent_seconds"], "process_wait_policy": plan["process_wait_policy"], "stages": copy.deepcopy(plan["_inherited_stages"])}
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
        content_identity.check()
        _pin(plan_path, expected_plan_sha256)
        checked = _parse(plan_path)
        _checked_prefix(checked)
        if checked != plan["_authorized"]:
            raise ValueError("Frozen recovery plan changed")
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
        _new(output / "inherited-content-acceptance.json", content_identity.record())
        write()
        snapshot = verify_snapshot(plan["model_snapshot"], plan["model_manifest_sha256"])
        if snapshot["repo_id"] != REPO_ID or snapshot["revision"] != REVISION:
            raise ValueError("Actual model snapshot differs from registered revision")
        cache_pin = {k: state["stages"][0]["reference_cache"][k] for k in ("path", "manifest_sha256", "identity_sha256")}
        for stage, seed in REMAINING_SEQUENCE:
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
