"""Read-only inheritance of seven accepted V6 policies after its closed failure.

V5 base output remains owned by V5; V6 owns its accepted official score. Partial
PPO44 output is never resumable evidence. No verifier launches a model or writes
any original artifact. Final analysis independently replays the official scores.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .evaluation_protocol_v2 import verified_bytes, _finite, _relative
from . import evaluation_inheritance_v6 as previous
from . import evaluation_inheritance_v7 as prior_version

ROOT = Path(__file__).resolve().parents[1]
PRIOR_PLAN_SHA256 = "c029d2427d329195a55c9ec17c8543bc6cb2da2a82b54d27d85e55d7984fee28"
PRIOR_SOURCE_COUNT = 154
PRESERVED_SOURCE_COUNT = 173
V8_EXECUTION = "alignment_lab/evaluation_execution_v8.py"
POLICIES = ("base", "sft", "dpo42", "ppo42", "dpo43", "ppo43", "dpo44")
SLOTS = frozenset("ifeval/" + name for name in POLICIES)
BINDING_FIELDS = frozenset({"prior_plan", "campaign", "closure", "request", "manifest",
                            "score_request", "score"})
COUNTS = {"records": 541, "prompts": 541, "instructions": 834}
DEADLINE = 1793673011
STARTED = 1792895411
WALL_SECONDS = 777600
_read = previous._read


def _frozen_v7_plan():
    from .evaluation_resources_v8 import V7_PLAN_SHA256
    return _read({"path": str(ROOT / "reports/alignlab-quality-v7-launch-20261029T033100Z/plan.json"),
                  "sha256": V7_PLAN_SHA256})


def _identity(value):
    """Accept exact integer or decimal-string ticks, never floating point ticks."""
    record = dict(value)
    if "created_utc_ticks" in record:
        ticks = record["created_utc_ticks"]
        if isinstance(ticks, str) and ticks.isascii() and ticks.isdecimal():
            ticks = int(ticks)
        if type(ticks) is not int:
            raise ValueError("Exact integer/string process ticks required")
        record["created_utc_ticks"] = ticks
    return previous._identity(record)


def _exited(expected, system):
    # Reuse the original PID-reuse-safe comparison with exact integer ticks.
    records = []
    for item in expected:
        _identity(item)
        record = dict(item)
        if "created_utc_ticks" in record:
            record["created_utc_ticks"] = int(record["created_utc_ticks"])
        records.append(record)
    return previous._exited(records, system)


def _same(value, expected, message):
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in expected.items()):
        raise ValueError(message)


def _sources(prior, current):
    preserved = _frozen_v7_plan().get("source_sha256", {})
    if (not isinstance(prior, dict) or len(prior) != PRIOR_SOURCE_COUNT
            or previous.V6_EXECUTION not in prior or V8_EXECUTION in prior
            or not isinstance(current, dict) or V8_EXECUTION not in current
            or any(current.get(name) != digest for name, digest in prior.items())):
        raise ValueError("Every original 154-file V6 source pin must remain unchanged")
    if (len(preserved) != PRESERVED_SOURCE_COUNT or prior_version.V7_EXECUTION not in preserved
            or V8_EXECUTION in preserved or any(preserved.get(k) != v for k, v in prior.items())
            or any(current.get(k) != v for k, v in preserved.items())):
        raise ValueError("All 173 frozen V7 source pins must remain byte-identical")
    for name, digest in current.items():
        verified_bytes(_relative(ROOT, name), digest)


def _scientific_identity(prior, plan):
    for key in ("kind", "schema_version", "authorization", "protocol", "roster", "decoder",
                "started_at_epoch", "deadline_epoch", "wall_seconds", "analysis_wall_seconds",
                "poll_seconds", "model_python_sha256", "scoring_python_sha256"):
        if key not in prior or plan.get(key) != prior[key]:
            raise ValueError("Inherited scientific identity or original budget changed: " + key)
    if (prior["started_at_epoch"] != STARTED or prior["deadline_epoch"] != DEADLINE
            or prior["wall_seconds"] != WALL_SECONDS
            or _finite(prior["deadline_epoch"], "deadline") - prior["started_at_epoch"] != WALL_SECONDS):
        raise ValueError("Original unextended evaluation budget required")
    authority = _read(prior["authorization"])
    _same(authority, {"kind": "alignlab-independent-quality-evaluation-authorization-v2",
        "started_at_epoch": STARTED, "deadline_epoch": DEADLINE, "wall_seconds": WALL_SECONDS,
        "automatic_extension": False, "reuses_old_budget": False, "no_new_training": True},
        "Original independent evaluation authorization required")
    for key in ("protocol", "roster"):
        _read(prior[key])
    decoder = prior["decoder"]
    for name, path in (("python", decoder["python"]), ("script", decoder["script"]),
                       ("tokenizer", decoder["tokenizer_path"])):
        verified_bytes(Path(path), decoder[name + "_sha256"])


def _accepted(record, phase, request_pin=None, request=None, result=None):
    _same(record, {"phase": phase, "status": "accepted", "exit_code": 0,
                   "active_processes": []}, "Only an accepted, exited original child may be inherited")
    if (record.get("monitoring_error") is not None or record.get("stop_reason") is not None
            or not isinstance(record.get("known_processes"), list) or not record["known_processes"]
            or result is not None and record.get("result") != result
            or request is not None and record.get("request") != request
            or request_pin is not None and {"path": record.get("request_path"),
                "sha256": record.get("request_sha256")} != request_pin):
        raise ValueError("Accepted child request, manifest or process evidence differs")


def _campaign(binding, prior, campaign):
    _same(campaign, {"kind": prior["kind"], "status": "failed", "capacity_verified": True,
        "started_at_epoch": STARTED, "deadline_epoch": DEADLINE,
        "primary_evaluation_complete": False, "research_complete": False, "formal_training": False},
        "The original failed V6 campaign is required")
    if ({"path": campaign.get("plan_path"), "sha256": campaign.get("plan_sha256")} != binding["prior_plan"]
            or set(campaign.get("scores", {})) != SLOTS
            or set(campaign.get("inherited_inference", {})) != {"ifeval/base"}
            or campaign.get("inherited_capacity") != {"bindings": prior.get("inherited_capacity"),
                "bridge": campaign.get("capacity_manifest")}):
        raise ValueError("V6 campaign plan, seven scores or original inheritance differs")
    expected = [("ifeval-base-score", "score")]
    for model in POLICIES[1:]:
        expected.extend((("ifeval-" + model, "inference"), ("ifeval-" + model + "-score", "score")))
    expected.append(("ifeval-ppo44", "inference"))
    children = campaign.get("children", [])
    if [(c.get("name"), c.get("phase")) for c in children] != expected:
        raise ValueError("Exact seven-policy V6 stage history followed by partial PPO44 required")
    for child in children[:-1]:
        _accepted(child, child["phase"])
    failed = children[-1]
    if (type(failed.get("exit_code")) is not int or failed["exit_code"] != 1
            or failed.get("status") == "accepted" or failed.get("result") is not None
            or failed.get("active_processes") != [] or not failed.get("known_processes")
            or Path(prior["output"], "ifeval-ppo44", "manifest.json").exists()):
        raise ValueError("PPO44 partial output must remain unaccepted and without a complete manifest")
    return {c["name"]: c for c in children}


def _closed_v6(binding, prior, campaign, closure):
    _same(closure, {"kind": "alignlab-independent-quality-evaluation-v6-failure-closure",
        "status": "failed_closed_services_restored", "deadline_epoch": DEADLINE,
        "fully_accepted_policy_inference_records": 3787, "failed_partial_records": 303,
        "failed_partial_accepted": False, "policy_scores_accepted": 7,
        "primary_evaluation_complete": False, "research_complete": False,
        "restore_pending": False, "all_28_fields_match": True,
        "all_owned_exact_identities_exited": True, "guardian_status": "complete",
        "awake_status": "released", "preserved_processes_match": True,
        "unknown_python_at_closure": [], "same_plan_output_relaunch_forbidden": True},
        "Complete safe V6 failure closure required")
    if closure.get("plan") != binding["prior_plan"] or closure.get("campaign") != binding["campaign"]:
        raise ValueError("Closure is not bound to the exact V6 plan and failed campaign")
    failure = closure.get("failure", {})
    _same(failure, {"stage": "ifeval-ppo44", "error_type": "RuntimeError", "exit_code": 1,
        "completed_records": 303, "unknown_process_origin_confirmed": False},
        "Original unknown-process guard failure must remain explicitly unresolved")
    if not failure.get("error", "").startswith(
            "Service-preserving evaluation guard: Unknown Python process outside exact frozen identities: "):
        raise ValueError("V6 was not stopped by its original unknown-process guard")
    partial = _read(closure["failed_progress"])
    _same(partial, {"status": "failed", "completed_records": 303, "inference_complete": False},
        "Original partial PPO44 failure must remain excluded")
    failed = campaign["children"][-1]
    failed_request = _read({"path": failed["request_path"], "sha256": failed["request_sha256"]})
    if (partial.get("error") != failure["error"]
            or partial.get("request") != failed_request or failed.get("request") != failed_request
            or partial.get("deadline_epoch") != DEADLINE
            or partial.get("pid") not in {p["pid"] for p in failed["known_processes"]}
            or Path(closure["failed_progress"]["path"]).resolve() != Path(prior["output"]) / "ifeval-ppo44/progress.json"):
        raise ValueError("Failure progress and closure disagree")
    profile, restoration = _read(prior["resource_profile"]), _read(closure["restoration"])
    health = {**next(s["json_equals"] for s in profile["services"] if s["name"] == "evidoc"), "busy": False}
    _same(restoration, {"kind": "alignlab-evidoc-restoration-complete-v1",
        "status": "restored_and_verified", "restoration_complete": True, "restore_pending": False,
        "all_28_fields_match": True, "deadline_epoch": DEADLINE}, "Original restoration must be complete")
    if (len(health) != 28 or restoration.get("health") != health
            or restoration.get("pause_authorization_sha256") != profile["authorized_pause"]["authorization"]["sha256"]
            or len(restoration.get("processes", [])) != 1
            or _identity(restoration["processes"][0]) != _identity(closure["restored_process"])):
        raise ValueError("All 28 health fields and the exact restored process are required")
    preserved = profile["background_processes"] + [p for s in profile["services"]
        if s["name"] != "evidoc" for p in s["processes"]]
    resource, monitor = _read(closure["resource"]), _read(closure["monitor"])
    _same(resource, {"all_preserved_identity_matches": True, "restored_identity_matches": True,
        "port_8083_owner_matches": True, "health_fields_verified": 28, "all_28_fields_match": True,
        "unmatched_python_processes": [], "all_owned_exact_identities_exited": True},
        "Closure resource observation must prove exact restored and preserved identities")
    if (resource.get("restoration") != closure["restoration"] or resource.get("health") != health
            or monitor.get("plan") != binding["prior_plan"]
            or _identity(resource["restored_process"]) != _identity(closure["restored_process"])):
        raise ValueError("Restoration, monitor and resource observations disagree")
    for records in (restoration.get("preserved_processes"), resource.get("preserved_processes")):
        if (not isinstance(records, list) or len(records) != 5
                or {_identity(p) for p in records} != {_identity(p) for p in preserved}):
            raise ValueError("All five frozen preserved processes must match")
    guardian, awake = monitor["guardian"], monitor["awake"]
    if (guardian.get("status") != "complete" or awake.get("status") != "released"
            or guardian.get("deadline_epoch") != DEADLINE or awake.get("deadline_epoch") != DEADLINE
            or guardian.get("plan") != binding["prior_plan"]):
        raise ValueError("Original guardian and awake must have completed their exact bindings")
    known = list(campaign["supervisor_processes"])
    for child in campaign["children"]:
        known.extend(child["known_processes"])
    recorded = resource.get("owned_processes", [])
    if ({_identity(v["expected"]) for v in recorded} != {_identity(p) for p in known}
            or any(v.get("exact_identity_present") is not False for v in recorded)):
        raise ValueError("Every historical evaluation identity must be recorded as exited")
    known.extend((guardian["guardian"], {"pid": awake["helper_pid"],
        "parent_pid": awake["helper_parent_pid"], "created_at_utc": awake["helper_created_utc"],
        "command_line": awake["helper_command"]}))
    pins = closure.get("verified_scorepins", [])
    if (len(pins) != 7 or {v.get("name"): v.get("pin") for v in pins} != campaign["scores"]
            or any(v.get("check") != "verified" or v.get("official") is not True
                or v.get("denominators") != {"prompts": 541, "instructions": 834} for v in pins)):
        raise ValueError("Closure must bind all seven complete official score pins")
    return known


def verify_recovery_context(binding, plan, *, system=None, check_live=True):
    """Common V6 provenance check used by inference and capacity inheritance."""
    prior, campaign, closure = (_read(binding[k]) for k in ("prior_plan", "campaign", "closure"))
    if binding["prior_plan"]["sha256"] != PRIOR_PLAN_SHA256:
        raise ValueError("Only the exact closed V6 plan may supply V8 inheritance")
    _scientific_identity(prior, plan)
    _sources(prior["source_sha256"], plan.get("source_sha256"))
    old_output, new_output = Path(prior["output"]), Path(plan["output"])
    if (not old_output.is_absolute() or not new_output.is_absolute()
            or old_output.resolve().parent != (ROOT / "runs").resolve()
            or new_output.resolve().parent != (ROOT / "runs").resolve()
            or old_output.resolve() == new_output.resolve()
            or Path(binding["campaign"]["path"]).resolve() != old_output.resolve() / "campaign.json"):
        raise ValueError("Exact original campaign and distinct fresh project output required")
    children = _campaign(binding, prior, campaign)
    known = _closed_v6(binding, prior, campaign, closure)
    from .evaluation_resources_v8 import load_profile
    profile = load_profile(plan["resource_profile"])
    baseline = _read(profile["authorized_pause"]["baseline_profile"])
    frozen_v7 = _frozen_v7_plan()
    if (baseline["authorized_pause"]["baseline_profile"] != prior["resource_profile"]
            or baseline["service_identity_succession"]["restoration_report"] != closure["restoration"]
            or frozen_v7["resource_profile"] != profile["authorized_pause"]["baseline_profile"]
            or frozen_v7.get("inherited_inference") != plan.get("inherited_inference")
            or frozen_v7.get("inherited_scores") != plan.get("inherited_scores")
            or frozen_v7.get("inherited_capacity") != plan.get("inherited_capacity")
            or {_identity(p) for p in next(s["processes"] for s in baseline["services"] if s["name"] == "evidoc")}
                != {_identity(closure["restored_process"])}):
        raise ValueError("V8 service succession must traverse V7 while retaining original V6 result bindings")
    if check_live:
        _exited(known, system or previous._LiveSystem())
    return prior, campaign, closure, children, known


def verify_versioned_evaluation_result(output, *, expected_manifest_sha256,
                                      expected_request=None, decode_tokens=None, codec=None):
    manifest = _read({"path": str(Path(output).resolve() / "manifest.json"), "sha256": expected_manifest_sha256})
    request = manifest.get("request", {})
    if expected_request is not None and request != expected_request:
        raise ValueError("Version routing request differs from the external request")
    sources = request.get("source_sha256", {})
    if V8_EXECUTION in sources:
        verified_bytes(_relative(ROOT, V8_EXECUTION), sources[V8_EXECUTION])
        from .evaluation_execution_v8 import verify_evaluation_result
    else:
        verify_evaluation_result = prior_version.verify_versioned_evaluation_result
    return verify_evaluation_result(output, expected_manifest_sha256=expected_manifest_sha256,
        expected_request=expected_request, decode_tokens=decode_tokens, codec=codec)


def _base_chain(binding, prior, campaign, request, *, system=None):
    slot = "ifeval/base"
    old = previous.verify_inherited_inference(prior["inherited_inference"][slot], prior, system=system)
    if (old["request"] != request or old["manifest_binding"] != binding["manifest"]
            or prior["inherited_inference"][slot]["request"] != binding["request"]
            or campaign["inherited_inference"][slot] != old):
        raise ValueError("Base must retain its exact V5 output to V6 acceptance chain")
    return old["accepted_record"]


def _verify_original_processes(accepted, manifest, campaign, closed_processes):
    """Bind the final admission snapshot separately from observed CPU encoders.

    V6 manifest.processes contains its runner pair and campaign supervisor pair.
    The accepted child's known_processes may additionally contain exited stdin
    encoders, which are historical descendants, not members of that snapshot.
    """
    from .evaluation_supervisor_v6 import _tokens

    def indexed(records):
        if not isinstance(records, list) or not records:
            raise ValueError("Original inference process identities are missing")
        identities = {_identity(p) for p in records}
        result = {p["pid"]: p for p in records}
        if len(result) != len(records) or len(identities) != len(records):
            raise ValueError("Duplicate original inference process identity")
        return result, identities

    observed, observed_ids = indexed(accepted["known_processes"])
    _, manifest_ids = indexed(manifest.get("processes"))
    supervisors, supervisor_ids = indexed(campaign.get("supervisor_processes"))
    closed_ids = {_identity(p) for p in closed_processes}
    wrapper, worker = observed.get(accepted.get("wrapper_pid")), observed.get(manifest.get("pid"))
    if (len(supervisors) != 2 or wrapper is None or worker is None or wrapper["pid"] == worker["pid"]
            or wrapper["parent_pid"] not in supervisors or worker["parent_pid"] != wrapper["pid"]
            or not observed_ids <= closed_ids or not supervisor_ids <= closed_ids):
        raise ValueError("Original inference runner/supervisor ancestry or closed identity differs")
    supervisor = supervisors[wrapper["parent_pid"]]
    if supervisor["parent_pid"] not in supervisors or supervisor["parent_pid"] == supervisor["pid"]:
        raise ValueError("Original campaign supervisor pair ancestry differs")
    command = accepted.get("command")
    if (not isinstance(command, list) or len(command) != 10
            or command[1:4] != ["-I", "-B", "-u"]
            or (ROOT / command[4]).resolve() != (ROOT / "scripts/run_alignlab_evaluation_stage_v6.py").resolve()
            or command[5:] != ["--request", accepted["request_path"], "--request-sha256",
                               accepted["request_sha256"], "--execute"]):
        raise ValueError("Original inference command does not bind its exact request")
    for process in (wrapper, worker):
        if _tokens(process["command_line"])[1:] != command[1:]:
            raise ValueError("Original inference runner invocation differs")
    runner_ids = {_identity(wrapper), _identity(worker)}
    if manifest_ids != runner_ids | supervisor_ids:
        raise ValueError("Original inference manifest must contain its exact runner and supervisor pairs")
    # Encoders are accepted only as exact recorded descendants of this runner,
    # with the original no-argument stdin CLI and complete closure evidence.
    remaining = {pid: p for pid, p in observed.items() if pid not in (wrapper["pid"], worker["pid"])}
    parents = {worker["pid"]}
    while remaining:
        attached = []
        for pid, process in remaining.items():
            if process["parent_pid"] not in parents:
                continue
            args = _tokens(process["command_line"])
            if (len(args) != 4 or args[1:3] != ["-I", "-B"]
                    or (ROOT / args[3]).resolve() != (ROOT / "scripts/encode_alignlab_evaluation_capacity_v2.py").resolve()):
                raise ValueError("Historical inference descendant is not the exact stdin encoder")
            attached.append(pid)
        if not attached:
            raise ValueError("Historical encoder ancestry is not bound to this inference worker")
        parents.update(attached)
        for pid in attached:
            del remaining[pid]


def verify_inherited_inference(binding, plan, *, system=None):
    if not isinstance(binding, dict) or set(binding) != BINDING_FIELDS:
        raise ValueError("Seven exact original inference/score provenance pins required")
    docs = {key: _read(value) for key, value in binding.items()}
    request, manifest, score_request = (docs[k] for k in ("request", "manifest", "score_request"))
    slot = request.get("task", "") + "/" + request.get("model_id", "")
    if (slot not in SLOTS or set(plan.get("inherited_inference", {})) != SLOTS
            or set(plan.get("inherited_scores", {})) != SLOTS
            or plan["inherited_inference"].get(slot) != binding
            or plan["inherited_scores"].get(slot) != binding):
        raise ValueError("Only the seven explicitly inherited complete policy slots are allowed")
    prior, campaign, closure, children, known = verify_recovery_context(binding, plan, system=system)
    model = request["model_id"]
    if (manifest.get("request") != request or manifest.get("counts") != COUNTS
            or request.get("deadline_epoch") != DEADLINE
            or {"path": request.get("protocol_path"), "sha256": request.get("protocol_sha256")} != prior["protocol"]
            or {"path": request.get("roster_path"), "sha256": request.get("roster_sha256")} != prior["roster"]):
        raise ValueError("Complete original inference identity and 541/834 denominator required")
    if model == "base":
        accepted = _base_chain(binding, prior, campaign, request, system=system)
    else:
        accepted = children["ifeval-" + model]
        _accepted(accepted, "inference", binding["request"], request, binding["manifest"])
        if (Path(binding["request"]["path"]).resolve() != Path(prior["output"]) / ("ifeval-" + model + ".request.json")
                or Path(binding["manifest"]["path"]).resolve() != Path(prior["output"]) / ("ifeval-" + model) / "manifest.json"
                or request.get("output") != accepted.get("output")
                or request.get("source_sha256") != prior["source_sha256"]
                or request.get("resource_profile") != prior["resource_profile"]
                or request.get("capacity_manifest") != campaign.get("capacity_manifest")):
            raise ValueError("Original V6 request/output/source/resource/capacity binding differs")
        _verify_original_processes(accepted, manifest, campaign, known)
    expected_score = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-request-v1",
        "inference_manifest": binding["manifest"]["path"], "inference_manifest_sha256": binding["manifest"]["sha256"],
        "source_sha256": prior["source_sha256"], "decoder": prior["decoder"], "deadline_epoch": DEADLINE}
    _accepted(children["ifeval-" + model + "-score"], "score", binding["score_request"], score_request, binding["score"])
    if (score_request != expected_score or campaign["scores"][slot] != binding["score"]
            or Path(binding["score_request"]["path"]).resolve() != Path(prior["output"]) / ("ifeval-" + model + "-score.request.json")
            or Path(binding["score"]["path"]).resolve() != Path(prior["output"]) / ("ifeval-" + model + "-score") / "scores.json"):
        raise ValueError("Only the original accepted score request and result can be inherited")
    # This preserved verifier checks complete file bytes, independent decoder
    # receipt, per-instruction official flags and original full inference proof.
    from .evaluation_supervisor_v6 import _verify_score
    if _verify_score(Path(binding["score"]["path"]).parent, score_request, prior, request) != binding["score"]:
        raise ValueError("Original complete inference and scoring proof differs")
    for key, pin in binding.items():
        if _read(pin) != docs[key]:
            raise ValueError("Inheritance evidence changed during verification")
    _sources(prior["source_sha256"], plan["source_sha256"])
    exited = _exited(known, system or previous._LiveSystem())
    return {"slot": slot, "request": copy.deepcopy(request), "manifest_binding": copy.deepcopy(binding["manifest"]),
        "score_binding": copy.deepcopy(binding["score"]), "score_request": copy.deepcopy(score_request),
        "accepted_record": copy.deepcopy(accepted), "source_campaign_binding": copy.deepcopy(binding["campaign"]),
        "verification": {"kind": "alignlab-complete-policy-inheritance-v8", "counts": dict(COUNTS),
            "prior_sources_verified": PRIOR_SOURCE_COUNT, "preserved_v7_sources_verified": PRESERVED_SOURCE_COUNT,
            "exact_old_processes_exited": exited,
            "frozen_output_reverified": True, "original_budget_unchanged": True, "restoration_verified": True,
            "official_scoring_complete": True, "requires_official_analysis_replay": True,
            "partial_ppo44_inherited": False, "unknown_process_origin_resolved": False,
            "outputs_copied_or_rewritten": False}}


def verify_inherited_score(binding, plan, *, system=None):
    result = verify_inherited_inference(binding, plan, system=system)
    return {"slot": result["slot"], "score_binding": result["score_binding"],
            "request": result["score_request"], "manifest_binding": result["manifest_binding"],
            "verification": result["verification"]}
