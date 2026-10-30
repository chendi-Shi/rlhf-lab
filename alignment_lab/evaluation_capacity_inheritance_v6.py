"""Explicit reuse of the completed V5 CUDA proof, never a new capacity claim.

The original request and evidence are verified by the original V5 verifier.
Only a byte-identical GPU implementation with the narrowly checked V6 resource
and receipt bridge can consume its peaks. Current live admission is separate.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import shlex

from .evaluation_protocol_v2 import parse, verified_bytes, _finite
from .evaluation_resources_v6 import load_profile, signature, DEADLINE, MIB

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-evaluation-capacity-inheritance-v6"
PRIOR_PLAN_SHA256 = "00b8a18afb5d75e5e6355a673bfcb4751bf1e0bbf53bf056ae07a5683fd6b2c9"
PIN_FIELDS = {"prior_plan", "campaign", "closure", "request", "manifest"}
POLICIES = ("base", "sft", "dpo42", "ppo42", "dpo43", "ppo43", "dpo44", "ppo44")
MODEL_IDS = {*POLICIES, "reward"}
TASKS = {( "ifeval", model) for model in POLICIES} | {
    ("helpsteer_rm", "reward"), ("rm_bench", "reward")}
EXTRA_EXECUTION_SOURCES = {
    "alignment_lab/evaluation_capacity_inheritance_v6.py",
    "tests_v2/test_evaluation_capacity_inheritance_v6.py",
}
VERSION_NAMES = (
    "evaluation_execution", "evaluation_resources", "evaluation_capacity",
    "evaluation_supervisor", "run_alignlab_evaluation_stage",
    "run_alignlab_evaluation_capacity", "supervise_alignlab_evaluation",
)
BRIDGE_STATEMENTS = '''
candidate = _read(binding["path"], binding["sha256"])
if candidate.get("kind") == "alignlab-evaluation-capacity-inheritance-v6":
    from .evaluation_capacity_inheritance_v6 import verify_capacity_inheritance_for_request
    return verify_capacity_inheritance_for_request(binding, request)
'''


def _read(binding):
    from .evaluation_resources_v6 import _binding
    _binding(binding)
    path = Path(binding["path"])
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Inheritance evidence cannot traverse symlinks")
    return parse(verified_bytes(path, binding["sha256"]))


def _version(value):
    for name in VERSION_NAMES:
        value = value.replace(name + "_v6", name + "_v5")
    return value


class _Versions(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = _version(node.value)
        return node

    def visit_ImportFrom(self, node):
        if node.module:
            node.module = _version(node.module)
        return self.generic_visit(node)

    def visit_alias(self, node):
        node.name = _version(node.name)
        return node


def _tree(source):
    tree = ast.parse(source)
    # Documentation has no effect on execution. No executable nodes are dropped.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                    and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return _Versions().visit(tree)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def verify_execution_ast(prior_source, current_source, *, execution=False):
    """Reject any GPU behavior change, including dtype, horizon and model imports."""
    old, new = _tree(prior_source), _tree(current_source)
    if execution:
        old_functions = [n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "_capacity_peak"]
        new_functions = [n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "_capacity_peak"]
        if len(old_functions) != 1 or len(new_functions) != 1:
            raise ValueError("Exactly one capacity admission function is required")
        expected = copy.deepcopy(old_functions[0])
        # The only executable addition: after the original missing-proof guard.
        expected.body[3:3] = ast.parse(BRIDGE_STATEMENTS).body
        if _dump(new_functions[0]) != _dump(expected):
            raise ValueError("Only the exact inherited-capacity admission bridge is permitted")
        new.body[new.body.index(new_functions[0])] = copy.deepcopy(old_functions[0])
        for tree, extra in ((old, set()), (new, EXTRA_EXECUTION_SOURCES)):
            assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                           and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                           and n.targets[0].id == "REQUIRED_SOURCES"]
            if len(assignments) != 1:
                raise ValueError("Exact explicit execution source inventory required")
            value = assignments[0].value
            if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
                    or value.func.id != "frozenset" or value.keywords or len(value.args) != 1
                    or not isinstance(value.args[0], ast.Set)
                    or any(not isinstance(n, ast.Constant) or not isinstance(n.value, str)
                           for n in value.args[0].elts)):
                raise ValueError("Execution source inventory must be a literal frozen set")
            names = [n.value for n in value.args[0].elts]
            if len(names) != len(set(names)) or not extra <= set(names):
                raise ValueError("Complete unique bridge source inventory required")
            value.args[0].elts = [ast.Constant(name) for name in sorted(set(names) - extra)]
    if _dump(old) != _dump(new):
        raise ValueError("Inherited CUDA proof cannot cover changed GPU execution semantics")


def _sources(prior, current):
    if (not isinstance(prior, dict) or len(prior) != 129 or not isinstance(current, dict)
            or any(current.get(name) != digest for name, digest in prior.items())):
        raise ValueError("All 129 historical source pins must remain byte-identical")
    raw = {}
    for name, digest in current.items():
        if (not isinstance(name, str) or "\\" in name or ":" in name
                or name.startswith("/") or ".." in Path(name).parts or not name.endswith(".py")):
            raise ValueError("Canonical pinned source path required")
        path = ROOT / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Pinned sources cannot traverse symlinks")
        raw[name] = verified_bytes(path, digest).decode("utf-8-sig")
    pairs = (
        ("alignment_lab/evaluation_execution", True),
        ("alignment_lab/evaluation_capacity", False),
        ("scripts/run_alignlab_evaluation_stage", False),
        ("scripts/run_alignlab_evaluation_capacity", False),
    )
    for stem, execution in pairs:
        old, new = stem + "_v5.py", stem + "_v6.py"
        if old not in prior or new not in raw:
            raise ValueError("Both frozen GPU execution versions must be pinned")
        verify_execution_ast(raw[old], raw[new], execution=execution)
    if not EXTRA_EXECUTION_SOURCES <= current.keys():
        raise ValueError("Capacity inheritance implementation and tests must be pinned")


def _tokens(command):
    if not isinstance(command, str) or not command:
        raise ValueError("Observed full command required")
    return [value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value
            for value in shlex.split(command, posix=False)]


def _command(process, args, interpreters):
    actual = _tokens(process["command_line"])
    script = Path(actual[4]) if len(actual) > 4 else Path("__missing__")
    script = script if script.is_absolute() else ROOT / script
    if (len(actual) != len(args) or Path(actual[0]).resolve() not in interpreters
            or actual[1:4] != ["-I", "-B", "-u"]
            or script.resolve() != (ROOT / args[4]).resolve()
            or actual[5:] != args[5:]):
        raise ValueError("Historical process command must exactly match its frozen dispatch")


def _provenance(binding, prior, campaign, closure, request, manifest, profile):
    if (campaign.get("kind") != "alignlab-primary-evaluation-campaign-v2"
            or campaign.get("status") != "failed" or campaign.get("capacity_verified") is not True
            or campaign.get("plan_path") != binding["prior_plan"]["path"]
            or campaign.get("plan_sha256") != binding["prior_plan"]["sha256"]
            or campaign.get("deadline_epoch") != DEADLINE
            or campaign.get("primary_evaluation_complete") is not False
            or campaign.get("research_complete") is not False
            or campaign.get("formal_training") is not False):
        raise ValueError("Exact failed V5 campaign with accepted capacity required")
    if (closure.get("kind") != "alignlab-quality-v5-failure-closure-v1"
            or closure.get("status") != "failed_closed_services_restored"
            or closure.get("plan") != binding["prior_plan"]
            or closure.get("campaign_status") != "failed"
            or closure.get("failure_phase") != "ifeval-base-score"
            or closure.get("failure_exit_code") != 1
            or closure.get("capacity_verified") is not True
            or closure.get("capacity_models_completed") != 9
            or closure.get("original_deadline_epoch") != DEADLINE
            or closure.get("all_evaluation_processes_exited") is not True
            or closure.get("restoration_complete") is not True
            or closure.get("restore_pending") is not False
            or closure.get("all_28_fields_match") is not True
            or closure.get("preserved_services_verified") is not True
            or closure.get("unknown_python") != []
            or closure.get("relaunch_permitted") is not False
            or closure.get("recovery_requires_fresh_launch_and_output") is not True
            or closure.get("restoration") != profile["service_identity_succession"]["restoration_report"]):
        raise ValueError("Pinned safe V5 failure closure and matching restoration are required")
    children = campaign.get("children", [])
    matches = [c for c in children if c.get("name") == "capacity" and c.get("phase") == "capacity"]
    if len(matches) != 1:
        raise ValueError("Exactly one historical capacity dispatch required")
    child = matches[0]
    if (child.get("status") != "accepted" or type(child.get("exit_code")) is not int
            or child["exit_code"] != 0 or child.get("monitoring_error") is not None
            or child.get("stop_reason") is not None or child.get("active_processes") != []
            or child.get("result") != binding["manifest"] or child.get("request") != request
            or child.get("request_path") != binding["request"]["path"]
            or child.get("request_sha256") != binding["request"]["sha256"]
            or child.get("output") != request["output"]
            or request.get("evaluation_plan_path") != binding["prior_plan"]["path"]
            or request.get("evaluation_plan_sha256") != binding["prior_plan"]["sha256"]
            or manifest.get("request") != binding["request"]):
        raise ValueError("Only an accepted, exited capacity child with all original pins can be inherited")
    if (Path(binding["campaign"]["path"]).resolve() != Path(prior["output"]) / "campaign.json"
            or Path(binding["request"]["path"]).resolve() != Path(prior["output"]) / "capacity.request.json"
            or Path(binding["manifest"]["path"]).resolve() != Path(prior["output"]) / "capacity" / "manifest.json"):
        raise ValueError("Capacity evidence must remain in the original campaign slots")
    known = child.get("known_processes", [])
    supervisors = campaign.get("supervisor_processes", [])
    if len(known) != 2 or len(supervisors) != 2:
        raise ValueError("Exact original worker/wrapper and supervisor identities required")
    identities = {signature(p) for p in known + supervisors}
    if len(identities) != 4 or {signature(p) for p in manifest["process_identity"]} != identities:
        raise ValueError("CUDA receipt identities differ from independently observed dispatch")
    by_pid = {p["pid"]: p for p in known}
    wrapper = by_pid.get(child.get("wrapper_pid"))
    if wrapper is None or wrapper["parent_pid"] not in {p["pid"] for p in supervisors}:
        raise ValueError("Historical capacity wrapper must descend from the exact supervisor")
    worker = next(p for p in known if p is not wrapper)
    if worker["parent_pid"] != wrapper["pid"]:
        raise ValueError("Historical CUDA worker must descend from its exact wrapper")
    previous_restore = _read(profile["service_identity_succession"]["restoration_manifest"])
    interpreter_rows = previous_restore.get("evaluation_interpreters", [])
    selected = [p for p in interpreter_rows if p.get("role") in {"base", "model"}]
    if len(selected) != 2 or {p["role"] for p in selected} != {"base", "model"}:
        raise ValueError("Pinned model and base interpreters required for historical identities")
    interpreters = set()
    for row in selected:
        verified_bytes(Path(row["path"]), row["sha256"])
        interpreters.add(Path(row["path"]).resolve())
        if row["role"] == "model" and row["sha256"] != prior["model_python_sha256"]:
            raise ValueError("Historical model interpreter pin differs")
    args = [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B", "-u",
            "scripts/run_alignlab_evaluation_capacity_v5.py", "--request", binding["request"]["path"],
            "--request-sha256", binding["request"]["sha256"], "--execute"]
    if child.get("command") != args:
        raise ValueError("Historical capacity dispatch argv differs from frozen command")
    for process in known:
        _command(process, args, interpreters)
    supervisor_args = [args[0], "-I", "-B", "-u", "scripts/supervise_alignlab_evaluation_v5.py",
                       "--plan", binding["prior_plan"]["path"], "--plan-sha256",
                       binding["prior_plan"]["sha256"], "--execute"]
    for process in supervisors:
        _command(process, supervisor_args, interpreters)
    all_known = {p["pid"] for p in supervisors}
    for row in children:
        if row.get("active_processes") != []:
            raise ValueError("Historical campaign children must all have exited")
        all_known.update(p["pid"] for p in row.get("known_processes", []))
    exited = closure.get("evaluation_processes", [])
    if ({p.get("pid") for p in exited} != all_known or len(exited) != len(all_known)
            or any(p.get("present") is not False for p in exited)
            or not closure.get("helper_processes")
            or any(p.get("present") is not False for p in closure["helper_processes"])):
        raise ValueError("Safe closure must cover every exact historical process")
    if any(campaign.get("capacity", {}).get(k) != v for k, v in manifest.items()):
        raise ValueError("Campaign's accepted capacity differs from the original manifest")
    if not (_finite(child["launch_epoch"], "Capacity launch") <= manifest["started_at_epoch"]
            <= manifest["completed_at_epoch"] <= _finite(child["finished_at_epoch"], "Capacity exit")
            < _finite(campaign["finished_at_epoch"], "Campaign finish") < DEADLINE):
        raise ValueError("Historical capacity must complete before the failed campaign and original deadline")


def verify_inherited_capacity(binding, plan):
    """Return the original verified receipt plus a per-model peak map, never live admission."""
    if not isinstance(binding, dict) or set(binding) != PIN_FIELDS:
        raise ValueError("Five explicit prior-plan/campaign/closure/request/manifest pins required")
    values = {key: _read(value) for key, value in binding.items()}
    prior, campaign, closure, request, manifest = (values[k] for k in
        ("prior_plan", "campaign", "closure", "request", "manifest"))
    if binding["prior_plan"]["sha256"] != PRIOR_PLAN_SHA256 or plan.get("inherited_capacity") != binding:
        raise ValueError("Only the explicitly frozen V5 campaign capacity may be inherited")
    for key in ("kind", "schema_version", "started_at_epoch", "deadline_epoch", "wall_seconds",
                "poll_seconds", "roster", "decoder", "model_python_sha256", "scoring_python_sha256"):
        if plan.get(key) != prior.get(key):
            raise ValueError("Capacity inheritance cannot change original identity or budget: " + key)
    if plan["deadline_epoch"] != DEADLINE or Path(plan["output"]).resolve() == Path(prior["output"]).resolve():
        raise ValueError("Fresh output and unchanged original deadline required")
    for key in ("protocol", "authorization"):
        if plan[key]["sha256"] != prior[key]["sha256"]:
            raise ValueError("Protocol and original authorization bytes cannot change")
        _read(plan[key])
        _read(prior[key])
    _sources(prior["source_sha256"], plan["source_sha256"])
    profile = load_profile(plan["resource_profile"])
    if profile["authorized_pause"]["baseline_profile"] != prior["resource_profile"]:
        raise ValueError("Only the prior campaign's completed restoration may change service identity")
    _provenance(binding, prior, campaign, closure, request, manifest, profile)
    from .evaluation_capacity_v5 import verify_evaluation_capacity
    result = verify_evaluation_capacity(Path(binding["manifest"]["path"]).parent,
        expected_manifest_sha256=binding["manifest"]["sha256"], expected_request=request,
        deadline_epoch=plan["deadline_epoch"])
    if (any(result.get(k) != v for k, v in manifest.items()) or set(result.get("models", {})) != MODEL_IDS
            or any(len(model.get("frozen_audits", [])) != 2 for model in result["models"].values())):
        raise ValueError("Full original nine-model and eighteen-audit verification required")
    peaks = {name: row["gpu_memory"]["max_reserved_bytes"] for name, row in result["models"].items()}
    if any(type(peak) is not int or not 0 < peak <= profile["allocator_limit_mib"] * MIB for peak in peaks.values()):
        raise ValueError("Every inherited peak must remain within the unchanged allocator ceiling")
    return {**result, "inheritance_verified": True, "inherited_capacity": copy.deepcopy(binding),
            "max_reserved_bytes": peaks, "current_live_admission_performed": False}


def verify_capacity_inheritance_for_request(capacity_binding, request):
    """Check a current V6 request against its pinned plan and return only its model's peak."""
    bridge = _read(capacity_binding)
    if not isinstance(bridge, dict) or set(bridge) != {"kind", "plan", "inherited_capacity"} or bridge["kind"] != KIND:
        raise ValueError("Exact current-plan capacity inheritance bridge required")
    plan = _read(bridge["plan"])
    if (request.get("capacity_manifest") != capacity_binding
            or Path(capacity_binding["path"]).resolve() != Path(plan["output"]) / "capacity-inheritance.json"
            or plan.get("inherited_capacity") != bridge["inherited_capacity"]):
        raise ValueError("Capacity bridge must belong to the current exact plan and output")
    if (request.get("schema_version") != 1 or request.get("kind") != "alignlab-evaluation-inference-stage-v2"
            or (request.get("task"), request.get("model_id")) not in TASKS
            or Path(request["output"]).resolve() != Path(plan["output"]) / (request["task"] + "-" + request["model_id"])):
        raise ValueError("Only a complete declared inference slot may consume inherited capacity")
    for key in ("source_sha256", "resource_profile", "deadline_epoch"):
        if request.get(key) != plan[key]:
            raise ValueError("Current request differs from its exact inherited-capacity plan: " + key)
    for key in ("protocol", "roster"):
        if {"path": request.get(key + "_path"), "sha256": request.get(key + "_sha256")} != plan[key]:
            raise ValueError("Current request changes its pinned protocol or roster")
    result = verify_inherited_capacity(bridge["inherited_capacity"], plan)
    return result["max_reserved_bytes"][request["model_id"]]
