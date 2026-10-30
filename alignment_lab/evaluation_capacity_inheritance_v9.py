"""Transitive V5 nine-model CUDA capacity proof for the fresh V9 continuation.

The original V6 bridge is reverified by its frozen verifier. New GPU execution
must be AST-identical except for the exact capacity/resource version routing.
This is a saved-evidence proof, never fresh GPU capacity or live admission.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

from .evaluation_protocol_v2 import verified_bytes, _relative
from . import evaluation_capacity_inheritance_v6 as previous
from .evaluation_capacity_inheritance_v7 import EXTRA_EXECUTION_SOURCES as PRESERVED_V7_EXECUTION_SOURCES
from .evaluation_capacity_inheritance_v8 import EXTRA_EXECUTION_SOURCES as PRESERVED_V8_EXECUTION_SOURCES
from .evaluation_inheritance_v9 import _read, verify_recovery_context

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-evaluation-capacity-inheritance-v9"
PIN_FIELDS = frozenset({"prior_plan", "campaign", "closure", "bridge"})
MODEL_IDS = previous.MODEL_IDS
TASKS = frozenset({("ifeval", "ppo44"), ("helpsteer_rm", "reward"), ("rm_bench", "reward")})
EXTRA_EXECUTION_SOURCES = frozenset({
    "alignment_lab/evaluation_execution_v9.py", "tests_v2/test_evaluation_execution_v9.py",
    "alignment_lab/evaluation_resources_v9.py", "tests_v2/test_evaluation_resources_v9.py",
    "scripts/run_alignlab_evaluation_stage_v9.py",
    "alignment_lab/evaluation_capacity_inheritance_v9.py", "tests_v2/test_evaluation_capacity_inheritance_v9.py",
})
VERSION_NAMES = ("evaluation_execution", "evaluation_resources", "evaluation_capacity_inheritance",
                 "evaluation_supervisor", "run_alignlab_evaluation_stage", "supervise_alignlab_evaluation")
CAPACITY_FUNCTION = '''
def _capacity_peak(request):
    binding = request["capacity_manifest"]
    if binding is None:
        raise ValueError("Real full nine-model capacity proof required before production inference")
    from .evaluation_capacity_inheritance_v9 import verify_capacity_inheritance_for_request
    return verify_capacity_inheritance_for_request(binding, request)
'''


def _version(value):
    for name in VERSION_NAMES:
        for version in ("v7", "v8", "v9"):
            value = value.replace(name + "_" + version, name + "_v6")
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
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return tree


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _one(tree, kind, name):
    nodes = [node for node in tree.body if isinstance(node, kind)
        and (getattr(node, "name", None) == name if kind is ast.FunctionDef else
             len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)]
    if len(nodes) != 1:
        raise ValueError("Exactly one explicit " + name + " declaration required")
    return nodes[0]


def _literal_sources(node):
    value = node.value
    if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
            or value.func.id != "frozenset" or value.keywords or len(value.args) != 1
            or not isinstance(value.args[0], ast.Set)
            or any(not isinstance(n, ast.Constant) or not isinstance(n.value, str) for n in value.args[0].elts)):
        raise ValueError("Literal explicit frozen source inventory required")
    names = [n.value for n in value.args[0].elts]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate execution source pins forbidden")
    return set(names)


def verify_execution_ast(prior_source, current_source, *, execution=False):
    """Permit exact admission routing only; retain every GPU executable node."""
    old, new = _tree(prior_source), _tree(current_source)
    if execution:
        old_function, new_function = (_one(t, ast.FunctionDef, "_capacity_peak") for t in (old, new))
        if _dump(new_function) != _dump(_tree(CAPACITY_FUNCTION).body[0]):
            raise ValueError("Only the exact V9 inherited-capacity function is permitted")
        new.body[new.body.index(new_function)] = copy.deepcopy(old_function)
        old_sources, new_sources = (_one(t, ast.Assign, "REQUIRED_SOURCES") for t in (old, new))
        if _literal_sources(new_sources) != (_literal_sources(old_sources)
                | PRESERVED_V7_EXECUTION_SOURCES | PRESERVED_V8_EXECUTION_SOURCES | EXTRA_EXECUTION_SOURCES):
            raise ValueError("V9 execution must retain V6, V7 and V8 sources and add only the exact V9 bridge sources")
        new.body[new.body.index(new_sources)] = copy.deepcopy(old_sources)
    if _dump(_Versions().visit(old)) != _dump(_Versions().visit(new)):
        raise ValueError("Inherited CUDA capacity cannot cover changed GPU execution semantics")


def _sources(prior, plan):
    # Common context already rehashes all 192 historical pins and all new pins.
    current = plan["source_sha256"]
    if not (EXTRA_EXECUTION_SOURCES | PRESERVED_V7_EXECUTION_SOURCES | PRESERVED_V8_EXECUTION_SOURCES) <= current.keys():
        raise ValueError("V9 capacity bridge and GPU execution implementation/tests must be pinned")
    for stem, execution in (("alignment_lab/evaluation_execution", True),
                             ("scripts/run_alignlab_evaluation_stage", False)):
        old, new = stem + "_v6.py", stem + "_v9.py"
        if old not in prior["source_sha256"] or new not in current:
            raise ValueError("Both original and current GPU entrypoints must be frozen")
        verify_execution_ast(verified_bytes(_relative(ROOT, old), prior["source_sha256"][old]).decode("utf-8-sig"),
            verified_bytes(_relative(ROOT, new), current[new]).decode("utf-8-sig"), execution=execution)


def verify_inherited_capacity(binding, plan):
    if not isinstance(binding, dict) or set(binding) != PIN_FIELDS:
        raise ValueError("Four exact V6 prior-plan/campaign/closure/bridge pins required")
    if plan.get("inherited_capacity") != binding:
        raise ValueError("Capacity inheritance must be explicitly frozen in the destination plan")
    prior, campaign, closure, _, _ = verify_recovery_context(binding, plan)
    bridge = _read(binding["bridge"])
    expected = {"kind": previous.KIND, "plan": binding["prior_plan"],
                "inherited_capacity": prior.get("inherited_capacity")}
    if (bridge != expected or campaign.get("capacity_manifest") != binding["bridge"]
            or Path(binding["bridge"]["path"]).resolve() != Path(prior["output"]) / "capacity-inheritance.json"):
        raise ValueError("Exact V6 bridge to its original full V5 CUDA capacity is required")
    _sources(prior, plan)
    # The original verifier proves V5 model identity, all eighteen full audits,
    # original source equivalence, accepted child and the V5 restoration chain.
    result = previous.verify_inherited_capacity(prior["inherited_capacity"], prior)
    if (result.get("inheritance_verified") is not True or set(result.get("models", {})) != MODEL_IDS
            or any(len(model.get("frozen_audits", [])) != 2 for model in result["models"].values())
            or campaign.get("capacity", {}).get("inherited_capacity") != prior["inherited_capacity"]
            or campaign.get("capacity", {}).get("inheritance_verified") is not True):
        raise ValueError("Only original complete nine-model and eighteen-audit capacity may be inherited")
    from .evaluation_resources_v9 import load_profile, MIB
    profile = load_profile(plan["resource_profile"])
    peaks = result["max_reserved_bytes"]
    if (set(peaks) != MODEL_IDS or any(type(peak) is not int
            or not 0 < peak <= profile["allocator_limit_mib"] * MIB for peak in peaks.values())):
        raise ValueError("All original capacity peaks must fit the unchanged allocator ceiling")
    # Re-read pinned bridge/context artifacts after full historical verification.
    for key, original in (("prior_plan", prior), ("campaign", campaign), ("closure", closure), ("bridge", bridge)):
        if _read(binding[key]) != original:
            raise ValueError("Capacity inheritance evidence changed during verification")
    return {**result, "inherited_capacity": copy.deepcopy(binding),
        "original_v5_inherited_capacity": copy.deepcopy(prior["inherited_capacity"]),
        "v6_bridge_reverified": True, "v9_execution_ast_equivalent": True,
        "current_live_admission_performed": False}


def verify_capacity_inheritance_for_request(capacity_binding, request):
    bridge = _read(capacity_binding)
    if not isinstance(bridge, dict) or set(bridge) != {"kind", "plan", "inherited_capacity"} or bridge["kind"] != KIND:
        raise ValueError("Exact current V9 capacity inheritance bridge required")
    plan = _read(bridge["plan"])
    if (request.get("capacity_manifest") != capacity_binding
            or Path(capacity_binding["path"]).resolve() != Path(plan["output"]) / "capacity-inheritance.json"
            or plan.get("inherited_capacity") != bridge["inherited_capacity"]):
        raise ValueError("Capacity bridge must belong to the current exact plan and output")
    if (request.get("schema_version") != 1 or request.get("kind") != "alignlab-evaluation-inference-stage-v2"
            or (request.get("task"), request.get("model_id")) not in TASKS
            or Path(request["output"]).resolve() != Path(plan["output"]) / (request["task"] + "-" + request["model_id"])):
        raise ValueError("Only the three remaining complete inference slots may consume V9 capacity")
    for key in ("source_sha256", "resource_profile", "deadline_epoch"):
        if request.get(key) != plan[key]:
            raise ValueError("Current request differs from exact V9 capacity plan: " + key)
    for key in ("protocol", "roster"):
        if {"path": request.get(key + "_path"), "sha256": request.get(key + "_sha256")} != plan[key]:
            raise ValueError("Current request changes its pinned protocol or roster")
    result = verify_inherited_capacity(bridge["inherited_capacity"], plan)
    return result["max_reserved_bytes"][request["model_id"]]


verify_capacity_bridge = verify_capacity_inheritance_for_request
