"""Transitive V5 nine-model CUDA capacity proof for the fresh V10 continuation.

The original V6 bridge is reverified by its frozen verifier. New GPU execution
retains that proof and adds a complete V9-to-V10 status-I/O-only AST proof.
This is a saved-evidence proof, never fresh GPU capacity or live admission.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

from .evaluation_protocol_v2 import verified_bytes, _relative
from . import evaluation_capacity_inheritance_v6 as previous
from . import evaluation_capacity_inheritance_v9 as frozen_v9
from .evaluation_capacity_inheritance_v7 import EXTRA_EXECUTION_SOURCES as PRESERVED_V7_EXECUTION_SOURCES
from .evaluation_capacity_inheritance_v8 import EXTRA_EXECUTION_SOURCES as PRESERVED_V8_EXECUTION_SOURCES
from .evaluation_inheritance_v10 import _read, verify_recovery_context
from .evaluation_status_semantics_v10 import validate_v10_execution_semantics, ADDED_SOURCES

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-evaluation-capacity-inheritance-v10"
PIN_FIELDS = frozenset({"prior_plan", "campaign", "closure", "bridge"})
MODEL_IDS = previous.MODEL_IDS
TASKS = frozenset({("ifeval", "ppo44"), ("helpsteer_rm", "reward"), ("rm_bench", "reward")})
EXTRA_EXECUTION_SOURCES = frozenset(ADDED_SOURCES) | frozenset({
    "alignment_lab/evaluation_execution_v10.py", "tests_v2/test_evaluation_execution_v10.py",
    "alignment_lab/evaluation_resources_v10.py", "tests_v2/test_evaluation_resources_v10.py",
    "scripts/run_alignlab_evaluation_stage_v10.py",
    "alignment_lab/evaluation_capacity_inheritance_v10.py", "tests_v2/test_evaluation_capacity_inheritance_v10.py",
})
def verify_execution_ast(prior_source, current_source, *, execution=False, v9_source=None):
    """Prove V6 -> frozen V9, then exactly reviewed V9 -> V10 status I/O."""
    if v9_source is None:
        raise ValueError("Externally pinned intermediate V9 source bytes are mandatory")
    if execution:
        validate_v10_execution_semantics(v9_source, current_source)
    else:
        before, after = ast.parse(v9_source), ast.parse(current_source)
        matches = [node for node in after.body if isinstance(node, ast.ImportFrom)
                   and node.module == "alignment_lab.evaluation_execution_v10"]
        expected = ast.parse(
            "from alignment_lab.evaluation_execution_v10 import run_evaluation_stage").body[0]
        if len(matches) != 1 or ast.dump(matches[0]) != ast.dump(expected):
            raise ValueError("Only the exact V10 worker entrypoint import is permitted")
        matches[0].module = "alignment_lab.evaluation_execution_v9"
        if ast.dump(before) != ast.dump(after):
            raise ValueError("V10 worker entrypoint may change only its execution-module version")
    # This frozen guard retains the original admission/capacity routing proof.
    frozen_v9.verify_execution_ast(prior_source, v9_source, execution=execution)


def _sources(prior, plan):
    # verify_recovery_context already rehashes all 211 V9 pins and all new pins.
    current = plan["source_sha256"]
    required = (EXTRA_EXECUTION_SOURCES | frozen_v9.EXTRA_EXECUTION_SOURCES
                | PRESERVED_V7_EXECUTION_SOURCES | PRESERVED_V8_EXECUTION_SOURCES)
    if not required <= current.keys():
        raise ValueError("V10 status publisher, semantics proof and all previous execution sources must be pinned")
    for stem, execution in (("alignment_lab/evaluation_execution", True),
                            ("scripts/run_alignlab_evaluation_stage", False)):
        old, intermediate, new = (stem + suffix for suffix in ("_v6.py", "_v9.py", "_v10.py"))
        if old not in prior["source_sha256"] or intermediate not in current or new not in current:
            raise ValueError("Original, intermediate and current GPU entrypoints must all be frozen")
        verify_execution_ast(
            verified_bytes(_relative(ROOT, old), prior["source_sha256"][old]).decode("utf-8-sig"),
            verified_bytes(_relative(ROOT, new), current[new]).decode("utf-8-sig"),
            v9_source=verified_bytes(_relative(ROOT, intermediate), current[intermediate]).decode("utf-8-sig"),
            execution=execution)

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
    from .evaluation_resources_v10 import load_profile, MIB
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
        "v6_bridge_reverified": True, "v9_execution_ast_equivalent": True, "v10_status_io_only_ast_verified": True,
        "current_live_admission_performed": False}


def verify_capacity_inheritance_for_request(capacity_binding, request):
    bridge = _read(capacity_binding)
    if not isinstance(bridge, dict) or set(bridge) != {"kind", "plan", "inherited_capacity"} or bridge["kind"] != KIND:
        raise ValueError("Exact current V10 capacity inheritance bridge required")
    plan = _read(bridge["plan"])
    if (request.get("capacity_manifest") != capacity_binding
            or Path(capacity_binding["path"]).resolve() != Path(plan["output"]) / "capacity-inheritance.json"
            or plan.get("inherited_capacity") != bridge["inherited_capacity"]):
        raise ValueError("Capacity bridge must belong to the current exact plan and output")
    if (request.get("schema_version") != 1 or request.get("kind") != "alignlab-evaluation-inference-stage-v2"
            or (request.get("task"), request.get("model_id")) not in TASKS
            or Path(request["output"]).resolve() != Path(plan["output"]) / (request["task"] + "-" + request["model_id"])):
        raise ValueError("Only the three remaining complete inference slots may consume V10 capacity")
    for key in ("source_sha256", "resource_profile", "deadline_epoch"):
        if request.get(key) != plan[key]:
            raise ValueError("Current request differs from exact V10 capacity plan: " + key)
    for key in ("protocol", "roster"):
        if {"path": request.get(key + "_path"), "sha256": request.get(key + "_sha256")} != plan[key]:
            raise ValueError("Current request changes its pinned protocol or roster")
    result = verify_inherited_capacity(bridge["inherited_capacity"], plan)
    return result["max_reserved_bytes"][request["model_id"]]


verify_capacity_bridge = verify_capacity_inheritance_for_request
