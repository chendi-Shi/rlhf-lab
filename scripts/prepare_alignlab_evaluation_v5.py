"""Freeze RM attention recovery and verified service identity succession.

The failed V4 plan, all its source pins, accepted models, scientific protocol
and absolute allocation remain immutable. A fresh proof covers changed V5 code.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_lab.evaluation_protocol_v2 import canonical, parse, verified_bytes
from alignment_lab.evaluation_supervisor_v5 import NEW_SOURCES, KIND, _static
from alignment_lab.evaluation_analysis_v5 import REQUIRED_SOURCES
from alignment_lab.evaluation_acceptance_v2 import SOURCES
from alignment_lab.evaluation_resources_v5 import load_profile

PRIOR_PLAN_SHA256 = "ce04b4e799d883d67f2dca4c920f07f93708b1475a228c53c84652231812176e"
RECOVERY_TEST_MODULES = frozenset({
    "tests_v2.test_evaluation_execution_v5", "tests_v2.test_evaluation_capacity_v5",
    "tests_v2.test_evaluation_supervisor_v5", "tests_v2.test_evaluation_analysis_v5",
    "tests_v2.test_evaluation_resources_v5", "tests_v2.test_evaluation_prepare_v5",
    "tests_v2.test_evaluation_attention_v5", "tests_v2.test_evaluation_model_v5",
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def read(binding):
    return parse(verified_bytes(Path(binding["path"]), binding["sha256"]))


def validate_reused_inputs(prior, candidate):
    for field in ("kind", "schema_version", "started_at_epoch", "deadline_epoch", "wall_seconds",
                  "analysis_wall_seconds", "poll_seconds", "roster", "decoder",
                  "model_python_sha256", "scoring_python_sha256"):
        if candidate.get(field) != prior.get(field):
            raise ValueError("Recovery changes an existing frozen field: " + field)
    for field in ("authorization", "protocol"):
        if candidate[field]["sha256"] != prior[field]["sha256"]:
            raise ValueError("Recovery changes original bound bytes: " + field)
    profile = load_profile(candidate["resource_profile"])
    pause = profile["authorized_pause"]
    if pause["baseline_profile"] != prior["resource_profile"]:
        raise ValueError("Authorized pause must bind the exact prior resource profile")
    if pause["deadline_epoch"] != prior["deadline_epoch"]:
        raise ValueError("Authorized pause cannot extend the existing deadline")
    if Path(candidate["output"]).resolve() == Path(prior["output"]).resolve():
        raise ValueError("Recovery requires a fresh output; failed output is historical evidence")


def verify_cpu_proof(cpu, sources):
    if (cpu.get("status") != "cpu_software_checks_passed" or cpu.get("source_unchanged") is not True
            or cpu.get("device") != "cpu" or cpu.get("cuda_initialized") is not False
            or type(cpu.get("tests_run")) is not int or cpu["tests_run"] <= 0
            or any(type(cpu.get(k)) is not int or cpu[k] != 0 for k in ("failures", "errors", "skipped"))
            or not RECOVERY_TEST_MODULES <= set(cpu.get("test_modules", []))):
        raise ValueError("Actual current V5 resource/integration regressions are required")
    if any(cpu.get("source_sha256", {}).get(name) != digest for name, digest in sources.items()):
        raise ValueError("CPU proof does not pin every current implementation/entrypoint/test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-plan", type=Path, required=True)
    parser.add_argument("--prior-plan-sha256", required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    args = parser.parse_args()
    if args.prior_plan_sha256 != PRIOR_PLAN_SHA256:
        raise ValueError("Recovery must start from the exact failed V4 plan")
    prior_binding = {"path": str(args.prior_plan.resolve()), "sha256": args.prior_plan_sha256}
    prior = read(prior_binding)
    if prior.get("kind") != KIND:
        raise ValueError("An explicitly pinned original evaluation plan is required")
    launch, output = args.launch.resolve(), args.output.resolve()
    if (launch.parent != ROOT / "reports" or not launch.is_dir() or output.parent != ROOT / "runs"
            or any(p.is_symlink() for p in (launch, output, *launch.parents, *output.parents))):
        raise ValueError("Use a fresh direct project launch/output without symlinks")
    if output.exists() or any((launch / name).exists() for name in (
            "plan.json", "process.json", "execution-launch.json")):
        raise FileExistsError("Existing attempt evidence may never be reused or overwritten")
    for name, digest in prior["source_sha256"].items():
        verified_bytes(ROOT / name, digest)
    candidate = copy.deepcopy(prior)
    candidate["output"] = str(output)
    for field, filename in (("authorization", "authorization-and-budget.json"),
                            ("resource_profile", "resource-profile.json"), ("protocol", "protocol.json")):
        candidate[field] = pin(launch / filename)
        read(candidate[field])
    validate_reused_inputs(prior, candidate)
    profile = load_profile(candidate["resource_profile"])
    pause = read(profile["authorized_pause"]["authorization"])
    if pause["evaluation_authorization"]["sha256"] != prior["authorization"]["sha256"]:
        raise ValueError("Pause permission must bind the original evaluation allocation")
    names = set(prior["source_sha256"]) | REQUIRED_SOURCES | set(SOURCES) | NEW_SOURCES | {
        "alignment_lab/evaluation_resources_v5.py", "tests_v2/test_evaluation_resources_v5.py",
        "alignment_lab/evaluation_model_v5.py", "tests_v2/test_evaluation_model_v5.py",
        "alignment_lab/evaluation_attention_v5.py", "tests_v2/test_evaluation_attention_v5.py",
        "alignment_lab/evaluation_attention_receipt_v5.py",
        "scripts/prepare_alignlab_evaluation_v5.py", "scripts/verify_alignlab_evaluation_v5.py",
        "tests_v2/test_evaluation_prepare_v5.py",
    }
    candidate["source_sha256"] = {name: sha(ROOT / name) for name in sorted(names)}
    candidate["cpu_check_report"] = pin(args.cpu_report)
    verify_cpu_proof(read(candidate["cpu_check_report"]), candidate["source_sha256"])
    for name in ("recovery-authorization.json", "source-versioning-provenance.json", "evidoc-pause-authorization.json"):
        binding = pin(launch / name)
        if not read(binding):
            raise ValueError("Recovery provenance is empty: " + name)
        candidate["prerequisites"].append({**binding, "json_equals": {}})
    candidate["prerequisites"].append({**prior_binding, "json_equals": {}})
    if "torch" in sys.modules:
        raise RuntimeError("Freezing recovery inputs must never import Torch")
    candidate_path = launch / "plan-candidate.json"
    with candidate_path.open("xb") as stream:
        stream.write(canonical(candidate))
    _static(candidate_path, sha(candidate_path))
    destination = launch / "plan.json"
    with destination.open("xb") as stream:
        stream.write(canonical(candidate))
    print(json.dumps({"status": "recovery_plan_frozen_static_validated", "plan": pin(destination),
        "protocol": candidate["protocol"], "roster": candidate["roster"], "output": str(output),
        "deadline_epoch": candidate["deadline_epoch"], "model_acceptance_repeated": False,
        "model_loaded": False, "cuda_used": False, "source_count": len(candidate["source_sha256"])}))


if __name__ == "__main__":
    main()
