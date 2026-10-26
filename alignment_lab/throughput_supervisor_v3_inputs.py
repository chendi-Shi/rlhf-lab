"""V3 with precisely scoped historical service observations in input closure.

The original V3 source is frozen. Its full semantic verifier and dispatch loop
are reused as identical bytecode with a local namespace, never monkey-patched.
Existing historical-admission pins resolve to their accepted immutable copies.
One exact historical verifier observation remains part of its pinned report,
rather than becoming a dependency on another project's current source file.
"""
from __future__ import annotations

from pathlib import Path
from types import FunctionType

from . import throughput_supervisor_v3 as frozen

ROOT, KIND, resume = frozen.ROOT, frozen.KIND, frozen.resume
sha, _parse = frozen.sha, frozen._parse
ADDED_SOURCES = {
    "alignment_lab/throughput_supervisor_v3_inputs.py",
    "scripts/throughput_inputs_v3/run_alignlab_throughput_campaign_v3.py",
    "scripts/run_alignlab_throughput_cpu_checks_v3_inputs.py",
    "scripts/run_alignlab_throughput_prepare_v3_inputs.py",
    "tests/test_throughput_supervisor_v3_inputs.py",
}
NEW_SOURCES = frozen.NEW_SOURCES | ADDED_SOURCES
HISTORY_PLAN = "configs/alignlab-v2-rm-recovery-r5-20261021T0848001676127Z/plan.json"
HISTORY_PLAN_SHA = "d516e70572b9a3a7fa987e0829d6f14b814613c9763258f7c82dfe2f495ef349"
PRIORITY_REPORT = "reports/alignlab-v2-priority-preflight-20261020T071145Z.json"
PRIORITY_REPORT_SHA = "746d402baff778c9f6e2226ea743c3cdb8dfe568ceff1f22e36613f0a4b12059"


def _historical_map(sentinel):
    """Reuse the already accepted historical-admission snapshot semantics."""
    plan_path = ROOT / HISTORY_PLAN
    frozen._pin(plan_path, HISTORY_PLAN_SHA)
    sentinel.add_file(plan_path, HISTORY_PLAN_SHA, always_hash=True)
    historical = _parse(plan_path)["historical_prerequisites"]
    mapped, leaves = {}, {}
    for original, item in historical.items():
        snapshot = item["snapshot"]
        path = Path(snapshot["path"]).resolve()
        if (item["original_path"] != original
                or item["semantics"] != "historical_admission_fact_not_current_resource_state"
                or snapshot["sha256"] != item["original_sha256"]
                or not path.is_relative_to(ROOT) or path == Path(original).resolve()):
            raise ValueError("Frozen historical prerequisite mapping is invalid")
        frozen._pin(path, snapshot["sha256"])
        if path.stat().st_size != snapshot["size_bytes"]:
            raise ValueError("Historical prerequisite snapshot size changed")
        mapped[(str(Path(original).resolve()), item["original_sha256"])] = path
        leaves[str(path)] = snapshot["sha256"]
    return mapped, leaves

def _cpu_proof(plan):
    frozen._cpu_proof(plan)
    report = _parse(frozen._pin_record(plan["cpu_report"]))
    if (not ADDED_SOURCES <= plan["source_sha256"].keys()
            or "tests.test_throughput_supervisor_v3_inputs" not in report.get("modules", [])):
        raise ValueError("Current source-bound input-closure regression proof required")


def _checked_prefix(plan):
    _cpu_proof(plan)
    return frozen._checked_prefix(plan)


def _historical_observation(document, trail, value):
    if document != (ROOT / PRIORITY_REPORT).resolve() or trail != ("revla", "verifier"):
        return False
    if (value.get("module_path") == str((ROOT.parent / "revla/windows/checkpointing.py").resolve())
            and value.get("module_sha256") == "6916f81a648d90fdff2dca8267f98d9f544c22654e1f57ce60e56580fd3899df"
            and value.get("training_state_loaded") is False and value.get("torch_imported") is False):
        frozen._pin(document, PRIORITY_REPORT_SHA)
        return True
    return False

def _add_document_dependencies(sentinel):
    """Preserve V3's input closure, except the exact historic observations."""
    visited = set()
    historical, historical_leaves = _historical_map(sentinel)

    def include(filename, expected=None):
        filename = Path(filename).resolve()
        filename = historical.get((str(filename), expected), filename)
        if not filename.is_relative_to(ROOT):
            raise ValueError("Acceptance dependency escaped the project: " + str(filename))
        sentinel.add_file(filename)
        if str(filename) in historical_leaves:
            frozen._pin(filename, historical_leaves[str(filename)])
            return
        key = str(filename)
        if filename.suffix.lower() != ".json" or key in visited:
            return
        visited.add(key)
        visit(_parse(filename), filename, ())

    def visit(value, document, trail):
        if type(value) is dict:
            if _historical_observation(document, trail, value):
                return
            if type(value.get("path")) is str and type(value.get("sha256")) is str:
                candidate = Path(value["path"])
                if candidate.is_absolute():
                    include(candidate, value["sha256"])
                else:
                    candidates = [document.parent / candidate, ROOT / candidate]
                    existing = [p for p in candidates if p.is_file()]
                    if not existing:
                        raise ValueError("Relative pinned acceptance input disappeared")
                    for item in existing:
                        include(item, value["sha256"])
            for name, digest in value.items():
                if name.endswith("_sha256") and type(digest) is str:
                    base = name[:-7]
                    filename = value.get(base, value.get(base + "_path"))
                    if type(filename) is str:
                        candidate = Path(filename)
                        candidate = candidate if candidate.is_absolute() else ROOT / candidate
                        if candidate.is_file() or (str(candidate.resolve()), digest) in historical:
                            include(candidate, digest)
                if type(digest) in (dict, list):
                    visit(digest, document, (*trail, name))
        elif type(value) is list:
            for index, item in enumerate(value):
                if type(item) in (dict, list):
                    visit(item, document, (*trail, index))

    for filename in list(sentinel.files):
        if Path(filename).suffix.lower() == ".json":
            include(filename)


def _own_processes(processes, plan_path):
    by_pid = {p["pid"]: p for p in processes}
    current, own = by_pid.get(frozen.os.getpid()), {}
    expected_sha = sha(plan_path)
    while current and frozen._command(current, str(ROOT / "scripts/throughput_inputs_v3/run_alignlab_throughput_campaign_v3.py").lower().replace("\\", "/"), plan_path):
        if current["pid"] in own or expected_sha not in current["command_line"]:
            raise RuntimeError("Supervisor command/ancestry mismatch")
        frozen._stamp(current["created_at_utc"])
        own[current["pid"]] = current
        current = by_pid.get(current.get("parent_pid"))
    if not own:
        raise RuntimeError("Cannot identify exact V3 input-closure supervisor chain")
    return own


def _clone(function, namespace):
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__doc__ = function.__doc__
    return clone


class _LiveIdentity:
    @classmethod
    def from_plan(cls, path, expected_sha, plan):
        return _clone(frozen._LiveIdentity.from_plan.__func__, _execution_namespace)(cls, path, expected_sha, plan)


# This namespace belongs exclusively to this version. All unlisted functions,
# guards, full artifact acceptance, stage selection and numerical code are V3.
_execution_namespace = dict(vars(frozen))
_execution_namespace.update(_LiveIdentity=_LiveIdentity, _add_document_dependencies=_add_document_dependencies,
    _own_processes=_own_processes, _checked_prefix=_checked_prefix, _cpu_proof=_cpu_proof)
_metadata_plan = _clone(frozen._metadata_plan, _execution_namespace)
validate_plan = _clone(frozen.validate_plan, _execution_namespace)
_execution_namespace.update(_metadata_plan=_metadata_plan, validate_plan=validate_plan)
run_alignment = _clone(frozen.run_alignment, _execution_namespace)
_ContentIdentity = frozen._ContentIdentity
