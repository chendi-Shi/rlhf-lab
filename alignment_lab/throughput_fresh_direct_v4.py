"""Separately authorized startup for fresh seed44; numerical bytecode unchanged.

The pinned R5 acceptance receipt and its complete downstream metadata checks are
used at startup. Reopening the historical R5 tensor/evidence inventory is deferred
to the supervisor's final acceptance. Current model/data/parent weight/cache
loading, content validation, strict process guards and new-stage acceptance are
unchanged. No previous CPU report covers this separate entry.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

from . import throughput_fresh_v3 as frozen
from . import alignment_stage_r3 as original
from . import recovered_parent_acceptance_r5 as recovered

ROOT = frozen.ROOT
CHILD_CLI = ROOT / "scripts/throughput_direct_v4/run_alignlab_throughput_fresh_v3.py"


def _clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__doc__ = function.__doc__
    return result


def _accepted_receipt(path, expected_sha256, *, system=None):
    """Pin acceptance/manifest metadata; do not assert a new historical audit."""
    report, _ = recovered.base._read(path, expected_sha256)
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("kind") != recovered.KIND or report.get("status") != recovered.ACCEPTED
            or any(report.get(k) is not True for k in (
                "all_checks_passed", "parents_verified", "formal_training", "processes_exited",
                "actual_cpu_payloads_verified"))
            or report.get("research_complete") is not False or report.get("cuda_initialized") is not False
            or report.get("device") != "cpu" or report.get("human_labels") != 0
            or report.get("human_win_rate") is not None
            or set(report.get("auditor_source_sha256", {})) != set(recovered.SOURCES)
            or set(report.get("stages", {})) != {"sft", "reward"}
            or type(report.get("evidence_files")) is not dict or not report["evidence_files"]):
        raise ValueError("Exact accepted independent R5 receipt required")
    for name, expected in report["auditor_source_sha256"].items():
        original.parent_stage._pin(ROOT / name, expected)
    recovery_manifest = recovered.base._read(
        report["recovery_manifest"]["path"], report["recovery_manifest"]["sha256"])[0]
    if (recovery_manifest["completion"] != report["completion"]
            or recovery_manifest["budget"] != report["budget"]):
        raise ValueError("Accepted R5 completion or historical budget differs")
    for name, record in report["stages"].items():
        rows, updates = original.parent_stage.EXPECTED[name][:2]
        if record.get("rows") != rows or record.get("optimizer_updates") != updates:
            raise ValueError("Accepted parent workload differs")
        manifest = Path(record["artifact_dir"]) / "manifest.json"
        original.parent_stage._pin(manifest, record["manifest_sha256"])
        value = original.read_json(manifest)
        if value.get("stage") != name or value.get("status") != "complete":
            raise ValueError("Current parent artifact manifest differs from its receipt")
    return report


def _parent_acceptance(request):
    namespace = dict(vars(recovered))
    namespace["verify_recovered_parents"] = _accepted_receipt
    # Retain the original downstream model/data/workload/parent/source bindings.
    return _clone(recovered.validate_recovered_for_request, namespace)(request)


def _namespace(wrapper, execution_manifest):
    from .throughput_supervisor_direct_v4 import validate_manifest

    manifest_pin = copy.deepcopy(execution_manifest)
    plan_pin = copy.deepcopy(wrapper["plan"])
    validate_manifest(manifest_pin, plan_pin)
    original_namespace = dict(vars(original))
    original_namespace["validate_parent_acceptance"] = _parent_acceptance
    original_proxy = SimpleNamespace(**vars(original))
    original_proxy.validate_request = _clone(original.validate_request, original_namespace)
    namespace = dict(vars(frozen))
    namespace["original"] = original_proxy
    request_validator = _clone(frozen.validate_request, namespace)

    def validate_request(value):
        validate_manifest(manifest_pin, value["plan"])
        if value["plan"] != plan_pin:
            raise ValueError("Direct child plan changed")
        return request_validator(value)

    def marker(value):
        result = frozen._marker(value)
        result.update(direct_execution_manifest=copy.deepcopy(manifest_pin),
            inherited_parent_history_reverification="deferred_until_final",
            new_entry_cpu_tests_run=False)
        return result

    def wrapper_inputs(value, output):
        manifest = validate_manifest(manifest_pin, value["plan"])
        files = frozen._wrapper_inputs(value, output)
        files["direct_v4/execution_manifest"] = Path(manifest_pin["path"])
        files["direct_v4/authorization"] = Path(manifest["authorization"]["path"])
        files.update({"direct_v4/source/" + name: ROOT / name for name in manifest["source_sha256"]})
        return files

    def owned(snapshot, request, request_path, request_sha256, *, pid=None):
        # Frozen ancestry verifier retains exact PID/create/parent/supervisor
        # identity. Additional checks bind the new absolute CLI and manifest.
        result = frozen._owned(snapshot, request, request_path, request_sha256, pid=pid)
        for record in result:
            command = (record.get("command_line") or "").lower().replace("\\", "/")
            if (str(CHILD_CLI.resolve()).lower().replace("\\", "/") not in command
                    or "--execution-manifest" not in command
                    or "--execution-manifest-sha256" not in command
                    or str(Path(manifest_pin["path"]).resolve()).lower().replace("\\", "/") not in command
                    or manifest_pin["sha256"].lower() not in command):
                raise RuntimeError("Direct child command lacks exact CLI/manifest binding")
        return result

    namespace.update(validate_request=validate_request, _marker=marker,
                     _wrapper_inputs=wrapper_inputs, _owned=owned)
    return namespace


def run_stage(wrapper, execution_manifest, *, execute=False):
    """Execute frozen numerical/guard bytecode with explicit startup authority."""
    namespace = _namespace(wrapper, execution_manifest)
    return _clone(frozen.run_stage, namespace)(wrapper, execute=execute)


def verify_stage(output, request, known, code, execution_manifest):
    """Keep full R3 new-stage/checkpoint/Adam/RNG/observation acceptance."""
    namespace = _namespace(request, execution_manifest)
    return _clone(frozen.verify_stage, namespace)(output, request, known, code)
