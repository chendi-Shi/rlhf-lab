"""Reviewed metadata wire-format repair around the frozen recovery engine.

No numerical loop, old evidence, or budget is replaced. Runtime tuples and the
corresponding JSON arrays compare through the complete JSON representation.
Every real field still participates; a new explicit dispatch authorization is
required. This module never retries a failed process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import parent_recovery as r1

REVISION = "parent-recovery-r2"
NEW_SOURCES = frozenset({"alignment_lab/parent_recovery_r2.py",
    "scripts/run_alignlab_parent_recovery_r2.py", "tests_v2/test_parent_recovery_r2.py"})


def canonical_metadata(value):
    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("Metadata keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) in (list, tuple):
            for child in item:
                check(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("Unsupported metadata type")
    check(value)
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compare_metadata(actual, expected):
    actual, expected = canonical_metadata(actual), canonical_metadata(expected)
    differences = []
    def visit(a, b, path):
        if type(a) is dict and type(b) is dict:
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append({"path": path + [key], "actual_present": key in a,
                                        "expected_present": key in b})
                else:
                    visit(a[key], b[key], path + [key])
        elif _wire(a) != _wire(b):
            differences.append({"path": path, "actual": a, "expected": b})
    visit(actual, expected, [])
    return {"schema_version": 1, "kind": "alignlab-loaded-model-metadata-comparison-v1",
        "execution_revision": REVISION, "normalization": "complete_json_roundtrip_no_fields_removed",
        "actual": actual, "expected": expected, "differences": differences,
        "all_fields_equal": not differences}


def validate_request(request, *, execute=False):
    context = r1.validate_request(request)
    plan = context["plan"]
    if plan.get("execution_revision") != REVISION or not NEW_SOURCES <= set(request["source_sha256"]):
        raise ValueError("The independently pinned r2 runtime and entry must be explicit")
    prior = plan.get("prior_failed_dispatch")
    if type(prior) is not dict or set(prior) != {"request", "progress"}:
        raise ValueError("The failed r1 dispatch must be pinned")
    previous = r1.read_json(r1._pinned(prior["request"]))
    progress_path = r1._pinned(prior["progress"])
    progress = r1.read_json(progress_path)
    r1.validate_request(previous)
    if (previous["budget"] != request["budget"] or previous["origin"] != request["origin"]
            or progress_path != Path(previous["output"]) / "progress.json"
            or progress.get("request") != previous or progress.get("status") != "failed"
            or progress.get("actual_optimizer_updates") != r1.ORIGIN_UPDATE
            or progress.get("latest_safe_checkpoint") is not None
            or progress.get("error") != "ValueError: Actual loaded model/runtime differs from the saved stage"
            or (Path(previous["output"]) / "manifest.json").exists()
            or context["output"].is_relative_to(Path(previous["output"]))):
        raise ValueError("Only the recorded pre-update metadata failure and identical absolute budget are eligible")
    if any(request["source_sha256"].get(name) != pin for name, pin in previous["source_sha256"].items()):
        raise ValueError("All frozen r1 sources must remain pinned unchanged")
    if execute:
        authorization = r1.read_json(r1._pinned(plan.get("retry_authorization")))
        if (authorization.get("kind") != "explicit-user-rm-r2-dispatch-authorization-v1"
                or authorization.get("source") != "direct_user_reply_in_this_chat"
                or authorization.get("allow_one_corrected_dispatch") is not True
                or authorization.get("budget") != request["budget"]
                or authorization.get("prior_request_sha256") != prior["request"]["sha256"]):
            raise ValueError("An explicit authorization for this one corrected dispatch is required")
    return context


class _EntryRuntime(r1._Runtime):
    def bind_processes(self, request_path, request_sha):
        """Same strict real-process binding, for the separately pinned r2 entry."""
        if os.name != "nt":
            raise RuntimeError("Production recovery requires real Windows process identities")
        import ctypes
        from ctypes import wintypes
        import sys
        parse = ctypes.windll.shell32.CommandLineToArgvW
        parse.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
        parse.restype = ctypes.POINTER(wintypes.LPWSTR)
        release = ctypes.windll.kernel32.LocalFree
        release.argtypes = (ctypes.c_void_p,)
        indexed = {p["pid"]: p for p in self.processes()}
        owned, current = [], os.getpid()
        while current in indexed:
            record = indexed[current]
            if not record.get("command_line"):
                raise RuntimeError("Own process command is unavailable")
            count = ctypes.c_int()
            argv = parse(record["command_line"], ctypes.byref(count))
            try:
                args = [argv[i] for i in range(count.value)]
            finally:
                release(argv)
            executable = Path(args.pop(0)).resolve()
            if executable not in {Path(sys.executable).resolve(), Path(sys._base_executable).resolve(),
                    (r1.ROOT / ".runtime/alignlab-v2/Scripts/python.exe").resolve()}:
                raise RuntimeError("Unexpected recovery Python executable")
            while args and args[0] in {"-u", "-I", "-B"}:
                args.pop(0)
            wanted = ["--request", str(Path(request_path).resolve()), "--request-sha256", request_sha, "--execute"]
            if (not args or Path(args[0]).resolve() != r1.ROOT / "scripts/run_alignlab_parent_recovery_r2.py"
                    or args[1:] != wanted):
                raise RuntimeError("A Python ancestor is not this exact pinned r2 recovery command")
            owned.append(record)
            current = record["parent_pid"]
        if not owned:
            raise RuntimeError("Own process identity was not observed")
        return owned


class _MetadataRuntime:
    def __init__(self, runtime, context):
        self.runtime, self.context = runtime, context

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def load(self, request):
        loaded = self.runtime.load(request)
        receipt = compare_metadata(loaded.metadata, self.context["old_identity"]["config"]["model"])
        receipt["runtime_target_modules_type"] = type(loaded.metadata.get("config", {}).get("target_modules")).__name__
        r1._write(self.context["output"] / "model-metadata-comparison.json", receipt, exclusive=True)
        if not receipt["all_fields_equal"]:
            raise ValueError("Actual loaded model metadata has a substantive difference; see comparison receipt")
        loaded.metadata = receipt["actual"]
        return loaded


def run_recovery(request_path, expected_request_sha256, *, execute=False, runtime=None):
    request = r1.read_json(r1._pinned({"path": str(request_path), "sha256": expected_request_sha256}))
    context = validate_request(request, execute=execute)
    return r1.run_recovery(request_path, expected_request_sha256, execute=execute,
                          runtime=_MetadataRuntime(runtime or _EntryRuntime(), context))


def verify_recovery_output(output, *, expected_manifest_sha256, expected_request=None):
    result = r1.verify_recovery_output(output, expected_manifest_sha256=expected_manifest_sha256,
                                       expected_request=expected_request)
    request = r1.read_json(r1._pinned(result["request"]))
    context = validate_request(request, execute=True)
    receipt = r1.read_json(Path(output) / "model-metadata-comparison.json")
    expected = compare_metadata(receipt["actual"], context["old_identity"]["config"]["model"])
    if ({k: v for k, v in receipt.items() if k != "runtime_target_modules_type"} != expected
            or expected["all_fields_equal"] is not True
            or r1.read_json(Path(output) / "progress.json")["model"] != expected["actual"]):
        raise ValueError("The full model metadata comparison receipt is inconsistent")
    return result
