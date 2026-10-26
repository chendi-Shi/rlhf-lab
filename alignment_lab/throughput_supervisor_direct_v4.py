"""Explicit user-directed entry using existing acceptance receipts at startup.

Frozen numerical code, scheduling, workload, process guards and old plans stay
unchanged. Historical payload re-verification is deferred until both new seed44
children have exited, before any primary-success publication. This entry has
not been covered by the historical CPU test report; its separate manifest says
so and binds the direct user instruction and every new source file.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import FunctionType

from . import throughput_supervisor_v3 as frozen
from .alignment_supervisor import _LiveIdentity as BaseIdentity

ROOT = frozen.ROOT
MANIFEST_KIND = "alignlab-user-directed-training-entry-v4"
SOURCE_FILES = {
    "alignment_lab/throughput_supervisor_direct_v4.py",
    "alignment_lab/throughput_fresh_direct_v4.py",
    "scripts/throughput_direct_v4/run_alignlab_throughput_campaign_v3.py",
    "scripts/throughput_direct_v4/run_alignlab_throughput_fresh_v3.py",
}
SUPERVISOR_CLI = ROOT / "scripts/throughput_direct_v4/run_alignlab_throughput_campaign_v3.py"
CHILD_CLI = ROOT / "scripts/throughput_direct_v4/run_alignlab_throughput_fresh_v3.py"


def _project_pin(pin):
    if type(pin) is not dict or set(pin) != {"path", "sha256"}:
        raise ValueError("Exact path/SHA pin required")
    raw = Path(pin["path"])
    if not raw.is_absolute():
        raise ValueError("Absolute manifest dependency required")
    BaseIdentity._metadata(raw)
    path = raw.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Direct-entry authority escaped the project")
    frozen._pin(path, pin["sha256"])
    return path


def validate_manifest(pin, plan_pin):
    """Authenticate the new entry independently of the unchanged old plan."""
    path = _project_pin(pin)
    value = frozen._parse(path)
    fields = {"schema_version", "kind", "plan", "authorization", "source_sha256",
              "defer_inherited_reverification_until_final", "no_new_tests_run"}
    if (type(value) is not dict or set(value) != fields
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != MANIFEST_KIND or value["plan"] != plan_pin
            or value["defer_inherited_reverification_until_final"] is not True
            or value["no_new_tests_run"] is not True
            or set(value["source_sha256"]) != SOURCE_FILES):
        raise ValueError("Exact separately authorized direct-entry manifest required")
    _project_pin(plan_pin)
    authorization = frozen._parse(_project_pin(value["authorization"]))
    if (authorization.get("source") != "direct_user_message_in_this_chat"
            or authorization.get("user_message") != "直接进入训练好吗 我最后说一遍"):
        raise ValueError("Direct user instruction for this startup replacement required")
    for name, digest in value["source_sha256"].items():
        _project_pin({"path": str(ROOT / name), "sha256": digest})
    return value


def _clone(function, namespace):
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__doc__ = function.__doc__
    return clone


def _receipt_view(path, expected_sha):
    """Check frozen plans and receipts, without reopening historical tensors."""
    frozen._pin(path, expected_sha)
    plan = frozen._parse(path)
    _, campaign, receipt = frozen._checked_prefix(plan)
    original_path = frozen._pin_record(plan["original_plan"])
    original = frozen._parse(original_path)
    if (original.get("kind") != frozen.legacy.KIND or original.get("ppo_rows") != 1024
            or original.get("continuation", {}).get("deadline_epoch") != frozen.resume.DEADLINE):
        raise ValueError("Original registered alignment plan differs")
    frozen.process_guard.validate_policy(original["process_wait_policy"])
    frozen.process_guard.validate_background(original["allowed_background_processes"])
    inherited = copy.deepcopy(campaign["stages"])
    for record in inherited[:3]:
        request = record["request"].get("legacy_request", record["request"])
        progress_path = Path(request["output"]).resolve() / "progress.json"
        if Path(record["progress_path"]).resolve() != progress_path:
            raise ValueError("Accepted progress path differs from the original stage")
        frozen._pin(progress_path, record["progress_sha256"])
        progress = frozen._parse(progress_path)
        expected = 822 if record["stage"] == "dpo" else 2048
        if (progress.get("status") != "formal_alignment_stage_passed"
                or progress.get("actual_optimizer_updates") != expected
                or progress.get("artifact") != record["artifact"]):
            raise ValueError("Existing accepted stage receipt differs")
        artifact = record["artifact"]
        frozen._pin(Path(artifact["artifact_dir"]) / "manifest.json", artifact["manifest_sha256"])
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
    view["continuation"] = {**original["continuation"],
        "process_wait_spent_seconds": plan["process_wait_spent_seconds"],
        "started_at_epoch": plan["execution_window"]["started_at_epoch"],
        "deadline_epoch": plan["execution_window"]["deadline_epoch"]}
    frozen._pin(path, expected_sha)
    return view


def run_alignment(plan_path, expected_plan_sha256, execution_manifest, *, execute=False, system=None):
    plan_path = Path(plan_path).resolve()
    plan_pin = {"path": str(plan_path), "sha256": expected_plan_sha256}
    manifest_pin = copy.deepcopy(execution_manifest)
    manifest = validate_manifest(manifest_pin, plan_pin)
    clock_system = system or frozen._System()
    context = {"final_verified": False}

    def metadata(path, digest):
        validate_manifest(manifest_pin, plan_pin)
        view = _receipt_view(path, digest)
        context["view"] = view
        return view

    class LiveIdentity(BaseIdentity):
        @classmethod
        def from_plan(cls, path, digest, plan):
            sentinel = cls({})
            pins = [plan_pin, manifest_pin, manifest["authorization"],
                    *(plan[k] for k in ("original_plan", "failed_v2_plan", "failed_v2_campaign",
                        "recovery_authorization", "budget_authorization", "supplemental_ppo43",
                        "cpu_report", "validation_report", "authorization", "original_request"))]
            for pin in pins:
                sentinel.add_file(pin["path"], pin["sha256"], always_hash=True)
            for name, source_sha in {**plan["source_sha256"], **manifest["source_sha256"]}.items():
                sentinel.add_file(ROOT / name, source_sha, always_hash=True)
            for key, digest_key in (("parent_plan_path", "parent_plan_sha256"),
                    ("parent_acceptance_path", "parent_acceptance_sha256"),
                    ("capacity_acceptance_path", "capacity_acceptance_sha256")):
                sentinel.add_file(plan[key], plan[digest_key], always_hash=True)
            for directory, filename, digest_key in ((plan["model_snapshot"], "snapshot-manifest.json", "model_manifest_sha256"),
                    (plan["audit_dir"], "manifest.json", "audit_manifest_sha256")):
                sentinel.add_file(Path(directory) / filename, plan[digest_key], always_hash=True)
            for directory in sorted({plan["model_snapshot"], plan["tokenizer_dir"], plan["audit_dir"],
                    *(pin["artifact_dir"] for pin in plan["parents"].values()),
                    plan["_inherited_stages"][0]["reference_cache"]["path"]}):
                sentinel.add_tree(directory)
            for record in plan["_inherited_stages"]:
                sentinel.add_file(record["progress_path"], record["progress_sha256"], always_hash=True)
                artifact = record["artifact"]
                sentinel.add_file(Path(artifact["artifact_dir"]) / "manifest.json",
                                  artifact["manifest_sha256"], always_hash=True)
            context["live_identity"] = sentinel
            return sentinel

        def record(self):
            result = super().record()
            result.update(kind="alignlab-direct-v4-receipt-and-current-input-guard",
                direct_execution_manifest=manifest_pin, full_hash_required=["final_inherited_acceptance"],
                historical_payloads_rehashed_at_startup=False,
                current_consumed_inputs_validated_by_frozen_loader=True)
            return result

    class ReceiptIdentity:
        def __init__(self, live):
            self.live = live
            live.check()

        def check(self):
            return self.live.check()

        def record(self):
            return {"kind": "alignlab-direct-v4-existing-acceptance-receipt-binding",
                "direct_execution_manifest": manifest_pin,
                "historical_payload_full_reverification_deferred": True,
                "full_content_reverified_at_startup": False,
                "large_file_content_proven_by_stat": False,
                "final_primary_success_requires_full_inherited_acceptance": True}

    class Writer(frozen.JsonWriter):
        def __call__(self, path, value, *, exclusive=False, terminal=False):
            if (Path(path).name == "campaign.json" and value.get("status") == "primary_alignment_passed"
                    and value.get("primary_alignment_verified") is True and not context["final_verified"]):
                value.update(status="verifying_inherited_prefix", primary_alignment_verified=False,
                    direct_execution_manifest=manifest_pin, inherited_prefix_reverified=False)
                super().__call__(path, value, exclusive=exclusive, terminal=False)
                validate_manifest(manifest_pin, plan_pin)
                started = clock_system.now()
                accepted = frozen.validate_plan(plan_path, expected_plan_sha256)
                if accepted["_inherited_stages"] != context["view"]["_inherited_stages"]:
                    raise ValueError("Final inherited acceptance differs from the startup receipts")
                context["live_identity"].check()
                validate_manifest(manifest_pin, plan_pin)
                evidence = {"kind": "alignlab-direct-v4-final-inherited-acceptance",
                    "status": "passed", "plan": plan_pin, "execution_manifest": manifest_pin,
                    "started_at_epoch": started, "finished_at_epoch": clock_system.now(),
                    "stage_names": [s["name"] for s in accepted["_inherited_stages"]]}
                receipt_path = Path(path).parent / "inherited-prefix-final-acceptance.json"
                super().__call__(receipt_path, evidence, exclusive=True, terminal=True)
                context["final_verified"] = True
                value.update(inherited_prefix_reverified=True,
                    inherited_prefix_final_acceptance={"path": str(receipt_path), "sha256": frozen.sha(receipt_path)})
                if clock_system.now() >= accepted["continuation"]["deadline_epoch"]:
                    value.update(status="budget_exhausted", primary_alignment_verified=False)
                elif (Path(path).parent / "STOP").exists() or Path(str(plan_path) + ".stop").exists():
                    value.update(status="paused", primary_alignment_verified=False)
                else:
                    value.update(status="primary_alignment_passed", primary_alignment_verified=True)
            if Path(path).name == "campaign.json":
                value["direct_execution_manifest"] = manifest_pin
                value["historical_payload_reverification_at_startup"] = False
                value["direct_entry_new_cpu_tests_run"] = False
            return super().__call__(path, value, exclusive=exclusive, terminal=terminal)

    def own_processes(processes, path):
        by_pid = {p["pid"]: p for p in processes}
        current, owned = by_pid.get(frozen.os.getpid()), {}
        script = str(SUPERVISOR_CLI).lower().replace("\\", "/")
        while current and frozen._command(current, script, path):
            command = current["command_line"]
            if (current["pid"] in owned or expected_plan_sha256 not in command
                    or manifest_pin["sha256"] not in command or "--execution-manifest-sha256" not in command):
                raise RuntimeError("Direct-entry supervisor identity or authority differs")
            frozen._stamp(current["created_at_utc"])
            owned[current["pid"]] = current
            current = by_pid.get(current.get("parent_pid"))
        if not owned:
            raise RuntimeError("Exact direct-entry supervisor ancestry required")
        return owned

    def build_request(*args):
        request, _ = frozen._build_request(*args)
        return request, "throughput_direct_v4/run_alignlab_throughput_fresh_v3.py"

    def verify_stage(output, request, known, code):
        from .throughput_fresh_direct_v4 import verify_stage as verify
        return verify(output, request, known, code, manifest_pin)

    def checked_prefix(plan):
        validate_manifest(manifest_pin, plan_pin)
        return frozen._checked_prefix(plan)

    class System:
        def __getattr__(self, name):
            return getattr(clock_system, name)

        def spawn(self, args, stdout, stderr):
            # The scheduler stores this same list in the campaign. Keep the
            # actual command, including the separate authority, fully recorded.
            if Path(args[4]).name != CHILD_CLI.name:
                raise ValueError("Only the exact remaining seed44 entry is allowed")
            args[4] = str(CHILD_CLI)
            args.extend(["--execution-manifest", str(manifest_pin["path"]),
                         "--execution-manifest-sha256", manifest_pin["sha256"]])
            validate_manifest(manifest_pin, plan_pin)
            return clock_system.spawn(args, stdout, stderr)

    namespace = dict(vars(frozen))
    namespace.update(_metadata_plan=metadata, validate_plan=metadata, _LiveIdentity=LiveIdentity,
        _ContentIdentity=ReceiptIdentity, JsonWriter=Writer, _own_processes=own_processes,
        _build_request=build_request, _verify_stage=verify_stage, _checked_prefix=checked_prefix)
    return _clone(frozen.run_alignment, namespace)(plan_path, expected_plan_sha256,
                                                    execute=execute, system=System())
