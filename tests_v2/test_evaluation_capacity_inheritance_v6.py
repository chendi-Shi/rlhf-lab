"""CPU-only evidence/AST fault injection; no model, CUDA or historical output is run."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_capacity_inheritance_v6 as inherit
from alignment_lab.evaluation_protocol_v2 import canonical


def pin(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def process(pid, parent, args):
    command = " ".join('"' + arg + '"' if " " in arg else arg for arg in args)
    return {"pid": pid, "parent_pid": parent,
            "created_at_utc": "2026-10-26T04:05:01.7426380Z", "command_line": command}


def fixture(root):
    """Tiny CPU receipts; the original full V5 verifier is explicitly mocked in these tests."""
    old_output, output = root / "runs/old", root / "runs/new"
    protocol = pin(root / "protocol.json", {"frozen": True})
    authorization = pin(root / "authority.json", {"deadline_epoch": inherit.DEADLINE})
    baseline = pin(root / "old-resource.json", {"kind": "fixture-old-profile"})
    new_profile = pin(root / "new-resource.json", {"kind": "fixture-new-profile"})
    model_python = root / ".runtime/alignlab-v2/Scripts/python.exe"
    base_python = root / "base-python.exe"
    model_python.parent.mkdir(parents=True)
    model_python.write_bytes(b"CPU-only interpreter pin fixture; never execute")
    base_python.write_bytes(b"CPU-only base interpreter pin fixture; never execute")
    model_sha = hashlib.sha256(model_python.read_bytes()).hexdigest()
    previous_restore = pin(root / "restoration-manifest.json", {"evaluation_interpreters": [
        {"role": "model", "path": str(model_python), "sha256": model_sha},
        {"role": "base", "path": str(base_python), "sha256": hashlib.sha256(base_python.read_bytes()).hexdigest()}]})
    restoration = pin(root / "restoration.json", {"restoration_complete": True})
    profile = {"authorized_pause": {"baseline_profile": baseline}, "allocator_limit_mib": 2560,
               "service_identity_succession": {"restoration_report": restoration,
                                               "restoration_manifest": previous_restore}}
    prior = {"kind": "alignlab-primary-evaluation-campaign-v2", "schema_version": 1,
        "started_at_epoch": 1, "deadline_epoch": inherit.DEADLINE, "wall_seconds": inherit.DEADLINE - 1,
        "poll_seconds": 10, "roster": pin(root / "roster.json", {"frozen_roster": True}),
        "decoder": {"fixture": "same-decoder"}, "model_python_sha256": model_sha,
        "scoring_python_sha256": "a" * 64, "output": str(old_output), "protocol": protocol,
        "authorization": authorization, "resource_profile": baseline, "source_sha256": {"old.py": "b" * 64}}
    prior_binding = pin(root / "old-plan.json", prior)
    request = {"output": str(old_output / "capacity"), "evaluation_plan_path": prior_binding["path"],
        "evaluation_plan_sha256": prior_binding["sha256"], "resource_profile": baseline}
    request_binding = pin(old_output / "capacity.request.json", request)
    args = [str(model_python), "-I", "-B", "-u", "scripts/run_alignlab_evaluation_capacity_v5.py",
            "--request", request_binding["path"], "--request-sha256", request_binding["sha256"], "--execute"]
    supervisor_args = [str(model_python), "-I", "-B", "-u", "scripts/supervise_alignlab_evaluation_v5.py",
                       "--plan", prior_binding["path"], "--plan-sha256", prior_binding["sha256"], "--execute"]
    supervisors = [process(10, 9, supervisor_args), process(11, 10, [str(base_python), *supervisor_args[1:]])]
    known = [process(12, 11, args), process(13, 12, [str(base_python), *args[1:]])]
    manifest = {"request": request_binding, "process_identity": known + supervisors,
        "started_at_epoch": 2, "completed_at_epoch": 3,
        "models": {name: {"frozen_audits": [{"unchanged": True}] * 2,
                           "gpu_memory": {"max_reserved_bytes": 100}} for name in inherit.MODEL_IDS}}
    manifest_binding = pin(old_output / "capacity/manifest.json", manifest)
    child = {"name": "capacity", "phase": "capacity", "status": "accepted", "exit_code": 0,
        "monitoring_error": None, "stop_reason": None, "active_processes": [], "result": manifest_binding,
        "request": request, "request_path": request_binding["path"], "request_sha256": request_binding["sha256"],
        "output": request["output"], "known_processes": known, "wrapper_pid": 12, "command": args,
        "launch_epoch": 1.5, "finished_at_epoch": 4}
    campaign = {"kind": prior["kind"], "status": "failed", "capacity_verified": True,
        "plan_path": prior_binding["path"], "plan_sha256": prior_binding["sha256"],
        "deadline_epoch": inherit.DEADLINE, "primary_evaluation_complete": False,
        "research_complete": False, "formal_training": False, "children": [child],
        "supervisor_processes": supervisors, "capacity": manifest, "finished_at_epoch": 5}
    closure = {"kind": "alignlab-quality-v5-failure-closure-v1", "status": "failed_closed_services_restored",
        "plan": prior_binding, "campaign_status": "failed", "failure_phase": "ifeval-base-score",
        "failure_exit_code": 1, "capacity_verified": True, "capacity_models_completed": 9,
        "original_deadline_epoch": inherit.DEADLINE, "all_evaluation_processes_exited": True,
        "restoration_complete": True, "restore_pending": False, "all_28_fields_match": True,
        "preserved_services_verified": True, "unknown_python": [], "relaunch_permitted": False,
        "recovery_requires_fresh_launch_and_output": True, "restoration": restoration,
        "evaluation_processes": [{"pid": p["pid"], "present": False} for p in known + supervisors],
        "helper_processes": [{"pid": 99, "present": False}]}
    bindings = {"prior_plan": prior_binding, "campaign": pin(old_output / "campaign.json", campaign),
                "closure": pin(root / "closure.json", closure), "request": request_binding, "manifest": manifest_binding}
    plan = {**prior, "output": str(output), "resource_profile": new_profile,
            "source_sha256": {**prior["source_sha256"], "new.py": "c" * 64}, "inherited_capacity": bindings}
    return plan, bindings, profile, manifest


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan, self.binding, self.profile, self.result = fixture(self.root)
        for target, value in (("ROOT", self.root), ("PRIOR_PLAN_SHA256", self.binding["prior_plan"]["sha256"])):
            patcher = patch.object(inherit, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, value in (("_sources", None), ("load_profile", self.profile)):
            patcher = patch.object(inherit, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        verifier = patch("alignment_lab.evaluation_capacity_v5.verify_evaluation_capacity", return_value=self.result)
        self.old_verify = verifier.start()
        self.addCleanup(verifier.stop)

    def change(self, key, mutate):
        value = inherit._read(self.binding[key])
        mutate(value)
        self.binding[key] = pin(Path(self.binding[key]["path"]), value)
        self.plan["inherited_capacity"] = self.binding

    def verify(self):
        return inherit.verify_inherited_capacity(self.binding, self.plan)

    def test_full_original_verifier_receives_original_request_and_shared_deadline(self):
        result = self.verify()
        self.assertTrue(result["inheritance_verified"])
        self.assertFalse(result["current_live_admission_performed"])
        self.assertEqual(result["max_reserved_bytes"], dict.fromkeys(inherit.MODEL_IDS, 100))
        self.old_verify.assert_called_once_with(Path(self.binding["manifest"]["path"]).parent,
            expected_manifest_sha256=self.binding["manifest"]["sha256"],
            expected_request=inherit._read(self.binding["request"]), deadline_epoch=inherit.DEADLINE)

    def test_full_original_verifier_failure_is_never_replaced_with_campaign_claim(self):
        self.old_verify.side_effect = ValueError("Original tokenizer or audit rejected")
        with self.assertRaisesRegex(ValueError, "Original tokenizer"):
            self.verify()

    def test_failed_or_unobserved_capacity_child_is_rejected(self):
        original = inherit._read(self.binding["campaign"])
        for key, value in (("exit_code", 1), ("status", "dispatching"), ("monitoring_error", "fault"),
                           ("stop_reason", "deadline"), ("active_processes", [{"pid": 13}])):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed["children"][0][key] = value
                self.binding["campaign"] = pin(Path(self.binding["campaign"]["path"]), changed)
                with self.assertRaisesRegex(ValueError, "accepted, exited"):
                    self.verify()
        self.old_verify.assert_not_called()

    def test_pid_only_reuse_or_wrong_frozen_command_is_rejected(self):
        self.change("campaign", lambda c: c["children"][0]["known_processes"][0].update(command_line="python other.py"))
        with self.assertRaisesRegex(ValueError, "identities"):
            self.verify()

    def test_missing_exit_or_wrong_restoration_closure_is_rejected(self):
        self.change("closure", lambda c: c["evaluation_processes"].pop())
        with self.assertRaisesRegex(ValueError, "Safe closure"):
            self.verify()

    def test_current_profile_must_succeed_exact_old_profile(self):
        self.profile["authorized_pause"]["baseline_profile"] = {"path": str(self.root / "foreign"), "sha256": "f" * 64}
        with self.assertRaisesRegex(ValueError, "prior campaign"):
            self.verify()

    def test_changed_protocol_budget_or_reused_output_rejected(self):
        original = copy.deepcopy(self.plan)
        changes = (lambda p: p.update(deadline_epoch=inherit.DEADLINE + 1),
                   lambda p: p.update(output=inherit._read(self.binding["prior_plan"])["output"]),
                   lambda p: p["protocol"].update(sha256="f" * 64),
                   lambda p: p["decoder"].update(fixture="different-tokenizer"))
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.plan = copy.deepcopy(original)
                change(self.plan)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_evidence_tampering_is_detected_before_verification(self):
        path = Path(self.binding["manifest"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.verify()
        self.old_verify.assert_not_called()

    def test_request_bridge_accepts_only_exact_current_slot(self):
        plan_binding = pin(self.root / "new-plan.json", self.plan)
        bridge = pin(Path(self.plan["output"]) / "capacity-inheritance.json", {
            "kind": inherit.KIND, "plan": plan_binding, "inherited_capacity": self.binding})
        request = {"schema_version": 1, "kind": "alignlab-evaluation-inference-stage-v2",
            "capacity_manifest": bridge, "task": "ifeval", "model_id": "sft",
            "output": str(Path(self.plan["output"]) / "ifeval-sft"),
            **{key: self.plan[key] for key in ("source_sha256", "resource_profile", "deadline_epoch")},
            **{key + suffix: self.plan[key][field] for key in ("protocol", "roster")
               for suffix, field in (("_path", "path"), ("_sha256", "sha256"))}}
        self.assertEqual(inherit.verify_capacity_inheritance_for_request(bridge, request), 100)
        for key, value in (("model_id", "reward"), ("output", str(self.root / "elsewhere")),
                           ("deadline_epoch", inherit.DEADLINE + 1), ("source_sha256", {}),
                           ("protocol_path", str(self.root / "different.json"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                inherit.verify_capacity_inheritance_for_request(bridge, {**request, key: value})


class ExecutionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.old = (root / "alignment_lab/evaluation_execution_v5.py").read_text(encoding="utf-8-sig")
        self.new = (root / "alignment_lab/evaluation_execution_v6.py").read_text(encoding="utf-8-sig")

    def test_actual_new_gpu_executor_has_only_the_explicit_bridge_changes(self):
        inherit.verify_execution_ast(self.old, self.new, execution=True)

    def test_changed_dtype_horizon_model_loader_or_execution_is_rejected(self):
        changes = (
            ('"compute_dtype": "bfloat16"', '"compute_dtype": "float16"'),
            ('max_new_tokens=2048', 'max_new_tokens=1024'),
            ('evaluation_model_v5 import', 'evaluation_model_v6 import'),
            ('    peak = _capacity_peak(request)', '    peak = 1'),
            ('    context = _context(request)', '    context = None'),
        )
        # Any missing anchor is a test error, not a silently skipped mutation.
        for before, after in changes:
            with self.subTest(before=before):
                self.assertIn(before, self.new)
                with self.assertRaises(ValueError):
                    inherit.verify_execution_ast(self.old, self.new.replace(before, after, 1), execution=True)

    def test_capacity_bridge_cannot_be_replaced_by_a_constant_or_unconditional_bypass(self):
        for replacement in ("return 1", "return verify_capacity_inheritance_for_request(binding, {**request, 'model_id': 'base'})"):
            changed = self.new.replace("return verify_capacity_inheritance_for_request(binding, request)", replacement)
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "exact inherited-capacity"):
                inherit.verify_execution_ast(self.old, changed, execution=True)

    def test_extra_executable_import_or_top_level_side_effect_is_rejected(self):
        with self.assertRaises(ValueError):
            inherit.verify_execution_ast(self.old, self.new + "\nimport torch\n", execution=True)

    def test_model_and_attention_names_are_never_normalized_to_old_versions(self):
        self.assertEqual(inherit._version("evaluation_model_v6 evaluation_attention_v6"),
                         "evaluation_model_v6 evaluation_attention_v6")

    def inventory(self, root):
        project = Path(__file__).resolve().parents[1]
        prior, current = {}, {}
        stems = ("alignment_lab/evaluation_execution", "alignment_lab/evaluation_capacity",
                 "scripts/run_alignlab_evaluation_stage", "scripts/run_alignlab_evaluation_capacity")
        names = [stem + suffix for stem in stems for suffix in ("_v5.py", "_v6.py")]
        names += sorted(inherit.EXTRA_EXECUTION_SOURCES)
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((project / name).read_bytes())
            current[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name.endswith("_v5.py"):
                prior[name] = current[name]
        for index in range(125):
            name = f"fixture_old_source_{index}.py"
            path = root / name
            path.write_bytes(b"FROZEN = True\n")
            prior[name] = current[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return prior, current

    def test_all_129_source_pins_and_complete_gpu_closure_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, current = self.inventory(root)
            with patch.object(inherit, "ROOT", root):
                inherit._sources(prior, current)
                altered = {**current, "fixture_old_source_0.py": "f" * 64}
                with self.assertRaisesRegex(ValueError, "129 historical"):
                    inherit._sources(prior, altered)
                (root / "fixture_old_source_1.py").write_bytes(b"FROZEN = False\n")
                with self.assertRaises(ValueError):
                    inherit._sources(prior, current)

    def test_repinning_modified_new_gpu_behavior_does_not_make_capacity_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, current = self.inventory(root)
            name = "alignment_lab/evaluation_execution_v6.py"
            changed = self.new.replace('max_new_tokens=2048', 'max_new_tokens=1024', 1)
            (root / name).write_text(changed, encoding="utf-8")
            current[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            with patch.object(inherit, "ROOT", root), self.assertRaisesRegex(ValueError, "GPU execution semantics"):
                inherit._sources(prior, current)


if __name__ == "__main__":
    unittest.main()
