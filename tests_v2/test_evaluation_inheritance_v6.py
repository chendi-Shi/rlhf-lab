"""Artificial complete saved-output evidence; no model execution or GPU use."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_inheritance_v6 as inheritance
from alignment_lab import evaluation_execution_v5 as execution
from alignment_lab import evaluation_analysis_v5 as analysis
from alignment_lab import evaluation_acceptance_v2 as acceptance
from alignment_lab import evaluation_supervisor_v5 as supervisor
from alignment_lab.evaluation_protocol_v2 import canonical, parse, EvaluationProtocol
from tests_v2 import test_evaluation_execution_v5 as fixture_module


def pin(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def proc(pid, parent=1):
    return {"pid": pid, "parent_pid": parent, "created_at_utc": "2026-10-26T04:00:00.1234560Z",
            "command_line": f"python frozen-fixture-{pid}.py --read-only"}


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        f = self.fixture = fixture_module.ExecutionTests(methodName="test_full_rm_bench_six_answers_per_record")
        f.setUp()
        self.addCleanup(f.tearDown)
        self.root = f.root
        change = patch.object(inheritance, "ROOT", self.root)
        change.start()
        self.addCleanup(change.stop)
        sources = analysis.REQUIRED_SOURCES | set(acceptance.SOURCES) | supervisor.NEW_SOURCES | {"scripts/verify_alignlab_cpu.py"}
        sources = set(sources)
        while len(sources) < 129:
            sources.add(f"fixture_pinned_source_{len(sources)}.py")
        self.assertEqual(len(sources), 129)
        for name in sources:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"# frozen synthetic CPU fixture\n")
        self.sources = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in sources}
        new_source = self.root / inheritance.V6_EXECUTION
        new_source.write_bytes(b"# synthetic V6 routing fixture, never executed\n")
        self.original = self.root / "runs/prior"
        self.output = self.original / "ifeval-base"
        self.output.mkdir(parents=True)
        self.started, self.deadline = 1792895411, 1793673011
        self.known = [proc(100 + n, 100 + n - 1 if n else 1) for n in range(8)]
        self.profile = parse(Path(f.resource_profile["path"]).read_bytes())
        health = self.profile["services"][0]["json_equals"]
        while len(health) < 27:
            health[f"unchanged_fixture_health_{len(health)}"] = "pinned"
        self.health = {**health, "busy": False}
        self.profile_pin = pin(self.root / "inheritance-profile.json", self.profile)
        # Historical resource-chain validation has its own fault-injection suite.
        # All full inference/output/parameter checks below remain real; this
        # synthetic resource-only seam is never available through a CLI.
        change = patch.object(execution, "load_profile", return_value=self.profile)
        change.start()
        self.addCleanup(change.stop)
        self.request = {"schema_version": 1, "kind": execution.KIND, "task": "ifeval", "model_id": "base",
            "output": str(self.output), "deadline_epoch": self.deadline,
            "protocol_path": str(f.protocol_path), "protocol_sha256": f.protocol_sha,
            "roster_path": str(f.roster_path), "roster_sha256": f.roster_sha,
            "source_sha256": self.sources, "resource_profile": self.profile_pin,
            "capacity_manifest": f.capacity_manifest}
        self.request_pin = pin(self.original / "ifeval-base.request.json", self.request)
        self.rows = [{"key": row["key"], "prompt": row["prompt"], "model_id": "base",
            "model_manifest_sha256": f.protocol["model_manifests"]["base"]["sha256"],
            "protocol_sha256": EvaluationProtocol(canonical(f.protocol)).sha256,
            "input_truncated": False, "prompt_token_ids": [10, 20], "prompt_tokens": 2,
            "generated_token_ids": [151645], "termination": "eos", "generation_seed": 0,
            "response": ""} for row in f.ifeval.rows]
        self.progress = {"status": "evaluation_inference_passed", "request": self.request, "pid": 105,
            "started_at_epoch": 1792990000, "finished_at_epoch": 1792990200,
            "deadline_epoch": self.deadline, "completed_records": 541, "inference_complete": True,
            "scoring_complete": False, "research_complete": False, "formal_training": False,
            **f.resource_evidence(1792990100), "attention_execution": fixture_module.attention_fixture("base")}
        fake = fixture_module.FakeModel("base")  # Metadata only; never calls generate, load, or score.
        self.manifest = {"schema_version": 1, "kind": execution.RESULT_KIND, **self.progress,
            "task": "ifeval", "model_id": "base", "protocol": {
                "path": str(f.protocol_path), "sha256": f.protocol_sha,
                "identity_sha256": EvaluationProtocol(canonical(f.protocol)).sha256},
            "accepted_roster": {"path": str(f.roster_path), "sha256": f.roster_sha},
            "model_manifest": f.protocol["model_manifests"]["base"], "dataset_identity": f.ifeval.identity,
            "counts": dict(inheritance.COUNTS), "model_runtime": {**fake.metadata, "cuda": True, "weights_unchanged": True},
            "frozen_audits": [fake.frozen_audit(full=True)] * 2, "processes": self.known[:2] + self.known[4:6],
            "quality_evaluation_complete": False}
        self.save_output()
        authority = pin(self.root / "authority.json", {"kind": "alignlab-independent-quality-evaluation-authorization-v2",
            "started_at_epoch": self.started, "deadline_epoch": self.deadline, "wall_seconds": self.deadline-self.started,
            "automatic_extension": False, "reuses_old_budget": False, "no_new_training": True})
        python = self.root / "fixture-python.exe"
        tokenizer = self.root / "fixture-tokenizer.json"
        for path in (python, tokenizer):
            path.write_bytes(b"opaque fixture; never executed or loaded")
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        decoder = {"python": str(python), "python_sha256": sha(python),
            "script": str(self.root / supervisor.DECODER_ENTRYPOINT),
            "script_sha256": self.sources[supervisor.DECODER_ENTRYPOINT],
            "tokenizer_path": str(tokenizer), "tokenizer_sha256": sha(tokenizer), "tokenizers_version": "fixture"}
        self.prior = {"schema_version": 1, "kind": supervisor.KIND, "output": str(self.original),
            "authorization": authority, "protocol": {"path": str(f.protocol_path), "sha256": f.protocol_sha},
            "roster": {"path": str(f.roster_path), "sha256": f.roster_sha}, "decoder": decoder,
            "source_sha256": self.sources, "resource_profile": self.profile_pin,
            "started_at_epoch": self.started, "deadline_epoch": self.deadline, "wall_seconds": self.deadline-self.started,
            "model_python_sha256": sha(python), "scoring_python_sha256": sha(python)}
        self.prior_pin = pin(self.root / "prior-plan.json", self.prior)
        self.plan = copy.deepcopy(self.prior)
        self.plan["output"] = str(self.root / "runs/fresh-v6")
        self.plan["source_sha256"][inheritance.V6_EXECUTION] = sha(new_source)
        self.score_request = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-request-v1",
            "inference_manifest": self.manifest_pin["path"], "inference_manifest_sha256": self.manifest_pin["sha256"],
            "source_sha256": self.sources, "decoder": decoder, "deadline_epoch": self.deadline}
        score_pin = pin(self.original / "ifeval-base-score.request.json", self.score_request)
        children = [
            {"name": "capacity", "phase": "capacity", "status": "accepted", "exit_code": 0,
             "active_processes": [], "known_processes": self.known[2:4], "result": f.capacity_manifest},
            {"name": "ifeval-base", "phase": "inference", "status": "accepted", "exit_code": 0,
             "active_processes": [], "known_processes": self.known[4:6], "request": self.request,
             "request_path": self.request_pin["path"], "request_sha256": self.request_pin["sha256"],
             "output": str(self.output), "result": self.manifest_pin},
            {"name": "ifeval-base-score", "phase": "score", "status": "dispatching", "exit_code": 1,
             "active_processes": [], "known_processes": self.known[6:8], "request": self.score_request,
             "request_path": score_pin["path"], "request_sha256": score_pin["sha256"]}]
        self.campaign = {"kind": supervisor.KIND, "status": "failed", "capacity_verified": True,
            "plan_path": self.prior_pin["path"], "plan_sha256": self.prior_pin["sha256"],
            "started_at_epoch": self.started, "deadline_epoch": self.deadline,
            "primary_evaluation_complete": False, "scores": {}, "children": children,
            "supervisor_processes": self.known[:2]}
        self.preserved = self.profile["background_processes"] + self.profile["services"][1]["processes"]
        self.restored = proc(200, 108)
        self.restoration = {"kind": "alignlab-evidoc-restoration-complete-v1", "restoration_complete": True,
            "restore_pending": False, "all_28_fields_match": True, "deadline_epoch": self.deadline,
            "pause_authorization_sha256": self.profile["authorized_pause"]["authorization"]["sha256"],
            "health": self.health, "processes": [self.restored], "preserved_processes": self.preserved}
        restore_pin = pin(self.root / "restoration.json", self.restoration)
        helper = proc(109)
        monitor_pin = pin(self.root / "monitor.json", {"plan": self.prior_pin,
            "guardian": {"guardian": proc(108)}, "awake": {"helper_pid": 109,
                "helper_parent_pid": helper["parent_pid"], "helper_created_utc": helper["created_at_utc"],
                "helper_command": helper["command_line"]}})
        self.closure = {"kind": "alignlab-quality-v5-failure-closure-v1", "status": "failed_closed_services_restored",
            "campaign_status": "failed", "failure_phase": "ifeval-base-score", "failure_exit_code": 1,
            "formal_generations_completed": 541, "formal_generations_expected": 4328,
            "base_inference_accepted": True, "base_inference_exit_code": 0, "official_scoring_complete": False,
            "primary_evaluation_complete": False, "research_complete": False, "capacity_verified": True,
            "capacity_models_completed": 9, "all_evaluation_processes_exited": True, "guardian_status": "complete",
            "awake_status": "released", "restoration_complete": True, "restore_pending": False,
            "health_field_count": 28, "all_28_fields_match": True, "preserved_services_verified": True,
            "relaunch_permitted": False, "recovery_requires_fresh_launch_and_output": True,
            "plan": self.prior_pin, "original_deadline_epoch": self.deadline, "unknown_python": [],
            "restoration": restore_pin, "health": self.health,
            "health_checks": [{"name": k, "matches": True} for k in self.health],
            "restored_processes": [self.restored], "restored_listener_owners": 200,
            "preserved_processes": self.preserved, "latest_monitor": monitor_pin,
            "evaluation_processes": [{"pid": p["pid"], "present": False} for p in self.known],
            "helper_processes": [{"pid": n, "present": False} for n in (108, 109)]}
        self.binding = {"prior_plan": self.prior_pin, "request": self.request_pin, "manifest": self.manifest_pin}
        self.save_state()
        self.live = []
        self.system = SimpleNamespace(processes=lambda: copy.deepcopy(self.live))

    def save_output(self):
        f = self.fixture
        payloads = {"request.snapshot.json": canonical(self.request), "protocol.snapshot.json": canonical(f.protocol),
            "roster.snapshot.json": canonical(f.roster), "progress.json": canonical(self.progress),
            "outputs.jsonl": b"\n".join(canonical(row) for row in self.rows) + b"\n"}
        for name, raw in payloads.items():
            (self.output / name).write_bytes(raw)
        self.manifest["files"] = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in payloads.items()}
        self.manifest_pin = pin(self.output / "manifest.json", self.manifest)

    def save_state(self):
        self.binding["campaign"] = pin(self.original / "campaign.json", self.campaign)
        self.binding["closure"] = pin(self.root / "closure.json", self.closure)

    def verify(self):
        return inheritance.verify_inherited_inference(self.binding, self.plan, system=self.system)

    def repin_output(self):
        self.save_output()
        self.binding["manifest"] = self.manifest_pin
        self.campaign["children"][1]["result"] = self.manifest_pin
        self.score_request["inference_manifest_sha256"] = self.manifest_pin["sha256"]
        score_pin = pin(self.original / "ifeval-base-score.request.json", self.score_request)
        self.campaign["children"][2].update(request_sha256=score_pin["sha256"])
        self.save_state()

    def test_full_541_reverified_without_writes_or_scoring_claim(self):
        before = {p: p.read_bytes() for p in self.original.rglob("*") if p.is_file()}
        result = self.verify()
        self.assertEqual(result["slot"], "ifeval/base")
        self.assertEqual(result["verification"]["counts"], inheritance.COUNTS)
        self.assertEqual(result["verification"]["exact_old_processes_exited"], 10)
        self.assertFalse(result["verification"]["independent_decoding_performed"])
        self.assertTrue(result["verification"]["requires_independent_decoder_at_scoring"])
        self.assertFalse(Path(self.plan["output"]).exists())
        self.assertEqual(before, {p: p.read_bytes() for p in self.original.rglob("*") if p.is_file()})

    def test_tampered_bytes_and_rehashed_truncated_output_fail(self):
        path = self.output / "outputs.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.verify()
        self.rows.pop()
        self.repin_output()
        with self.assertRaises(ValueError):
            self.verify()

    def test_parameter_audit_change_and_inconsistent_saved_text_fail(self):
        self.manifest["frozen_audits"][1] = {**self.manifest["frozen_audits"][1], "base_sha256": "b" * 64}
        self.repin_output()
        with self.assertRaisesRegex(ValueError, "frozen CUDA"):
            self.verify()
        self.manifest["frozen_audits"][1] = copy.deepcopy(self.manifest["frozen_audits"][0])
        self.rows[0]["response"] = "inconsistent same actions"
        self.repin_output()
        with self.assertRaisesRegex(ValueError, "inconsistent text"):
            self.verify()

    def test_budget_protocol_decoder_and_roster_replacements_fail(self):
        for key, value in (("deadline_epoch", self.deadline + 1), ("started_at_epoch", self.started + 1),
                           ("authorization", {}), ("protocol", {}), ("roster", {}), ("decoder", {})):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.plan)
                bad[key] = value
                with self.assertRaisesRegex(ValueError, "identity, decoder or original budget"):
                    inheritance.verify_inherited_inference(self.binding, bad, system=self.system)

    def test_original_source_changed_or_pin_removed_fails(self):
        name = "alignment_lab/evaluation_execution_v5.py"
        original = (self.root / name).read_bytes()
        (self.root / name).write_bytes(original + b"# changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.verify()
        (self.root / name).write_bytes(original)
        self.plan["source_sha256"].pop(name)
        with self.assertRaisesRegex(ValueError, "every unchanged"):
            self.verify()

    def test_unaccepted_nonzero_inference_or_other_failure_slot_fails(self):
        accepted = self.campaign["children"][1]
        for key, value in (("status", "dispatching"), ("exit_code", 1)):
            before = accepted[key]
            accepted[key] = value
            self.save_state()
            with self.assertRaisesRegex(ValueError, "accepted exit-zero"):
                self.verify()
            accepted[key] = before
        self.closure["failure_phase"] = "ifeval-base"
        self.save_state()
        with self.assertRaisesRegex(ValueError, "closed V5 score failure"):
            self.verify()

    def test_mixed_request_model_and_old_output_reuse_fail(self):
        self.campaign["children"][1]["request"] = {**self.request, "model_id": "sft"}
        self.save_state()
        with self.assertRaisesRegex(ValueError, "original request"):
            self.verify()
        self.campaign["children"][1]["request"] = self.request
        self.save_state()
        self.plan["output"] = self.prior["output"]
        with self.assertRaisesRegex(ValueError, "distinct new campaign"):
            self.verify()

    def test_old_process_alive_blocks_but_unrelated_pid_reuse_is_allowed(self):
        self.live = [copy.deepcopy(self.known[5])]
        with self.assertRaisesRegex(ValueError, "old process is still alive"):
            self.verify()
        self.live[0].update(created_at_utc="2026-10-26T08:40:00.1234567Z", parent_pid=999,
                            command_line=None)
        self.assertEqual(self.verify()["verification"]["exact_old_processes_exited"], 10)
        self.live = [proc(108)]
        with self.assertRaisesRegex(ValueError, "old process is still alive"):
            self.verify()

    def test_submicrosecond_identity_and_inconsistent_ticks(self):
        prior = proc(500)
        now = {**prior, "created_at_utc": "2026-10-26T04:00:00.1234561Z"}
        self.assertNotEqual(inheritance._identity(prior), inheritance._identity(now))
        with self.assertRaisesRegex(ValueError, "ticks"):
            inheritance._identity({**prior, "created_utc_ticks": 1})

    def test_restore_health_or_absent_helper_receipt_fails(self):
        self.closure["health_checks"][0]["matches"] = False
        self.save_state()
        with self.assertRaisesRegex(ValueError, "28 original health"):
            self.verify()
        self.closure["health_checks"][0]["matches"] = True
        self.closure["helper_processes"].pop()
        self.save_state()
        with self.assertRaisesRegex(ValueError, "every old process"):
            self.verify()

    def test_version_router_runs_real_v5_and_rejects_older_or_other_models(self):
        result = inheritance.verify_versioned_evaluation_result(self.output,
            expected_manifest_sha256=self.manifest_pin["sha256"], expected_request=self.request,
            decode_tokens=lambda ids: "")
        self.assertEqual(result["counts"], inheritance.COUNTS)
        self.manifest["request"] = {**self.request, "model_id": "sft"}
        changed = pin(self.output / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "Only original V5"):
            inheritance.verify_versioned_evaluation_result(self.output, expected_manifest_sha256=changed["sha256"])
        self.manifest["request"] = {**self.request, "source_sha256": {"alignment_lab/evaluation_execution_v2.py": "a" * 64}}
        changed = pin(self.output / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "129-file V5"):
            inheritance.verify_versioned_evaluation_result(self.output, expected_manifest_sha256=changed["sha256"])

    def test_version_router_uses_explicit_v6_validator_and_rejects_bad_pin(self):
        value = copy.deepcopy(self.manifest)
        value["request"]["source_sha256"] = self.plan["source_sha256"]
        changed = pin(self.output / "manifest.json", value)
        calls = []
        def verify(*args, **kwargs):
            calls.append((args, kwargs))
            return {"called": "V6 full verifier fixture"}
        with patch.dict(sys.modules, {"alignment_lab.evaluation_execution_v6": SimpleNamespace(verify_evaluation_result=verify)}):
            inheritance.verify_versioned_evaluation_result(self.output, expected_manifest_sha256=changed["sha256"])
        self.assertEqual(len(calls), 1)
        (self.root / inheritance.V6_EXECUTION).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            inheritance.verify_versioned_evaluation_result(self.output, expected_manifest_sha256=changed["sha256"])


if __name__ == "__main__":
    unittest.main()
