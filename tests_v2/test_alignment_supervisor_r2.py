"""CPU-only continuation evidence and mocked child lifecycle tests; no GPU claims."""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import alignment_supervisor_r2 as ss
from alignment_lab import reference_cache as storage
from tests_v2.test_study_supervisor import FakeSystem as OldFakeSystem, write, stamp


def pin(path):
    return {"path": str(path), "sha256": ss.sha(path)}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.oldout = self.root / "runs/old"
        (self.oldout / "dpo42").mkdir(parents=True)
        self.planpath = self.root / "old-plan.json"
        self.oldplan = {"kind": ss.previous.KIND, "output": str(self.oldout), "source_sha256": {"old.py": "a" * 64},
            **{k: k for k in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
            "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path",
            "parent_plan_sha256", "parent_acceptance_path", "parent_acceptance_sha256")},
            "parents": {"sft": {}}, "ppo_rows": 1024, "wall_seconds": 3000, "poll_seconds": 10}
        write(self.planpath, self.oldplan)
        self.request = {"kind": ss.previous.STAGE_KIND, "stage": "dpo", "seed": 42,
            "alignment_plan_path": str(self.planpath), "alignment_plan_sha256": ss.sha(self.planpath),
            "deadline_epoch": 3100, "workload": ss.workload("dpo", 1024)}
        write(self.oldout / "dpo42.request.json", self.request)
        write(self.oldout / "dpo42/request.json", self.request)
        self.progress = {"status": "paused", "stage": "dpo", "actual_optimizer_updates": 0,
            "latest_safe_checkpoint": None, "request": self.request,
            "process_guard": {"total_waited_seconds": 300.0664498806}}
        write(self.oldout / "dpo42/progress.json", self.progress)
        self.campaign = {"kind": ss.previous.KIND, "status": "failed", "started_at_epoch": 100,
            "deadline_epoch": 3100, "plan_sha256": ss.sha(self.planpath),
            "supervisor_processes": [{"pid": 771, "parent_pid": 1, "created_at_utc": stamp(1), "command_line": "python old.py"}],
            "stages": [{"name": "dpo42", "request": self.request, "exit_code": 1, "active_processes": []}]}
        write(self.oldout / "campaign.json", self.campaign)
        auth = {"kind": "explicit-user-primary-alignment-continuation-authorization-v1",
            "source": "direct_user_reply_in_this_chat", "allow_continuous_training": True,
            "origin_plan_sha256": ss.sha(self.planpath), "origin_campaign_sha256": ss.sha(self.oldout / "campaign.json"),
            "original_started_at_epoch": 100, "original_deadline_epoch": 3100}
        write(self.root / "auth.json", auth)
        for name in ("journal.jsonl", "boundary.json"):
            write(self.oldout / "dpo42" / name, {})
        prefix = {"request": pin(self.oldout / "dpo42/request.json"),
            "progress": pin(self.oldout / "dpo42/progress.json"),
            "journal": pin(self.oldout / "dpo42/journal.jsonl"), "boundary_receipt": pin(self.oldout / "dpo42/boundary.json"),
            "scored_sequences": 2, "expected_unique_sequences": 3}
        self.plan = {**self.oldplan, "kind": ss.KIND, "output": str(self.root / "runs/new"),
            "source_sha256": {**self.oldplan["source_sha256"], "new.py": "b" * 64},
            "continuation": {"prior_plan": pin(self.planpath), "prior_campaign": pin(self.oldout / "campaign.json"),
                "prior_stage_request": pin(self.oldout / "dpo42.request.json"),
                "prior_stage_progress": pin(self.oldout / "dpo42/progress.json"), "authorization": pin(self.root / "auth.json"),
                "started_at_epoch": 100, "deadline_epoch": 3100, "process_wait_spent_seconds": 300.0664498806,
                "cache_prefix": prefix}}
        self.rootpatch = patch.object(ss, "ROOT", self.root)
        self.rootpatch.start()
        self.addCleanup(self.rootpatch.stop)

    def validate(self):
        # Prefix mathematical and hash-chain audit has its separate real-file suite.
        with patch("alignment_lab.reference_cache_recovery.audit_prefix") as audit:
            result = ss.validate_continuation(self.plan)
            audit.assert_called_once_with(self.plan["continuation"]["cache_prefix"])
            return result

    def test_separate_external_and_internal_request_pins_are_valid(self):
        self.assertEqual(self.validate()["origin_campaign"]["status"], "failed")

    def test_budget_wait_and_work_cannot_change(self):
        for change in (lambda p: p["continuation"].update(deadline_epoch=3200),
                       lambda p: p["continuation"].update(process_wait_spent_seconds=0),
                       lambda p: p.update(ppo_rows=4096),
                       lambda p: p["source_sha256"].update({"old.py": "c" * 64})):
            original = copy.deepcopy(self.plan)
            change(self.plan)
            with self.assertRaises(ValueError):
                self.validate()
            self.plan = original

    def test_auth_hash_drift_rejected(self):
        (self.root / "auth.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.validate()

    def test_internal_request_mismatch_rejected_even_when_rehashed(self):
        write(self.oldout / "dpo42/request.json", {**self.request, "seed": 43})
        self.plan["continuation"]["cache_prefix"]["request"] = pin(self.oldout / "dpo42/request.json")
        with self.assertRaises(ValueError):
            self.validate()

    def test_live_historical_instance_blocks_but_pid_reuse_does_not(self):
        old = self.campaign["supervisor_processes"][0]
        with self.assertRaises(RuntimeError):
            ss._prior_processes_gone(self.plan, [old])
        with self.assertRaises(RuntimeError):
            ss._prior_processes_gone(self.plan, [{**old, "command_line": ""}])
        ss._prior_processes_gone(self.plan, [{**old, "created_at_utc": stamp(500)}])


class FakeSystem(OldFakeSystem):
    def processes(self):
        rows = super().processes()
        for row in rows:
            if row["pid"] in (os.getpid(), 91):
                row["command_line"] = row["command_line"].replace("run_alignlab_parents.py", "run_alignlab_alignment_r2.py")
        if getattr(self, "snapshot_failure_until", 0) > self.clock and self.child:
            raise RuntimeError("Fixture CIM unavailable")
        return rows


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / "reports").mkdir()
        (self.root / "runs").mkdir()
        self.path = self.root / "plan.json"
        self.plan = {"output": str(self.root / "runs/new"), "ppo_rows": 1024, "wall_seconds": 3000, "poll_seconds": 10,
            "parents": {"sft": {}, "reward": {}}, "source_sha256": {}, "model_snapshot": "fixture", "prerequisites": [],
            "allowed_background_processes": [], "process_wait_policy": ss.process_guard.POLICY,
            "continuation": {"started_at_epoch": 100, "deadline_epoch": 3100, "process_wait_spent_seconds": 300.0664498806},
            **{k: "fixture" for k in ("model_manifest_sha256", "tokenizer_dir", "audit_dir", "audit_manifest_sha256",
             "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path", "parent_plan_sha256",
             "parent_acceptance_path", "parent_acceptance_sha256")}}
        write(self.path, self.plan)
        self.system = FakeSystem(self)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("ROOT", self.root), ("validate_plan", lambda *_: self.plan),
            ("_parents_gone", lambda *_: None), ("_prior_processes_gone", lambda *_: None),
            ("_cache", lambda *_: None), ("_verify_stage", self.accept_fixture),
            ("verify_snapshot", lambda *_: {"repo_id": ss.REPO_ID, "revision": ss.REVISION})):
            self.stack.enter_context(patch.object(ss, name, value))
        self.stack.enter_context(patch.object(ss._LiveIdentity, "from_plan", return_value=SimpleNamespace(
            check=lambda: None, add_file=lambda *a, **k: None, add_tree=lambda *a: None, record=lambda: {})))

    def publish(self, request, pid):
        output = Path(request["output"])
        output.mkdir()
        write(output / "progress.json", {"fixture": True})

    def accept_fixture(self, output, request, known, code):
        # Orchestration tests never substitute this fixture for real stage acceptance.
        if code:
            raise RuntimeError("Fixture child failed")
        cache = request["reference_cache"]
        return {"artifact": {}, "process_guard": {"total_waited_seconds": request["process_wait_spent_seconds"] + 200},
            "reference_cache": None if cache is None else {"path": cache["path"],
                "manifest_sha256": "a" * 64, "identity_sha256": "b" * 64}}

    def run_campaign(self):
        return ss.run_alignment(self.path, ss.sha(self.path), execute=True, system=self.system)

    def test_all_six_share_original_clock_and_cumulative_wait_above_900(self):
        result = self.run_campaign()
        self.assertEqual(result["status"], "primary_alignment_passed")
        self.assertEqual(result["started_at_epoch"], 100)
        self.assertEqual(result["attempt_started_at_epoch"], 1000)
        self.assertEqual(result["deadline_epoch"], 3100)
        self.assertEqual(result["total_process_wait_seconds"], 1500.0664498806)
        requests = [s["request"] for s in result["stages"]]
        self.assertEqual([(r["stage"], r["seed"]) for r in requests], list(ss.SEQUENCE))
        self.assertTrue(all(r["deadline_epoch"] == 3100 and r["continuation"] == self.plan["continuation"] for r in requests))
        self.assertEqual([r["reference_cache"]["mode"] for r in requests if r["stage"] == "dpo"], ["create", "reuse", "reuse"])
        self.assertFalse(result["research_complete"])

    def test_snapshot_uncertainty_over_300_does_not_stop_live_child(self):
        self.system.child_duration = 400
        self.system.snapshot_failure_until = 1350
        self.plan["continuation"]["deadline_epoch"] = 5000
        self.plan["wall_seconds"] = 4900
        # First snapshot proves the wrapper/child; temporary observations fail afterwards.
        original = self.system.processes
        calls = [0]
        def observations():
            if self.system.child:
                calls[0] += 1
                if calls[0] == 1:
                    bound = self.system.snapshot_failure_until
                    self.system.snapshot_failure_until = 0
                    rows = original()
                    self.system.snapshot_failure_until = bound
                    return rows
            return original()
        self.system.processes = observations
        self.assertEqual(self.run_campaign()["status"], "primary_alignment_passed")
        self.assertFalse((Path(self.plan["output"]) / "STOP").exists())

    def test_expired_original_budget_dispatches_nothing(self):
        self.system.clock = 3100
        self.assertEqual(self.run_campaign()["status"], "budget_exhausted")
        self.assertEqual(self.system.commands, [])
        self.assertFalse(Path(self.plan["output"]).exists())

    def test_stop_lets_current_child_exit_without_dispatching_next(self):
        def stop(system):
            target = Path(self.plan["output"]) / "STOP"
            if not target.exists():
                target.write_text("user pause")
        self.system.on_sleep = stop
        self.assertEqual(self.run_campaign()["status"], "paused")
        self.assertEqual(len(self.system.commands), 1)
        self.assertGreaterEqual(self.system.clock, self.system.child_end)

    def test_foreign_lock_never_deleted_or_retried(self):
        lock = self.root / "reports/.alignlab-alignment.lock"
        lock.write_text("foreign")
        with self.assertRaises(FileExistsError):
            self.run_campaign()
        self.assertEqual(lock.read_text(), "foreign")
        self.assertEqual(self.system.commands, [])

    def test_failure_keeps_failed_output_and_does_not_retry(self):
        self.system.child_code = 2
        with self.assertRaises(RuntimeError):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)
        self.assertEqual(json.loads((Path(self.plan["output"]) / "campaign.json").read_text())["status"], "failed")
        with self.assertRaises(FileExistsError):
            self.run_campaign()


class MigrationVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(ss, "ROOT", self.root))
        for name in ("base", "tokenizer", "source", "new-source"):
            (self.root / name).write_text(name)
        manifest = self.root / "data.json"
        write(manifest, {"status": "audit_complete", "data_ready": True, "readiness_blockers": [], "training_started": False})
        config = {"template": "fixture", "serialization": "raw", "precision": "float32", "quantization": "none", "library_versions": {"fixture": "1"}}
        self.old = storage.build_identity(base_files={"b": self.root / "base"}, adapter_files={},
            tokenizer_files={"t": self.root / "tokenizer"}, source_files={"s": self.root / "source"},
            dataset_manifest=manifest, accepted_dataset_manifest_sha256=ss.sha(manifest), scoring_config=config)
        self.new = copy.deepcopy(self.old)
        self.new["files"]["source"]["new"] = storage._file_record(self.root / "new-source")
        self.records = [{"prompt_token_ids": [i], "completion_token_ids": [10, 11], "sum_logprob": -float(i), "token_count": 2} for i in (1, 2, 3)]
        historical = [{**r, "key": storage.cache_key(self.old, r["prompt_token_ids"], r["completion_token_ids"])} for r in self.records]
        historical.sort(key=lambda r: r["key"])
        self.audit = SimpleNamespace(identity=self.old, records=historical[:2], summary={"verified": True})
        self.stack.enter_context(patch("alignment_lab.reference_cache_recovery.audit_prefix", return_value=self.audit))
        self.cache = storage.write_or_reuse_cache(self.root / "cache", self.new, self.records)
        self.spec = {"scored_sequences": 2, "expected_unique_sequences": 3,
            "journal": {"sha256": "a" * 64}, "boundary_receipt": {"sha256": "b" * 64}}
        self.request = {"continuation": {"cache_prefix": self.spec}}
        added = {"new": self.new["files"]["source"]["new"]}
        keys = sorted(storage.cache_key(self.new, r["prompt_token_ids"], r["completion_token_ids"]) for r in self.audit.records)
        self.report = {**self.audit.summary, "kind": "alignlab-reference-cache-prefix-migration-v1",
            "actual_loaded_reference_verified": True, "new_identity_sha256": storage.identity_sha256(self.new),
            "added_sources": added, "new_record_keys_sha256": hashlib.sha256(ss._canonical(keys)).hexdigest(),
            "score_values_unchanged": True, "sequence_tokens_unchanged": True,
            "identity_migration": "strict_source_inventory_superset_with_explicit_rekey", "new_optimizer_updates": 0,
            "formal_training_complete": False, "imported_sequences": 2, "new_sequences_to_score": 1,
            "source_journal_sha256": "a" * 64, "source_boundary_receipt_sha256": "b" * 64,
            "all_original_sources_unchanged": True, "source_inventory_added": added,
            "old_reference_identity": self.old, "new_reference_identity": self.new}
        self.save_report()

    def save_report(self):
        path = self.root / "cache-prefix-migration.json"
        write(path, self.report)
        self.result = {"cache_prefix_recovery_report": pin(path), "reference_cache": {
            "migration_report_path": str(path), "migration_report_sha256": ss.sha(path),
            "imported_sequences": 2, "newly_scored_sequences": 1}}

    def check(self):
        return ss._verify_prefix_migration(self.root, self.request, self.result, self.cache, self.new)

    def test_real_completed_cache_matches_every_carried_score_and_rekey(self):
        self.assertEqual(self.check(), self.result["cache_prefix_recovery_report"])

    def test_rehashed_report_count_or_old_identity_lie_rejected(self):
        self.report["new_sequences_to_score"] = 0
        self.save_report()
        with self.assertRaises(ValueError):
            self.check()

    def test_finite_changed_carried_score_rejected(self):
        self.audit.records[0]["sum_logprob"] -= .25
        with self.assertRaises(ValueError):
            self.check()

    def test_actual_reference_protocol_difference_rejected(self):
        self.new["scoring_config"]["precision"] = "bfloat16"
        with self.assertRaises(ValueError):
            self.check()


class ActualFormatTests(unittest.TestCase):
    def test_receipt_six_mandatory_extra_source_labels_checked_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "alignment_lab").mkdir()
            inputs, cache_sources = {}, {}
            for filename in ("gpu_reference.py", "reference_bridge.py", "shared_qwen.py",
                             "token_logprobs.py", "training_steps.py", "reference_cache.py"):
                path = root / "alignment_lab" / filename
                path.write_text(filename)
                record = storage._file_record(path)
                inputs["source/alignment_lab/" + filename] = record
                cache_sources["gpu_reference/" + filename] = record
            cache_sources.update(inputs)
            identity = {"inputs": inputs}
            cache = {"files": {"source": cache_sources, "base": {}, "tokenizer": {}}}
            with patch.object(ss, "ROOT", root):
                ss._verify_cache_files(identity, cache)
                cache_sources["untracked"] = next(iter(inputs.values()))
                with self.assertRaises(ValueError):
                    ss._verify_cache_files(identity, cache)
                del cache_sources["untracked"]
                del cache_sources["gpu_reference/reference_cache.py"]
                with self.assertRaises(ValueError):
                    ss._verify_cache_files(identity, cache)

    def test_live_and_returned_objectives_require_exact_same_numerics_and_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "telemetry").mkdir()
            objectives = [{"update": 1, "objective": 1.25, "gradient_norm": .5, "rows": 16}]
            event = {"event": "optimizer_end", **objectives[0]}
            path = root / "telemetry/cost-events.jsonl"
            path.write_text(json.dumps(event) + "\n")
            ss._verify_live_objectives(root, "dpo", objectives)
            event["objective"] += .01
            path.write_text(json.dumps(event) + "\n")
            with self.assertRaises(ValueError):
                ss._verify_live_objectives(root, "dpo", objectives)
            path.write_text("")
            with self.assertRaises(ValueError):
                ss._verify_live_objectives(root, "dpo", objectives)


if __name__ == "__main__":
    unittest.main()
