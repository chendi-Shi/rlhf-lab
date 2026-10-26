"""Tiny actual CPU RM continuation; no pretrained model, CUDA, or real launch."""
import contextlib
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from alignment_lab import parent_recovery as recovery
from alignment_lab import study_stage as old
from alignment_lab import checkpointing as ck
from alignment_lab.stage_artifacts import import_stage
from alignment_lab.stage_engine import run_supervised, parameter_mapping, SupervisedConfig, _records
import test_study_stage as fixtures
from test_stage_artifacts import backend_for


class Runtime:
    def __init__(self, request, old_request, mode=None):
        self.request, self.old_request, self.mode = request, old_request, mode
        self.epoch = request["budget"]["started_at_epoch"] + 1
        self.calls = 0
        self.resource_calls = 0
        self.gpu_samples = 0
        self.own = {"pid": 900001, "parent_pid": 0, "created_at_utc": "2026-10-21T00:00:00.0000000Z",
                    "command_line": "explicit tiny CPU fixture process"}

    def now(self):
        return self.epoch

    def sleep(self, value):
        self.epoch += value

    def bind_processes(self, *args):
        return [self.own]

    def processes(self):
        self.calls += 1
        # Third observation is after the new origin checkpoint and before the
        # first new update. This tests clean zero-new-step pause/time boundaries.
        if self.calls == 3 and self.mode == "stop":
            (Path(self.request["output"]) / "STOP").write_text("fixture pause")
        if self.calls == 3 and self.mode == "deadline":
            self.epoch = self.request["budget"]["deadline_epoch"]
        if self.calls >= 3 and self.mode == "missing_owned":
            return []
        if self.calls >= 3 and self.mode == "query_failure":
            raise OSError("fixture CIM snapshot failed")
        return [self.own]

    def resources(self):
        self.resource_calls += 1
        if self.resource_calls == 2 and self.mode == "preload_deadline":
            self.epoch = self.request["budget"]["deadline_epoch"]
        return {"admitted": self.mode != "busy", "fixture_only": True}

    def load(self, request):
        backend, _ = backend_for("reward")
        return SimpleNamespace(backend=backend, role="reward", heads=("reward",),
            metadata={"fixture": "tiny random CPU, no CUDA evidence"},
            device=torch.device("cpu"), autocast=contextlib.nullcontext)

    def sample_gpu(self):
        self.gpu_samples += 1
        if self.gpu_samples == 2 and self.mode == "source_drift":
            (recovery.ROOT / "alignment_lab/parent_recovery.py").write_text("changed fixture source")
        return {"fixture_only": True, "cuda": False}


class ParentRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.fixture = fixtures.StudyStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("ROOT", self.root), ("ORIGIN_UPDATE", 1), ("ROWS", 19), ("UPDATES", 2)):
            self.stack.enter_context(patch.object(recovery, name, value))
        tiny = {"sft": (17, 2, "sft_train", 1e-4), "reward": (19, 2, "preference_train", 5e-5)}
        self.stack.enter_context(patch.object(old, "EXPECTED", tiny))
        self.stack.enter_context(patch.object(old, "PAD", 0))
        self.stack.enter_context(patch.object(old, "EOS", 2))
        self.old_sft_request = copy.deepcopy(self.fixture.request)
        with self.fixture.cpu_runtime(self.old_sft_request):
            self.sft = old.run_study_stage(self.old_sft_request, execute=True)
        self.old_request = dict(self.old_sft_request, stage="reward", expected_rows=19, expected_updates=2,
            output=str(self.root / "runs/formal/reward"),
            parents={"sft": {k: self.sft["artifact"][k] for k in ("artifact_dir", "manifest_sha256")}})
        def paused(*args, **kwargs):
            original_callback = kwargs["checkpoint_callback"]
            done = False
            def callback(cursor, extra):
                nonlocal done
                original_callback(cursor, extra)
                done = True
            kwargs.update(checkpoint_callback=callback, stop_requested=lambda: done)
            return run_supervised(*args, **kwargs)
        with self.fixture.cpu_runtime(self.old_request, engine=paused):
            with self.assertRaises(InterruptedError):
                old.run_study_stage(self.old_request, execute=True)
        old_output = Path(self.old_request["output"])
        self.old_progress = old.read_json(old_output / "progress.json")
        self.old_identity = self.old_progress["identity"]
        self.partition = self.fixture.partition(self.old_request)
        self.stack.enter_context(patch("alignment_lab.training_data.load_training_partition", return_value=self.partition))
        old._write(old_output.parent / "campaign.json", {"status": "failed", "plan_sha256": self.old_request["parent_plan_sha256"]})
        old._write(old_output.parent / "STOP", {"owner": "alignlab-formal-parents-campaign-v1",
            "reason": "identity_failure", "plan_sha256": self.old_request["parent_plan_sha256"]})
        origins = {"plan": recovery.pin(self.old_request["parent_plan_path"]),
            "campaign": recovery.pin(old_output.parent / "campaign.json"),
            "stop": recovery.pin(old_output.parent / "STOP")}
        for name in ("request", "progress", "identity", "selection", "metrics"):
            origins[name] = recovery.pin(old_output / (name + (".jsonl" if name == "metrics" else ".json")))
        origins["checkpoint"] = {"path": str(old_output / "update-000001"),
            "manifest_sha256": old.sha(old_output / "update-000001/manifest.json")}
        sources = dict(self.old_request["source_sha256"])
        for name in recovery.NEW_SOURCES:
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("new recovery CPU fixture source", encoding="utf-8")
            sources[name] = old.sha(p)
        auth = self.root / "reports/authorization.json"
        old._write(auth, {"kind": "explicit-user-rm-recovery-budget-authorization-v1",
            "evidence": {"source": "direct_user_reply_in_this_chat"}, "authorized_scope": {
                "stage": "reward", "new_wall_seconds": 28800, "resume_origin_optimizer_update": 1,
                "target_total_optimizer_updates": 2, "target_total_pairs": 19, "automatic_retries": False,
                "original_budget_changed": False, "old_campaign_mutation_allowed": False,
                "new_output_directory_required": True}})
        self.plan_path = self.root / "recovery-plan.json"
        output = str(self.root / "runs/recovery/reward")
        self.plan = {"schema_version": 1, "kind": recovery.PLAN_KIND, "output": output,
            "wall_seconds": 28800, "source_sha256": sources, "origin": origins,
            "budget_authorization": recovery.pin(auth), "allowed_background_processes": []}
        old._write(self.plan_path, self.plan)
        start = time.time()
        self.request = {"schema_version": 1, "kind": recovery.KIND, "stage": "reward", "output": output,
            "recovery_plan_path": str(self.plan_path), "recovery_plan_sha256": old.sha(self.plan_path),
            "source_sha256": sources, "origin": origins,
            "budget": {"started_at_epoch": start, "wall_seconds": 28800, "deadline_epoch": start + 28800}}
        self.request_path = self.root / "recovery.request.json"
        old._write(self.request_path, self.request)

    def run_(self, mode=None):
        return recovery.run_recovery(self.request_path, old.sha(self.request_path), execute=True,
                                     runtime=Runtime(self.request, self.old_request, mode))

    def test_real_rm_restoration_matches_continuous_adam_cursor_and_final_partial_batch(self):
        backend, _ = backend_for("reward")
        sft = self.old_request["parents"]["sft"]
        import_stage(sft["artifact_dir"], backend, "reward", expected_manifest_sha256=sft["manifest_sha256"],
                     expected_identity=self.old_identity)
        mapping = parameter_mapping(backend, "reward", heads=("reward",))
        optimizer = torch.optim.AdamW(mapping.values(), lr=5e-5)
        config = SupervisedConfig(**self.old_identity["config"]["work"])
        continuous = run_supervised(backend, "reward", optimizer, self.partition.rows, config, device="cpu")
        result = self.run_()
        output = Path(self.request["output"])
        saved = torch.load(output / "update-000002/state.pt", map_location="cpu", weights_only=True)
        self.assertTrue(recovery._tree_equal(optimizer.state_dict(), saved["optimizer"]))
        self.assertTrue(all(torch.equal(p.detach(), saved["parameters"][n]) for n, p in mapping.items()))
        self.assertEqual(continuous.cursor["order"], saved["cursor"]["order"])
        self.assertEqual(saved["cursor"]["offset"], 19)
        self.assertEqual(result["completion"]["segment_optimizer_updates"], 1)
        self.assertEqual(result["completion"]["segment_rows"], 3)
        self.assertEqual(result["completion"]["historical_rows"], 16)
        audited = recovery.verify_recovery_output(output, expected_manifest_sha256=old.sha(output / "manifest.json"),
                                                 expected_request=self.request)
        self.assertEqual(audited, result)
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(old.read_json(Path(self.old_request["output"]).parent / "campaign.json")["status"], "failed")

    def test_transition_rejects_other_config_or_cursor_changes_and_does_not_mutate_origin(self):
        backend, _ = backend_for("reward")
        rows = _records(self.partition.rows, backend, preference=True, eos_id=2)
        config = SupervisedConfig(**self.old_identity["config"]["work"])
        cursor = self.old_progress["cursor"]
        unchanged = copy.deepcopy(cursor)
        migrated, receipt = recovery.migrate_cursor(cursor, rows, config, replace(config, deadline_epoch=config.deadline_epoch + 50))
        self.assertEqual(cursor, unchanged)
        self.assertEqual(receipt["changed_cursor_fields"], ["config_sha256"])
        self.assertNotEqual(migrated["config_sha256"], cursor["config_sha256"])
        for changed in (replace(config, seed=43), replace(config, max_updates=3), replace(config, effective_batch=8)):
            with self.assertRaises(ValueError):
                recovery.migrate_cursor(cursor, rows, config, replace(changed, deadline_epoch=config.deadline_epoch + 50))
        corrupted = copy.deepcopy(cursor); corrupted["offset"] -= 1
        with self.assertRaises(ValueError):
            recovery.migrate_cursor(corrupted, rows, config, replace(config, deadline_epoch=config.deadline_epoch + 50))

    def test_new_origin_checkpoint_exists_and_is_bitwise_equal_before_clean_stop(self):
        with self.assertRaises(InterruptedError):
            self.run_("stop")
        output = Path(self.request["output"])
        status = old.read_json(output / "progress.json")
        self.assertEqual(status["status"], "paused")
        self.assertEqual(status["segment_optimizer_updates"], 0)
        self.assertTrue((output / "origin-000001/state.pt").is_file())
        self.assertFalse((output / "manifest.json").exists())
        transition = old.read_json(output / "migration-manifest.json")
        audit = recovery.audit_migration_checkpoints(self.request["origin"]["checkpoint"], status["origin_checkpoint"],
                                                    transition, status["identity"])
        self.assertTrue(audit["rng_bitwise_equal"])
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())

    def test_deadline_at_clean_boundary_preserves_origin_and_never_claims_completion(self):
        with self.assertRaises(TimeoutError):
            self.run_("deadline")
        output = Path(self.request["output"])
        status = old.read_json(output / "progress.json")
        self.assertEqual(status["status"], "budget_exhausted")
        self.assertEqual(status["segment_optimizer_updates"], 0)
        self.assertTrue((output / "origin-000001/state.pt").exists())
        self.assertFalse((output / "manifest.json").exists())

    def test_missing_owned_identity_preserves_clean_boundary_instead_of_failed_exception(self):
        with self.assertRaises(InterruptedError):
            self.run_("missing_owned")
        output = Path(self.request["output"])
        status = old.read_json(output / "progress.json")
        self.assertEqual(status["status"], "paused")
        self.assertEqual(status["objective_records"], 0)
        self.assertTrue(status["process_guard_blocked"])
        self.assertTrue((output / "origin-000001/state.pt").is_file())
        events = [json.loads(line) for line in (output / "process-guard.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["missing_or_changed_owned_pids"], [900001])
        self.assertFalse((output / "manifest.json").exists())

    def test_failed_snapshot_preserves_clean_boundary_instead_of_unknown_exception(self):
        with self.assertRaises(InterruptedError):
            self.run_("query_failure")
        output = Path(self.request["output"])
        status = old.read_json(output / "progress.json")
        self.assertEqual(status["status"], "paused")
        self.assertTrue(status["process_guard_blocked"])
        self.assertTrue((output / "origin-000001/state.pt").is_file())
        self.assertFalse(status["emergency_checkpoint_attempted"])

    def test_code_drift_at_completed_update_pauses_and_preserves_objective_and_checkpoint(self):
        with self.assertRaises(InterruptedError):
            self.run_("source_drift")
        output = Path(self.request["output"])
        status = old.read_json(output / "progress.json")
        self.assertEqual(status["status"], "paused")
        self.assertEqual(status["objective_records"], 1)
        self.assertIn("identity_guard_error", status)
        self.assertFalse((output / "update-000002/state.pt").exists())
        self.assertEqual(status["latest_safe_checkpoint"], status["origin_checkpoint"])
        self.assertFalse((output / "manifest.json").exists())

    def test_admission_query_crossing_deadline_does_not_load_model(self):
        with patch.object(Runtime, "load", side_effect=AssertionError("must not load")) as load:
            with self.assertRaises(TimeoutError):
                self.run_("preload_deadline")
        self.assertEqual(load.call_count, 0)
        status = old.read_json(Path(self.request["output"]) / "progress.json")
        self.assertEqual(status["status"], "budget_exhausted")

    def test_origin_hash_tamper_is_rejected_before_new_output(self):
        state = Path(self.request["origin"]["checkpoint"]["path"]) / "state.pt"
        with state.open("ab") as stream:
            stream.write(b"tampered fixture")
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.run_()
        self.assertFalse(Path(self.request["output"]).exists())

    def test_foreign_lock_existing_output_and_busy_admission_do_not_resume(self):
        output = Path(self.request["output"])
        with self.assertRaisesRegex(RuntimeError, "admission"):
            self.run_("busy")
        self.assertFalse(output.exists())
        lock = self.root / "reports/.alignlab-gpu.lock"
        lock.write_text('foreign lock')
        with self.assertRaises(FileExistsError):
            self.run_()
        self.assertEqual(lock.read_text(), 'foreign lock')
        lock.unlink()
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_()

    def test_undeclared_old_budget_reset_or_false_total_is_rejected(self):
        changed = copy.deepcopy(self.request)
        changed["budget"]["deadline_epoch"] += 1
        with self.assertRaises(ValueError):
            recovery.validate_request(changed)
        changed["budget"] = dict(self.request["budget"], wall_seconds=43200)
        with self.assertRaises(ValueError):
            recovery.validate_request(changed)


if __name__ == "__main__":
    unittest.main()
