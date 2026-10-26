"""CPU-only formal-stage integration fixtures; these do not train the real model."""
import contextlib
import copy
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from alignment_lab import study_stage as stage
from alignment_lab.checkpointing import load_checkpoint
from alignment_lab.stage_engine import StageOutcome, parameter_mapping, run_supervised
from alignment_lab.training_data import TrainingPartition, TrainingRow
from test_stage_artifacts import backend_for


class StudyStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("reports", "runs", "models/base", "models/tokenizer", "audit", "docs"):
            (self.root / name).mkdir(parents=True)
        for name in stage.REQUIRED_SOURCES | {"docs/ALIGNLAB_V2.md"}:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test fixture", encoding="utf-8")
        self.json("parent-plan.json", {"kind": "CPU formal parent plan fixture"})
        self.json("audit/manifest.json", {})
        for part in ("sft_train", "preference_train"):
            (self.root / f"audit/{part}.jsonl").write_text("{}\n", encoding="utf-8")
        self.json("models/base/snapshot-manifest.json", {"files": {"weights.bin": {}}})
        (self.root / "models/base/weights.bin").write_bytes(b"CPU fixture")
        self.json("models/tokenizer/source-manifest.json", {"artifacts": {"tokenizer.json": {}}})
        self.json("models/tokenizer/tokenizer.json", {"fixture_vocab_size": 97})
        self.json("reports/source-snapshot.json", {"fixture": True})
        self.json("reports/capacity-acceptance.json", {
            "status": "accepted_registered_full_capacity_workload_only",
            "all_checks_passed": True, "capacity_verified": True, "formal_training": False,
            "model_snapshot": {"sha256": stage.sha(self.root / "models/base/snapshot-manifest.json")},
            "data": {"manifest": {"sha256": stage.sha(self.root / "audit/manifest.json")}},
            "source_snapshot": {"path": str(self.root / "reports/source-snapshot.json"),
                                "sha256": stage.sha(self.root / "reports/source-snapshot.json")}})
        self.request = {
            "schema_version": 1, "kind": stage.KIND, "stage": "sft",
            "output": str(self.root / "runs/formal/sft"), "deadline_epoch": time.time() + 600,
            "parent_plan_path": str(self.root / "parent-plan.json"),
            "parent_plan_sha256": stage.sha(self.root / "parent-plan.json"),
            "source_sha256": {name: stage.sha(self.root / name) for name in stage.REQUIRED_SOURCES},
            "model_snapshot": str(self.root / "models/base"),
            "model_manifest_sha256": stage.sha(self.root / "models/base/snapshot-manifest.json"),
            "tokenizer_dir": str(self.root / "models/tokenizer"), "audit_dir": str(self.root / "audit"),
            "audit_manifest_sha256": stage.sha(self.root / "audit/manifest.json"),
            "capacity_acceptance_path": str(self.root / "reports/capacity-acceptance.json"),
            "capacity_acceptance_sha256": stage.sha(self.root / "reports/capacity-acceptance.json"),
            "parents": {}, "expected_rows": 4000, "expected_updates": 250, "seed": 42}
        self.root_patch = patch.object(stage, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def partition(self, request, count=None):
        count = request["expected_rows"] if count is None else count
        reward = request["stage"] == "reward"
        rows = tuple(TrainingRow(f"record-{i}", f"cluster-{i}", f"prompt-{i}",
            (4, 8 + i % 4), (20 + i % 7, stage.EOS), (40 + i % 7, stage.EOS) if reward else None)
            for i in range(count))
        return TrainingPartition(rows, stage.EXPECTED[request["stage"]][2],
            request["audit_manifest_sha256"], "a" * 64, "CPU-fixture", b"{}")

    @contextlib.contextmanager
    def cpu_runtime(self, request, *, engine=None, gpu_inventory=None):
        """Mock device admission only: engine, gradients, files and imports are real."""
        tiny_expected = {"sft": (17, 2, "sft_train", 1e-4),
                         "reward": (19, 2, "preference_train", 5e-5)}
        request.update(expected_rows=tiny_expected[request["stage"]][0], expected_updates=2)
        backend, _ = backend_for(request["stage"])
        role = "reward" if request["stage"] == "reward" else "policy"
        heads = ("reward",) if role == "reward" else ()
        loaded = SimpleNamespace(backend=backend, role=role, heads=heads,
            metadata={"fixture": "tiny random CPU, no CUDA evidence"},
            device=torch.device("cpu"), autocast=contextlib.nullcontext)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(stage, "EXPECTED", tiny_expected))
            stack.enter_context(patch.object(stage, "PAD", 0))
            stack.enter_context(patch.object(stage, "EOS", 2))
            partition = self.partition(request)
            stack.enter_context(patch("alignment_lab.training_data.load_training_partition", return_value=partition))
            stack.enter_context(patch("alignment_lab.gpu_probe.resources", return_value={"blockers": []}))
            stack.enter_context(patch("alignment_lab.model_loader.load_quantized_stage", return_value=loaded))
            stack.enter_context(patch("rlhf_lab.runtime.gpu_inventory",
                side_effect=gpu_inventory if gpu_inventory else lambda: {"fixture_only": True}))
            for name in ("set_device", "reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=None))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            if engine:
                stack.enter_context(patch("alignment_lab.stage_engine.run_supervised", side_effect=engine))
            yield loaded, partition
        self.assertFalse(torch.cuda.is_initialized())

    def test_dry_run_explicit_dependencies_allow_unrelated_later_modules(self):
        with patch("alignment_lab.gpu_probe.resources", side_effect=AssertionError("Dry-run must not admit CUDA")):
            result = stage.run_study_stage(self.request)
        self.assertEqual(result["status"], "plan_only")
        self.assertFalse(result["formal_training"])
        self.assertFalse(Path(self.request["output"]).exists())
        (self.root / "alignment_lab/later_evaluation.py").write_text("unrelated new code")
        stage.validate_request(self.request)
        (self.root / "alignment_lab/stage_engine.py").write_text("changed frozen code")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            stage.validate_request(self.request)

    def test_exact_registered_counts_seed_deadline_and_source_inventory(self):
        self.assertEqual(stage.EXPECTED, {"sft": (4000, 250, "sft_train", 1e-4),
                                         "reward": (13139, 822, "preference_train", 5e-5)})
        for key, invalid in (("seed", True), ("seed", 43), ("expected_rows", 3999),
                ("expected_updates", 251), ("deadline_epoch", True), ("deadline_epoch", float("nan")),
                ("deadline_epoch", 0), ("stage", "dpo")):
            with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                stage.validate_request(dict(self.request, **{key: invalid}))
        request = copy.deepcopy(self.request)
        request["source_sha256"].pop("alignment_lab/stage_engine.py")
        with self.assertRaisesRegex(ValueError, "inventory"):
            stage.validate_request(request)
        with self.assertRaises(ValueError):
            stage.validate_request(dict(self.request, unexpected="not accepted"))

    def test_capacity_acceptance_and_original_manifest_cannot_change(self):
        (self.root / "reports/source-snapshot.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            stage.validate_request(self.request)
        self.json("reports/source-snapshot.json", {"fixture": True})
        acceptance = stage.read_json(self.request["capacity_acceptance_path"])
        acceptance["formal_training"] = True
        self.json("reports/capacity-acceptance.json", acceptance)
        request = dict(self.request, capacity_acceptance_sha256=stage.sha(self.request["capacity_acceptance_path"]))
        with self.assertRaisesRegex(ValueError, "full-capacity evidence"):
            stage.validate_request(request)

    def test_real_selection_requires_every_accepted_row_and_no_synthetic(self):
        partition = self.partition(self.request)
        summary, records = stage.real_selection(partition, self.request)
        self.assertEqual((summary["real_rows"], summary["synthetic_rows"], len(records)), (4000, 0, 4000))
        self.assertEqual(summary["scope"], stage.SCOPE)
        for bad in (dataclasses.replace(partition, rows=partition.rows[:-1]),
                    dataclasses.replace(partition, manifest_sha256="b" * 64),
                    dataclasses.replace(partition, partition="final_holdout"),
                    dataclasses.replace(partition, rows=(partition.rows[1],) + partition.rows[1:]),
                    dataclasses.replace(partition, rows=(dataclasses.replace(partition.rows[0],
                        record_id="synthetic-worst-case"),) + partition.rows[1:]),
                    dataclasses.replace(partition, rows=(dataclasses.replace(partition.rows[0],
                        chosen_ids=(9,)),) + partition.rows[1:])):
            with self.subTest(partition=bad.partition, first=bad.rows[0].record_id), self.assertRaises(ValueError):
                stage.real_selection(bad, self.request)

    def test_schedule_combines_reasons_and_never_overwrites_partial_checkpoint(self):
        schedule = stage.CheckpointSchedule(250)
        writer = Mock(return_value={"checkpoint_path": "fresh", "manifest_sha256": "a" * 64})
        first = schedule.save(1, 3601, schedule.reasons(1, 3601), writer)
        self.assertEqual(first["reasons"], ["first_update", "cost_1h"])
        schedule.save(1, 3602, ["safe_engine_return"], writer)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(schedule.reasons(50, 7201), ["every_50_updates", "cost_2h"])
        self.assertEqual(schedule.reasons(250, 8 * 3600), [
            "every_50_updates", "full_data_pass", "cost_2h", "cost_4h", "cost_8h"])
        failed = Mock(side_effect=OSError("disk full"))
        with self.assertRaises(OSError):
            schedule.save(50, 7201, schedule.reasons(50, 7201), failed)
        with self.assertRaisesRegex(RuntimeError, "cannot be retried"):
            schedule.save(50, 7202, ["every_50_updates"], writer)
        self.assertNotIn(2, schedule.costs_recorded)

    def test_existing_output_foreign_lock_stop_and_deadline_never_launch(self):
        output = Path(self.request["output"])
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            stage.run_study_stage(self.request, execute=True)
        output.rmdir()
        stop = output.parent / "STOP"
        stop.write_text("user pause")
        with patch("alignment_lab.gpu_probe.resources", side_effect=AssertionError("STOP must precede admission")):
            with self.assertRaises(InterruptedError):
                stage.run_study_stage(self.request, execute=True)
        stop.unlink()
        with self.assertRaises(TimeoutError):
            stage.run_study_stage(dict(self.request, deadline_epoch=time.time() - 1), execute=True)
        lock = self.root / "reports/.alignlab-gpu.lock"
        lock.write_text('{"foreign":true}')
        with patch("alignment_lab.gpu_probe.resources", return_value={"blockers": []}):
            with self.assertRaises(FileExistsError):
                stage.run_study_stage(self.request, execute=True)
        self.assertEqual(lock.read_text(), '{"foreign":true}')
        self.assertFalse(output.exists())

    def test_no_admission_on_busy_resources(self):
        with patch("alignment_lab.gpu_probe.resources", return_value={"blockers": ["other training"]}):
            with self.assertRaisesRegex(RuntimeError, "admission blocked"):
                stage.run_study_stage(self.request, execute=True)
        self.assertFalse(Path(self.request["output"]).exists())

    def test_partial_outcome_cannot_claim_full_data_pass(self):
        rows = self.partition(self.request).rows
        good = StageOutcome({"offset": 4000, "optimizer_updates": 250, "response_tokens": 8000,
            "active_rollout": False}, None, "data_exhausted", tuple({} for _ in range(250)))
        self.assertTrue(stage.verify_completion(good, rows, self.request)["full_accepted_dataset_pass"])
        for bad in (dataclasses.replace(good, stop_reason="budget_exhausted"),
                    dataclasses.replace(good, metrics=good.metrics[:-1]),
                    dataclasses.replace(good, cursor={**good.cursor, "response_tokens": 7999}),
                    dataclasses.replace(good, extra_state={"unfinished": True})):
            with self.assertRaises(RuntimeError):
                stage.verify_completion(bad, rows, self.request)

    def test_tiny_sft_to_reward_full_pass_and_final_partial_batch_imports_real_artifact(self):
        with self.cpu_runtime(self.request) as (sft_loaded, sft_partition):
            sft = stage.run_study_stage(self.request, execute=True)
            sft_mapping = parameter_mapping(sft_loaded.backend, "policy")
            final_weights = {n: p.detach().clone() for n, p in sft_mapping.items()}
            fresh, _ = backend_for("sft")
            fresh_mapping = parameter_mapping(fresh, "policy")
            optimizer = torch.optim.AdamW(fresh_mapping.values(), lr=1e-4)
            generator = torch.Generator().manual_seed(999)
            checkpoint = sft["latest_safe_checkpoint"]
            restored = load_checkpoint(checkpoint["checkpoint_path"], expected_identity=sft["identity"],
                parameters=fresh_mapping, optimizer=optimizer, generators={"sampling": generator},
                expected_manifest_sha256=checkpoint["manifest_sha256"])
            self.assertEqual(restored["cursor"], sft["cursor"])
            self.assertTrue(optimizer.state)
            for name, value in fresh_mapping.items():
                self.assertTrue(torch.equal(value, final_weights[name]), name)
        self.assertEqual(sft["status"], "formal_parent_stage_passed")
        self.assertTrue(sft["formal_training"])
        self.assertFalse(sft["research_complete"])
        self.assertEqual(sft["completion"]["rows_processed"], 17)
        self.assertEqual(sft["completion"]["response_tokens"], 34)
        self.assertEqual([item["optimizer_updates"] for item in sft["checkpoints"]], [1, 2])
        self.assertEqual(sft["identity"]["config"]["work"]["deadline_epoch"], self.request["deadline_epoch"])
        self.assertEqual(sft["identity"]["config"]["optimizer"]["lr"], 1e-4)
        self.assertEqual(sft["selection"]["synthetic_rows"], 0)
        events = [json.loads(line) for line in (Path(self.request["output"]) / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([item["event"] for item in events],
            ["optimizer_boundary", "optimizer_boundary", "objective_after_engine_return", "objective_after_engine_return"])
        reward_request = dict(self.request, stage="reward", output=str(self.root / "runs/formal/reward"),
            parents={"sft": {key: sft["artifact"][key] for key in ("artifact_dir", "manifest_sha256")}})
        with self.cpu_runtime(reward_request):
            reward = stage.run_study_stage(reward_request, execute=True)
        self.assertEqual(reward["status"], "formal_parent_stage_passed")
        self.assertEqual(reward["completion"]["rows_processed"], 19)  # second batch contains three actual pairs
        self.assertEqual(reward["completion"]["response_tokens"], 76)
        self.assertTrue(reward["scalar_heads_changed"]["reward"])
        self.assertEqual(reward["identity"]["config"]["optimizer"]["lr"], 5e-5)
        self.assertEqual(reward["imports"]["reward"]["manifest_sha256"], sft["artifact"]["manifest_sha256"])
        self.assertEqual(reward["identity"]["config"]["work"]["deadline_epoch"], self.request["deadline_epoch"])
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())

    def test_capacity_or_incomplete_parent_is_rejected_before_cuda(self):
        with self.cpu_runtime(self.request):
            result = stage.run_study_stage(self.request, execute=True)
            parent = {key: result["artifact"][key] for key in ("artifact_dir", "manifest_sha256")}
            manifest_path = Path(parent["artifact_dir"]) / "manifest.json"
            manifest = stage.read_json(manifest_path)
            manifest["identity"]["config"]["purpose"] = "alignlab-capacity-stage-v1"
            manifest_path.write_text(json.dumps(manifest))
            parent["manifest_sha256"] = stage.sha(manifest_path)
            with self.assertRaisesRegex(ValueError, "capacity parents forbidden"):
                stage.validate_parent(parent, self.request)

    def test_stop_at_complete_boundary_saves_checkpoint_without_publishing_artifact(self):
        output = Path(self.request["output"])
        def engine(*args, **kwargs):
            callback = kwargs["checkpoint_callback"]
            def pause(cursor, extra):
                callback(cursor, extra)
                (output.parent / "STOP").write_text("fixture pause")
            kwargs["checkpoint_callback"] = pause
            return run_supervised(*args, **kwargs)
        with self.cpu_runtime(self.request, engine=engine):
            with self.assertRaises(InterruptedError):
                stage.run_study_stage(self.request, execute=True)
        report = stage.read_json(output / "progress.json")
        self.assertEqual(report["status"], "paused")
        self.assertEqual(report["actual_optimizer_updates"], 1)
        self.assertEqual(report["cursor"]["offset"], 16)
        self.assertTrue(report["safe_resume_available"])
        self.assertIn("safe_engine_return", report["latest_safe_checkpoint"]["reasons"])
        self.assertFalse(report["emergency_checkpoint_attempted"])
        self.assertFalse((output / "final-artifact").exists())

    def test_budget_return_keeps_original_deadline_and_saves_only_clean_boundary(self):
        def engine(*args, **kwargs):
            completed = []
            callback = kwargs["checkpoint_callback"]
            def tick(cursor, extra):
                callback(cursor, extra)
                completed.append(cursor)
            kwargs.update(checkpoint_callback=tick,
                clock=lambda: self.request["deadline_epoch"] + 1 if completed else time.time())
            return run_supervised(*args, **kwargs)
        with self.cpu_runtime(self.request, engine=engine):
            with self.assertRaises(TimeoutError):
                stage.run_study_stage(self.request, execute=True)
        report = stage.read_json(Path(self.request["output"]) / "progress.json")
        self.assertEqual(report["status"], "budget_exhausted")
        self.assertEqual(report["actual_optimizer_updates"], 1)
        self.assertEqual(report["deadline_epoch"], self.request["deadline_epoch"])
        self.assertTrue(report["safe_resume_available"])

    def test_exception_never_creates_speculative_checkpoint_after_last_committed_boundary(self):
        def engine(*args, **kwargs):
            callback = kwargs["checkpoint_callback"]
            def fail(cursor, extra):
                callback(cursor, extra)
                raise RuntimeError("unknown subsequent backward failure")
            kwargs["checkpoint_callback"] = fail
            return run_supervised(*args, **kwargs)
        with self.cpu_runtime(self.request, engine=engine):
            with self.assertRaisesRegex(RuntimeError, "backward failure"):
                stage.run_study_stage(self.request, execute=True)
        output = Path(self.request["output"])
        report = stage.read_json(output / "progress.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["emergency_checkpoint_attempted"])
        self.assertEqual(report["latest_safe_checkpoint"]["optimizer_updates"], 1)
        self.assertNotIn("safe_engine_return", report["latest_safe_checkpoint"]["reasons"])
        self.assertFalse((output / "update-000002").exists())

    def test_source_change_during_training_prevents_success_and_keeps_safe_checkpoint(self):
        def engine(*args, **kwargs):
            outcome = run_supervised(*args, **kwargs)
            (self.root / "alignment_lab/stage_engine.py").write_text("changed during execution")
            return outcome
        with self.cpu_runtime(self.request, engine=engine):
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                stage.run_study_stage(self.request, execute=True)
        output = Path(self.request["output"])
        report = stage.read_json(output / "progress.json")
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["safe_resume_available"])
        self.assertFalse((output / "final-artifact").exists())


if __name__ == "__main__":
    unittest.main()
