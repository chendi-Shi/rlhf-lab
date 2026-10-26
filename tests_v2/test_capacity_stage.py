"""CPU-only stage checks with tiny random Qwen; never CUDA capacity evidence."""
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
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from alignment_lab import capacity_stage as stage
from alignment_lab.checkpointing import build_identity, save_checkpoint, load_checkpoint
from alignment_lab.stage_engine import StageOutcome, PPOConfig, parameter_mapping, run_ppo
from alignment_lab.training_data import TrainingRow
from test_shared_qwen import fixture


class CapacityStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("alignment_lab", "scripts", "rlhf_lab", "reports", "runs", "models/base", "models/tokenizer", "audit", "docs"):
            (self.root / name).mkdir(parents=True)
        for name in ("alignment_lab/fixture.py", "scripts/run_alignlab_capacity_stage.py", "rlhf_lab/runtime.py", "docs/ALIGNLAB_V2.md"):
            (self.root / name).write_text("fixture", encoding="utf-8")
        (self.root / "capacity-plan.json").write_text('{"capacity_only":true}')
        (self.root / "audit/manifest.json").write_text("{}")
        (self.root / "audit/sft_train.jsonl").write_text("{}\n")
        (self.root / "models/base/snapshot-manifest.json").write_text('{"files":{"weights.bin":{}}}')
        (self.root / "models/base/weights.bin").write_bytes(b"tiny-test-artifact")
        (self.root / "models/tokenizer/source-manifest.json").write_text('{"artifacts":{"tokenizer.json":{}}}')
        (self.root / "models/tokenizer/tokenizer.json").write_text("{}")
        sources = {name: stage.sha(self.root / name) for name in
            ("alignment_lab/fixture.py", "scripts/run_alignlab_capacity_stage.py", "rlhf_lab/runtime.py")}
        self.request = {"schema_version": 1, "kind": stage.KIND, "stage": "sft",
            "output": str(self.root / "runs/capacity/sft"), "deadline_epoch": time.time() + 600,
            "capacity_plan_path": str(self.root / "capacity-plan.json"),
            "capacity_plan_sha256": stage.sha(self.root / "capacity-plan.json"), "source_sha256": sources,
            "model_snapshot": str(self.root / "models/base"),
            "model_manifest_sha256": stage.sha(self.root / "models/base/snapshot-manifest.json"),
            "tokenizer_dir": str(self.root / "models/tokenizer"), "audit_dir": str(self.root / "audit"),
            "audit_manifest_sha256": stage.sha(self.root / "audit/manifest.json"), "parents": {}}
        self.root_patch = patch.object(stage, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_plan_only_and_exact_frozen_sources(self):
        with patch("alignment_lab.gpu_probe.resources", side_effect=AssertionError("No admission in dry-run")):
            self.assertEqual(stage.run_capacity_stage(self.request)["status"], "plan_only")
        self.assertFalse(Path(self.request["output"]).exists())
        (self.root / "alignment_lab/fixture.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            stage.validate_request(self.request)

    def test_new_untracked_module_manifest_and_parent_changes_fail(self):
        (self.root / "alignment_lab/new.py").write_text("new")
        with self.assertRaisesRegex(ValueError, "inventory"):
            stage.validate_request(self.request)
        self.request["source_sha256"]["alignment_lab/new.py"] = stage.sha(self.root / "alignment_lab/new.py")
        (self.root / "audit/manifest.json").write_text('{"tampered":true}')
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            stage.validate_request(self.request)
        self.request["audit_manifest_sha256"] = stage.sha(self.root / "audit/manifest.json")
        self.request["stage"] = "ppo"
        with self.assertRaisesRegex(ValueError, "lineage"):
            stage.validate_request(self.request)

    def test_boundary_budget_and_stop_are_independent(self):
        stage.boundary(100, lambda: False, clock=lambda: 99)
        with self.assertRaises(TimeoutError):
            stage.boundary(100, lambda: False, clock=lambda: 100)
        with self.assertRaises(InterruptedError):
            stage.boundary(100, lambda: True, clock=lambda: 10)
        for invalid in (True, float("nan"), -1):
            self.request["deadline_epoch"] = invalid
            with self.assertRaises(ValueError):
                stage.validate_request(self.request)

    def test_fixed_work_preserves_shared_deadline_and_hyperparameters(self):
        for name, count in stage.COUNTS.items():
            config = stage.stage_config(name, 12345.67)
            self.assertEqual(config.deadline_epoch, 12345.67)
            if name == "ppo":
                self.assertEqual((config.rollout_batch, config.epochs, config.optimizer_batch,
                                  config.max_responses, config.max_updates, config.max_new_tokens),
                                 (16, 2, 1, 128, 256, 256))
            else:
                self.assertEqual((config.effective_batch, config.microbatch, config.max_updates), (16, 1, count // 16))
        self.assertEqual(stage.LEARNING_RATES, {"sft": 1e-4, "reward": 5e-5, "dpo": 5e-6, "ppo": 1e-6})

    def test_incomplete_outcome_never_counts_as_capacity(self):
        cursor = {"offset": 320, "optimizer_updates": 20, "active_rollout": False}
        good = StageOutcome(cursor, None, "data_exhausted", tuple({} for _ in range(20)))
        self.assertFalse(stage.verify_outcome("dpo", good, 320)["full_accepted_dataset_pass"])
        for bad in (dataclasses.replace(good, stop_reason="budget_exhausted"),
                    dataclasses.replace(good, extra_state={"rollout": "unfinished"}),
                    dataclasses.replace(good, metrics=({},))):
            with self.assertRaises(RuntimeError):
                stage.verify_outcome("dpo", bad, 320)

    def test_generation_must_have_real_full_horizon_and_final_kv(self):
        complete = {"quality_evaluation": False, "capacity_only": True, "termination": "forced_max_new_tokens",
            "final_action_flushed_to_kv": True, "generated_tokens": 2048, "prompt_tokens": 1024,
            "max_cache_tokens": 3072, "ignore_eos": True}
        stage.verify_generation(complete, forced=True)
        for key, value in (("generated_tokens", 7), ("termination", "eos"),
                           ("max_cache_tokens", 3071), ("final_action_flushed_to_kv", False)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                stage.verify_generation(dict(complete, **{key: value}), forced=True)

    def test_existing_output_foreign_lock_and_stop_do_not_launch(self):
        output = Path(self.request["output"])
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            stage.run_capacity_stage(self.request, execute=True)
        output.rmdir()
        (output.parent / "STOP").write_text("paused")
        with self.assertRaises(InterruptedError):
            stage.run_capacity_stage(self.request, execute=True)
        (output.parent / "STOP").unlink()
        lock = self.root / "reports/.alignlab-gpu.lock"
        lock.write_text('{"foreign":true}')
        with patch("alignment_lab.gpu_probe.resources", return_value={"blockers": []}):
            with self.assertRaises(FileExistsError):
                stage.run_capacity_stage(self.request, execute=True)
        self.assertEqual(lock.read_text(), '{"foreign":true}')
        self.assertFalse(output.exists())

    def test_gpu_resource_failure_does_not_create_output(self):
        with patch("alignment_lab.gpu_probe.resources", return_value={"blockers": ["busy"]}):
            with self.assertRaisesRegex(RuntimeError, "admission blocked"):
                stage.run_capacity_stage(self.request, execute=True)
        self.assertFalse(Path(self.request["output"]).exists())

    def test_no_step_ppo_shape_backward_changes_no_weights_or_optimizer(self):
        backend, *_ = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy", heads=("value",)), lr=1e-3)
        before = {name: p.detach().clone() for name, p in parameter_mapping(backend, "policy", heads=("value",)).items()}
        with patch.object(optimizer, "step", side_effect=AssertionError("Shape stress must not step")):
            result = stage.ppo_shape_stress(backend, "policy", optimizer,
                [dict(prompt_ids=(4, 8, 9)), dict(prompt_ids=(5, 6))], device="cpu",
                special_token_ids=(0, 2), prompt_length=6, action_length=3, chunk_tokens=2)
        self.assertTrue(result["parameters_unchanged"])
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertFalse(optimizer.state)
        for name, p in parameter_mapping(backend, "policy", heads=("value",)).items():
            self.assertTrue(torch.equal(before[name], p))
        self.assertTrue(all(p.grad is None for p in backend._all_parameters()))

    def test_replay_state_guard_exact_oldstats_and_tolerant_weights(self):
        a = torch.tensor([1.0])
        stage.compare_state(a + 1e-6, a)
        with self.assertRaises(AssertionError):
            stage.compare_state(a + 1e-6, a, exact_float=True)
        with self.assertRaises(AssertionError):
            stage.compare_state({"cursor": True}, {"cursor": 1})

    def test_tiny_ppo_disk_midrollout_one_update_matches_continuous(self):
        backend, *_ = fixture()
        mapping = parameter_mapping(backend, "policy", heads=("value",))
        optimizer = torch.optim.AdamW(mapping.values(), lr=1e-4)
        generator = torch.Generator().manual_seed(42)
        rows = [dict(record_id=f"p{i}", prompt_ids=(4 + i, 8)) for i in range(2)]
        config = PPOConfig(seed=42, rollout_batch=2, optimizer_batch=1, epochs=2,
            max_responses=2, max_updates=4, max_new_tokens=3, deadline_epoch=time.time() + 100,
            pad_id=0, eos_id=2, chunk_tokens=2)
        identity = build_identity({"fixture": self.root / "capacity-plan.json"}, {"purpose": "CPU-replay-test"})
        saved = {}

        def checkpoint(cursor, extra):
            if cursor["optimizer_updates"] in (1, 2):
                saved[cursor["optimizer_updates"]] = save_checkpoint(self.root / f"ckpt{cursor['optimizer_updates']}",
                    identity=identity, parameters=mapping, optimizer=optimizer, cursor=cursor,
                    generators={"sampling": generator}, extra_state=extra)

        result = run_ppo(backend, "policy", optimizer, rows, config, device="cpu", generator=generator,
            reference_role="reference", reward_role="reward", checkpoint_callback=checkpoint)
        self.assertEqual(result.cursor["optimizer_updates"], 4)
        restored = load_checkpoint(saved[1]["checkpoint_path"], expected_identity=identity,
            parameters=mapping, optimizer=optimizer, generators={"sampling": generator},
            expected_manifest_sha256=saved[1]["manifest_sha256"])
        observed = []
        replay = run_ppo(backend, "policy", optimizer, rows, config, device="cpu", generator=generator,
            reference_role="reference", reward_role="reward", state=restored["cursor"],
            extra_state=restored["extra_state"], checkpoint_callback=lambda c, e: observed.append((c, e)),
            stop_requested=lambda: bool(observed))
        self.assertEqual((len(replay.metrics), replay.stop_reason), (1, "paused"))
        receipt = save_checkpoint(self.root / "replay", identity=identity, parameters=mapping,
            optimizer=optimizer, cursor=replay.cursor, generators={"sampling": generator}, extra_state=replay.extra_state)
        actual = torch.load(Path(receipt["checkpoint_path"]) / "state.pt", map_location="cpu", weights_only=True)
        expected = torch.load(Path(saved[2]["checkpoint_path"]) / "state.pt", map_location="cpu", weights_only=True)
        stage.compare_state(actual, expected, exact_float=True)

    def test_tiny_sft_stage_orchestration_exports_completed_artifact_with_cpu_mocks(self):
        backend, *_ = fixture()
        loaded = SimpleNamespace(backend=backend, role="policy", heads=(), metadata={"fixture": "tiny-random-CPU"},
            device=torch.device("cpu"), autocast=contextlib.nullcontext)
        rows = tuple(TrainingRow(f"r{i}", f"c{i}", f"p{i}", (4, 8), (9 + i % 10, 2)) for i in range(16))
        selection = SimpleNamespace(rows=rows, metadata={"fixture": "CPU only", "serialization_version": "fixture"})
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(stage, "_selection", return_value=(selection, None)))
            stack.enter_context(patch.object(stage, "PAD", 0))
            stack.enter_context(patch.object(stage, "EOS", 2))
            stack.enter_context(patch("alignment_lab.gpu_probe.resources", return_value={"blockers": []}))
            stack.enter_context(patch("alignment_lab.model_loader.load_quantized_stage", return_value=loaded))
            stack.enter_context(patch("rlhf_lab.runtime.gpu_inventory", return_value={"mock": "CPU"}))
            for name in ("set_device", "reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=None))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            result = stage.run_capacity_stage(self.request, execute=True)
        self.assertEqual(result["status"], "capacity_stage_passed")
        self.assertFalse(result["capacity_verified"])
        self.assertEqual(result["actual_optimizer_updates"], 1)
        self.assertTrue(result["changed_parameter_names"])
        self.assertEqual(stage.sha(result["artifact"]["manifest_path"]), result["artifact"]["manifest_sha256"])
        self.assertTrue((Path(self.request["output"]) / "update-000001/state.pt").is_file())
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
