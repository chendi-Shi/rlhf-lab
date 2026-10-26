"""CPU-only fresh wrapper/cache identity and terminal-stop regression checks."""
import ast
import copy
import inspect
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from alignment_lab import throughput_fresh_v1 as fresh
from alignment_lab import alignment_stage_r3 as original
from alignment_lab import alignment_process_guard_r3 as guard


class FreshThroughputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def plan(self):
        return {"source_sha256": {"old.py": "a", "new.py": "b"}, "selected_backend": "deferred_scalar",
                "future_ppo_backend": "deferred_scalar", "authorization": {}, "validation_report": {},
                "original_request": {}, "history": {}, "original_started_at_epoch": fresh.resume.STARTED,
                "original_deadline_epoch": fresh.resume.DEADLINE}

    def test_all_four_fresh_branches_preserve_inner_sources_and_dpo_original_backend(self):
        for stage in ("dpo", "ppo"):
            for seed in (43, 44):
                inner = {"stage": stage, "seed": seed, "source_sha256": {"old.py": "a"}}
                before = copy.deepcopy(inner)
                wrapper = fresh.make_request(inner, self.plan(), {})
                self.assertEqual(inner, before)
                self.assertEqual(wrapper["legacy_request"], before)
                self.assertEqual(wrapper["source_sha256"], self.plan()["source_sha256"])
                self.assertEqual(wrapper["selected_backend"], "original" if stage == "dpo" else "deferred_scalar")
        with self.assertRaisesRegex(ValueError, "limited"):
            fresh.make_request({"stage": "dpo", "seed": 42}, self.plan(), {})

    def test_dpo_capture_retains_actual_original_cache_identity(self):
        from tests_v2.test_shared_qwen import fixture
        from tests_v2.test_reference_bridge import identity_inputs
        from alignment_lab.gpu_reference import capture_frozen_reference
        backend, *_ = fixture()
        inputs = identity_inputs(self.root / "reference", backend)
        artifacts = {k: inputs[k] for k in ("base_files", "tokenizer_files", "source_files", "adapter_files", "dataset_manifest")}
        old_files = {"source/" + k: v for k, v in inputs["source_files"].items()}
        artifacts["source_files"] = old_files
        before = capture_frozen_reference(backend, "reference", artifact_files=artifacts,
            accepted_data_manifest_sha256=inputs["accepted_dataset_manifest_sha256"], precision="float32", quantization="none")
        files = {**old_files, "throughput/source/new.py": Path(fresh.__file__), "throughput/plan": Path(__file__)}
        artifacts["source_files"] = fresh._cache_source_files(files)
        after = capture_frozen_reference(backend, "reference", artifact_files=artifacts,
            accepted_data_manifest_sha256=inputs["accepted_dataset_manifest_sha256"], precision="float32", quantization="none")
        self.assertEqual(before.artifact_identity, after.artifact_identity)
        self.assertEqual(before.anchor, after.anchor)
        loaded = SimpleNamespace(backend=backend)
        self.assertIs(fresh._adopt_backend(loaded, {"legacy_request": {"stage": "dpo"}, "selected_backend": "original"}), backend)

    def test_deferred_scalar_maps_to_original_guard_without_replacing_parameters(self):
        from tests_v2.test_shared_qwen import fixture
        backend, *_ = fixture()
        loaded = SimpleNamespace(backend=backend)
        parameter_ids = [id(p) for p in backend._all_parameters()]
        adopted = fresh._adopt_backend(loaded, {"legacy_request": {"stage": "ppo"}, "selected_backend": "deferred_scalar"})
        self.assertEqual([id(p) for p in adopted._all_parameters()], parameter_ids)
        self.assertIs(loaded.backend, adopted)

    def test_original_numerical_engine_optimizer_rng_and_acceptance_calls_are_unchanged(self):
        names = {"run_supervised", "run_ppo", "AdamW", "manual_seed", "stage_config", "verify_completion"}
        def calls(function):
            nodes = ast.walk(ast.parse(inspect.getsource(function)))
            return sorted(ast.dump(node, include_attributes=False) for node in nodes if isinstance(node, ast.Call)
                          and (node.func.id if isinstance(node.func, ast.Name) else
                               node.func.attr if isinstance(node.func, ast.Attribute) else None) in names)
        self.assertEqual(calls(original.run_alignment_stage), calls(fresh.run_stage))

    def test_real_stop_before_cuda_saves_receipt_and_terminal_progress_despite_stop(self):
        output = self.root / "ppo43"
        (self.root / "reports").mkdir()
        request_path = self.root / "ppo43.request.json"
        request_path.write_text("{}", encoding="utf-8")
        own = {"pid": 101, "parent_pid": 100, "created_at_utc": "2026-10-21T08:00:00Z",
               "command_line": "python scripts/run_alignlab_throughput_fresh_v1.py --request fixture --execute"}
        inner = {"stage": "ppo", "seed": 43, "output": str(output), "deadline_epoch": time.time() + 3600,
            "alignment_plan_path": str(self.root / "old-plan.json"), "alignment_plan_sha256": "unused",
            "allowed_background_processes": [], "supervisor_processes": [], "source_sha256": {},
            "process_wait_spent_seconds": 0, "process_wait_policy": guard.POLICY, "workload": {"scope": "fixture"}}
        wrapper = {"legacy_request": inner, "plan": {"path": str(self.root / "new-plan.json")}, "source_sha256": {}}
        def stop_after_request(*args):
            (output / "STOP").write_text("fixture explicit stop", encoding="utf-8")
            return {}
        with patch.object(fresh, "ROOT", self.root), patch.object(fresh, "validate_request", return_value=(output, inner)), \
             patch.object(fresh, "_owned", return_value=[own]), patch.object(fresh, "_marker", return_value={}), \
             patch.object(fresh, "_wrapper_inputs", side_effect=stop_after_request), \
             patch("alignment_lab.probe_dispatch._System", return_value=SimpleNamespace(processes=lambda: [own])), \
             patch.object(guard, "strict_admission", return_value={"caller_admission_allowed": True}), \
             patch("alignment_lab.gpu_probe.resources") as resources:
            with self.assertRaisesRegex(InterruptedError, "after saving current state"):
                fresh.run_stage(wrapper, execute=True)
            resources.assert_not_called()
        status = fresh.read_json(output / "progress.json")
        self.assertEqual(status["status"], "paused")
        receipt = status["process_guard"]["stopping_boundary"]
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["optimizer_updates"], 0)
        self.assertEqual(receipt["sha256"], fresh.sha(output / "request.json"))
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())


if __name__ == "__main__":
    unittest.main()
