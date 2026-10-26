"""CPU-only regression for a real metadata tuple/JSON round-trip recovery."""
import contextlib
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from alignment_lab import parent_recovery as r1
from alignment_lab import parent_recovery_r2 as r2
from alignment_lab.model_loader import ModelLoadConfig
from alignment_lab.stage_artifacts import import_stage
from alignment_lab.stage_engine import SupervisedConfig, parameter_mapping, run_supervised
import test_parent_recovery as previous_tests
from test_stage_artifacts import backend_for

TARGETS = ("q_proj", "k_proj", "v_proj")


class MetadataTests(unittest.TestCase):
    def test_actual_dataclass_tuple_json_roundtrip_and_full_semantic_comparison(self):
        actual = {"config": asdict(ModelLoadConfig("reward")), "parameters_selected": 17434625,
                  "quantization": {"bits": 4, "compute_dtype": "bfloat16"}, "capacity_verified": False}
        persisted = json.loads(json.dumps(actual))
        self.assertNotEqual(actual, persisted)  # The exact r1 bug.
        self.assertTrue(r2.compare_metadata(actual, persisted)["all_fields_equal"])
        for key, value in (("lora_r", 8), ("seed", 43), ("device", "cuda:1")):
            changed = copy.deepcopy(actual); changed["config"][key] = value
            self.assertFalse(r2.compare_metadata(changed, persisted)["all_fields_equal"])
        changed = copy.deepcopy(actual); changed["capacity_verified"] = 0
        self.assertFalse(r2.compare_metadata(changed, persisted)["all_fields_equal"])
        changed = copy.deepcopy(actual); changed.pop("quantization")
        self.assertFalse(r2.compare_metadata(changed, persisted)["all_fields_equal"])

    def test_unknown_types_and_nonfinite_metadata_cannot_be_normalized_away(self):
        for invalid in ({"x": float("nan")}, {"x": object()}, {1: "coerced key"}):
            with self.assertRaises(ValueError):
                r2.canonical_metadata(invalid)


class RecoveryR2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        original = previous_tests.fixtures.StudyStageTests.cpu_runtime
        @contextlib.contextmanager
        def with_runtime_tuple(fixture, *args, **kwargs):
            with original(fixture, *args, **kwargs) as (loaded, partition):
                loaded.metadata["config"] = {"target_modules": TARGETS}
                yield loaded, partition
        self.legacy = previous_tests.ParentRecoveryTests()
        self.addCleanup(self.legacy.doCleanups)
        with patch.object(previous_tests.fixtures.StudyStageTests, "cpu_runtime", with_runtime_tuple):
            self.legacy.setUp()
        self.root = self.legacy.root
        class Runtime(previous_tests.Runtime):
            def load(inner, request):
                loaded = super().load(request)
                loaded.metadata["config"] = {"target_modules": TARGETS}
                if inner.mode == "changed_metadata":
                    loaded.metadata["config"]["target_modules"] = tuple(reversed(TARGETS))
                return loaded
        self.Runtime = Runtime
        with self.assertRaisesRegex(ValueError, "Actual loaded model/runtime"):
            r1.run_recovery(self.legacy.request_path, r1.sha(self.legacy.request_path), execute=True,
                runtime=Runtime(self.legacy.request, self.legacy.old_request))
        self.prior_output = Path(self.legacy.request["output"])
        self.plan = copy.deepcopy(self.legacy.plan)
        self.plan.update(execution_revision=r2.REVISION, output=str(self.root / "runs/recovery-r2"),
            prior_failed_dispatch={"request": r1.pin(self.legacy.request_path),
                                   "progress": r1.pin(self.prior_output / "progress.json")})
        for name in r2.NEW_SOURCES:
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("new r2 fixture source", encoding="utf8")
            self.plan["source_sha256"][name] = r1.sha(path)
        authorization = self.root / "reports/r2-authorization.json"
        r1._write(authorization, {"kind": "explicit-user-rm-r2-dispatch-authorization-v1",
            "source": "direct_user_reply_in_this_chat", "allow_one_corrected_dispatch": True,
            "budget": self.legacy.request["budget"],
            "prior_request_sha256": self.plan["prior_failed_dispatch"]["request"]["sha256"]})
        self.plan["retry_authorization"] = r1.pin(authorization)
        self.plan_path = self.root / "r2-plan.json"
        self.request = copy.deepcopy(self.legacy.request)
        self.request.update(output=self.plan["output"], source_sha256=self.plan["source_sha256"],
                            recovery_plan_path=str(self.plan_path))
        self.request_path = self.root / "r2-request.json"
        self.write_request()

    def write_request(self):
        r1._write(self.plan_path, self.plan)
        self.request["recovery_plan_sha256"] = r1.sha(self.plan_path)
        r1._write(self.request_path, self.request)

    def run_(self, mode=None, execute=True):
        return r2.run_recovery(self.request_path, r1.sha(self.request_path), execute=execute,
            runtime=self.Runtime(self.request, self.legacy.old_request, mode))

    def test_real_tiny_restore_matches_continuous_adam_and_persists_full_comparison(self):
        backend, _ = backend_for("reward")
        parent = self.legacy.old_request["parents"]["sft"]
        import_stage(parent["artifact_dir"], backend, "reward", expected_manifest_sha256=parent["manifest_sha256"],
                     expected_identity=self.legacy.old_identity)
        parameters = parameter_mapping(backend, "reward", heads=("reward",))
        optimizer = torch.optim.AdamW(parameters.values(), lr=5e-5)
        run_supervised(backend, "reward", optimizer, self.legacy.partition.rows,
            SupervisedConfig(**self.legacy.old_identity["config"]["work"]), device="cpu")
        result = self.run_()
        output = Path(self.request["output"])
        saved = torch.load(output / "update-000002/state.pt", map_location="cpu", weights_only=True)
        self.assertTrue(r1._tree_equal(optimizer.state_dict(), saved["optimizer"]))
        self.assertTrue(all(torch.equal(p.detach(), saved["parameters"][n]) for n, p in parameters.items()))
        receipt = r1.read_json(output / "model-metadata-comparison.json")
        self.assertTrue(receipt["all_fields_equal"])
        self.assertEqual(receipt["runtime_target_modules_type"], "tuple")
        self.assertEqual(receipt["actual"], receipt["expected"])
        self.assertEqual(result, r2.verify_recovery_output(output,
            expected_manifest_sha256=r1.sha(output / "manifest.json"), expected_request=self.request))
        self.assertEqual(r1.read_json(self.prior_output / "progress.json")["status"], "failed")
        self.assertEqual(result["budget"], self.legacy.request["budget"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_metadata_change_is_recorded_and_rejected_before_restore(self):
        with self.assertRaisesRegex(ValueError, "substantive difference"):
            self.run_("changed_metadata")
        output = Path(self.request["output"])
        receipt = r1.read_json(output / "model-metadata-comparison.json")
        self.assertFalse(receipt["all_fields_equal"])
        self.assertEqual(receipt["differences"][0]["path"], ["config", "target_modules"])
        self.assertFalse((output / "origin-000001").exists())
        self.assertFalse((output / "manifest.json").exists())

    def test_no_implicit_retry_and_no_budget_reset(self):
        self.plan.pop("retry_authorization")
        self.write_request()
        self.assertEqual(self.run_(execute=False)["status"], "plan_only")
        self.assertFalse(Path(self.request["output"]).exists())
        with self.assertRaises(ValueError):
            self.run_()
        self.request["budget"]["started_at_epoch"] += 1
        self.request["budget"]["deadline_epoch"] += 1
        self.write_request()
        with self.assertRaisesRegex(ValueError, "identical absolute budget"):
            self.run_(execute=False)


if __name__ == "__main__":
    unittest.main()
