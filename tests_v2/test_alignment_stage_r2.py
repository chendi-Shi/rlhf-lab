"""CPU-only admission tests for the explicitly authorized r2 stage envelope."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab import alignment_stage_r2 as stage
from alignment_lab import alignment_process_guard_r2 as guard


class AlignmentStageR2AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "runs" / "fresh" / "dpo42"
        self.plan_path = self.root / "plan.json"
        continuation = {"deadline_epoch": 1792894045.7777236,
                        "started_at_epoch": 1792634845.7777236,
                        "cache_prefix": {"scored_sequences": 17152}}
        self.request = {key: "fixture" for key in (
            "parent_plan_path", "parent_plan_sha256", "model_snapshot",
            "model_manifest_sha256", "tokenizer_dir", "audit_dir",
            "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256",
            "parent_acceptance_path", "parent_acceptance_sha256")}
        self.request.update(schema_version=1, kind=stage.KIND, stage="dpo", seed=42,
            output=str(self.output), deadline_epoch=continuation["deadline_epoch"],
            source_sha256={name: "a" * 64 for name in stage.REQUIRED_SOURCES},
            parents={"sft": {"artifact_dir": str(self.root / "sft"), "manifest_sha256": "b" * 64}},
            alignment_plan_path=str(self.plan_path), alignment_plan_sha256="c" * 64,
            workload={"scope": stage.FULL_SCOPE, "rows": 13139, "optimizer_updates": 822, "beta": 0.1},
            reference_cache={"mode": "create", "path": str(self.output / "reference-cache")},
            process_wait_policy=copy.deepcopy(guard.POLICY), allowed_background_processes=[],
            supervisor_processes=[{"pid": 900001, "parent_pid": 1,
                "created_at_utc": "2026-10-22T05:00:00.0000000Z",
                "command_line": "python scripts/run_alignlab_alignment_r2.py --plan fixture --execute"}],
            process_wait_spent_seconds=300.0664498806, continuation=copy.deepcopy(continuation))
        self.plan = {"kind": "alignlab-formal-alignment-campaign-r2-v1",
                     "continuation": continuation,
                     "process_wait_policy": copy.deepcopy(guard.POLICY),
                     "allowed_background_processes": []}
        self.write_plan()
        self.addCleanup(patch.stopall)
        patch.object(stage.parent_stage, "_pin").start()
        patch.object(stage.parent_stage, "validate_request", return_value=self.output).start()
        patch.object(stage, "validate_parent_acceptance", return_value={}).start()
        self.continuation_validator = patch(
            "alignment_lab.alignment_supervisor_r2.validate_continuation", return_value={}).start()

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def test_authorized_envelope_preserves_deadline_and_inherited_wait(self):
        self.assertEqual(stage.validate_request(self.request), self.output)
        self.continuation_validator.assert_called_once_with(self.plan)
        self.assertEqual(self.request["process_wait_spent_seconds"], 300.0664498806)

    def test_old_request_kind_is_rejected(self):
        self.request["kind"] = "alignlab-formal-alignment-stage-v1"
        with self.assertRaises(ValueError):
            stage.validate_request(self.request)

    def test_missing_authorization_envelope_is_rejected(self):
        self.request.pop("continuation")
        with self.assertRaises(ValueError):
            stage.validate_request(self.request)

    def test_child_cannot_replace_plan_continuation(self):
        self.request["continuation"]["cache_prefix"]["scored_sequences"] = 1
        with self.assertRaisesRegex(ValueError, "continuation must equal"):
            stage.validate_request(self.request)

    def test_child_cannot_extend_absolute_deadline(self):
        self.request["deadline_epoch"] += 1
        with self.assertRaisesRegex(ValueError, "absolute deadline"):
            stage.validate_request(self.request)

    def test_origin_proof_failure_propagates(self):
        self.continuation_validator.side_effect = ValueError("origin SHA mismatch")
        with self.assertRaisesRegex(ValueError, "origin SHA"):
            stage.validate_request(self.request)

    def test_wait_over_old_900_seconds_is_not_a_new_stage_failure(self):
        self.request["process_wait_spent_seconds"] = 901.0
        self.assertEqual(stage.validate_request(self.request), self.output)

    def test_wait_policy_cannot_be_changed_by_child(self):
        self.request["process_wait_policy"]["poll_interval_seconds"] = 1
        with self.assertRaises(ValueError):
            stage.validate_request(self.request)

    def test_fresh_cache_path_cannot_overwrite_origin(self):
        self.request["reference_cache"]["path"] = str(self.root / "old-cache")
        with self.assertRaisesRegex(ValueError, "fresh branch output"):
            stage.validate_request(self.request)

    def test_full_dpo_workload_cannot_be_shortened(self):
        self.request["workload"]["rows"] = 100
        with self.assertRaisesRegex(ValueError, "entire accepted"):
            stage.validate_request(self.request)

    def test_extra_request_fields_are_rejected(self):
        self.request["ignore_guard"] = True
        with self.assertRaisesRegex(ValueError, "Exact formal"):
            stage.validate_request(self.request)


if __name__ == "__main__":
    unittest.main()
