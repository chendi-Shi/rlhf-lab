"""CPU acceptance fixtures; never launch recovery or touch pretrained weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from alignment_lab import recovered_parent_acceptance as acceptance


class PayloadMigrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tests_v2.test_parent_acceptance import tensor_bundle
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_cursor = {"schema": 1, "optimizer_updates": 256, "offset": 4096,
            "response_tokens": 8192, "completed_rollouts": 0, "active_rollout": False,
            "order": list(range(13139)), "data_sha256": "a" * 64, "config_sha256": "b" * 64}
        self.new_cursor = {**self.old_cursor, "config_sha256": "c" * 64}
        self.transition = {"kind": "alignlab-explicit-deadline-transition-v1",
            "changed_cursor_fields": ["config_sha256"], "changed_work_fields": ["deadline_epoch"],
            "parameter_optimizer_rng_change_permitted": False,
            "old_cursor": self.old_cursor, "new_cursor": self.new_cursor}
        pins = []
        for name, cursor in (("old256", self.old_cursor), ("migrated256", self.new_cursor)):
            directory = self.root / name
            directory.mkdir()
            manifest = {"format": "alignlab-optimizer-checkpoint-v1", "status": "complete",
                        "identity": {"config": {"stage": "reward"}}, "artifacts": {}}
            digest = tensor_bundle(directory, manifest, stage="reward", cursor=cursor)
            pins.append({"path": str(directory), "manifest_sha256": digest})
        self.origin = pins[0]
        self.migration = {"checkpoint_path": pins[1]["path"], "manifest_sha256": pins[1]["manifest_sha256"]}
        self.assertFalse(torch.cuda.is_initialized())

    def rewrite_new(self, mutate):
        import torch
        from alignment_lab.alignment_supervisor import _cpu_payload
        path = Path(self.migration["checkpoint_path"])
        manifest = json.loads((path / "manifest.json").read_text())
        payload = _cpu_payload(path / "state.pt", manifest["artifacts"]["state.pt"]["sha256"])
        mutate(payload)
        torch.save(payload, path / "state.pt")
        item = acceptance._file(path / "state.pt")
        manifest["artifacts"]["state.pt"] = {k: item[k] for k in ("sha256", "size_bytes")}
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.migration["manifest_sha256"] = acceptance._file(path / "manifest.json")["sha256"]

    def test_actual_parameters_optimizer_rng_preserved(self):
        result = acceptance._migration_payloads(self.origin, self.migration, self.transition)
        self.assertTrue(result["parameters_optimizer_rng_exact"])

    def test_rehashed_changed_parameter_is_not_a_valid_migration(self):
        self.rewrite_new(lambda p: next(iter(p["parameters"].values())).add_(1))
        with self.assertRaisesRegex(ValueError, "parameters"):
            acceptance._migration_payloads(self.origin, self.migration, self.transition)

    def test_rehashed_changed_optimizer_moment_is_rejected(self):
        self.rewrite_new(lambda p: p["optimizer"]["state"][0]["exp_avg"].add_(1))
        with self.assertRaisesRegex(ValueError, "optimizer"):
            acceptance._migration_payloads(self.origin, self.migration, self.transition)

    def test_rehashed_changed_rng_is_rejected(self):
        self.rewrite_new(lambda p: p["rng"]["torch_cpu"].bitwise_xor_(1))
        with self.assertRaisesRegex(ValueError, "rng"):
            acceptance._migration_payloads(self.origin, self.migration, self.transition)

    def test_rehashed_wrong_adam_step_is_rejected(self):
        self.rewrite_new(lambda p: p["optimizer"]["state"][0]["step"].add_(1))
        with self.assertRaisesRegex(ValueError, "Adam step"):
            acceptance._migration_payloads(self.origin, self.migration, self.transition)

    def test_cursor_order_or_offset_cannot_be_migrated(self):
        for field, changed in (("offset", 4112), ("order", list(reversed(self.new_cursor["order"])))):
            value = copy.deepcopy(self.transition)
            value["new_cursor"][field] = changed
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Migration changed"):
                    acceptance._migration_payloads(self.origin, self.migration, value)


class CombinedWorkTests(unittest.TestCase):
    def setUp(self):
        self.completion = {"rows_processed": 13139, "optimizer_updates": 822,
            "historical_optimizer_updates": 256, "segment_optimizer_updates": 566,
            "historical_rows": 4096, "segment_rows": 9043, "synthetic_rows": 0,
            "data_pass_complete": True, "protocol_complete": True, "full_accepted_dataset_pass": True,
            "scope": "full_accepted_real_training_partition", "stop_reason": "data_exhausted",
            "historical_response_tokens": 8192, "segment_response_tokens": 18086, "response_tokens": 26278}
        self.budget = {"started_at_epoch": 10000., "wall_seconds": 28800, "deadline_epoch": 38800.}
        self.transition = {"old_work": {"deadline_epoch": 9000., "lr": 5e-5},
                           "new_work": {"deadline_epoch": 38800., "lr": 5e-5},
                           "old_cursor": {"response_tokens": 8192}}

    def test_exact_total_and_segment_denominators(self):
        acceptance._combined_work(self.completion, self.transition, self.budget, 20000.)

    def test_partial_or_duplicated_history_is_rejected(self):
        for key, value in (("segment_optimizer_updates", 822), ("historical_optimizer_updates", 0),
                           ("rows_processed", 9043), ("segment_rows", 13139),
                           ("protocol_complete", False), ("response_tokens", 26279)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    acceptance._combined_work({**self.completion, key: value}, self.transition, self.budget, 20000.)

    def test_new_budget_cannot_reset_or_overrun(self):
        for budget, end in (({**self.budget, "wall_seconds": 43200}, 20000.),
                            ({**self.budget, "deadline_epoch": 50000.}, 20000.),
                            (self.budget, 38800.001), (self.budget, float("nan"))):
            with self.assertRaises(ValueError):
                acceptance._combined_work(self.completion, self.transition, budget, end)

    def test_training_hyperparameters_cannot_change_with_deadline(self):
        transition = copy.deepcopy(self.transition)
        transition["new_work"]["lr"] = 1e-4
        with self.assertRaisesRegex(ValueError, "training protocol"):
            acceptance._combined_work(self.completion, transition, self.budget, 20000.)

    def test_process_exit_distinguishes_pid_reuse_from_identity_drift(self):
        process = {"pid": 1234, "parent_pid": 1000, "created_at_utc": "2026-10-20T10:00:00+00:00",
                   "command_line": "python recovery.py --request known"}
        self.assertEqual(acceptance._process_history([process], [])[0]["state"], "exited")
        reused = {**process, "created_at_utc": "2026-10-21T10:00:00+00:00", "command_line": "python unrelated.py"}
        self.assertEqual(acceptance._process_history([process], [reused])[0]["state"], "pid_reused_original_exited")
        for actual in (process, {**process, "command_line": None}):
            with self.assertRaises(ValueError):
                acceptance._process_history([process], [actual])


class HistoricalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.original_path = str(root / "old-priority.json")
        self.snapshot = root / "priority.snapshot"
        self.snapshot.write_text(json.dumps({"vla": {"paused": True}}), encoding="utf-8")
        saved = acceptance._file(self.snapshot)
        self.original = {"prerequisites": [{"path": self.original_path, "sha256": saved["sha256"],
                                            "json_equals": {"vla.paused": True}}]}
        self.recovery = {"historical_prerequisites": {self.original_path: {
            "original_path": self.original_path, "original_sha256": saved["sha256"],
            "snapshot": saved, "semantics": "historical_admission_fact_not_current_resource_state"}}}

    def test_changed_live_priority_is_not_reinterpreted_as_old_admission(self):
        Path(self.original_path).write_text('{"vla":{"paused":false}}', encoding="utf-8")
        evidence = {}
        acceptance._historical_admission(self.original, self.recovery, evidence)
        self.assertEqual(set(evidence), {str(self.snapshot)})
        self.assertNotIn(self.original_path, evidence)

    def test_missing_snapshot_or_false_size_is_rejected(self):
        missing = {"historical_prerequisites": {}}
        wrong_size = copy.deepcopy(self.recovery)
        wrong_size["historical_prerequisites"][self.original_path]["snapshot"]["size_bytes"] += 1
        for recovery in (missing, wrong_size):
            with self.assertRaises(ValueError):
                acceptance._historical_admission(self.original, recovery, {})

    def test_snapshot_tamper_and_wrong_json_field_are_rejected(self):
        original = copy.deepcopy(self.original)
        original["prerequisites"][0]["json_equals"]["vla.paused"] = False
        with self.assertRaisesRegex(ValueError, "JSON field differs"):
            acceptance._historical_admission(original, self.recovery, {})
        self.snapshot.write_text('{"vla":{"paused":false}}', encoding="utf-8")
        with self.assertRaises(ValueError):
            acceptance._historical_admission(self.original, self.recovery, {})


if __name__ == "__main__":
    unittest.main()
