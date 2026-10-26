"""Independent acceptance tests; run only after all real RM processes exit.

These tests use tiny local CPU tensors or recorded process snapshots. They do
not load the research model, execute a recovery, launch a process or use CUDA.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from alignment_lab import recovered_parent_acceptance_r5 as acceptance
from alignment_lab.process_guard_v2 import inspect_processes


OWNED = {"pid": 101, "parent_pid": 100, "created_at_utc": "2026-10-21T08:00:00.1234567Z",
         "command_line": "python run_alignlab_parent_recovery_r5.py --request known --execute"}
UNKNOWN = {"pid": 202, "parent_pid": 200, "created_at_utc": "2026-10-21T08:00:01.1234567Z",
           "command_line": "python -m pip install --target .runtime Pillow"}
BUDGET = {"started_at_epoch": 100.0, "wall_seconds": 28800, "deadline_epoch": 28900.0}


def guard(snapshot, at):
    return inspect_processes(lambda: copy.deepcopy(snapshot), allowed_identities=[OWNED],
        max_attempts=2, retry_delay_seconds=0, deadline_epoch=BUDGET["deadline_epoch"],
        clock=lambda: at, sleep=lambda _: None).to_dict()


def row(update, cycle, at, snapshot, waited, total):
    return {"optimizer_updates": update, "cycle": cycle, "at_epoch": at,
            "guard": guard(snapshot, at), "waited_seconds": waited, "total_waited_seconds": total}


def fixture(waiting=True):
    rows = [row(4, 1, 200.0, [OWNED], 0.0, 0.0)]
    if waiting:
        rows.extend((row(5, 1, 201.0, [OWNED, UNKNOWN], 0.0, 0.0),
                     row(5, 2, 203.0, [OWNED], 2.0, 2.0),
                     row(5, 1, 204.0, [OWNED], 0.0, 2.5)))
        total = 2.5
    else:
        rows.extend((row(5, 1, 201.0, [OWNED], 0.0, 0.0), row(5, 1, 202.0, [OWNED], 0.0, 0.0)))
        total = 0.0
    observation = {k: v for k, v in rows[-1].items() if k != "optimizer_updates"}
    progress = {"process_identity": [copy.deepcopy(OWNED)], "total_process_wait_seconds": total,
        "process_guard_blocked": False, "last_process_wait_result": {
            "schema_version": 1, "kind": "alignlab-process-guard-v3-boundary-wait",
            "admission_allowed": True, "reason": "cleared", "waited_seconds": 0.0,
            "total_waited_seconds": total, "observations": [observation],
            "last_guard": copy.deepcopy(rows[-1]["guard"]), "unknown_processes_allowed": False,
            "training_during_wait": False}}
    plan = {"process_wait_policy": copy.deepcopy(acceptance.WAIT_POLICY), "allowed_background_processes": []}
    return rows, progress, plan


class WaitEvidenceTests(unittest.TestCase):
    def check(self, rows, progress, plan, checkpoints=(4, 5)):
        # Actual acceptance supplies [684,685,...,822,822], not this tiny order.
        return acceptance._validate_wait_log(rows, progress, plan, BUDGET, [4, 5, 5], set(checkpoints))

    def test_clean_and_unknown_then_clear_have_exact_denominators(self):
        for waiting, count, total in ((False, 0, 0.0), (True, 1, 2.5)):
            with self.subTest(waiting=waiting):
                result = self.check(*fixture(waiting))
                self.assertEqual(result["boundaries"], 3)
                self.assertEqual(result["waiting_boundaries"], count)
                self.assertEqual(result["total_waited_seconds"], total)
                self.assertFalse(result["unknown_processes_allowed"])

    def test_rehashed_unknown_snapshot_cannot_be_forged_as_clear(self):
        rows, progress, plan = fixture()
        rows[1]["guard"]["admission_allowed"] = True
        rows[1]["guard"]["requires_safe_stop"] = False
        rows[1]["guard"]["decision"] = "allow"
        rows[1]["guard"]["blockers"] = []
        for item in rows[1]["guard"]["observations"]:
            item["blockers"] = []
        with self.assertRaisesRegex(ValueError, "classified snapshot"):
            self.check(rows, progress, plan)

    def test_owned_pid_reuse_and_creation_precision_are_not_accepted(self):
        rows, progress, plan = fixture(False)
        reused = {**OWNED, "created_at_utc": "2026-10-21T08:00:00.1234568Z"}
        rows[0]["guard"] = guard([reused], 200.0)
        with self.assertRaisesRegex(ValueError, "owned process"):
            self.check(rows, progress, plan)

    def test_known_competing_training_is_not_a_waitable_unknown(self):
        rows, progress, plan = fixture()
        competing = {**UNKNOWN, "command_line": "python windows.train --updates 2000"}
        rows[1]["guard"] = guard([OWNED, competing], 201.0)
        with self.assertRaisesRegex(ValueError, "hard process conflict"):
            self.check(rows, progress, plan)

    def test_wait_needs_current_checkpoint_and_later_clear_snapshot(self):
        rows, progress, plan = fixture()
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.check(rows, progress, plan, checkpoints=(4,))
        rows.pop(2)
        with self.assertRaises(ValueError):
            self.check(rows, progress, plan)

    def test_missing_duplicate_or_reordered_optimizer_guards_are_rejected(self):
        rows, progress, plan = fixture(False)
        variants = [rows[1:], [*rows, copy.deepcopy(rows[-1])], [rows[1], rows[0], rows[2]]]
        for changed in variants:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    self.check(changed, progress, plan)

    def test_wait_accounting_cannot_reset_hide_time_or_increase_limits(self):
        for mutation in (lambda r, p, c: r[-1].update(total_waited_seconds=0.0),
                         lambda r, p, c: r[2].update(waited_seconds=1.0, total_waited_seconds=1.0),
                         lambda r, p, c: p.update(total_process_wait_seconds=900.0),
                         lambda r, p, c: c["process_wait_policy"].update(max_wait_seconds=600)):
            rows, progress, plan = fixture()
            mutation(rows, progress, plan)
            with self.assertRaises(ValueError):
                self.check(rows, progress, plan)

    def test_own_identity_or_source_failure_cannot_be_labeled_completed(self):
        for key, value in (("hard_guard_stop", True), ("identity_guard_error", "source drift"),
                           ("process_guard_blocked", True)):
            rows, progress, plan = fixture(False)
            progress[key] = value
            with self.assertRaisesRegex(ValueError, "Final process clearance"):
                self.check(rows, progress, plan)

    def test_final_receipt_needs_exact_recorded_observations(self):
        rows, progress, plan = fixture()
        progress["last_process_wait_result"]["observations"] = []
        with self.assertRaisesRegex(ValueError, "Final process clearance"):
            self.check(rows, progress, plan)

    def test_guard_time_must_remain_inside_absolute_budget(self):
        rows, progress, plan = fixture(False)
        rows[-1]["at_epoch"] = BUDGET["deadline_epoch"]
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.check(rows, progress, plan)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.completion = {"optimizer_updates": 822, "rows_processed": 13139,
            "historical_optimizer_updates": 684, "segment_optimizer_updates": 138,
            "historical_rows": 10944, "segment_rows": 2195, "synthetic_rows": 0,
            "data_pass_complete": True, "protocol_complete": True, "full_accepted_dataset_pass": True,
            "scope": "full_accepted_real_training_partition", "stop_reason": "data_exhausted",
            "response_tokens": 26278, "historical_response_tokens": 21888, "segment_response_tokens": 4390}
        work = {"deadline_epoch": BUDGET["deadline_epoch"], "lr": 5e-5}
        cursor = {"optimizer_updates": 684, "offset": 10944, "response_tokens": 21888,
                  "config_sha256": "a" * 64, "data_sha256": "b" * 64, "order": list(range(13139))}
        self.transition = {"kind": "alignlab-same-budget-continuation-v1", "changed_cursor_fields": [],
            "changed_work_fields": [], "parameter_optimizer_rng_change_permitted": False,
            "old_work": copy.deepcopy(work), "new_work": copy.deepcopy(work),
            "old_cursor": copy.deepcopy(cursor), "new_cursor": copy.deepcopy(cursor)}

    def test_full_exact_r5_work_and_original_absolute_deadline(self):
        acceptance._combined_work(self.completion, self.transition, BUDGET, 1000.0)
        self.assertEqual(sum(end - begin for _, begin, end in acceptance.CHAIN) + 256, 822)

    def test_paused_partial_duplicate_or_original_r2_scope_is_rejected(self):
        for key, value in (("optimizer_updates", 684), ("segment_optimizer_updates", 566),
                           ("historical_rows", 4096), ("rows_processed", 2195),
                           ("protocol_complete", False), ("response_tokens", 26279)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    acceptance._combined_work({**self.completion, key: value}, self.transition, BUDGET, 1000.0)

    def test_no_budget_reset_or_hyperparameter_or_cursor_transition(self):
        for mutation in (lambda t: t["new_work"].update(lr=1e-4),
                         lambda t: t["new_work"].update(deadline_epoch=50000),
                         lambda t: t["new_cursor"].update(offset=10960),
                         lambda t: t.update(changed_cursor_fields=["config_sha256"])):
            changed = copy.deepcopy(self.transition)
            mutation(changed)
            with self.assertRaises(ValueError):
                acceptance._combined_work(self.completion, changed, BUDGET, 1000.0)
        for budget, finished in (({**BUDGET, "wall_seconds": 43200}, 1000.0),
                                  (BUDGET, 28900.001), (BUDGET, float("nan"))):
            with self.assertRaises(ValueError):
                acceptance._combined_work(self.completion, self.transition, budget, finished)


class NoopPayloadTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tests_v2.test_parent_acceptance import tensor_bundle
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        cursor = {"schema": 1, "optimizer_updates": 684, "offset": 10944,
            "response_tokens": 21888, "completed_rollouts": 0, "active_rollout": False,
            "order": list(range(13139)), "data_sha256": "a" * 64, "config_sha256": "b" * 64}
        self.transition = {"old_cursor": copy.deepcopy(cursor), "new_cursor": copy.deepcopy(cursor),
            "old_work": {"deadline_epoch": 28900.0}, "new_work": {"deadline_epoch": 28900.0},
            "changed_cursor_fields": [], "changed_work_fields": []}
        pins = []
        for name in ("source684", "origin684"):
            directory = root / name
            directory.mkdir()
            manifest = {"format": "alignlab-optimizer-checkpoint-v1", "status": "complete",
                "identity": {"config": {"stage": "reward"}}, "artifacts": {},
                "parameter_specs": [{"name": "base.layer.lora_A.reward.weight", "shape": [1], "dtype": "torch.float32"},
                                    {"name": "scalar_heads.reward.weight", "shape": [1], "dtype": "torch.float32"}],
                "optimizer": {"class": "torch.optim.adamw.AdamW", "groups": []},
                "generator_specs": {}, "software": {"fixture": "cpu-only-tensor-comparison"}}
            digest = tensor_bundle(directory, manifest, stage="reward", cursor=cursor)
            pins.append({"path": str(directory), "manifest_sha256": digest})
        self.source = pins[0]
        self.migrated = {"checkpoint_path": pins[1]["path"], "manifest_sha256": pins[1]["manifest_sha256"]}
        self.assertFalse(torch.cuda.is_initialized())

    def rewrite(self, mutate):
        import torch
        from alignment_lab.alignment_supervisor import _cpu_payload
        directory = Path(self.migrated["checkpoint_path"])
        manifest = json.loads((directory / "manifest.json").read_text())
        payload = _cpu_payload(directory / "state.pt", manifest["artifacts"]["state.pt"]["sha256"])
        mutate(payload)
        torch.save(payload, directory / "state.pt")
        pin = acceptance.base._file(directory / "state.pt")
        manifest["artifacts"]["state.pt"] = {k: pin[k] for k in ("sha256", "size_bytes")}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.migrated["manifest_sha256"] = acceptance.base._file(directory / "manifest.json")["sha256"]

    def test_complete_noop_parameters_adam_rng_cursor(self):
        result = acceptance._continuation_payloads(self.source, self.migrated, self.transition)
        self.assertTrue(result["parameters_optimizer_rng_cursor_exact"])

    def test_changed_adam_rng_or_weights_rejected_even_after_rehash(self):
        mutations = (lambda p: next(iter(p["parameters"].values())).add_(1),
                     lambda p: p["optimizer"]["state"][0]["exp_avg"].add_(1),
                     lambda p: p["rng"]["torch_cpu"].bitwise_xor_(1))
        # Each mutation persists; every resulting difference must remain invalid.
        for mutate in mutations:
            self.rewrite(mutate)
            with self.assertRaisesRegex(ValueError, "Continuation changed"):
                acceptance._continuation_payloads(self.source, self.migrated, self.transition)

    def test_wrong_adam_step_or_nonfinite_parameter_rejected(self):
        self.rewrite(lambda p: p["optimizer"]["state"][0]["step"].add_(1))
        with self.assertRaisesRegex(ValueError, "Adam step"):
            acceptance._continuation_payloads(self.source, self.migrated, self.transition)


if __name__ == "__main__":
    unittest.main()
