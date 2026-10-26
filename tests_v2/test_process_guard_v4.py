"""Deadline-only waiting never treats unknown Python as admitted GPU work."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from alignment_lab import process_guard_v4 as guard
from alignment_lab import alignment_process_guard_r2 as adapter


OWN = {"pid": 1, "parent_pid": 10, "created_at_utc": "2026-10-22T05:00:00.1234567Z",
       "command_line": "python scripts/run_alignlab_alignment_stage_r2.py --request frozen --execute"}
UNKNOWN = {"pid": 2, "parent_pid": 20, "created_at_utc": "2026-10-22T05:00:01.1234567Z",
           "command_line": "python -m pip install --target .runtime Pillow"}


class Scenario:
    def __init__(self, clear_after=0, snapshot=None):
        self.now, self.reads, self.sleeps = 100.0, 0, []
        self.clear_after, self.snapshot = clear_after, snapshot

    def read(self):
        self.reads += 1
        if self.snapshot is not None:
            if isinstance(self.snapshot, Exception):
                raise self.snapshot
            return copy.deepcopy(self.snapshot)
        return copy.deepcopy([OWN, UNKNOWN] if self.now < 100 + self.clear_after else [OWN])

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def run(self, **kwargs):
        values = dict(owned_identities=[OWN], deadline_epoch=5000,
            budget=guard.WaitBudget(300.0664498806), clock=lambda: self.now, sleep=self.sleep)
        values.update(kwargs)
        return guard.wait_for_clearance(self.read, **values)


class ProcessGuardV4Tests(unittest.TestCase):
    def test_unknown_installation_exceeds_both_old_limits_then_clears(self):
        scenario, saved, seen = Scenario(clear_after=1210), [], []
        result = scenario.run(on_wait_start=lambda: saved.append(scenario.now),
            on_observation=lambda row: seen.append(copy.deepcopy(row)))
        self.assertTrue(result.admission_allowed)
        self.assertEqual(result.waited_seconds, 1210)
        self.assertAlmostEqual(result.total_waited_seconds, 1510.0664498806)
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(seen), 122)
        self.assertTrue(all(row["guard"]["admission_allowed"] is False for row in seen[:-1]))
        self.assertTrue(seen[-1]["guard"]["admission_allowed"])
        self.assertFalse(result.to_dict()["unknown_processes_allowed"])
        self.assertFalse(result.to_dict()["training_during_wait"])

    def test_persistent_unknown_stops_at_original_deadline_without_new_snapshot(self):
        scenario = Scenario(clear_after=10000)
        result = scenario.run(deadline_epoch=1421)
        self.assertFalse(result.admission_allowed)
        self.assertEqual(result.reason, "deadline_exhausted")
        self.assertEqual(scenario.now, 1421)
        self.assertEqual(result.waited_seconds, 1321)
        self.assertEqual(scenario.sleeps[-1], 1)

    def test_shared_ledger_carries_more_than_old_900_across_calls(self):
        budget = guard.WaitBudget(890)
        first, second = Scenario(clear_after=600), Scenario(clear_after=400)
        self.assertTrue(first.run(budget=budget).admission_allowed)
        self.assertTrue(second.run(budget=budget).admission_allowed)
        self.assertEqual(budget.spent_seconds, 1890)

    def test_clean_snapshot_has_no_wait_or_save(self):
        scenario = Scenario()
        result = scenario.run(on_wait_start=lambda: self.fail("Clean boundary cannot wait"))
        self.assertTrue(result.admission_allowed)
        self.assertEqual((scenario.reads, scenario.sleeps, result.waited_seconds), (1, [], 0))

    def test_expired_deadline_blocks_even_clean_snapshot_before_query(self):
        scenario = Scenario()
        result = scenario.run(deadline_epoch=100)
        self.assertFalse(result.admission_allowed)
        self.assertEqual(scenario.reads, 0)

    def test_hard_gpu_training_conflict_never_waits(self):
        competitor = {**UNKNOWN, "command_line": "python -m windows.train --updates 2000"}
        scenario = Scenario(snapshot=[OWN, competitor])
        result = scenario.run(on_wait_start=lambda: self.fail("Hard conflict must not save"))
        self.assertFalse(result.admission_allowed)
        self.assertEqual(result.reason, "hard_blocker")
        self.assertEqual(scenario.sleeps, [])

    def test_unknown_then_hard_conflict_cannot_release_boundary(self):
        scenario = Scenario(clear_after=1000)
        original = scenario.sleep
        def sleep(seconds):
            original(seconds)
            scenario.snapshot = [OWN, {**UNKNOWN, "command_line": "python scripts/run_campaign.py --execute"}]
        result = scenario.run(sleep=sleep)
        self.assertFalse(result.admission_allowed)
        self.assertEqual(result.reason, "hard_blocker")
        self.assertEqual(scenario.sleeps, [10])

    def test_stop_or_source_change_during_wait_never_reads_again(self):
        for field in ("stop_requested", "identity_unchanged"):
            scenario = Scenario(clear_after=1000)
            callback = (lambda: scenario.now >= 110) if field == "stop_requested" else (lambda: scenario.now < 110)
            result = scenario.run(**{field: callback})
            self.assertFalse(result.admission_allowed)
            self.assertEqual(scenario.reads, 2)
            self.assertEqual(scenario.sleeps, [10])

    def test_owned_and_allowed_ancestry_pid_reuse_and_command_are_strict(self):
        for changed in (dict(OWN, parent_pid=9), dict(OWN, command_line=None),
                        dict(OWN, created_at_utc="2026-10-22T05:00:00.1234568Z")):
            scenario = Scenario(snapshot=[changed])
            self.assertFalse(scenario.run().admission_allowed)
            self.assertEqual(scenario.sleeps, [])
        scenario = Scenario(snapshot=[OWN, dict(UNKNOWN, parent_pid=99)])
        result = scenario.run(allowed_identities=[UNKNOWN])
        self.assertFalse(result.admission_allowed)
        self.assertEqual(result.reason, "allowed_identity_changed")
        self.assertEqual(scenario.sleeps, [])

    def test_unavailable_snapshot_waits_but_does_not_assume_empty(self):
        scenario = Scenario(snapshot=OSError("snapshot unavailable"))
        sleep = scenario.sleep
        def clear(seconds):
            sleep(seconds)
            scenario.snapshot = [OWN]
        result = scenario.run(sleep=clear)
        self.assertTrue(result.admission_allowed)
        self.assertEqual((scenario.reads, scenario.sleeps), (3, [10]))

    def test_failed_save_or_failed_evidence_callback_is_closed(self):
        for name in ("on_wait_start", "on_observation"):
            scenario = Scenario(clear_after=1000)
            def fail(*args):
                raise OSError("cannot persist evidence")
            result = scenario.run(**{name: fail})
            self.assertFalse(result.admission_allowed)
            self.assertEqual(result.reason, "callback_failed")
            self.assertEqual(scenario.sleeps, [])

    def test_clock_stall_or_backwards_fails_without_busy_loop(self):
        for delta in (0, -1):
            scenario = Scenario(clear_after=1000)
            def sleep(seconds):
                scenario.now += delta
            result = scenario.run(sleep=sleep)
            self.assertFalse(result.admission_allowed)
            self.assertIn(result.reason, ("clock_not_advancing", "clock_moved_backwards"))
            self.assertLessEqual(scenario.reads, 2)

    def test_slow_clear_query_after_deadline_does_not_admit(self):
        scenario = Scenario()
        def read():
            scenario.now = 5010
            return [OWN]
        result = guard.wait_for_clearance(read, owned_identities=[OWN], deadline_epoch=5000,
            budget=guard.WaitBudget(), clock=lambda: scenario.now, sleep=scenario.sleep)
        self.assertFalse(result.admission_allowed)
        self.assertEqual(result.reason, "deadline_exhausted")

    def test_adapter_frozen_policy_and_strict_initial_admission(self):
        adapter.validate_policy(copy.deepcopy(adapter.POLICY))
        for change in ({"mode": "allow_pip"}, {"max_wait_seconds": 300},
                       {"max_total_wait_seconds": 900}, {"poll_interval_seconds": 10.0}):
            with self.assertRaises(ValueError):
                adapter.validate_policy({**adapter.POLICY, **change})
        scenario = Scenario(clear_after=1000)
        result = adapter.strict_admission(scenario.read, owned_identities=[OWN], deadline_epoch=5000,
            clock=lambda: scenario.now, sleep=scenario.sleep)
        self.assertFalse(result["caller_admission_allowed"])
        self.assertEqual(scenario.sleeps, [])

    def test_real_cpu_adam_remains_unchanged_through_long_wait(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        parameter = torch.nn.Parameter(torch.tensor([0.25], device="cpu"))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        def step():
            optimizer.zero_grad(set_to_none=True)
            (parameter - 1).square().sum().backward()
            optimizer.step()
        step()
        saved = parameter.detach().clone()
        adam = copy.deepcopy(optimizer.state_dict())
        scenario, boundaries = Scenario(clear_after=1210), []
        reader, sleeper = scenario.read, scenario.sleep
        def unchanged():
            self.assertTrue(torch.equal(parameter.detach(), saved))
            actual = optimizer.state_dict()
            self.assertEqual(actual["param_groups"], adam["param_groups"])
            for key, value in adam["state"][0].items():
                self.assertTrue(torch.equal(actual["state"][0][key], value))
        def read():
            unchanged()
            return reader()
        def sleep(seconds):
            self.assertEqual(boundaries, [1])
            unchanged()
            sleeper(seconds)
        def save():
            unchanged()
            boundaries.append(1)
        result = guard.wait_for_clearance(read, owned_identities=[OWN], deadline_epoch=5000,
            budget=guard.WaitBudget(300.0664498806), clock=lambda: scenario.now, sleep=sleep,
            on_wait_start=save)
        self.assertTrue(result.admission_allowed)
        unchanged()
        step()
        self.assertEqual(optimizer.state_dict()["state"][0]["step"].item(), 2)
        self.assertFalse(torch.equal(parameter.detach(), saved))
        self.assertFalse(torch.cuda.is_initialized())

    def test_adapter_long_wait_checkpoint_and_final_ledger_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            manifest = checkpoint / "manifest.json"
            identity = {"test": "real complete boundary"}
            manifest.write_text(json.dumps({"status": "complete", "identity": identity}), encoding="utf-8")
            receipt = {"complete": True, "optimizer_updates": 1, "kind": "optimizer_checkpoint",
                "path": str(manifest.resolve()), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            scenario, events = Scenario(), []
            session = adapter.GuardSession(scenario.read, owned_identities=[OWN], deadline_epoch=5000,
                policy=adapter.POLICY, spent_seconds=300.0664498806,
                clock=lambda: scenario.now, sleep=scenario.sleep,
                on_event=lambda event: events.append(copy.deepcopy(event)))
            for boundary, update in (("before_cuda", 0), ("engine_origin", 0),
                                     ("optimizer_complete", 1), ("engine_final", 1)):
                scenario.clear_after = 1210 if boundary == "optimizer_complete" else 0
                outcome = session.check(boundary=boundary, optimizer_updates=update,
                    save_boundary=lambda: copy.deepcopy(receipt))
                self.assertTrue(outcome["admission_allowed"])
            log = root / "process-guard.jsonl"
            log.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
            request = {"output": str(root), "stage": "ppo", "deadline_epoch": 5000,
                "process_wait_policy": copy.deepcopy(adapter.POLICY), "process_wait_spent_seconds": 300.0664498806,
                "supervisor_processes": [], "allowed_background_processes": [], "workload": {"optimizer_updates": 1}}
            progress = {"started_at_epoch": 100, "identity": identity,
                "checkpoints": [{"checkpoint_path": str(checkpoint), "manifest_sha256": receipt["sha256"], "optimizer_updates": 1}],
                "process_guard": {"owned_identities": [OWN], "last": session.last,
                    "total_waited_seconds": session.budget.spent_seconds}}
            result = adapter.verify_guard_log(log, request=request, progress=progress)
            self.assertTrue(result["verified"])
            self.assertEqual(result["segment_waited_seconds"], 1210)
            self.assertEqual(result["boundary_calls"], 4)
            # Coherently rewriting the final self-reported totals cannot erase elapsed evidence.
            events[-1]["total_waited_seconds"] -= 100
            progress["process_guard"]["last"] = events[-1]
            progress["process_guard"]["total_waited_seconds"] -= 100
            log.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
            with self.assertRaises(ValueError):
                adapter.verify_guard_log(log, request=request, progress=progress)


if __name__ == "__main__":
    unittest.main()
