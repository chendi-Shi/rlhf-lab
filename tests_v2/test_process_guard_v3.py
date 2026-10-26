"""Deterministic boundary-wait guards and a real tiny CPU optimizer integration.

The guard remains a strict process classifier. Waiting cannot turn a live
unknown process into an allowed one and never advances an optimizer itself.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from alignment_lab.process_guard_v3 import WaitPolicy, WaitBudget, wait_for_clearance


def process(pid=1, command="python scripts/run_alignlab_recovery.py --execute", *, parent=10,
            created="2026-10-21T08:00:00.0000001Z"):
    return {"pid": pid, "parent_pid": parent, "created_at_utc": created, "command_line": command}


OWN = process()
UNKNOWN = process(2, "python -m pip install --target .runtime Pillow")


class Scenario:
    def __init__(self, snapshots):
        self.snapshots, self.calls, self.now, self.sleeps = snapshots, 0, 100.0, []

    def read(self):
        item = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def run(self, **kwargs):
        defaults = dict(owned_identities=[OWN], deadline_epoch=1000.0, budget=WaitBudget(),
                        clock=lambda: self.now, sleep=self.sleep)
        defaults.update(kwargs)
        return wait_for_clearance(self.read, **defaults)


class ProcessGuardV3Tests(unittest.TestCase):
    def test_clean_exact_owned_identity_admits_without_sleep_or_save(self):
        case, saves = Scenario([[OWN]]), []
        result = case.run(on_wait_start=lambda *args: saves.append(args))
        self.assertTrue(result.admission_allowed)
        self.assertEqual(case.calls, 1)
        self.assertEqual(case.sleeps, [])
        self.assertEqual(saves, [])
        self.assertEqual(result.waited_seconds, 0)

    def test_unknown_pip_must_disappear_and_new_full_snapshot_is_required(self):
        case = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
        budget, saves, observations = WaitBudget(), [], []
        result = case.run(budget=budget, on_wait_start=lambda *args: saves.append(args),
                          on_observation=lambda *args: observations.append(args))
        self.assertTrue(result.admission_allowed)
        self.assertEqual(case.calls, 3)
        self.assertEqual(case.sleeps, [2.0])
        self.assertEqual(len(saves), 1)
        self.assertGreaterEqual(len(observations), 2)
        self.assertEqual(result.waited_seconds, 2.0)
        self.assertEqual(budget.spent_seconds, 2.0)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_persistent_unknown_hits_local_limit_without_harmless_cpu_assumption(self):
        case = Scenario([[OWN, UNKNOWN]])
        budget = WaitBudget()
        result = case.run(budget=budget, policy=WaitPolicy(max_wait_seconds=3.0, total_wait_seconds=5.0, poll_seconds=2.0))
        self.assertFalse(result.admission_allowed)
        self.assertEqual(sum(case.sleeps), 3.0)
        self.assertEqual(budget.spent_seconds, 3.0)
        self.assertLessEqual(case.calls, 6)

    def test_wait_budget_is_cumulative_across_optimizer_boundaries(self):
        policy = WaitPolicy(max_wait_seconds=3.0, total_wait_seconds=5.0, poll_seconds=2.0)
        budget = WaitBudget()
        first = Scenario([[OWN, UNKNOWN]])
        self.assertFalse(first.run(budget=budget, policy=policy).admission_allowed)
        second = Scenario([[OWN, UNKNOWN]])
        self.assertFalse(second.run(budget=budget, policy=policy).admission_allowed)
        self.assertEqual(sum(first.sleeps) + sum(second.sleeps), 5.0)
        self.assertEqual(budget.spent_seconds, 5.0)
        # Exhausting the wait allocation cannot forbid an already clear snapshot,
        # but it must forbid any further wait when an unknown process remains.
        self.assertTrue(Scenario([[OWN]]).run(budget=budget, policy=policy).admission_allowed)
        third = Scenario([[OWN, UNKNOWN]])
        self.assertFalse(third.run(budget=budget, policy=policy).admission_allowed)
        self.assertEqual(third.sleeps, [])

    def test_deadline_before_and_during_wait_never_reads_or_updates_after_expiry(self):
        case = Scenario([[OWN]])
        self.assertFalse(case.run(deadline_epoch=100.0).admission_allowed)
        self.assertEqual(case.calls, 0)
        case = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
        result = case.run(deadline_epoch=101.0)
        self.assertFalse(result.admission_allowed)
        self.assertEqual(case.calls, 2)
        self.assertEqual(case.sleeps, [1.0])
        self.assertEqual(case.now, 101.0)

    def test_slow_clear_query_crossing_absolute_deadline_never_admits(self):
        case = Scenario([[OWN]])
        def slow_read():
            case.now += 3
            return [OWN]
        result = wait_for_clearance(slow_read, owned_identities=[OWN], deadline_epoch=102.0,
            budget=WaitBudget(), clock=lambda: case.now, sleep=case.sleep)
        self.assertFalse(result.admission_allowed)
        self.assertEqual(case.sleeps, [])

    def test_stop_and_frozen_identity_drift_interrupt_wait_before_next_read(self):
        for key in ("stop_requested", "identity_unchanged"):
            with self.subTest(key=key):
                case = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
                condition = (lambda: case.now >= 102.0) if key == "stop_requested" else (lambda: case.now < 102.0)
                result = case.run(**{key: condition})
                self.assertFalse(result.admission_allowed)
                self.assertEqual(case.calls, 2)
                self.assertEqual(case.sleeps, [2.0])

    def test_stop_or_source_drift_at_entry_does_not_query(self):
        for kwargs in ({"stop_requested": lambda: True}, {"identity_unchanged": lambda: False}):
            case = Scenario([[OWN]])
            self.assertFalse(case.run(**kwargs).admission_allowed)
            self.assertEqual(case.calls, 0)

    def test_real_competing_training_or_allowed_command_drift_never_waits(self):
        competing = process(3, "python -m windows.train --updates 2000")
        background = process(4, "python -m uvicorn app.api:app")
        changed = dict(background, command_line=background["command_line"] + " --different")
        for snapshot, allowed in (([OWN, competing], []), ([OWN, changed], [background])):
            with self.subTest(snapshot=snapshot):
                case, saves = Scenario([snapshot]), []
                result = case.run(allowed_identities=allowed, on_wait_start=lambda *args: saves.append(args))
                self.assertFalse(result.admission_allowed)
                self.assertEqual(case.calls, 1)
                self.assertEqual(case.sleeps, [])
                self.assertEqual(saves, [])

    def test_owned_identity_missing_pid_reused_parent_or_command_changed_never_waits(self):
        changes = [[], [dict(OWN, parent_pid=11)], [dict(OWN, command_line=OWN["command_line"] + " --other")],
                   [dict(OWN, created_at_utc="2026-10-21T08:00:00.0000002Z")], [dict(OWN, command_line=None)]]
        for snapshot in changes:
            with self.subTest(snapshot=snapshot):
                case = Scenario([snapshot])
                self.assertFalse(case.run().admission_allowed)
                self.assertEqual(case.sleeps, [])

    def test_unknown_disappearing_into_known_competitor_does_not_allow_training(self):
        competing = process(3, "python scripts/run_campaign.py --execute")
        case = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN, competing]])
        self.assertFalse(case.run().admission_allowed)
        self.assertEqual(case.calls, 3)
        self.assertEqual(case.sleeps, [2.0])

    def test_snapshot_failure_waits_for_real_clean_snapshot_instead_of_empty_assumption(self):
        case = Scenario([OSError("CIM temporarily unavailable"), OSError("CIM unavailable"), [OWN]])
        self.assertTrue(case.run().admission_allowed)
        self.assertEqual(case.calls, 3)
        self.assertEqual(case.sleeps, [2.0])

    def test_save_callback_failure_never_continues_or_sleeps(self):
        case = Scenario([[OWN, UNKNOWN]])
        def failed_save(*args):
            raise OSError("cannot commit boundary checkpoint")
        self.assertFalse(case.run(on_wait_start=failed_save).admission_allowed)
        self.assertEqual(case.sleeps, [])

    def test_clock_not_advancing_has_a_hard_iteration_bound(self):
        case = Scenario([[OWN, UNKNOWN]])
        case.sleep = lambda seconds: case.sleeps.append(seconds)
        result = case.run(policy=WaitPolicy(max_wait_seconds=4.0, total_wait_seconds=4.0, poll_seconds=2.0))
        self.assertFalse(result.admission_allowed)
        self.assertLessEqual(case.calls, 20)

    def test_clock_moving_backwards_never_admits(self):
        case = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
        def backwards(seconds):
            case.sleeps.append(seconds)
            case.now -= seconds
        self.assertFalse(case.run(sleep=backwards).admission_allowed)
        self.assertEqual(case.calls, 2)

    def test_invalid_policy_budget_or_missing_owned_identity_rejects_before_query(self):
        bad = [lambda: Scenario([[OWN]]).run(policy=WaitPolicy(max_wait_seconds=-1)),
               lambda: Scenario([[OWN]]).run(policy=WaitPolicy(poll_seconds=0)),
               lambda: Scenario([[OWN]]).run(policy=WaitPolicy(total_wait_seconds=float("nan"))),
               lambda: Scenario([[OWN]]).run(owned_identities=[]),
               lambda: Scenario([[OWN]]).run(deadline_epoch=float("nan")),
               lambda: Scenario([[OWN]]).run(budget=WaitBudget(spent_seconds=-1))]
        for call in bad:
            with self.subTest(call=call), self.assertRaises((ValueError, TypeError)):
                call()

    def test_isolated_import_never_loads_gpu_frameworks(self):
        root = Path(__file__).resolve().parents[1]
        code = ("import json,sys; sys.path.insert(0," + repr(str(root)) + "); "
                "import alignment_lab.process_guard_v3; "
                "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] "
                "in {'torch','tensorflow','jax','flax','bitsandbytes'})))")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=root,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, capture_output=True, text=True, timeout=15, check=True)
        self.assertEqual(json.loads(result.stdout), [])


class BoundaryOptimizerIntegrationTests(unittest.TestCase):
    def test_real_cpu_optimizer_does_not_advance_while_waiting_or_after_conflict(self):
        import torch
        from alignment_lab import parent_recovery as r1
        from alignment_lab.stage_engine import run_supervised, parameter_mapping
        from test_stage_engine import examples, supervised
        from test_stage_artifacts import backend_for
        torch.set_num_threads(2)
        self.assertFalse(torch.cuda.is_available())
        self.assertFalse(torch.cuda.is_initialized())
        for mode in ("clears", "competing", "deadline"):
            with self.subTest(mode=mode):
                backend, _ = backend_for("reward")
                parameters = parameter_mapping(backend, "reward", heads=("reward",))
                optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3)
                case = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
                if mode == "competing":
                    case = Scenario([[OWN, process(3, "python -m windows.train")]])
                blocked, boundaries, checks = False, [], []
                def checkpoint(cursor, extra):
                    nonlocal blocked
                    boundaries.append(cursor["optimizer_updates"])
                    if cursor["optimizer_updates"] != 1:
                        return
                    saved_parameters = {n: p.detach().clone() for n, p in parameters.items()}
                    saved_optimizer = copy.deepcopy(optimizer.state_dict())
                    def assert_unchanged():
                        self.assertTrue(all(torch.equal(p.detach(), saved_parameters[n]) for n, p in parameters.items()))
                        self.assertTrue(r1._tree_equal(optimizer.state_dict(), saved_optimizer))
                        checks.append(True)
                    reader, sleeper = case.read, case.sleep
                    def read():
                        assert_unchanged()
                        return reader()
                    def sleep(seconds):
                        assert_unchanged()
                        sleeper(seconds)
                    result = wait_for_clearance(read, owned_identities=[OWN], deadline_epoch=101.0 if mode == "deadline" else 1000.0,
                        budget=WaitBudget(), clock=lambda: case.now, sleep=sleep,
                        on_wait_start=lambda *args: assert_unchanged())
                    assert_unchanged()
                    blocked = not result.admission_allowed
                outcome = run_supervised(backend, "reward", optimizer, examples(), supervised("reward"), device="cpu",
                    checkpoint_callback=checkpoint, stop_requested=lambda: blocked, clock=lambda: case.now)
                self.assertGreater(len(checks), 0)
                if mode == "clears":
                    self.assertEqual(outcome.stop_reason, "data_exhausted")
                    self.assertEqual(boundaries, [1, 2])
                else:
                    self.assertEqual(outcome.stop_reason, "paused")
                    self.assertEqual(boundaries, [1])
                self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
