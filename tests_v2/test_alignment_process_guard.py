"""Pure CPU/stdlib regression for branch admission and committed wait receipts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from alignment_lab import alignment_process_guard as guard


OWN = {"pid": 101, "parent_pid": 100, "created_at_utc": "2026-10-21T08:00:00.1234567Z",
       "command_line": "python scripts/run_alignlab_alignment_stage.py --request known --execute"}
UNKNOWN = {"pid": 202, "parent_pid": 200, "created_at_utc": "2026-10-21T08:00:01.1234567Z",
           "command_line": "python -m pip install --target .runtime Pillow"}


class Scenario:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = 0
        self.epoch = 100.0
        self.sleeps = []
        self.on_sleep = lambda: None

    def read(self):
        snapshot = self.snapshots[min(self.calls, len(self.snapshots)-1)]
        self.calls += 1
        if isinstance(snapshot, Exception):
            raise snapshot
        return copy.deepcopy(snapshot)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.epoch += seconds
        self.on_sleep()


class AlignmentProcessGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.saved = self.root / "committed-boundary.json"
        self.saved.write_bytes(b'{"optimizer_updates": 7, "complete": true}\n')
        self.receipt = {"complete": True, "optimizer_updates": 7, "kind": "optimizer_checkpoint",
            "path": str(self.saved), "sha256": hashlib.sha256(self.saved.read_bytes()).hexdigest()}
        self.events = []

    def session(self, scenario, **kwargs):
        options = dict(owned_identities=[OWN], allowed_identities=[], deadline_epoch=1000.0,
            policy=copy.deepcopy(guard.POLICY), clock=lambda: scenario.epoch, sleep=scenario.sleep,
            on_event=lambda event: self.events.append(copy.deepcopy(event)))
        options.update(kwargs)
        return guard.GuardSession(scenario.read, **options)

    def check(self, session, receipt=None):
        return session.check(boundary="optimizer", optimizer_updates=7,
            save_boundary=lambda: copy.deepcopy(self.receipt if receipt is None else receipt))

    def test_exact_policy_and_typed_background_identity(self):
        guard.validate_policy(copy.deepcopy(guard.POLICY))
        for change in ({"max_total_wait_seconds": 901}, {"max_wait_seconds": 300.0},
                       {"no_unknown_process_whitelist": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                guard.validate_policy({**guard.POLICY, **change})
        self.assertEqual(guard.validate_background([OWN]), [OWN])
        for rows in (None, [OWN, OWN], [{**OWN, "command_line": " "}], [{**OWN, "extra": True}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                guard.validate_background(rows)

    def test_strict_admission_needs_complete_owned_and_no_unknown(self):
        for snapshot, expected in (([OWN], True), ([OWN, UNKNOWN], False),
                ([{**OWN, "created_at_utc": "2026-10-21T08:00:00.1234568Z"}], False), ([], False)):
            scenario = Scenario([snapshot])
            result = guard.strict_admission(scenario.read, owned_identities=[OWN],
                deadline_epoch=1000, clock=lambda: scenario.epoch, sleep=scenario.sleep)
            self.assertEqual(result["caller_admission_allowed"], expected)
            self.assertEqual(scenario.sleeps, [])

    def test_unknown_wait_commits_verified_file_then_requires_fresh_clear_snapshot(self):
        scenario = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
        before = self.saved.read_bytes()
        scenario.on_sleep = lambda: self.assertEqual(self.saved.read_bytes(), before)
        result = self.check(self.session(scenario))
        self.assertTrue(result["admission_allowed"])
        self.assertFalse(result["unknown_processes_allowed"])
        self.assertFalse(result["training_during_wait"])
        self.assertEqual(result["saved_boundary"], self.receipt)
        self.assertEqual((scenario.calls, scenario.sleeps), (3, [2.0]))
        self.assertEqual([event["event"] for event in self.events].count("wait_start"), 1)
        self.assertEqual(self.events[-1]["event"], "result")

    def test_pinned_background_parent_drift_denies_before_save_or_sleep(self):
        changed = {**UNKNOWN, "parent_pid": 999}
        scenario = Scenario([[OWN, changed]])
        result = guard.strict_admission(scenario.read, owned_identities=[OWN], allowed_identities=[UNKNOWN],
            deadline_epoch=1000, clock=lambda: scenario.epoch, sleep=scenario.sleep)
        self.assertFalse(result["caller_admission_allowed"])
        self.assertFalse(result["allowed_ancestry_verified"])
        scenario = Scenario([[OWN, changed]])
        result = self.session(scenario, allowed_identities=[UNKNOWN]).check(boundary="optimizer_complete",
            optimizer_updates=7, save_boundary=lambda: self.fail("Identity drift must not save a wait checkpoint"))
        self.assertFalse(result["admission_allowed"])
        self.assertEqual(result["reason"], "identity_changed")
        self.assertEqual(scenario.sleeps, [])

    def test_clean_boundary_never_invokes_save_or_wait(self):
        scenario = Scenario([[OWN]])
        result = self.session(scenario).check(boundary="before_cuda", optimizer_updates=0,
            save_boundary=lambda: self.fail("Clean admission must not need a wait checkpoint"))
        self.assertTrue(result["admission_allowed"])
        self.assertIsNone(result["saved_boundary"])
        self.assertEqual(scenario.sleeps, [])

    def test_receipt_file_hash_cursor_kind_and_complete_are_strict(self):
        for kind in ("optimizer_checkpoint", "cache_record_boundary", "before_cuda"):
            self.assertEqual(guard.verify_boundary_receipt({**self.receipt, "kind": kind}, 7)["kind"], kind)
        for change in ({"complete": False}, {"optimizer_updates": 8}, {"optimizer_updates": True},
                       {"path": "relative.json"}, {"sha256": "0"*64}, {"kind": "partial"}, {"extra": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                guard.verify_boundary_receipt({**self.receipt, **change}, 7)

    def test_receipt_tamper_during_sleep_blocks_and_cannot_automatically_resume(self):
        scenario = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
        scenario.on_sleep = lambda: self.saved.write_bytes(b"tampered boundary")
        session = self.session(scenario)
        result = self.check(session)
        self.assertFalse(result["admission_allowed"])
        self.assertEqual(scenario.sleeps, [2.0])
        with self.assertRaisesRegex(RuntimeError, "cannot be resumed"):
            self.check(session)

    def test_campaign_wait_ledger_is_inherited_without_reset(self):
        first = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
        result = self.check(self.session(first, spent_seconds=897.0))
        self.assertTrue(result["admission_allowed"])
        self.assertEqual(result["total_waited_seconds"], 899.0)
        self.assertEqual(result["inherited_waited_seconds"], 897.0)
        second = Scenario([[OWN, UNKNOWN]])
        denied = self.check(self.session(second, spent_seconds=result["total_waited_seconds"]))
        self.assertFalse(denied["admission_allowed"])
        self.assertEqual((second.sleeps, denied["total_waited_seconds"]), ([1.0], 900.0))
        for invalid in (-1, 900, float("nan"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.session(Scenario([[OWN]]), spent_seconds=invalid)

    def test_stop_source_change_and_deadline_during_wait_all_deny(self):
        for mode in ("stop", "source", "deadline"):
            with self.subTest(mode=mode):
                scenario = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
                session = self.session(scenario,
                    stop_requested=lambda: mode == "stop" and bool(scenario.sleeps),
                    identity_unchanged=lambda: not (mode == "source" and scenario.sleeps),
                    deadline_epoch=101.0 if mode == "deadline" else 1000.0)
                result = self.check(session)
                self.assertFalse(result["admission_allowed"])
                self.assertEqual(len(scenario.sleeps), 1)
                self.assertEqual(scenario.calls, 2)

    def test_hard_conflict_or_owned_identity_drift_never_saves_for_wait(self):
        competitor = {**UNKNOWN, "command_line": "python windows.train --updates 2000"}
        for snapshot in ([OWN, competitor], [{**OWN, "parent_pid": 99}], []):
            with self.subTest(snapshot=snapshot):
                scenario = Scenario([snapshot])
                result = self.session(scenario).check(boundary="optimizer", optimizer_updates=7,
                    save_boundary=lambda: self.fail("A hard conflict must not use uncertainty waiting"))
                self.assertFalse(result["admission_allowed"])
                self.assertEqual(scenario.sleeps, [])

    def test_missing_or_incomplete_wait_receipt_cannot_allow_training(self):
        for receipt in ({}, {**self.receipt, "complete": False}, {**self.receipt, "path": str(self.root / "missing")}):
            with self.subTest(receipt=receipt):
                scenario = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
                result = self.check(self.session(scenario), receipt)
                self.assertFalse(result["admission_allowed"])
                self.assertEqual(scenario.sleeps, [])

    def test_observation_write_failure_denies_instead_of_allowing_unlogged_wait(self):
        scenario = Scenario([[OWN, UNKNOWN], [OWN, UNKNOWN], [OWN]])
        def broken_log(event):
            if event["event"] == "observation":
                raise OSError("fixture full disk")
        result = self.check(self.session(scenario, on_event=broken_log))
        self.assertFalse(result["admission_allowed"])
        self.assertEqual(scenario.sleeps, [])

    def test_child_wrapper_chain_requires_exact_pinned_supervisor(self):
        output = self.root / "run" / "stage"
        request_path = output.parent / "stage.request.json"
        supervisor = {**OWN, "pid": 301, "parent_pid": 0, "command_line": "python supervisor.py"}
        command = f'python scripts/run_alignlab_alignment_stage.py --request "{request_path}" --execute'
        wrapper = {**OWN, "pid": 302, "parent_pid": 301, "command_line": command}
        child = {**OWN, "pid": 303, "parent_pid": 302, "command_line": command}
        request = {"output": str(output), "supervisor_processes": [supervisor]}
        self.assertEqual(guard.child_identities([supervisor, wrapper, child], request, pid=303), [child, wrapper])
        for changed in ([{**supervisor, "created_at_utc": "2026-10-21T08:00:00.1234568Z"}, wrapper, child],
                        [supervisor, {**wrapper, "command_line": "python unrelated.py"}, child],
                        [supervisor, child]):
            with self.subTest(snapshot=changed), self.assertRaises(RuntimeError):
                guard.child_identities(changed, request, pid=303)


class CompletedGuardLogTests(unittest.TestCase):
    """Replay actual helper events, then mutate evidence rather than validators."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "process-guard.jsonl"
        identity = {"fixture": "tiny CPU guard evidence only"}
        checkpoint = self.root / "update-000001"
        checkpoint.mkdir()
        manifest = checkpoint / "manifest.json"
        manifest.write_text(json.dumps({"status": "complete", "identity": identity}))
        receipt = {"complete": True, "optimizer_updates": 1, "kind": "optimizer_checkpoint",
            "path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        self.request = {"output": str(self.root), "stage": "ppo", "workload": {"optimizer_updates": 1},
            "process_wait_policy": copy.deepcopy(guard.POLICY), "allowed_background_processes": [],
            "supervisor_processes": [], "process_wait_spent_seconds": 0.0, "deadline_epoch": 1000.0}
        scenario = Scenario([[OWN], [OWN], [OWN, UNKNOWN], [OWN, UNKNOWN], [OWN], [OWN]])
        self.rows = []
        session = guard.GuardSession(scenario.read, owned_identities=[OWN], deadline_epoch=1000.0,
            policy=copy.deepcopy(guard.POLICY), clock=lambda: scenario.epoch, sleep=scenario.sleep,
            on_event=lambda event: self.rows.append(copy.deepcopy(event)))
        for boundary, update in (("before_cuda", 0), ("engine_origin", 0), ("optimizer_complete", 1), ("engine_final", 1)):
            result = session.check(boundary=boundary, optimizer_updates=update, save_boundary=lambda: receipt)
            self.assertTrue(result["admission_allowed"])
        self.progress = {"started_at_epoch": 100.0, "identity": identity,
            "checkpoints": [{"optimizer_updates": 1, "checkpoint_path": str(checkpoint), "manifest_sha256": receipt["sha256"]}],
            "process_guard": {"owned_identities": [OWN], "last": self.rows[-1], "total_waited_seconds": 2.0}}

    def check(self):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        return guard.verify_guard_log(self.path, request=self.request, progress=self.progress)

    def sync_result(self, call):
        observations = [{k: row[k] for k in ("cycle", "at_epoch", "guard", "waited_seconds", "total_waited_seconds")}
            for row in self.rows if row["call"] == call and row["event"] == "observation"]
        result = next(row for row in self.rows if row["call"] == call and row["event"] == "result")
        result["observations"] = copy.deepcopy(observations)
        result["last_guard"] = copy.deepcopy(observations[-1]["guard"])
        self.progress["process_guard"]["last"] = self.rows[-1]

    def test_real_helper_log_is_accepted_with_exact_wait_ledger(self):
        result = self.check()
        self.assertTrue(result["verified"])
        self.assertEqual((result["boundary_calls"], result["segment_waited_seconds"]), (4, 2.0))

    def test_both_reused_cache_endpoints_wait_with_distinct_receipts_and_reject_replay(self):
        self.request.update(stage="dpo", reference_cache={"mode": "reuse"})
        journal = self.root / "reference-cache-records.jsonl"
        entries, receipts, chain = [], [], "0" * 64
        for index in range(2):
            payload = {"schema_version": 1, "previous_sha256": chain, "scored_sequences": 0,
                "expected_unique_sequences": 0, "record": None}
            chain = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            row = {**payload, "record_sha256": chain}
            entries.append(row)
            path = self.root / f"cache-safe-{index}.json"
            path.write_text(json.dumps({"kind": "alignlab-safe-cache-record-boundary-v1", "complete": True,
                "optimizer_updates": 0, "cache_journal": str(journal), "committed_record": row}))
            receipts.append({"complete": True, "optimizer_updates": 0, "kind": "cache_record_boundary",
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        journal.write_text("".join(json.dumps(row) + "\n" for row in entries))
        scenario = Scenario([[OWN], [OWN, UNKNOWN], [OWN, UNKNOWN], [OWN],
            [OWN, UNKNOWN], [OWN, UNKNOWN], [OWN], [OWN], [OWN], [OWN]])
        self.rows = []
        session = guard.GuardSession(scenario.read, owned_identities=[OWN], deadline_epoch=1000,
            policy=copy.deepcopy(guard.POLICY), clock=lambda: scenario.epoch, sleep=scenario.sleep,
            on_event=lambda event: self.rows.append(copy.deepcopy(event)))
        for boundary, update, receipt in (("before_cuda", 0, None), ("cache_record:0", 0, receipts[0]),
                ("cache_record:0", 0, receipts[1]), ("engine_origin", 0, None),
                ("optimizer_complete", 1, None), ("engine_final", 1, None)):
            self.assertTrue(session.check(boundary=boundary, optimizer_updates=update,
                save_boundary=lambda receipt=receipt: receipt)["admission_allowed"])
        self.progress["process_guard"].update(last=self.rows[-1], total_waited_seconds=4.0)
        self.assertEqual(self.check()["segment_waited_seconds"], 4.0)
        for row in self.rows:
            if row["call"] == 3 and row.get("saved_boundary"):
                row["saved_boundary"] = receipts[0]
        with self.assertRaisesRegex(ValueError, "committed complete scoring record"):
            self.check()

    def test_outer_clear_cannot_hide_unknown_final_snapshot_observation(self):
        observations = [row for row in self.rows if row["event"] == "observation" and row["call"] == 3]
        observations[-1]["guard"]["observations"] = copy.deepcopy(observations[0]["guard"]["observations"])
        # Outer final_snapshot/blockers still claim clear; the real inner
        # observation now contains the correctly classified unknown process.
        self.sync_result(3)
        with self.assertRaises(ValueError):
            self.check()

    def test_hard_training_conflict_followed_by_clear_cannot_be_accepted(self):
        hard = {**UNKNOWN, "command_line": "python windows.train --updates 2000"}
        first = next(row for row in self.rows if row["event"] == "observation" and row["call"] == 3)
        first["guard"] = guard.strict.inspect_processes(lambda: [OWN, hard], allowed_identities=[OWN],
            max_attempts=2, retry_delay_seconds=0, clock=lambda: 100.0, sleep=lambda _: None).to_dict()
        self.sync_result(3)
        with self.assertRaises(ValueError):
            self.check()

    def test_wait_cycle_or_boundary_identity_changes_are_rejected(self):
        original = copy.deepcopy(self.rows)
        for change in ({"cycle": 5}, {"boundary": "engine_final"}, {"optimizer_updates": 0}):
            with self.subTest(change=change):
                self.rows = copy.deepcopy(original)
                row = next(row for row in self.rows if row["event"] == "observation" and row["call"] == 3)
                row.update(change)
                self.sync_result(3)
                with self.assertRaises(ValueError):
                    self.check()

    def test_unknown_without_wait_start_even_when_charge_forged_zero_is_rejected(self):
        self.rows = [row for row in self.rows if row["event"] != "wait_start"]
        for row in self.rows:
            if "waited_seconds" in row:
                row["waited_seconds"] = 0.0
                row["total_waited_seconds"] = 0.0
            if row["event"] == "result":
                row["saved_boundary"] = None
        self.sync_result(3)
        self.sync_result(4)
        self.progress["process_guard"]["total_waited_seconds"] = 0.0
        with self.assertRaises(ValueError):
            self.check()

    def test_elapsed_unknown_wait_cannot_be_hidden_by_zero_charge(self):
        for row in self.rows:
            if row["call"] >= 3 and row.get("at_epoch", 0) > 100:
                row["at_epoch"] = 200.0
            if "waited_seconds" in row:
                row["waited_seconds"] = 0.0
                row["total_waited_seconds"] = 0.0
        self.sync_result(3)
        self.sync_result(4)
        self.progress["process_guard"]["total_waited_seconds"] = 0.0
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
