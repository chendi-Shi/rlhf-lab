"""Pure stdlib deterministic race/identity decisions; no training or GPU."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError

from alignment_lab.process_guard_v2 import inspect_processes, ProcessIdentity


def process(pid=21068, *, command=None, created="2026-10-20T13:35:43.7933190Z"):
    return {"pid": pid, "parent_pid": 13832, "created_at_utc": created, "command_line": command}


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

    def inspect(self, **kwargs):
        return inspect_processes(self.read, clock=lambda: self.now, sleep=self.sleep, **kwargs)


class ProcessGuardV2Tests(unittest.TestCase):
    def test_incident_transient_blank_disappears_requires_fresh_snapshot(self):
        api = process(10, command="python -m uvicorn app.api:app")
        case = Scenario([[api, process()], [api]])
        result = case.inspect(allowed_identities=[api])
        self.assertTrue(result.admission_allowed)
        self.assertEqual((case.calls, result.attempt_count), (2, 2))
        self.assertEqual(result.observations[0].blockers[0].code, "unknown_command")
        self.assertEqual([p.pid for p in result.final_snapshot], [10])
        self.assertFalse(result.requires_safe_stop)

    def test_blank_persists_bounded_then_requests_safe_stop_without_claiming_gpu(self):
        case = Scenario([[process()]])
        result = case.inspect()
        self.assertFalse(result.admission_allowed)
        self.assertTrue(result.requires_safe_stop)
        self.assertEqual(case.calls, 3)
        self.assertEqual(case.sleeps, [.25, .25])
        self.assertEqual(result.blockers[0].code, "unknown_command")
        self.assertFalse(result.to_dict()["gpu_resource_admission_verified"])

    def test_known_identity_blank_recovers_only_to_exact_command(self):
        known = process(command="python scripts/run_alignlab_stage.py --execute")
        self.assertTrue(Scenario([[process()], [known]]).inspect(allowed_identities=[known]).admission_allowed)
        changed = process(command=known["command_line"] + " --different")
        result = Scenario([[process()], [changed]]).inspect(allowed_identities=[known])
        self.assertEqual(result.blockers[0].code, "allowed_command_changed")

    def test_reused_pid_is_never_the_old_child_even_with_same_command(self):
        old = process(command="python scripts/run_alignlab_study_stage.py --execute", created="2026-10-20T10:39:40.9430490Z")
        new = process(command=old["command_line"])
        case = Scenario([[new]])
        result = case.inspect(allowed_identities=[old])
        self.assertEqual((case.calls, result.blockers[0].code), (1, "competing_process"))
        result = Scenario([[process()], []]).inspect(allowed_identities=[old])
        self.assertTrue(result.admission_allowed)
        self.assertEqual(result.attempt_count, 2)

    def test_snapshot_after_disappearance_still_detects_new_training(self):
        case = Scenario([[process()], [process(456, command="python -m windows.train")]])
        result = case.inspect()
        self.assertFalse(result.admission_allowed)
        self.assertEqual(result.blockers[0].process.pid, 456)

    def test_unknown_nonempty_script_is_not_assumed_cpu_and_parent_is_not_allowlist(self):
        own = process(13832, command="python approved-supervisor.py")
        unknown = process(command="python arbitrary_worker.py")
        result = Scenario([[own, unknown]]).inspect(allowed_identities=[own])
        self.assertEqual(result.attempt_count, 3)
        self.assertEqual(result.blockers[0].code, "unclassified_process")

    def test_explicit_vetted_background_and_own_busy_commands_allowed(self):
        own = process(1, command="python scripts/run_alignlab_resume.py --execute")
        api = process(2, command="python -m uvicorn app.api:app")
        self.assertTrue(Scenario([[own, api]]).inspect(allowed_identities=[own, api]).admission_allowed)
        # A script-name match without the exact process instance never grants trust.
        clone = process(3, command=api["command_line"])
        self.assertFalse(Scenario([[clone]]).inspect(allowed_identities=[api]).admission_allowed)

    def test_reader_failure_and_invalid_snapshot_reobserve_but_never_mean_empty(self):
        result = Scenario([RuntimeError("WMI unavailable"), []]).inspect()
        self.assertTrue(result.admission_allowed)
        self.assertFalse(result.observations[0].snapshot_valid)
        for snapshot in (None, {}, [process(), process()], [process(created="missing")]):
            with self.subTest(snapshot=snapshot):
                result = Scenario([snapshot]).inspect()
                self.assertFalse(result.admission_allowed)
                self.assertEqual(result.attempt_count, 3)
                self.assertEqual(result.blockers[0].code, "snapshot_unavailable")

    def test_deadline_before_between_and_after_reads_never_allows_dispatch(self):
        case = Scenario([[]])
        self.assertEqual(case.inspect(deadline_epoch=100).blockers[-1].code, "deadline_exhausted")
        self.assertEqual(case.calls, 0)
        case = Scenario([[process()], []])
        result = case.inspect(deadline_epoch=100.1)
        self.assertEqual(result.blockers[-1].code, "deadline_exhausted")
        self.assertEqual(case.calls, 1)
        self.assertAlmostEqual(sum(case.sleeps), .1)
        case = Scenario([[]])
        def slow_read():
            case.now += 3
            return []
        result = inspect_processes(slow_read, clock=lambda: case.now, deadline_epoch=102)
        self.assertFalse(result.admission_allowed)
        self.assertEqual(result.blockers[-1].code, "deadline_exhausted")

    def test_creation_time_keeps_windows_submicrosecond_identity(self):
        a = process(command="python approved.py", created="2026-10-20T13:35:43.7933191Z")
        equivalent = process(command=a["command_line"], created="2026-10-20T21:35:43.793319100+08:00")
        different = process(command=a["command_line"], created="2026-10-20T13:35:43.7933192Z")
        self.assertTrue(Scenario([[equivalent]]).inspect(allowed_identities=[a]).admission_allowed)
        self.assertEqual(Scenario([[different]]).inspect(allowed_identities=[a]).blockers[0].code, "unclassified_process")

    def test_result_copy_is_immutable(self):
        result = Scenario([[process()]]).inspect()
        with self.assertRaises(FrozenInstanceError):
            result.final_snapshot[0].pid = 7
        copy_result = result.to_dict()
        copy_result["blockers"][0]["process"]["pid"] = 7
        self.assertEqual(result.blockers[0].process.pid, 21068)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_isolated_import_does_not_load_any_gpu_framework(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            "import json, sys; sys.path.insert(0, " + repr(str(root)) + "); "
            "from alignment_lab.process_guard_v2 import inspect_processes; "
            "r = inspect_processes(lambda: []); "
            "print(json.dumps({'allowed': r.admission_allowed, "
            "'frameworks': sorted(n for n in sys.modules "
            "if n.split('.')[0] in {'torch','tensorflow','jax','flax','bitsandbytes'})}))"
        )
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "USE_TORCH": "0", "USE_TF": "0"}
        run = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                             cwd=root, env=env, capture_output=True, text=True,
                             timeout=15, check=True)
        evidence = json.loads(run.stdout)
        self.assertTrue(evidence["allowed"])
        self.assertEqual(evidence["frameworks"], [])

    def test_invalid_configuration_rejected_before_query(self):
        for kwargs in ({"max_attempts": 1}, {"max_attempts": 6}, {"retry_delay_seconds": float("nan")},
                       {"busy_markers": []}, {"allowed_identities": [process()]},
                       {"deadline_epoch": True}, {"retry_delay_seconds": 3}):
            with self.subTest(kwargs=kwargs):
                case = Scenario([[]])
                with self.assertRaises(ValueError):
                    case.inspect(**kwargs)
                self.assertEqual(case.calls, 0)


if __name__ == "__main__":
    unittest.main()
