"""Mocked process/resource/filesystem tests; never launch any real subprocess."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab import probe_dispatch as dispatch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class MockChild:
    pid = 101

    def __init__(self, system, plan):
        self.system, self.plan = system, plan

    def poll(self):
        if self.system.now() < self.system.child_end:
            return None
        output = Path(self.plan["output"])
        if self.system.child_code == 0 and self.system.publish_result:
            result = {"status": "initial_gpu_training_probe_passed", "capacity_verified": False,
                      "formal_training": False, "training_started": True, "recovery_verified": True,
                      "actual_optimizer_updates": 3, "plan": self.plan, "pid": 102,
                      "finished_at_utc": timestamp(self.system.now())}
            result.update(self.system.result_overrides)
            (output / "probe.json").write_text(json.dumps(result), encoding="utf-8")
        return self.system.child_code


class MockSystem:
    def __init__(self, test):
        self.test, self.clock = test, 1000.0
        self.download_until = 1020
        self.resource_ready = 1000
        self.child_duration = 20
        self.child_code = 0
        self.publish_manifest = True
        self.publish_result = True
        self.result_overrides = {}
        self.process_overrides = {}
        self.extra_processes = []
        self.child, self.child_end = None, None
        self.commands, self.sleeps = [], []
        self.on_sleep = None
        self.kill_called = False

    def now(self):
        return self.clock

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.clock += seconds
        if self.on_sleep:
            self.on_sleep(self)

    def processes(self):
        records = [{"pid": os.getpid(), "parent_pid": 1, "created_at_utc": timestamp(900),
                    "command_line": "python scripts/dispatch_alignlab_probe.py --execute"}]
        if self.clock < self.download_until:
            for record in self.test.request["download_processes"]:
                value = {**record, "parent_pid": 2}
                value.update(self.process_overrides.get(record["pid"], {}))
                records.append(value)
        elif self.publish_manifest and not self.test.manifest.exists():
            self.test.manifest.write_text('{"complete":"pinned-fixture"}', encoding="utf-8")
        records += self.extra_processes
        if self.child and self.clock < self.child_end:
            command = " ".join(self.commands[-1])
            records.extend([
                {"pid": 101, "parent_pid": os.getpid(), "created_at_utc": timestamp(self.child_start),
                 "command_line": command},
                {"pid": 102, "parent_pid": 101, "created_at_utc": timestamp(self.child_start + 0.001),
                 "command_line": command},
            ])
        return records

    def resources(self):
        return {"blockers": ["busy"] if self.clock < self.resource_ready else [],
                "gpus": [{"index": 0, "free_mb": 7500, "utilization_percent": 0}],
                "available_ram_gb": 12}

    def spawn(self, args, stdout, stderr):
        self.commands.append(args)
        plan = json.loads(Path(args[args.index("--plan") + 1]).read_text())
        Path(plan["output"]).mkdir()
        self.child_start, self.child_end = self.clock, self.clock + self.child_duration
        self.child = MockChild(self, plan)
        stdout.write(b"mock probe stdout\n")
        stdout.flush()
        return self.child


class ProbeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for path in ("reports", "runs", "models/snapshot", "models/tokenizer", "data/audit",
                     "alignment_lab", "scripts", "tests_v2"):
            (self.root / path).mkdir(parents=True, exist_ok=True)
        for name in dispatch.SOURCE_FILES:
            (self.root / name).write_text("# frozen fixture\n", encoding="utf-8")
        source = self.root / "alignment_lab/stage.py"
        source.write_text("# stage fixture\n", encoding="utf-8")
        audit = self.root / "data/audit/manifest.json"
        audit.write_text('{"data_ready":true}', encoding="utf-8")
        prior = self.root / "reports/prior.json"
        prior.write_text('{"state":"paused"}', encoding="utf-8")
        self.request = {
            "schema_version": 1, "kind": dispatch.KIND, "report_dir": str(self.root / "reports/dispatch-1"),
            "model": {"repo": dispatch.REPO_ID, "revision": dispatch.REVISION},
            "download_processes": [
                {"pid": 20116, "created_at_utc": timestamp(800),
                 "command_line": "python scripts/fetch_alignlab_model.py --execute --timeout 90"},
                {"pid": 7244, "created_at_utc": timestamp(800.001),
                 "command_line": "basepython scripts/fetch_alignlab_model.py --execute --timeout 90"},
            ],
            "max_wait_seconds": 90, "poll_seconds": 10,
            "dispatcher_source_sha256": {name: digest(self.root / name) for name in dispatch.SOURCE_FILES},
            "prerequisites": [{"path": str(prior), "sha256": digest(prior), "json_equals": {"state": "paused"}}],
            "plan_template": {
                "kind": "alignlab-nf4-sft-initial-probe-v1", "formal_training": False,
                "output": str(self.root / "runs/probe-1"), "model_snapshot": str(self.root / "models/snapshot"),
                "tokenizer_dir": str(self.root / "models/tokenizer"), "audit_dir": str(audit.parent),
                "audit_manifest_sha256": digest(audit),
                "source_sha256": {"alignment_lab/stage.py": digest(source)},
                "work": {"wall_seconds": 1800},
            },
        }
        self.path = self.root / "reports/request.json"
        self.manifest = self.root / "models/snapshot/snapshot-manifest.json"
        self.report = Path(self.request["report_dir"])
        self.system = MockSystem(self)
        self.root_patch = patch.object(dispatch, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

        def validate(plan):
            for name, sha in plan["source_sha256"].items():
                if digest(self.root / name) != sha:
                    raise ValueError("Frozen probe source changed")
            return Path(plan["output"])

        self.validator = patch.object(dispatch, "validate_plan", side_effect=validate)
        self.validator.start()
        self.addCleanup(self.validator.stop)
        self.snapshot = patch.object(dispatch, "verify_snapshot",
            side_effect=lambda path, sha: self.verify_snapshot(path, sha))
        self.verify_mock = self.snapshot.start()
        self.addCleanup(self.snapshot.stop)
        self.save_request()

    def save_request(self):
        self.path.write_text(json.dumps(self.request), encoding="utf-8")
        self.sha = digest(self.path)

    def verify_snapshot(self, path, sha):
        self.assertEqual(Path(path), self.manifest.parent)
        self.assertEqual(sha, digest(self.manifest))
        return {"repo_id": dispatch.REPO_ID, "revision": dispatch.REVISION}

    def run_dispatch(self, execute=True):
        return dispatch.run_dispatch(self.path, self.sha, execute=execute, system=self.system)

    def status(self):
        return json.loads((self.report / "dispatch.json").read_text())

    def test_plan_default_does_not_create_files_wait_or_launch(self):
        result = self.run_dispatch(execute=False)
        self.assertEqual(result["status"], "plan_only")
        self.assertFalse(self.report.exists())
        self.assertFalse(self.system.commands)
        self.assertFalse(self.system.sleeps)

    def test_download_exit_full_verification_resources_then_one_exact_probe_command(self):
        self.system.resource_ready = 1040
        result = self.run_dispatch()
        self.assertEqual(result["status"], "initial_probe_completed")
        self.assertFalse(result["capacity_verified"])
        self.assertEqual(len(self.system.commands), 1)
        command = self.system.commands[0]
        self.assertEqual(command[1:5], ["-I", "-B", "-u", "scripts/run_alignlab_probe.py"])
        self.assertEqual(command[-1], "--execute")
        self.assertNotIn("--resume", command)
        self.assertGreaterEqual(self.verify_mock.call_count, 2)
        self.assertGreaterEqual(result["probe_launch_epoch"], 1040)
        self.assertEqual({p["pid"] for p in result["probe_processes"]}, {101, 102})
        self.assertEqual(result["probe_result"]["plan"]["work"]["wall_seconds"], 1800)
        self.assertFalse((self.root / "reports/.alignlab-probe-dispatch.lock").exists())

    def test_manifest_presence_does_not_bypass_active_download_wrapper_or_child(self):
        self.manifest.write_text("{}", encoding="utf-8")
        self.system.download_until = 1030
        result = self.run_dispatch()
        self.assertGreaterEqual(result["probe_launch_epoch"], 1030)

    def test_download_exit_without_manifest_fails_without_launch_or_retry(self):
        self.system.publish_manifest = False
        with self.assertRaisesRegex(RuntimeError, "without a complete"):
            self.run_dispatch()
        self.assertEqual(self.status()["status"], "failed")
        self.assertFalse(self.system.commands)
        with self.assertRaises(FileExistsError):
            self.run_dispatch()

    def test_download_identity_pid_reuse_or_changed_command_is_rejected(self):
        for change in ({"created_at_utc": timestamp(999)}, {"command_line": "other python process"}):
            with self.subTest(change=change):
                self.system.process_overrides[20116] = change
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    self.run_dispatch()
                self.assertFalse(self.system.commands)
                # Separate fresh request/report, never retry the same attempt.
                self.request["report_dir"] = str(self.root / f"reports/next-{len(list(self.root.iterdir()))}-{uuid_token(change)}")
                self.report = Path(self.request["report_dir"])
                self.save_request()

    def test_snapshot_verification_failure_or_retained_download_lock_prevents_launch(self):
        self.system.download_until = 1000
        self.verify_mock.side_effect = ValueError("actual LFS mismatch")
        with self.assertRaisesRegex(ValueError, "LFS"):
            self.run_dispatch()
        self.assertFalse(self.system.commands)

    def test_download_retained_failure_lock_rejected_even_with_manifest(self):
        self.system.download_until = 1000
        (self.manifest.parent.parent / ".snapshot.download.lock").write_text("failed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "failure lock"):
            self.run_dispatch()
        self.assertFalse(self.system.commands)

    def test_changed_source_while_waiting_is_fatal_and_no_child_is_dispatched(self):
        def changed(system):
            (self.root / "alignment_lab/stage.py").write_text("# changed\n", encoding="utf-8")
        self.system.on_sleep = changed
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_dispatch()
        self.assertFalse(self.system.commands)

    def test_foreign_dispatch_lock_preserved_without_new_report_or_launch(self):
        lock = self.root / "reports/.alignlab-probe-dispatch.lock"
        lock.write_text('{"pid":999,"owner":"foreign"}', encoding="utf-8")
        original = lock.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_dispatch()
        self.assertEqual(lock.read_bytes(), original)
        self.assertFalse(self.report.exists())
        self.assertFalse(self.system.commands)

    def test_existing_probe_output_or_probe_lock_is_not_reused(self):
        Path(self.request["plan_template"]["output"]).mkdir()
        with self.assertRaisesRegex(FileExistsError, "output already exists"):
            self.run_dispatch()
        self.assertFalse(self.system.commands)

    def test_stop_request_while_downloading_stops_dispatch_only(self):
        self.system.on_sleep = lambda system: Path(str(self.path) + ".stop").write_text("stop")
        result = self.run_dispatch()
        self.assertEqual(result["status"], "stopped_before_dispatch")
        self.assertFalse(self.system.commands)
        self.assertFalse(self.system.kill_called)

    def test_total_wait_budget_expires_without_reset_after_download(self):
        self.system.resource_ready = 2000
        result = self.run_dispatch()
        self.assertEqual(result["status"], "dispatch_wait_budget_exhausted")
        self.assertEqual(result["elapsed_seconds"], 90)
        self.assertFalse(self.system.commands)

    def test_deadline_during_full_verification_blocks_launch(self):
        def slow(path, sha):
            result = self.verify_snapshot(path, sha)
            self.system.clock += 100
            return result
        self.verify_mock.side_effect = slow
        result = self.run_dispatch()
        self.assertEqual(result["status"], "dispatch_wait_budget_exhausted")
        self.assertFalse(self.system.commands)

    def test_stop_during_child_writes_owned_marker_and_waits_without_killing(self):
        def stop_after_start(system):
            if system.child:
                (self.report / "STOP").write_text("user stop")
        self.system.on_sleep = stop_after_start
        result = self.run_dispatch()
        self.assertEqual(result["status"], "stopped_after_dispatch")
        marker = Path(self.request["plan_template"]["output"]) / "STOP"
        self.assertEqual(json.loads(marker.read_text())["request_sha256"], self.sha)
        self.assertGreaterEqual(self.system.clock, self.system.child_end)
        self.assertFalse(self.system.kill_called)

    def test_dispatch_wait_deadline_during_child_never_stops_or_restarts_probe(self):
        self.system.child_duration = 100
        result = self.run_dispatch()
        self.assertEqual(result["status"], "initial_probe_completed")
        self.assertTrue(result["dispatcher_wait_deadline_reached"])
        self.assertFalse((Path(self.request["plan_template"]["output"]) / "STOP").exists())
        self.assertEqual(len(self.system.commands), 1)

    def test_child_failure_or_missing_artifacts_never_implicitly_retries(self):
        self.system.child_code = 2
        with self.assertRaisesRegex(RuntimeError, "exited 2"):
            self.run_dispatch()
        self.assertEqual(len(self.system.commands), 1)
        self.assertEqual(self.status()["status"], "failed")
        with self.assertRaises(FileExistsError):
            self.run_dispatch()

    def test_success_claim_requires_capacity_false_complete_probe_and_observed_pid(self):
        self.system.result_overrides = {"capacity_verified": True}
        with self.assertRaisesRegex(RuntimeError, "complete verified short-probe"):
            self.run_dispatch()
        self.assertEqual(len(self.system.commands), 1)

    def test_unexpected_downloader_and_prerequisite_changes_fail_closed(self):
        self.system.extra_processes = [{"pid": 333, "parent_pid": 1, "created_at_utc": timestamp(990),
            "command_line": "python scripts/fetch_alignlab_model.py --execute"}]
        with self.assertRaisesRegex(RuntimeError, "Unexpected downloader"):
            self.run_dispatch()
        self.assertFalse(self.system.commands)


def uuid_token(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()[:8]


if __name__ == "__main__":
    unittest.main()
