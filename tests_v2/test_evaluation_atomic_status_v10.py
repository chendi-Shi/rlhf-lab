"""Synthetic CPU-only regressions, including actual Windows/PS7 readers.

No datasets, output samples, model weights, CUDA or model packages are loaded.
The fake clocks test retry policy; native handle tests separately exercise the
actual Windows filesystem behavior without claiming kernel I/O is hard timed.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_atomic_status_v10 as atomic
from alignment_lab.evaluation_protocol_v2 import canonical


class Clock:
    def __init__(self):
        self.wall, self.mono, self.sleeps = 100.0, 50.0, []

    def now(self): return self.wall
    def monotonic(self): return self.mono

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.wall += seconds
        self.mono += seconds


def winerror(code):
    error = PermissionError(f"synthetic native Windows error {code}")
    error.winerror = code
    return error


if os.name == "nt":
    KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    KERNEL32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    KERNEL32.CreateFileW.restype = wintypes.HANDLE
    KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    KERNEL32.CloseHandle.restype = wintypes.BOOL
    KERNEL32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    KERNEL32.ReadFile.restype = wintypes.BOOL
    INVALID_HANDLE = ctypes.c_void_p(-1).value

    def open_handle(path, allow_delete):
        handle = KERNEL32.CreateFileW(str(path), 0x80000000,
            1 | 2 | (4 if allow_delete else 0), None, 3, 0x80, None)
        if handle == INVALID_HANDLE: raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def shared_read(path):
        handle = open_handle(path, True)
        try:
            result = bytearray()
            while True:
                buffer = ctypes.create_string_buffer(65536)
                count = wintypes.DWORD()
                if not KERNEL32.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                result.extend(buffer.raw[:count.value])
                if count.value == 0: return bytes(result)
        finally:
            KERNEL32.CloseHandle(handle)
else:
    def shared_read(path): return Path(path).read_bytes()


class AtomicStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "campaign.json"
        self.old = {"complete": True, "count": 1, "text": "旧完整状态"}
        self.new = {"complete": True, "count": 2, "text": "新完整状态"}
        self.clock = Clock()

    def publisher(self, **kwargs):
        options = dict(deadline_epoch=200, stop_requested=lambda: False,
                       serialize=canonical, now_fn=self.clock.now,
                       monotonic_fn=self.clock.monotonic, sleep_fn=self.clock.sleep)
        options.update(kwargs)
        return atomic.AtomicStatusPublisher(self.target, **options)

    def seeded(self, **kwargs):
        publisher = self.publisher(**kwargs)
        publisher.publish(self.old)
        return publisher

    def temporary(self):
        matches = list(self.root.glob("campaign.json.*.tmp"))
        self.assertEqual(len(matches), 1)
        return matches[0]

    def assert_preserved(self):
        self.assertEqual(self.target.read_bytes(), canonical(self.old))
        self.assertEqual(self.temporary().read_bytes(), canonical(self.new))

    def test_original_canonical_bytes_without_format_or_unicode_changes(self):
        result = self.publisher().publish(self.old)
        self.assertEqual(self.target.read_bytes(), canonical(self.old))
        self.assertEqual(result["sha256"], hashlib.sha256(canonical(self.old)).hexdigest())
        self.assertTrue(result["exclusive"])
        self.assertFalse(result["temporary_file_remaining"])

    def test_complete_fsync_and_closed_temp_precede_replace(self):
        publisher = self.seeded()
        original_fsync, original_replace, original_open = os.fsync, atomic._replace_existing, Path.open
        events, streams = [], []
        def open_file(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if args and args[0] == "xb": streams.append(stream)
            return stream
        def fsync(fd): events.append("fsync"); original_fsync(fd)
        def replace(source, target, backup):
            self.assertEqual(events, ["fsync"])
            self.assertEqual(Path(source).read_bytes(), canonical(self.new))
            self.assertEqual(Path(target).read_bytes(), canonical(self.old))
            self.assertEqual(len(streams), 1)
            self.assertTrue(streams[0].closed)
            events.append("replace")
            original_replace(source, target, backup)
        with patch.object(Path, "open", new=open_file), patch.object(os, "fsync", side_effect=fsync), patch.object(atomic, "_replace_existing", side_effect=replace):
            result = publisher.publish(self.new)
        self.assertEqual(events, ["fsync", "replace"])
        self.assertFalse(result["exclusive"])
        self.assertEqual(result["attempts"], 1)

    def test_fresh_instance_cannot_adopt_existing_attempt(self):
        self.seeded()
        with self.assertRaises(FileExistsError): self.publisher().publish(self.new)
        self.assertEqual(self.target.read_bytes(), canonical(self.old))
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_published_target_missing_before_update_fails_without_recreating(self):
        publisher = self.seeded()
        self.target.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "disappeared") as first:
            publisher.publish(self.new)
        with self.assertRaises(FileNotFoundError) as second:
            publisher.publish(self.new, terminal=True)
        self.assertIs(first.exception, second.exception)
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_published_target_deleted_after_fsync_retains_unpublished_temp(self):
        publisher = self.seeded()
        original = os.fsync
        def fsync(fd):
            original(fd)
            self.target.unlink()
        with patch.object(os, "fsync", side_effect=fsync), patch.object(atomic, "_replace_existing") as operation:
            with self.assertRaises(FileNotFoundError): publisher.publish(self.new)
        operation.assert_not_called()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.temporary().read_bytes(), canonical(self.new))

    @unittest.skipUnless(os.name == "nt", "Requires real Windows existing-only replacement")
    def test_real_windows_target_deleted_at_native_commit_is_not_recreated(self):
        publisher = self.seeded()
        original = atomic._replace_existing
        def replace(source, target, backup):
            target.unlink()
            original(source, target, backup)
        with patch.object(atomic, "_replace_existing", side_effect=replace) as operation:
            with self.assertRaises(OSError) as caught: publisher.publish(self.new)
        self.assertIn(caught.exception.winerror, (2, 3))
        self.assertEqual(operation.call_count, 1)
        self.assertFalse(self.clock.sleeps)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.temporary().read_bytes(), canonical(self.new))
        self.assertEqual(publisher.failure["failure_file_states"]["target"], {"exists": False})
        self.assertIsNone(publisher.failure["committed"])

    def test_replace_preserves_old_backup_until_success_then_cleans_only_own_backup(self):
        publisher = self.seeded()
        original = atomic._replace_existing
        unrelated = self.root / "campaign.json.unrelated.previous"
        unrelated.write_bytes(b"unrelated evidence")
        backups = []
        previous = [canonical(self.old)]
        def replace(source, target, backup):
            self.assertEqual(backup.parent, target.parent)
            self.assertFalse(backup.exists())
            original(source, target, backup)
            self.assertEqual(backup.read_bytes(), previous[0])
            backups.append(backup)
        with patch.object(atomic, "_replace_existing", side_effect=replace):
            for value in (self.new, self.old, self.new):
                result = publisher.publish(value)
                previous[0] = canonical(value)
                self.assertFalse(result["backup_file_remaining"])
                self.assertEqual(result["phase"], "complete")
        self.assertEqual(len(set(backups)), 3)
        self.assertTrue(all(not path.exists() for path in backups))
        self.assertEqual(unrelated.read_bytes(), b"unrelated evidence")

    def test_backup_cleanup_error_is_latched_as_committed_with_old_backup(self):
        publisher = self.seeded()
        original_unlink = Path.unlink
        error = PermissionError("synthetic owned-backup cleanup failure")
        def unlink(path, *args, **kwargs):
            if path.suffix == ".previous": raise error
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", new=unlink):
            with self.assertRaises(PermissionError) as first: publisher.publish(self.new)
            with self.assertRaises(PermissionError) as second: publisher.publish(self.old, terminal=True)
        self.assertIs(first.exception, error)
        self.assertIs(second.exception, error)
        self.assertTrue(publisher.failure["committed"])
        self.assertEqual(publisher.failure["phase"], "cleanup_own_backup")
        self.assertTrue(publisher.failure["backup_file_remaining"])
        self.assertFalse(publisher.failure["temporary_file_remaining"])
        self.assertEqual(Path(publisher.failure["backup_path"]).read_bytes(), canonical(self.old))
        self.assertEqual(self.target.read_bytes(), canonical(self.new))

    def test_stop_after_commit_preserves_backup_without_second_publication(self):
        stopped = []
        publisher = self.seeded(stop_requested=lambda: bool(stopped))
        original = atomic._replace_existing
        def replace(source, target, backup):
            original(source, target, backup)
            stopped.append(True)
        with patch.object(atomic, "_replace_existing", side_effect=replace) as operation:
            with self.assertRaises(InterruptedError) as first: publisher.publish(self.new)
            with self.assertRaises(InterruptedError) as second: publisher.publish(self.old, terminal=True)
        self.assertIs(first.exception, second.exception)
        self.assertEqual(operation.call_count, 1)
        self.assertTrue(publisher.failure["committed"])
        self.assertTrue(publisher.failure["stop_observed"])
        self.assertEqual(self.target.read_bytes(), canonical(self.new))
        self.assertEqual(Path(publisher.failure["backup_path"]).read_bytes(), canonical(self.old))

    def test_native_partial_move_1177_retains_old_backup_and_new_temp(self):
        publisher = self.seeded()
        error = winerror(1177)
        def replace(source, target, backup):
            os.rename(target, backup)
            raise error
        with patch.object(atomic, "_replace_existing", side_effect=replace) as operation:
            with self.assertRaises(PermissionError) as caught: publisher.publish(self.new)
        self.assertIs(caught.exception, error)
        self.assertEqual(operation.call_count, 1)
        self.assertFalse(self.clock.sleeps)
        self.assertIsNone(publisher.failure["committed"])
        self.assertFalse(self.target.exists())
        self.assertEqual(self.temporary().read_bytes(), canonical(self.new))
        self.assertEqual(Path(publisher.failure["backup_path"]).read_bytes(), canonical(self.old))
        self.assertEqual(publisher.failure["failure_file_states"]["target"], {"exists": False})

    def test_native_partial_failure_codes_never_retry_even_when_names_are_unchanged(self):
        for code in (1175, 1176, 1177):
            with self.subTest(code=code):
                self.target.unlink(missing_ok=True)
                publisher = self.seeded()
                error = winerror(code)
                with patch.object(atomic, "_replace_existing", side_effect=error) as operation:
                    with self.assertRaises(PermissionError) as caught: publisher.publish(self.new)
                self.assertIs(caught.exception, error)
                self.assertEqual(operation.call_count, 1)
                self.assertFalse(self.clock.sleeps)
                self.assertIsNone(publisher.failure["committed"])
                self.assert_preserved()
                self.temporary().unlink()

    def test_allowlisted_error_requires_source_target_and_backup_proof_before_retry(self):
        for changed in ("source", "source_identity", "target", "backup"):
            with self.subTest(changed=changed):
                self.target.unlink(missing_ok=True)
                publisher = self.seeded()
                error = winerror(32)
                def replace(source, target, backup):
                    if changed == "source": source.write_bytes(b"changed source")
                    elif changed == "source_identity":
                        os.rename(source, source.with_suffix(".displaced"))
                        source.write_bytes(canonical(self.new))
                    elif changed == "target": target.write_bytes(b"changed target")
                    else: backup.write_bytes(canonical(self.old))
                    raise error
                with patch.object(atomic, "_replace_existing", side_effect=replace) as operation:
                    with self.assertRaises(PermissionError) as caught: publisher.publish(self.new)
                self.assertIs(caught.exception, error)
                self.assertEqual(operation.call_count, 1)
                self.assertFalse(self.clock.sleeps)
                self.assertFalse(publisher.failure["unchanged_before_retry_proven"])
                self.assertIsNone(publisher.failure["committed"])
                self.temporary().unlink()

    def test_failure_observation_error_does_not_mask_native_error_or_retry(self):
        publisher = self.seeded()
        original = atomic._file_state
        native_called = []
        error = winerror(5)
        def replace(*args):
            native_called.append(True)
            raise error
        def observe(path):
            if native_called: raise OSError("synthetic failure evidence read error")
            return original(path)
        with patch.object(atomic, "_replace_existing", side_effect=replace) as operation, patch.object(atomic, "_file_state", side_effect=observe):
            with self.assertRaises(PermissionError) as caught: publisher.publish(self.new)
        self.assertIs(caught.exception, error)
        self.assertEqual(operation.call_count, 1)
        self.assertIn("read error", publisher.failure["failure_observation_error"])
        self.assertIsNone(publisher.failure["committed"])
        self.assert_preserved()

    def test_existing_backup_collision_is_not_overwritten(self):
        publisher = self.seeded()
        backup = self.root / "campaign.json.reserved.previous"
        backup.write_bytes(b"prior unrelated evidence")
        with patch.object(atomic.uuid, "uuid4") as identifier, patch.object(atomic, "_replace_existing") as operation:
            identifier.return_value.hex = "reserved"
            with self.assertRaises(FileExistsError): publisher.publish(self.new)
        operation.assert_not_called()
        self.assertEqual(backup.read_bytes(), b"prior unrelated evidence")
        self.assert_preserved()

    def test_first_publish_race_does_not_overwrite_winner(self):
        publisher = self.publisher()
        original_fsync = os.fsync
        def fsync(fd):
            original_fsync(fd)
            self.target.write_bytes(canonical(self.old))
        with patch.object(os, "fsync", side_effect=fsync):
            with self.assertRaises(FileExistsError): publisher.publish(self.new)
        self.assert_preserved()

    def test_replace_allowlist_only_retries_native_5_32_33(self):
        publisher = self.seeded()
        original = atomic._replace_existing
        for code in (5, 32, 33):
            attempts = []
            def replace(source, target, backup):
                attempts.append(code)
                if len(attempts) < 3: raise winerror(code)
                original(source, target, backup)
            with patch.object(atomic, "_replace_existing", side_effect=replace):
                result = publisher.publish(self.new)
            self.assertEqual(result["attempts"], 3)
            self.assertEqual([r["winerror"] for r in result["retry_observations"]], [code, code])
        self.assertTrue(all(v <= .1 for v in self.clock.sleeps))

    def test_permanent_errors_do_not_retry_or_destroy_old_and_temp(self):
        for error in (winerror(112), winerror(123), PermissionError("plain denial"),
                      FileNotFoundError("missing path")):
            with self.subTest(error=error):
                self.target.unlink(missing_ok=True)
                publisher = self.seeded()
                with patch.object(atomic, "_replace_existing", side_effect=error) as operation:
                    with self.assertRaises(type(error)) as caught: publisher.publish(self.new)
                self.assertIs(caught.exception, error)
                self.assertEqual(operation.call_count, 1)
                self.assert_preserved()
                self.temporary().unlink()

    def test_eight_attempt_limit_and_first_failure_latched_across_finally(self):
        publisher = self.seeded()
        with patch.object(atomic, "_replace_existing", side_effect=winerror(32)) as operation:
            with self.assertRaises(TimeoutError) as first: publisher.publish(self.new)
            before = list(self.clock.sleeps)
            self.clock.sleep(20)
            with self.assertRaises(TimeoutError) as second:
                publisher.publish({"status": "failed"}, terminal=True)
        self.assertIs(first.exception, second.exception)
        self.assertIs(publisher.first_error, first.exception)
        self.assertEqual(operation.call_count, 8)
        self.assertLessEqual(sum(before), 1)
        self.assertTrue(all(v <= .1 for v in before))
        self.assert_preserved()
        failure = publisher.failure
        failure["attempts"] = -1
        self.assertEqual(publisher.failure["attempts"], 8)

    def test_monotonic_elapsed_includes_native_attempt_time(self):
        publisher = self.seeded()
        def replacement(*_):
            self.clock.mono += .6
            raise winerror(5)
        with patch.object(atomic, "_replace_existing", side_effect=replacement) as operation:
            with self.assertRaisesRegex(TimeoutError, "budget"): publisher.publish(self.new)
        self.assertEqual(operation.call_count, 2)
        self.assert_preserved()

    def test_original_deadline_caps_sleep_and_attempts(self):
        publisher = self.seeded(deadline_epoch=100.025)
        with patch.object(atomic, "_replace_existing", side_effect=winerror(32)) as operation:
            with self.assertRaisesRegex(TimeoutError, "Original"): publisher.publish(self.new)
        self.assertLessEqual(sum(self.clock.sleeps), .025 + 1e-12)
        self.assertEqual(operation.call_count, 2)
        self.assert_preserved()

    def test_stop_and_deadline_checked_before_creating_temp(self):
        for options in (dict(stop_requested=lambda: True), dict(deadline_epoch=99)):
            with self.subTest(options=options):
                publisher = self.publisher(**options)
                with self.assertRaises((InterruptedError, TimeoutError)): publisher.publish(self.new)
                self.assertFalse(self.target.exists())
                self.assertFalse(list(self.root.glob("*.tmp")))

    def test_stop_during_backoff_prevents_second_attempt(self):
        publisher = self.seeded(stop_requested=lambda: bool(self.clock.sleeps))
        with patch.object(atomic, "_replace_existing", side_effect=winerror(32)) as operation:
            with self.assertRaises(InterruptedError): publisher.publish(self.new)
        self.assertEqual(operation.call_count, 1)
        self.assertTrue(publisher.failure["stop_observed"])
        self.assert_preserved()

    def test_terminal_evidence_observes_stop_but_never_extends_deadline(self):
        publisher = self.publisher(stop_requested=lambda: True)
        result = publisher.publish(self.old, terminal=True)
        self.assertTrue(result["terminal"])
        self.assertTrue(result["stop_observed"])
        self.clock.wall = 200
        with self.assertRaisesRegex(TimeoutError, "Original"):
            publisher.publish(self.new, terminal=True)
        self.assertEqual(self.target.read_bytes(), canonical(self.old))

    def test_successful_terminal_calls_share_attempt_and_time_budget(self):
        publisher = self.seeded()
        first = publisher.publish(self.new, terminal=True)
        second = publisher.publish(self.old, terminal=True)
        self.assertEqual((first["attempts"], second["attempts"]), (1, 2))
        self.clock.sleep(1)
        with self.assertRaisesRegex(TimeoutError, "budget"):
            publisher.publish(self.new, terminal=True)
        self.assertEqual(self.target.read_bytes(), canonical(self.old))

    def test_normal_mode_cannot_resume_after_terminal_publication(self):
        publisher = self.publisher()
        publisher.publish(self.old, terminal=True)
        with self.assertRaisesRegex(RuntimeError, "cannot resume"): publisher.publish(self.new)
        self.assertEqual(self.target.read_bytes(), canonical(self.old))

    def test_stalled_clocks_still_have_eight_attempt_bound(self):
        publisher = self.seeded(sleep_fn=lambda seconds: None)
        with patch.object(atomic, "_replace_existing", side_effect=winerror(5)) as operation:
            with self.assertRaisesRegex(TimeoutError, "attempt"): publisher.publish(self.new)
        self.assertEqual(operation.call_count, 8)
        self.assertLessEqual(publisher.failure["requested_sleep_seconds"], 1)

    def test_wall_and_monotonic_rollback_fail_closed(self):
        for field in ("wall", "mono"):
            with self.subTest(clock=field):
                self.target.unlink(missing_ok=True)
                publisher = self.seeded()
                setattr(self.clock, field, getattr(self.clock, field) - 1)
                with self.assertRaisesRegex(RuntimeError, "backwards"): publisher.publish(self.new)
                self.assertEqual(self.target.read_bytes(), canonical(self.old))

    def test_nonfinite_clock_and_nonbool_stop_fail_closed(self):
        publisher = self.seeded()
        self.clock.mono = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"): publisher.publish(self.new)
        self.clock.mono = 50
        self.target.unlink()
        publisher = self.publisher(stop_requested=lambda: "false")
        with self.assertRaisesRegex(ValueError, "bool"): publisher.publish(self.new)

    def test_serialize_error_never_retries_or_creates_temp(self):
        publisher = self.seeded()
        with self.assertRaises(ValueError): publisher.publish({"invalid": float("nan")})
        self.assertEqual(self.target.read_bytes(), canonical(self.old))
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_serializer_must_return_original_bytes_and_is_called_once(self):
        calls = []
        def serialize(value): calls.append(value); return canonical(value)
        publisher = self.seeded(serialize=serialize)
        calls.clear()
        with patch.object(atomic, "_replace_existing", side_effect=winerror(32)):
            with self.assertRaises(TimeoutError): publisher.publish(self.new)
        self.assertEqual(calls, [self.new])
        self.target.unlink()
        with self.assertRaisesRegex(TypeError, "bytes"):
            self.publisher(serialize=lambda value: "{}").publish(self.old)

    def test_write_and_flush_errors_preserve_old_and_any_partial_temp(self):
        original_open = Path.open
        for phase in ("write", "flush"):
            with self.subTest(phase=phase):
                self.target.unlink(missing_ok=True)
                publisher = self.seeded()
                failure = OSError("synthetic " + phase + " disk failure")
                class FailingStream:
                    def __init__(self, wrapped): self.wrapped = wrapped
                    def __enter__(self): self.wrapped.__enter__(); return self
                    def __exit__(self, *args): return self.wrapped.__exit__(*args)
                    def write(self, payload):
                        if phase == "write":
                            self.wrapped.write(payload[:3])
                            raise failure
                        return self.wrapped.write(payload)
                    def flush(self):
                        if phase == "flush": raise failure
                        return self.wrapped.flush()
                    def fileno(self): return self.wrapped.fileno()
                def open_file(path, *args, **kwargs):
                    stream = original_open(path, *args, **kwargs)
                    return FailingStream(stream) if args and args[0] == "xb" else stream
                with patch.object(Path, "open", new=open_file), patch.object(atomic, "_replace_existing") as operation:
                    with self.assertRaises(OSError) as caught: publisher.publish(self.new)
                self.assertIs(caught.exception, failure)
                operation.assert_not_called()
                self.assertEqual(self.target.read_bytes(), canonical(self.old))
                self.assertTrue(self.temporary().exists())
                self.temporary().unlink()

    def test_stop_or_deadline_after_fsync_prevents_commit(self):
        original = os.fsync
        for reason in ("stop", "deadline"):
            with self.subTest(reason=reason):
                self.target.unlink(missing_ok=True)
                self.clock = Clock()
                stopped = []
                publisher = self.seeded(stop_requested=lambda: bool(stopped))
                def fsync(fd):
                    original(fd)
                    if reason == "stop": stopped.append(True)
                    else: self.clock.wall = 200
                with patch.object(os, "fsync", side_effect=fsync), patch.object(atomic, "_replace_existing") as operation:
                    with self.assertRaises((InterruptedError, TimeoutError)): publisher.publish(self.new)
                operation.assert_not_called()
                self.assert_preserved()
                self.temporary().unlink()

    def test_fsync_error_preserves_first_error_and_complete_unpublished_temp(self):
        publisher = self.seeded()
        error = OSError("synthetic fsync disk failure")
        with patch.object(os, "fsync", side_effect=error), patch.object(atomic, "_replace_existing") as operation:
            with self.assertRaises(OSError) as first: publisher.publish(self.new)
            with self.assertRaises(OSError) as second: publisher.publish(self.old, terminal=True)
        self.assertIs(first.exception, error)
        self.assertIs(second.exception, error)
        operation.assert_not_called()
        self.assert_preserved()

    def test_post_commit_clock_failure_is_explicit_and_cannot_republish(self):
        publisher = self.seeded()
        original = atomic._replace_existing
        def replace(source, target, backup):
            original(source, target, backup)
            self.clock.mono -= 1
        with patch.object(atomic, "_replace_existing", side_effect=replace) as operation:
            with self.assertRaises(RuntimeError) as first: publisher.publish(self.new)
            with self.assertRaises(RuntimeError) as second: publisher.publish(self.old, terminal=True)
        self.assertIs(first.exception, second.exception)
        self.assertEqual(operation.call_count, 1)
        self.assertTrue(publisher.failure["committed"])
        self.assertEqual(self.target.read_bytes(), canonical(self.new))

    def test_native_error_with_disappeared_temp_is_uncertain_and_not_retried(self):
        publisher = self.seeded()
        original = atomic._replace_existing
        error = winerror(5)
        def replace(source, target, backup):
            original(source, target, backup)
            raise error
        with patch.object(atomic, "_replace_existing", side_effect=replace) as operation:
            with self.assertRaises(PermissionError) as caught: publisher.publish(self.new)
        self.assertIs(caught.exception, error)
        self.assertEqual(operation.call_count, 1)
        self.assertIsNone(publisher.failure["committed"])
        self.assertFalse(self.clock.sleeps)
        self.assertEqual(self.target.read_bytes(), canonical(self.new))

    def test_old_wb_partial_string_race_is_reproduced_at_a_barrier(self):
        self.target.write_bytes(canonical(self.old))
        half_ready, finish = threading.Event(), threading.Event()
        errors = []
        def writer():
            try:
                with self.target.open("wb") as stream:
                    stream.write(b'{"complete":true,"text":"unterminated')
                    stream.flush()
                    half_ready.set()
                    if not finish.wait(3): raise TimeoutError("Synthetic old-writer barrier")
                    stream.write(b'"}')
            except BaseException as error: errors.append(error)
        thread = threading.Thread(target=writer)
        thread.start()
        try:
            self.assertTrue(half_ready.wait(3))
            with self.assertRaises(json.JSONDecodeError): json.loads(shared_read(self.target))
        finally:
            finish.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)

    def test_new_atomic_writer_barrier_preserves_complete_old_then_new(self):
        publisher = self.seeded()
        ready, finish = threading.Event(), threading.Event()
        original_fsync = os.fsync
        errors = []
        def fsync(fd):
            original_fsync(fd)
            ready.set()
            if not finish.wait(3): raise TimeoutError("Synthetic complete-temp barrier")
        def writer():
            try: publisher.publish(self.new)
            except BaseException as error: errors.append(error)
        with patch.object(os, "fsync", side_effect=fsync):
            thread = threading.Thread(target=writer)
            thread.start()
            try:
                self.assertTrue(ready.wait(3))
                for _ in range(10): self.assertEqual(shared_read(self.target), canonical(self.old))
            finally:
                finish.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(shared_read(self.target), canonical(self.new))

    @unittest.skipUnless(os.name == "nt", "Requires actual Windows filesystem sharing")
    def test_real_windows_transient_sharing_with_and_without_delete(self):
        for allow_delete in (False, True):
            with self.subTest(allow_delete=allow_delete):
                self.target.unlink(missing_ok=True)
                closed = []
                def sleep(seconds):
                    self.assert_preserved()
                    self.assertTrue(KERNEL32.CloseHandle(handle))
                    closed.append(True)
                    self.clock.sleep(seconds)
                publisher = self.seeded(sleep_fn=sleep)
                handle = open_handle(self.target, allow_delete)
                try:
                    result = publisher.publish(self.new)
                    # Both outcomes are valid with SHARE_DELETE on different
                    # Windows filesystems: immediate commit or a native retry.
                    self.assertIn(result["attempts"], (1, 2))
                    if not allow_delete: self.assertEqual(result["attempts"], 2)
                    self.assertEqual(self.target.read_bytes(), canonical(self.new))
                finally:
                    if not closed: KERNEL32.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Requires actual permanent Windows sharing denial")
    def test_real_windows_permanent_sharing_preserves_complete_evidence(self):
        publisher = self.seeded()
        handle = open_handle(self.target, False)
        try:
            with self.assertRaises(TimeoutError): publisher.publish(self.new)
            self.assertEqual(publisher.failure["attempts"], 8)
            self.assertLessEqual(sum(self.clock.sleeps), 1)
            self.assert_preserved()
        finally:
            KERNEL32.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Requires actual PS7 shared reader on Windows")
    def test_actual_ps7_cross_process_reader_sees_only_complete_old_or_new(self):
        bundled = Path(os.environ.get("USERPROFILE", "")) / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe"
        pwsh = str(bundled) if bundled.is_file() else shutil.which("pwsh")
        self.assertTrue(pwsh, "PS7 must be available for the required offline proof")
        fixture = Path(__file__).parent / "fixtures/v10_atomic_reader.ps1"
        ready, seen, done, result = [self.root / name for name in ("ready", "new", "done", "result.json")]
        self.old["payload"] = "A" * 100000
        self.new["payload"] = "B" * 100000
        publisher = self.publisher(now_fn=time.time, monotonic_fn=time.monotonic,
                                   sleep_fn=time.sleep, deadline_epoch=time.time() + 15)
        publisher.publish(self.old)
        command = [pwsh, "-NoProfile", "-File", str(fixture), "-Target", str(self.target),
                   "-Ready", str(ready), "-SeenNew", str(seen), "-Done", str(done), "-Result", str(result),
                   "-OldSha256", hashlib.sha256(canonical(self.old)).hexdigest(),
                   "-NewSha256", hashlib.sha256(canonical(self.new)).hexdigest()]
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        def wait_for(path):
            until = time.monotonic() + 5
            while not path.exists() and child.poll() is None and time.monotonic() < until:
                time.sleep(.005)
            self.assertTrue(path.exists(), f"Synthetic PS7 reader did not produce {path.name}")
        try:
            wait_for(ready)
            publisher.publish(self.new)
            wait_for(seen)
        finally:
            done.write_text("done", encoding="utf-8")
            try: stdout, stderr = child.communicate(timeout=12)
            except subprocess.TimeoutExpired:
                # This is exclusively the synthetic fixture process we own.
                child.kill()
                stdout, stderr = child.communicate(timeout=3)
                self.fail("Synthetic PS7 reader did not exit after its deadline")
        self.assertEqual(child.returncode, 0, (stdout + stderr).decode(errors="replace"))
        report = json.loads(result.read_text(encoding="utf-8-sig"))
        self.assertFalse(report["errors"])
        self.assertGreater(report["counts"]["old"], 0)
        self.assertGreater(report["counts"]["new"], 0)
        self.assertFalse(report["model_loaded"])


if __name__ == "__main__":
    unittest.main()
