"""CPU-only writer regressions including real Windows sharing violations."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from alignment_lab import windows_atomic_io as atomic


class Clock:
    def __init__(self): self.value = 100.0; self.sleeps = []
    def __call__(self): return self.value
    def sleep(self, seconds): self.sleeps.append(seconds); self.value += seconds


def winerror(code):
    error = PermissionError(f"simulated winerror {code}")
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

    def open_handle(path, *, allow_delete):
        handle = KERNEL32.CreateFileW(str(path), 0x80000000, 1 | 2 | (4 if allow_delete else 0), None, 3, 0x80, None)
        if handle == INVALID_HANDLE:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def shared_read(path):
        handle = open_handle(path, allow_delete=True)
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


class WindowsAtomicJSONTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "progress.json"
        self.old = {"complete": True, "updates": 10, "text": "旧完整 JSON"}
        self.new = {"complete": True, "updates": 11, "text": "新完整 JSON"}
        self.target.write_text(json.dumps(self.old), encoding="utf-8")
        self.clock = Clock()

    def write(self, **kwargs):
        return atomic.write_json_atomic(self.target, self.new, deadline_epoch=200,
            clock=self.clock, sleep=self.clock.sleep, **kwargs)

    def temporary(self):
        matches = list(self.root.glob("progress.json.*.tmp"))
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_complete_json_flush_fsync_before_publication_and_no_temp_on_success(self):
        original_fsync, original_replace = os.fsync, os.replace
        seen = []
        def fsync(fd): seen.append("fsync"); return original_fsync(fd)
        def replace(source, target):
            self.assertEqual(seen, ["fsync"])
            self.assertEqual(json.loads(Path(source).read_text(encoding="utf-8")), self.new)
            self.assertEqual(json.loads(Path(target).read_text(encoding="utf-8")), self.old)
            seen.append("replace")
            return original_replace(source, target)
        with patch.object(os, "fsync", side_effect=fsync):
            result = self.write(replace=replace)
        self.assertTrue(result["committed"])
        self.assertEqual((result["attempts"], result["retried"]), (1, False))
        self.assertEqual(seen, ["fsync", "replace"])
        self.assertFalse(list(self.root.glob("*.tmp")))
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.new)

    def test_only_transient_windows_codes_retry_then_publish(self):
        for code in (5, 32, 33):
            with self.subTest(code=code):
                attempts = []
                def replace(source, target):
                    attempts.append(code)
                    if len(attempts) <= 2: raise winerror(code)
                    os.replace(source, target)
                result = self.write(replace=replace)
                self.assertEqual(result["attempts"], 3)
                self.assertEqual([r["winerror"] for r in result["retry_observations"]], [code, code])
                self.assertFalse(list(self.root.glob("*.tmp")))

    def test_permanent_error_and_plain_permission_error_are_not_retried(self):
        for error in (winerror(123), PermissionError("no Windows transient code"), FileNotFoundError("missing")):
            with self.subTest(error=str(error)):
                with patch.object(os, "replace", side_effect=error) as replace:
                    with self.assertRaises(type(error)) as caught: self.write()
                self.assertEqual(replace.call_count, 1)
                self.assertFalse(self.clock.sleeps)
                temporary = self.temporary()
                self.assertEqual(json.loads(temporary.read_text(encoding="utf-8")), self.new)
                self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)
                self.assertTrue(any(str(temporary) in note for note in caught.exception.__notes__))
                temporary.unlink()

    def test_retry_window_and_absolute_deadline_are_both_hard_bounds(self):
        for deadline, max_wait in ((100.025, 5), (200, .025)):
            with self.subTest(deadline=deadline):
                clock = Clock()
                with patch.object(os, "replace", side_effect=winerror(5)):
                    with self.assertRaises(TimeoutError):
                        atomic.write_json_atomic(self.target, self.new, deadline_epoch=deadline,
                            max_wait_seconds=max_wait, clock=clock, sleep=clock.sleep)
                self.assertLessEqual(clock.value, min(deadline, 100 + max_wait) + 1e-12)
                self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)
                self.assertEqual(json.loads(self.temporary().read_text(encoding="utf-8")), self.new)
                self.temporary().unlink()

    def test_stop_and_expired_deadline_before_temporary_creation(self):
        for kwargs in ({"deadline_epoch": 99}, {"deadline_epoch": 200, "stop_requested": lambda: True}):
            with self.subTest(kwargs=kwargs):
                with patch.object(os, "replace") as replace:
                    with self.assertRaises((TimeoutError, InterruptedError)):
                        atomic.write_json_atomic(self.target, self.new, clock=self.clock, sleep=self.clock.sleep, **kwargs)
                    replace.assert_not_called()
                self.assertFalse(list(self.root.glob("*.tmp")))
                self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)

    def test_stop_during_wait_retains_complete_temp_without_another_publish(self):
        def sleep(seconds): self.clock.sleep(seconds)
        with patch.object(os, "replace", side_effect=winerror(32)) as replace:
            with self.assertRaises(InterruptedError):
                atomic.write_json_atomic(self.target, self.new, deadline_epoch=200, clock=self.clock,
                    sleep=sleep, stop_requested=lambda: bool(self.clock.sleeps))
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(json.loads(self.temporary().read_text(encoding="utf-8")), self.new)
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)

    def test_attempt_bound_also_handles_nonadvancing_clock(self):
        with patch.object(os, "replace", side_effect=winerror(5)) as replace:
            with self.assertRaisesRegex(TimeoutError, "attempt limit"):
                atomic.write_json_atomic(self.target, self.new, deadline_epoch=200, clock=self.clock,
                    sleep=lambda _: None, max_attempts=3)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)

    def test_exclusive_existing_never_overwritten_and_new_file_is_complete(self):
        with self.assertRaises(FileExistsError): self.write(exclusive=True)
        self.assertFalse(list(self.root.glob("*.tmp")))
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)
        target = self.root / "new-receipt.json"
        result = atomic.write_json_atomic(target, self.new, exclusive=True, deadline_epoch=200,
            clock=self.clock, sleep=self.clock.sleep)
        self.assertTrue(result["exclusive"])
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), self.new)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_nonfinite_json_and_fsync_failure_never_publish(self):
        with self.assertRaises(ValueError):
            atomic.write_json_atomic(self.target, {"value": float("nan")}, deadline_epoch=200, clock=self.clock)
        self.assertFalse(list(self.root.glob("*.tmp")))
        with patch.object(os, "fsync", side_effect=OSError("disk failure")):
            with patch.object(os, "replace") as replace:
                with self.assertRaisesRegex(OSError, "disk failure"): self.write()
                replace.assert_not_called()
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)
        self.assertEqual(json.loads(self.temporary().read_text(encoding="utf-8")), self.new)

    @unittest.skipUnless(os.name == "nt", "Real Windows sharing test")
    def test_real_windows_no_share_delete_blocks_then_release_allows_replace(self):
        handle = open_handle(self.target, allow_delete=False)
        closed, retries = [], []
        before = self.target.read_bytes()
        def sleep(seconds):
            self.assertEqual(self.target.read_bytes(), before)
            self.assertEqual(json.loads(self.temporary().read_text(encoding="utf-8")), self.new)
            self.assertTrue(KERNEL32.CloseHandle(handle))
            closed.append(True)
            time.sleep(seconds)
        try:
            result = atomic.write_json_atomic(self.target, self.new, deadline_epoch=time.time() + 5,
                sleep=sleep, on_retry=retries.append)
        finally:
            if not closed: KERNEL32.CloseHandle(handle)
        self.assertEqual(len(closed), 1)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(retries), 1)
        self.assertIn(retries[0]["winerror"], {5, 32, 33})
        self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.new)

    @unittest.skipUnless(os.name == "nt", "Real Windows shared-delete handle retry test")
    def test_real_windows_delete_shared_handle_still_recovers_when_reader_closes(self):
        # This host demonstrably denies replacement even with SHARE_DELETE.
        # Do not assume that adding the share flag eliminates every lock race.
        handle = open_handle(self.target, allow_delete=True)
        closed, observations = [], []
        def release_then_sleep(seconds):
            self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)
            self.assertEqual(json.loads(self.temporary().read_text(encoding="utf-8")), self.new)
            self.assertTrue(KERNEL32.CloseHandle(handle))
            closed.append(True)
            time.sleep(seconds)
        try:
            result = atomic.write_json_atomic(self.target, self.new, deadline_epoch=time.time() + 5,
                sleep=release_then_sleep, on_retry=observations.append)
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(len(closed), 1)
            self.assertIn(observations[0]["winerror"], {5, 32, 33})
            self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.new)
        finally:
            if not closed: KERNEL32.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Real Windows permanent sharing test")
    def test_real_windows_permanent_lock_exits_bounded_preserving_original_and_temp(self):
        handle = open_handle(self.target, allow_delete=False)
        started = time.time()
        try:
            with self.assertRaises(TimeoutError):
                atomic.write_json_atomic(self.target, self.new, deadline_epoch=started + 5,
                    max_wait_seconds=.07, initial_delay_seconds=.01, max_delay_seconds=.02)
            self.assertLess(time.time() - started, 1.5)
            self.assertEqual(json.loads(self.target.read_text(encoding="utf-8")), self.old)
            self.assertEqual(json.loads(self.temporary().read_text(encoding="utf-8")), self.new)
        finally:
            KERNEL32.CloseHandle(handle)

    @unittest.skipUnless(os.name == "nt", "Real Windows shared-delete reader test")
    def test_real_windows_shared_delete_reader_observes_only_complete_old_or_new_json(self):
        self.old["payload"] = "A" * 200000
        self.new["payload"] = "B" * 200000
        self.target.write_text(json.dumps(self.old), encoding="utf-8")
        ready, stop = threading.Event(), threading.Event()
        pause, released, resume = threading.Event(), threading.Event(), threading.Event()
        samples, errors = [], []
        def reader():
            try:
                while not stop.is_set():
                    if pause.is_set():
                        released.set()
                        if not resume.wait(3): raise TimeoutError("test reader release barrier")
                    samples.append(json.loads(shared_read(self.target)))
                    ready.set()
            except BaseException as error:
                errors.append(error); ready.set()
        thread = threading.Thread(target=reader)
        thread.start()
        try:
            self.assertTrue(ready.wait(3))
            def release_reader(observation):
                pause.set()
                self.assertTrue(released.wait(2))
            atomic.write_json_atomic(self.target, self.new, deadline_epoch=time.time() + 5, on_retry=release_reader)
            resume.set()
            until = time.time() + 3
            while not any(v == self.new for v in samples) and time.time() < until: time.sleep(.002)
        finally:
            stop.set(); resume.set(); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertIn(self.old, samples)
        self.assertIn(self.new, samples)
        self.assertTrue(all(row == self.old or row == self.new for row in samples))


if __name__ == "__main__": unittest.main()



