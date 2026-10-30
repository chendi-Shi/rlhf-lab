"""Atomic campaign/progress publication with one bounded failure budget.

This module does not import a model runtime.  The caller supplies the existing
canonical serializer, the original campaign deadline and its STOP predicate.
Keep one publisher per newly allocated target for the whole execution.  A
failure is latched: except/finally cannot retry with a fresh clock or overwrite
the first error.  Terminal evidence may acknowledge STOP, but cannot extend the
original deadline.  All terminal calls also share their first terminal budget.

The retry clock bounds attempts and sleeps, not a synchronous kernel call:
write/fsync/rename can block inside the OS.  The independent guardian remains
responsible for the original campaign deadline.  No in-place write fallback is
provided, and failed temporary evidence is never deleted here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import ctypes
import hashlib
import math
import os
from pathlib import Path
import stat
import time
import uuid


SOURCE_FILES = ("alignment_lab/evaluation_atomic_status_v10.py",
                "tests_v2/test_evaluation_atomic_status_v10.py",
                "tests_v2/fixtures/v10_atomic_reader.ps1")
MAX_ATTEMPTS = 8
MAX_RETRY_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 0.1
# Access denied may be a sharing denial on Windows.  Disk-full, missing-path,
# malformed-data and ordinary POSIX permission errors are deliberately absent.
TRANSIENT_WINDOWS_REPLACE_ERRORS = frozenset({5, 32, 33})


def _replace_existing(source, target, backup):
    """Windows existing-only replacement; no destination-creating fallback.

    ReplaceFileW's documented partial failures can move the old file to the
    supplied backup.  The caller therefore preserves all names on failure and
    does not assume that the old bytes still occupy the target path.
    """
    if os.name != "nt":
        raise NotImplementedError("Existing-only status replacement requires Windows ReplaceFileW")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    operation = kernel.ReplaceFileW
    operation.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                         wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID]
    operation.restype = wintypes.BOOL
    # Flags 0 preserve ACL/metadata error handling. WRITE_THROUGH is unsupported
    # by this API; the replacement bytes were already flushed and fsynced.
    if not operation(str(target.absolute()), str(source.absolute()),
                     str(backup.absolute()), 0, None, None):
        raise ctypes.WinError(ctypes.get_last_error())


def _file_state(path):
    """Observe a closed, complete file for failure evidence, never health."""
    try:
        entry = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    if (not stat.S_ISREG(entry.st_mode) or path.is_symlink()
            or getattr(entry, "st_file_attributes", 0) & 0x400):
        raise ValueError("Status evidence must be an ordinary non-reparse file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        after = os.fstat(stream.fileno())
    current = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    # Windows CRT fstat and path stat can expose different ctime semantics.
    # Compare ctime only within the same query family, never across them.
    if (any(getattr(entry, key) != getattr(before, key) for key in fields)
            or any(getattr(before, key) != getattr(after, key) for key in fields)
            or any(getattr(after, key) != getattr(current, key) for key in fields)
            or before.st_ctime_ns != after.st_ctime_ns
            or entry.st_ctime_ns != current.st_ctime_ns
            or len(payload) != after.st_size):
        raise RuntimeError("Status evidence changed during one complete read")
    return {"exists": True, "device": str(after.st_dev), "file_id": str(after.st_ino),
            "size_bytes": after.st_size, "modified_ns": str(after.st_mtime_ns),
            "change_ns": str(after.st_ctime_ns), "sha256": hashlib.sha256(payload).hexdigest()}


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass
class _Budget:
    started: float
    attempts: int = 0
    requested_sleep: float = 0.0
    observed_sleep: float = 0.0


class AtomicStatusPublisher:
    """Publish canonical bytes to a fresh, single-writer mutable status path.

    ``publish(..., terminal=True)`` permits evidence of an observed STOP; it is
    not permission to dispatch work.  It never permits an expired deadline.
    The first successful call uses exclusive publication even if another file
    appears between construction and commit.  A fresh instance cannot adopt an
    existing status target.  Failure diagnostics identify an uncertain commit
    explicitly; callers must never re-dispatch the operation producing value.

    Clock/sleep injection is for CPU-only validation and fake-system lifecycle
    tests.  Limits are constants, not caller-selectable runtime relaxations.
    """

    def __init__(self, path, *, deadline_epoch, stop_requested, serialize,
                 now_fn=time.time, sleep_fn=time.sleep,
                 monotonic_fn=time.monotonic):
        if not all(callable(fn) for fn in (stop_requested, serialize, now_fn,
                                          sleep_fn, monotonic_fn)):
            raise ValueError("Publisher controls and serializer must be callable")
        self.path = Path(path)
        self.deadline_epoch = _finite(deadline_epoch, "Original deadline")
        self._stop = stop_requested
        self._serialize = serialize
        self._now = now_fn
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._last_wall = _finite(self._now(), "UTC clock")
        self._last_mono = _finite(self._monotonic(), "Monotonic clock")
        self._deadline_mono = self._last_mono + max(0.0, self.deadline_epoch - self._last_wall)
        self._published = False
        self._published_sha256 = None
        self._terminal_budget = None
        self._failure = None
        self.first_error = None

    @property
    def failure(self):
        """A detached diagnostic snapshot, or None; never a live-health cache."""
        return None if self._failure is None else copy.deepcopy(self._failure)

    def _clocks(self):
        wall = _finite(self._now(), "UTC clock")
        mono = _finite(self._monotonic(), "Monotonic clock")
        if wall < self._last_wall:
            raise RuntimeError("UTC clock moved backwards during status publication")
        if mono < self._last_mono:
            raise RuntimeError("Monotonic clock moved backwards during status publication")
        self._last_wall, self._last_mono = wall, mono
        return wall, mono

    def _boundary(self, budget, terminal, diagnostic):
        wall, mono = self._clocks()
        stopped = self._stop()
        if type(stopped) is not bool:
            raise ValueError("STOP predicate must return bool")
        diagnostic["stop_observed"] |= stopped
        if stopped and not terminal:
            raise InterruptedError("Status publication STOP requested before commit")
        if wall >= self.deadline_epoch or mono >= self._deadline_mono:
            raise TimeoutError("Original campaign deadline reached during status publication")
        if mono - budget.started >= MAX_RETRY_SECONDS or budget.requested_sleep >= MAX_RETRY_SECONDS:
            raise TimeoutError("Shared status publication retry budget exhausted")
        return min(MAX_RETRY_SECONDS - (mono - budget.started),
                   MAX_RETRY_SECONDS - budget.requested_sleep,
                   self.deadline_epoch - wall, self._deadline_mono - mono)

    def _ordinary_target(self):
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("Status target must be an ordinary file or fresh path")
        if self._published and not self.path.exists():
            raise FileNotFoundError(f"Previously published status target disappeared: {self.path}")

    def publish(self, value, *, terminal=False):
        if self.first_error is not None:
            # Do not allocate another temp, serialize again or reset any budget.
            raise self.first_error
        diagnostic = {"path": str(self.path), "temporary_path": None, "backup_path": None,
                      "attempts": 0, "phase": "begin", "committed": False,
                      "terminal": terminal, "stop_observed": False,
                      "deadline_epoch": self.deadline_epoch}
        temporary = None
        backup = None
        budget = None
        try:
            if type(terminal) is not bool:
                raise ValueError("Explicit terminal flag must be bool")
            if self._terminal_budget is not None and not terminal:
                raise RuntimeError("Normal status publication cannot resume after terminal evidence")
            _, mono = self._clocks()
            if terminal and self._terminal_budget is not None:
                budget = self._terminal_budget
            else:
                budget = _Budget(mono)
                if terminal:
                    self._terminal_budget = budget
            self._boundary(budget, terminal, diagnostic)
            self._ordinary_target()
            if not self._published and self.path.exists():
                raise FileExistsError(f"Initial status target already exists: {self.path}")
            diagnostic["phase"] = "serialize"
            payload = self._serialize(value)
            if type(payload) is not bytes:
                raise TypeError("Existing canonical serializer must return bytes")
            self._boundary(budget, terminal, diagnostic)
            original_state = None
            if self._published:
                original_state = _file_state(self.path)
                if (not original_state["exists"]
                        or original_state["sha256"] != self._published_sha256):
                    raise RuntimeError("Previously published status target changed or disappeared")
                self._boundary(budget, terminal, diagnostic)
            temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
            diagnostic.update(temporary_path=str(temporary), phase="write_temporary")
            with temporary.open("xb") as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise OSError("Short status temporary write")
                stream.flush()
                os.fsync(stream.fileno())
            self._boundary(budget, terminal, diagnostic)
            temporary_state = _file_state(temporary)
            if (temporary_state.get("sha256") != hashlib.sha256(payload).hexdigest()
                    or temporary_state.get("size_bytes") != len(payload)):
                raise RuntimeError("Complete status temporary bytes differ from canonical payload")
            if self._published:
                # The native API creates this unique, same-directory backup.
                # Never adopt or overwrite a pre-existing backup name.
                backup = temporary.with_suffix(".previous")
                diagnostic["backup_path"] = str(backup)
                if backup.exists() or backup.is_symlink():
                    raise FileExistsError(f"Status backup name already exists: {backup}")
            diagnostic["phase"] = "publish"
            delay = 0.01
            observations = []
            exclusive = not self._published
            while True:
                self._boundary(budget, terminal, diagnostic)
                self._ordinary_target()
                if budget.attempts >= MAX_ATTEMPTS:
                    raise TimeoutError("Shared status publication attempt limit exhausted")
                budget.attempts += 1
                diagnostic["attempts"] = budget.attempts
                try:
                    if exclusive:
                        # Windows rename refuses an existing destination.  The
                        # POSIX hard-link path also commits without replacement.
                        self._boundary(budget, terminal, diagnostic)
                        if os.name == "nt":
                            os.rename(temporary, self.path)
                        else:
                            os.link(temporary, self.path)
                    else:
                        if backup.exists() or backup.is_symlink():
                            raise FileExistsError(f"Status backup name already exists: {backup}")
                        self._boundary(budget, terminal, diagnostic)
                        _replace_existing(temporary, self.path, backup)
                except OSError as error:
                    code = getattr(error, "winerror", None)
                    diagnostic["native_error_code"] = code
                    # 1175/1176/1177 and all non-allowlisted errors fail closed.
                    # Even an allowlisted error is retryable only after proving
                    # old target unchanged, new source intact and backup absent.
                    uncommitted = False
                    try:
                        states = {"temporary": _file_state(temporary),
                                  "target": _file_state(self.path),
                                  "backup": _file_state(backup) if backup is not None else None}
                        diagnostic["failure_file_states"] = states
                        uncommitted = (not exclusive and states["target"] == original_state
                            and states["temporary"] == temporary_state
                            and states["backup"] == {"exists": False})
                    except BaseException as observation_error:
                        diagnostic["failure_observation_error"] = (
                            f"{type(observation_error).__name__}: {observation_error}")
                    if not exclusive and (not uncommitted or code in (1175, 1176, 1177)):
                        diagnostic["committed"] = None
                    elif exclusive and not temporary.exists():
                        diagnostic["committed"] = None
                    diagnostic["unchanged_before_retry_proven"] = uncommitted
                    if exclusive or code not in TRANSIENT_WINDOWS_REPLACE_ERRORS or not uncommitted:
                        raise
                    remaining = self._boundary(budget, terminal, diagnostic)
                    if budget.attempts >= MAX_ATTEMPTS:
                        raise TimeoutError("Shared status publication attempt limit exhausted") from error
                    duration = min(delay, MAX_BACKOFF_SECONDS, remaining)
                    if duration <= 0:
                        raise TimeoutError("Shared status publication retry budget exhausted") from error
                    observations.append({"attempt": budget.attempts, "winerror": code,
                                         "sleep_seconds": duration})
                    # Recheck immediately before sleeping; both callback work
                    # and native I/O elapsed time consume the same budget.
                    remaining = self._boundary(budget, terminal, diagnostic)
                    duration = min(duration, remaining)
                    before = self._last_mono
                    budget.requested_sleep += duration
                    self._sleep(duration)
                    self._boundary(budget, terminal, diagnostic)
                    budget.observed_sleep += self._last_mono - before
                    delay = min(delay * 2, MAX_BACKOFF_SECONDS)
                    continue
                diagnostic.update(committed=True, phase="post_commit")
                self._published = True
                self._published_sha256 = hashlib.sha256(payload).hexdigest()
                # POSIX exclusive publication leaves a second name.  Cleanup
                # failure is a committed failure, never a reason to republish.
                if exclusive and os.name != "nt":
                    temporary.unlink()
                self._boundary(budget, terminal, diagnostic)
                if backup is not None:
                    # Cleanup only this successful call's unique backup. A
                    # cleanup failure remains a committed, latched failure;
                    # it cannot be hidden or used to retry publication.
                    diagnostic["phase"] = "cleanup_own_backup"
                    backup.unlink()
                    self._boundary(budget, terminal, diagnostic)
                diagnostic["phase"] = "complete"
                return {"kind": "alignlab-atomic-status-publication-v10",
                        **diagnostic, "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload), "exclusive": exclusive,
                        "retry_observations": observations,
                        "requested_sleep_seconds": budget.requested_sleep,
                        "observed_sleep_seconds": budget.observed_sleep,
                        "elapsed_seconds": self._last_mono - budget.started,
                        "max_attempts": MAX_ATTEMPTS,
                        "max_retry_seconds": MAX_RETRY_SECONDS,
                        "temporary_file_remaining": temporary.exists(),
                        "backup_file_remaining": backup.exists() if backup is not None else False}
        except BaseException as error:
            if budget is not None:
                diagnostic.update(attempts=budget.attempts,
                                  requested_sleep_seconds=budget.requested_sleep)
            diagnostic.update(first_error_type=type(error).__name__, first_error_message=str(error))
            # Observations are diagnostic only: never mask the first failure
            # or claim an unavailable file is absent.
            for key, evidence in (("temporary_file_remaining", temporary),
                                  ("backup_file_remaining", backup)):
                try:
                    diagnostic[key] = evidence.exists() if evidence is not None else False
                except OSError:
                    diagnostic[key] = None
            self._failure = diagnostic
            self.first_error = error
            error.add_note("Atomic status publication failure latched; retained temporary "
                           f"evidence when present: {temporary}; backup: {backup}; destination: {self.path}; "
                           f"attempts: {diagnostic['attempts']}; committed: {diagnostic['committed']}")
            raise
