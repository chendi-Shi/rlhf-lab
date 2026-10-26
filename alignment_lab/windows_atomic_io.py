"""Bounded, durable JSON publication on Windows without weakening train deadlines.

Only transient Windows access/sharing/lock denials (5/32/33) while publishing a
complete same-directory temporary file are retried. Every failure preserves the
temporary evidence. Callers must propagate errors for weights/receipts; this
module never treats failed publication as success. Terminal-only evidence may
use an explicitly separate short flush deadline selected by its caller; this
writer never invents one or extends the supplied deadline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

SOURCE_FILES = ("alignment_lab/windows_atomic_io.py", "tests_v2/test_windows_atomic_io.py")
TRANSIENT_WINDOWS_ERRORS = frozenset({5, 32, 33})


def _finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return float(value)


def _exclusive_publish(temporary, target):
    # Windows rename is atomic and refuses an existing destination. The POSIX
    # fallback uses an atomic exclusive hard-link publication, never replace.
    if os.name == "nt":
        os.rename(temporary, target)
    else:
        os.link(temporary, target)
        temporary.unlink()


def write_json_atomic(path, value, *, deadline_epoch, stop_requested=lambda: False,
        exclusive=False, max_wait_seconds=5.0, initial_delay_seconds=.01,
        max_delay_seconds=.25, max_attempts=64, clock=time.time, sleep=time.sleep,
        replace=None, on_retry=None):
    """Publish complete finite JSON, or raise while retaining any temporary file.

    The elapsed cap includes serialization, fsync, publication attempts, and
    retry waits. Temporary publication failures never cause a numerical update
    or a retry of the operation that generated ``value``. ``stop_requested`` is
    checked before creating the temporary file and before every publication.
    Callback errors also fail closed. The receipt identifies only JSON I/O, not
    a successful training stage or independently verified checkpoint.
    """
    deadline = _finite(deadline_epoch, "deadline_epoch")
    maximum = _finite(max_wait_seconds, "max_wait_seconds", positive=True)
    initial = _finite(initial_delay_seconds, "initial_delay_seconds", positive=True)
    ceiling = _finite(max_delay_seconds, "max_delay_seconds", positive=True)
    if initial > ceiling:
        raise ValueError("initial_delay_seconds must not exceed max_delay_seconds")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 100000:
        raise ValueError("max_attempts must be a bounded positive integer")
    if type(exclusive) is not bool or not callable(stop_requested) or not callable(clock) or not callable(sleep):
        raise ValueError("Explicit exclusive flag and callable controls are required")
    if replace is not None and not callable(replace):
        raise ValueError("replace must be callable")
    if on_retry is not None and not callable(on_retry):
        raise ValueError("on_retry must be callable")
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("JSON destination must be an ordinary file or a new path")
    if exclusive and target.exists():
        raise FileExistsError(f"Exclusive JSON destination already exists: {target}")
    started = _finite(clock(), "clock")
    until = min(deadline, started + maximum)
    latest = started
    temporary = None
    attempts = 0
    waited = 0.0
    observations = []

    def boundary():
        nonlocal latest
        now = _finite(clock(), "clock")
        if now < latest:
            raise RuntimeError("Clock moved backwards during JSON publication")
        latest = now
        if stop_requested():
            raise InterruptedError("JSON publication stopped before commit")
        if now >= deadline:
            raise TimeoutError("Original absolute deadline reached before JSON publication")
        if now >= until:
            raise TimeoutError("Bounded JSON publication window exhausted")
        return now

    try:
        boundary()
        payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        boundary()
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        publish = _exclusive_publish if exclusive else (os.replace if replace is None else replace)
        delay = initial
        while True:
            boundary()
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("JSON destination changed to a nonordinary file")
            attempts += 1
            try:
                publish(temporary, target)
            except OSError as error:
                winerror = getattr(error, "winerror", None)
                if winerror not in TRANSIENT_WINDOWS_ERRORS:
                    raise
                now = boundary()
                if attempts >= max_attempts:
                    raise TimeoutError("JSON publication attempt limit exhausted") from error
                duration = min(delay, until - now)
                if duration <= 0:
                    raise TimeoutError("Bounded JSON publication window exhausted") from error
                observation = {"attempt": attempts, "winerror": winerror,
                    "elapsed_seconds": now - started, "sleep_seconds": duration,
                    "temporary_path": str(temporary), "target_path": str(target)}
                observations.append(observation)
                if on_retry is not None:
                    on_retry(dict(observation))
                # Callback work is charged too; never sleep beyond either bound.
                now = boundary()
                duration = min(duration, until - now)
                if duration <= 0:
                    raise TimeoutError("Bounded JSON publication window exhausted") from error
                sleep_started = now
                sleep(duration)
                after = _finite(clock(), "clock")
                if after < sleep_started:
                    raise RuntimeError("Clock moved backwards during JSON retry sleep")
                waited += after - sleep_started
                delay = min(delay * 2, ceiling)
                continue
            finished = _finite(clock(), "clock")
            if finished < latest:
                # Publication has already committed; make this explicit rather
                # than suggesting the caller can repeat a state-changing task.
                raise RuntimeError("Clock moved backwards after JSON commit; destination may be committed")
            return {"kind": "alignlab-atomic-json-publication-v1", "committed": True,
                "path": str(target), "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload), "exclusive": exclusive, "attempts": attempts,
                "retried": bool(observations), "retry_observations": observations,
                "waited_seconds": waited, "elapsed_seconds": finished - started,
                "deadline_epoch": deadline, "publication_window_seconds": maximum,
                "temporary_path": str(temporary), "temporary_file_remaining": temporary.exists()}
    except BaseException as error:
        if temporary is not None:
            error.add_note(f"Atomic JSON evidence retained when present: {temporary}; destination: {target}; attempts: {attempts}")
        raise
