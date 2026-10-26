"""Bounded, read-only process identity decisions for a new execution identity.

This does not change the historical guard, acquire GPU resources, launch/kill a
process, or write STOP. A supervisor must still enforce source/artifact/STOP and
GPU resource checks, and request a clean child boundary when requires_safe_stop
is true. The reader must itself have a bounded OS-query timeout. An empty or
keyword-free command is never evidence that a process is harmless CPU work.

Only explicitly vetted PID + creation + exact-command identities are allowed.
Include vetted background APIs as well as owned children; never allow a script
name globally. Caller remains responsible for authenticating child ancestry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import re
import time
from typing import Callable, Iterable, Mapping


VERSION = "alignlab-process-guard-v2"
DEFAULT_BUSY_MARKERS = (
    "run_alignlab_", "dispatch_alignlab_", "supervise_alignlab_",
    "run_campaign.py", "local-train", "local-evaluate", "windows.train",
    "train_sft", "train_dpo", "train_ppo", "run_training", "serve_adapter",
    "bootstrap_run.py", "fetch_alignlab_model.py", "forgelm.campaign",
)
_STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)$")


def _stamp(value: str) -> str:
    """Normalize offsets without discarding Windows' seventh fractional digit."""
    match = _STAMP.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("An explicit timezone-aware creation timestamp is required")
    whole, fraction, zone = match.groups()
    utc = datetime.fromisoformat(whole + zone.replace("Z", "+00:00")).astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + "." + (fraction or "").ljust(9, "0") + "Z"


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int | None
    created_at_utc: str
    command_line: str | None

    @property
    def instance(self):
        return self.pid, self.created_at_utc

    @classmethod
    def from_record(cls, value):
        if isinstance(value, cls):
            value = asdict(value)
        if not isinstance(value, Mapping):
            raise ValueError("Process record must be a mapping")
        pid, parent, command = value.get("pid"), value.get("parent_pid"), value.get("command_line")
        if type(pid) is not int or pid <= 0:
            raise ValueError("Process PID must be a positive integer")
        if parent is not None and (type(parent) is not int or parent < 0):
            raise ValueError("Invalid parent PID")
        if command is not None and not isinstance(command, str):
            raise ValueError("Process command must be a string or null")
        return cls(pid, parent, _stamp(value.get("created_at_utc")), command)


@dataclass(frozen=True)
class GuardBlocker:
    code: str
    message: str
    process: ProcessIdentity | None = None


@dataclass(frozen=True)
class SnapshotObservation:
    attempt: int
    started_at_epoch: float
    finished_at_epoch: float
    snapshot: tuple[ProcessIdentity, ...]
    blockers: tuple[GuardBlocker, ...]
    snapshot_valid: bool


@dataclass(frozen=True)
class GuardResult:
    admission_allowed: bool
    blockers: tuple[GuardBlocker, ...]
    observations: tuple[SnapshotObservation, ...]
    final_snapshot: tuple[ProcessIdentity, ...]
    elapsed_seconds: float

    @property
    def requires_safe_stop(self):
        return not self.admission_allowed

    @property
    def decision(self):
        return "allow" if self.admission_allowed else "block_new_dispatch_and_request_safe_stop"

    @property
    def attempt_count(self):
        return len(self.observations)

    def to_dict(self):
        """Return an independent JSON-compatible copy, with no live objects."""
        return {"schema_version": 1, "kind": VERSION, **asdict(self),
                "decision": self.decision, "requires_safe_stop": self.requires_safe_stop,
                "attempt_count": self.attempt_count,
                "gpu_resource_admission_verified": False}


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def inspect_processes(snapshot_reader: Callable, *, allowed_identities: Iterable = (),
                      busy_markers=DEFAULT_BUSY_MARKERS, max_attempts=3,
                      retry_delay_seconds=0.25, deadline_epoch=None,
                      clock=time.time, sleep=time.sleep) -> GuardResult:
    """Reobserve uncertain identities; never grandfather a reused PID.

    Persistent unknown/nonempty commands also block: arbitrary Python can use
    CUDA. A known competing command or drift of an allowed exact command blocks
    immediately. Unknown instances that disappear are resolved only by a fresh
    successful complete snapshot. Allowed records absent from a snapshot are not
    treated as alive; caller separately checks its own required process tree.
    """
    if type(max_attempts) is not int or not 2 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 2 and 5")
    delay = _finite(retry_delay_seconds, "retry_delay_seconds")
    if not 0 <= delay <= 2:
        raise ValueError("retry_delay_seconds must be between 0 and 2")
    if deadline_epoch is not None:
        deadline_epoch = _finite(deadline_epoch, "deadline_epoch")
    if isinstance(busy_markers, str):
        raise ValueError("busy_markers must be a sequence, not a string")
    markers = tuple(busy_markers)
    if not markers or any(not isinstance(m, str) or not m.strip() for m in markers):
        raise ValueError("Nonempty competing-command markers required")
    markers = tuple(m.lower() for m in markers)
    allowed = {}
    for raw in allowed_identities:
        record = ProcessIdentity.from_record(raw)
        if not record.command_line or not record.command_line.strip():
            raise ValueError("Allowed identities need their original full command")
        previous = allowed.get(record.instance)
        if previous is not None and previous != record:
            raise ValueError("Conflicting allowed identities for one process instance")
        allowed[record.instance] = record
    observations, final_snapshot = [], ()
    started = _finite(clock(), "clock")
    last_clock = started

    def finish(blockers):
        return GuardResult(not blockers, tuple(blockers), tuple(observations), final_snapshot,
                           max(0.0, last_clock - started))

    def boundary():
        nonlocal last_clock
        now = _finite(clock(), "clock")
        if now < last_clock:
            return GuardBlocker("clock_moved_backwards", "Cannot trust the process admission time boundary")
        last_clock = now
        if deadline_epoch is not None and now >= deadline_epoch:
            return GuardBlocker("deadline_exhausted", "No dispatch or further identity query after the absolute deadline")
        return None

    for attempt in range(1, max_attempts + 1):
        expired = boundary()
        if expired:
            return finish((expired,))
        before, blockers, snapshot, valid = last_clock, [], (), False
        try:
            raw_snapshot = snapshot_reader()
            if not isinstance(raw_snapshot, (list, tuple)):
                raise ValueError("Reader must return a complete list/tuple, including [] when empty")
            snapshot = tuple(ProcessIdentity.from_record(p) for p in raw_snapshot)
            if len({p.pid for p in snapshot}) != len(snapshot):
                raise ValueError("Duplicate PID records make the snapshot ambiguous")
            valid, final_snapshot = True, snapshot
            for record in snapshot:
                expected = allowed.get(record.instance)
                if not record.command_line or not record.command_line.strip():
                    blockers.append(GuardBlocker("unknown_command", "Full command unavailable; reobserve this process instance", record))
                elif expected is not None:
                    if record.command_line != expected.command_line:
                        blockers.append(GuardBlocker("allowed_command_changed", "Same PID/creation has a different full command", record))
                elif any(marker in record.command_line.lower() for marker in markers):
                    blockers.append(GuardBlocker("competing_process", "A non-allowed training/probe/service/queue command is active", record))
                else:
                    blockers.append(GuardBlocker("unclassified_process", "Command is not an explicitly vetted identity; CPU-only use is unproven", record))
        except Exception as exc:
            blockers = [GuardBlocker("snapshot_unavailable", f"{type(exc).__name__}: {exc}")]
        expired = boundary()
        observations.append(SnapshotObservation(attempt, before, last_clock, snapshot, tuple(blockers), valid))
        if expired:
            return finish((*blockers, expired))
        if not blockers:
            return finish(())
        if any(b.code in ("allowed_command_changed", "competing_process") for b in blockers):
            return finish(blockers)
        if attempt == max_attempts:
            return finish(blockers)
        if delay:
            remaining = delay if deadline_epoch is None else min(delay, max(0.0, deadline_epoch - last_clock))
            if remaining:
                sleep(remaining)
    raise AssertionError("Unreachable bounded identity guard")
