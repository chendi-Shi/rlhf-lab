"""Bounded waiting at an already-complete optimizer boundary.

The frozen v2 classifier remains authoritative: unknown Python is never allowed.
Only a fresh complete clear snapshot may release a blocked boundary. A caller
must not compute training work while this synchronous function is running, and
must commit its safe optimizer boundary in on_wait_start before any waiting.
No process is stopped, restarted, whitelisted dynamically, or given more budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

from . import process_guard_v2 as strict

VERSION = "alignlab-process-guard-v3-boundary-wait"
WAITABLE = frozenset({"unknown_command", "unclassified_process", "snapshot_unavailable"})


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return float(value)


@dataclass(frozen=True)
class WaitPolicy:
    max_wait_seconds: float = 300.0
    total_wait_seconds: float = 900.0
    poll_seconds: float = 2.0

    def __post_init__(self):
        maximum = _finite(self.max_wait_seconds, "max_wait_seconds")
        total = _finite(self.total_wait_seconds, "total_wait_seconds")
        poll = _finite(self.poll_seconds, "poll_seconds")
        if not 0 < maximum <= 300 or not 0 < total <= 900 or not 0 < poll <= 5:
            raise ValueError("Finite bounded wait policy required")


@dataclass
class WaitBudget:
    spent_seconds: float = 0.0

    def __post_init__(self):
        if _finite(self.spent_seconds, "spent_seconds") < 0:
            raise ValueError("Spent waiting time cannot be negative")


@dataclass(frozen=True)
class BoundaryWaitResult:
    admission_allowed: bool
    reason: str
    waited_seconds: float
    total_waited_seconds: float
    observations: tuple
    last_guard: dict | None

    def to_dict(self):
        return {"schema_version": 1, "kind": VERSION, **asdict(self),
                "unknown_processes_allowed": False, "training_during_wait": False}


class _ClockBackwards(ValueError):
    pass


def wait_for_clearance(snapshot_reader, *, owned_identities, allowed_identities=(),
                       deadline_epoch, budget, policy=WaitPolicy(),
                       stop_requested=lambda: False, identity_unchanged=lambda: True,
                       clock=time.time, sleep=time.sleep,
                       on_wait_start=None, on_observation=None):
    """Block a safe boundary for transient uncertainty, without authorizing it.

    Hard conflicts and own-process identity changes stop immediately. External
    unknown identities and unavailable snapshots get bounded re-observation.
    Query/checkpoint/logging/sleep time during an uncertainty episode all count
    toward the same per-episode, cumulative and absolute wall-clock deadlines.
    The OS snapshot reader must itself have a bounded timeout.
    """
    if not isinstance(policy, WaitPolicy) or not isinstance(budget, WaitBudget):
        raise ValueError("Typed wait policy and persistent cumulative budget required")
    deadline = _finite(deadline_epoch, "deadline_epoch")
    spent_before = _finite(budget.spent_seconds, "spent_seconds")
    if spent_before < 0:
        raise ValueError("Invalid spent wait budget")
    owned = tuple(strict.ProcessIdentity.from_record(p) for p in owned_identities)
    if not owned or any(not p.command_line or not p.command_line.strip() for p in owned):
        raise ValueError("Complete real owned process identities required")
    if len({p.pid for p in owned}) != len(owned):
        raise ValueError("Duplicate owned process IDs")
    allowed = [*allowed_identities, *owned]
    observations, last_guard = [], None
    last_time = _finite(clock(), "clock")
    started = last_time
    waiting = False
    wait_hook_called = False

    def now():
        nonlocal last_time
        value = _finite(clock(), "clock")
        if value < last_time:
            raise _ClockBackwards("Clock moved backwards")
        last_time = value
        return value

    def account():
        elapsed = max(0.0, last_time - started) if waiting else 0.0
        budget.spent_seconds = spent_before + elapsed
        return elapsed

    def finish(allowed, reason):
        # A failed save/log callback can consume wall time before raising.
        # Retain that time in the same bounded wait ledger as successful work.
        try:
            now()
        except _ClockBackwards:
            allowed, reason = False, "clock_moved_backwards"
        except Exception:
            allowed, reason = False, "callback_failed"
        elapsed = account()
        if allowed and last_time >= deadline:
            allowed, reason = False, "deadline_exhausted"
        if allowed and waiting and (elapsed >= policy.max_wait_seconds
                                    or budget.spent_seconds >= policy.total_wait_seconds):
            allowed, reason = False, "wait_limit_exhausted"
        return BoundaryWaitResult(allowed, reason, elapsed, budget.spent_seconds,
                                  tuple(observations), last_guard)

    def boundary():
        current = now()
        if current >= deadline:
            return "deadline_exhausted"
        try:
            if stop_requested():
                return "stop_requested"
            if identity_unchanged() is not True:
                return "identity_changed"
        except Exception:
            return "callback_failed"
        current = now()  # Caller checks may consume real wall time.
        if current >= deadline:
            return "deadline_exhausted"
        if waiting and (current - started >= policy.max_wait_seconds
                        or account() + spent_before >= policy.total_wait_seconds):
            return "wait_limit_exhausted"
        return None

    # Independent count bound also fails closed if a custom clock/sleeper stalls.
    max_cycles = math.ceil(policy.max_wait_seconds / policy.poll_seconds) + 3
    try:
        for cycle in range(1, max_cycles + 1):
            reason = boundary()
            if reason:
                return finish(False, reason)
            decision = strict.inspect_processes(snapshot_reader, allowed_identities=allowed,
                max_attempts=2, retry_delay_seconds=0, deadline_epoch=deadline,
                clock=now, sleep=sleep)
            last_guard = decision.to_dict()
            reason = boundary()
            if reason:
                return finish(False, reason)
            # A valid fresh snapshot that loses or changes our own process is a
            # hard identity failure, unlike an unavailable whole-system query.
            fresh = bool(decision.observations and decision.observations[-1].snapshot_valid)
            if fresh:
                actual = {p.pid: p for p in decision.observations[-1].snapshot}
                if any(actual.get(p.pid) != p for p in owned):
                    return finish(False, "own_identity_changed")
            observation = {"cycle": cycle, "at_epoch": last_time, "guard": last_guard,
                           "waited_seconds": account(), "total_waited_seconds": budget.spent_seconds}
            observations.append(observation)
            if on_observation is not None:
                try:
                    on_observation(observation)
                except Exception:
                    return finish(False, "callback_failed")
            reason = boundary()
            if reason:
                return finish(False, reason)
            if decision.admission_allowed and fresh:
                return finish(True, "cleared")
            codes = {item.code for item in decision.blockers}
            if not codes or not codes <= WAITABLE:
                return finish(False, "hard_blocker")
            waiting = True
            if not wait_hook_called:
                # No later update can happen before this caller's synchronous
                # safe checkpoint completes, and a failed save never resumes.
                if on_wait_start is not None:
                    try:
                        on_wait_start()
                    except Exception:
                        return finish(False, "callback_failed")
                wait_hook_called = True
            reason = boundary()
            if reason:
                return finish(False, reason)
            remaining = min(deadline - last_time, policy.max_wait_seconds - (last_time - started),
                            policy.total_wait_seconds - (spent_before + last_time - started))
            if remaining <= 0:
                return finish(False, "wait_limit_exhausted")
            sleep(min(policy.poll_seconds, remaining))
        return finish(False, "wait_limit_exhausted")
    except _ClockBackwards:
        return finish(False, "clock_moved_backwards")
    except Exception:
        return finish(False, "callback_failed")
