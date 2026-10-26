"""Strict process admission and synchronous safe-boundary waits for six branches.

Unknown Python is never approved. Waiting performs no training, preserves the
same campaign deadline and a cumulative wait ledger shared by successive stages.
The caller must commit the current complete optimizer or cache-record boundary
before sleeping. This module never changes STOP, launches or kills a process.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import re
import os
from pathlib import Path
import time

from . import process_guard_v2 as strict
from .process_guard_v3 import WaitPolicy, WaitBudget, wait_for_clearance

KIND = "alignlab-alignment-process-boundary-v1"
CACHE_RECORD_INTERVAL = 16
POLICY = {"no_unknown_process_whitelist": True,
    "training_stops_computing_while_unknown": True,
    "safe_boundary_checkpoint_before_wait": True,
    "max_wait_seconds": 300, "max_total_wait_seconds": 900,
    "poll_interval_seconds": 2,
    "stop_deadline_and_identity_failures_not_ignored": True}
SOURCE_FILES = {"alignment_lab/alignment_process_guard.py", "alignment_lab/process_guard_v2.py",
    "alignment_lab/process_guard_v3.py", "tests_v2/test_alignment_process_guard.py"}


def _json_equal(left, right):
    """Compare all persisted fields, normalizing only JSON tuple/list encoding."""
    return json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def validate_policy(policy):
    if type(policy) is not dict or policy != POLICY or any(type(policy[k]) is not type(v) for k, v in POLICY.items()):
        raise ValueError("Exact frozen 300/900-second alignment boundary policy required")
    return WaitPolicy(policy["max_wait_seconds"], policy["max_total_wait_seconds"], policy["poll_interval_seconds"])


def validate_background(rows):
    if type(rows) is not list:
        raise ValueError("Explicit vetted background identity list required")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"pid", "parent_pid", "created_at_utc", "command_line"}:
            raise ValueError("Exact PID/create/full-command/parent identities required")
        item = strict.ProcessIdentity.from_record(row)
        if not item.command_line or not item.command_line.strip() or item.pid in seen:
            raise ValueError("Duplicate or incomplete background identity")
        seen.add(item.pid)
    return rows


def _owned_present(owned, snapshot):
    actual = {p.pid: p for p in snapshot}
    return bool(owned) and all(actual.get(p.pid) == p for p in owned)


def _allowed_ancestry_unchanged(allowed, snapshot):
    pinned = {(item.pid, item.created_at_utc): item.parent_pid
              for item in (strict.ProcessIdentity.from_record(row) for row in allowed)}
    return all((item.pid, item.created_at_utc) not in pinned
               or item.parent_pid == pinned[(item.pid, item.created_at_utc)] for item in snapshot)


def strict_admission(snapshot_reader, *, owned_identities, allowed_identities=(),
                     deadline_epoch, clock=time.time, sleep=time.sleep):
    owned = tuple(strict.ProcessIdentity.from_record(p) for p in owned_identities)
    if not owned or any(not p.command_line or not p.command_line.strip() for p in owned):
        raise ValueError("Complete nonempty owned process identities required")
    decision = strict.inspect_processes(snapshot_reader,
        allowed_identities=[*allowed_identities, *owned], deadline_epoch=deadline_epoch,
        max_attempts=2, retry_delay_seconds=0, clock=clock, sleep=sleep)
    own_ok = _owned_present(owned, decision.final_snapshot)
    ancestry_ok = all(_allowed_ancestry_unchanged(allowed_identities, observation.snapshot)
                      for observation in decision.observations if observation.snapshot_valid)
    return {**decision.to_dict(), "owned_identity_verified": own_ok,
            "allowed_ancestry_verified": ancestry_ok,
            "caller_admission_allowed": decision.admission_allowed and own_ok and ancestry_ok}


def child_identities(snapshot, request, *, pid=None):
    """Identify only the exact stage process/wrapper chain under its supervisor."""
    from .study_supervisor import _command
    pid = os.getpid() if pid is None else pid
    supervisors = {p["pid"]: strict.ProcessIdentity.from_record(p) for p in request["supervisor_processes"]}
    if not supervisors:
        raise ValueError("Pinned active supervisor identities required")
    by_pid = {p["pid"]: p for p in snapshot}
    request_path = Path(request["output"]).parent / (Path(request["output"]).name + ".request.json")
    current, result, seen = by_pid.get(pid), [], set()
    while current is not None:
        identity = strict.ProcessIdentity.from_record(current)
        if identity.pid in seen:
            raise RuntimeError("Cyclic child ancestry")
        seen.add(identity.pid)
        if identity.pid in supervisors:
            if identity != supervisors[identity.pid] or not result:
                raise RuntimeError("Supervisor identity changed or missing stage process")
            return result
        if not _command(current, "run_alignlab_alignment_stage.py", request_path):
            raise RuntimeError("Unverified child or wrapper command")
        result.append(current)
        current = by_pid.get(identity.parent_pid)
    raise RuntimeError("Child does not descend from a pinned active supervisor")


def verify_boundary_receipt(receipt, optimizer_updates):
    fields = {"complete", "optimizer_updates", "kind", "path", "sha256"}
    if (type(receipt) is not dict or set(receipt) != fields or receipt.get("complete") is not True
            or type(receipt.get("optimizer_updates")) is not int or receipt["optimizer_updates"] != optimizer_updates
            or receipt.get("kind") not in {"optimizer_checkpoint", "cache_record_boundary", "before_cuda"}
            or not isinstance(receipt.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is None):
        raise ValueError("Exact pinned complete boundary receipt required")
    path = Path(receipt["path"])
    if (not path.is_absolute() or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
            or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]):
        raise ValueError("Committed safe boundary file identity changed")
    return receipt

class GuardSession:
    def __init__(self, snapshot_reader, *, owned_identities, allowed_identities=(),
                 deadline_epoch, policy, spent_seconds=0.0,
                 stop_requested=lambda: False, identity_unchanged=lambda: True,
                 clock=time.time, sleep=time.sleep, on_event=lambda event: None):
        self.reader = snapshot_reader
        self.owned = list(owned_identities)
        self.allowed = list(allowed_identities)
        self.deadline = deadline_epoch
        self.policy = validate_policy(policy)
        if type(spent_seconds) not in (int, float) or not math.isfinite(spent_seconds) or not 0 <= spent_seconds < policy["max_total_wait_seconds"]:
            raise ValueError("Inherited campaign wait ledger must remain below 900 seconds")
        self.initial_spent_seconds = float(spent_seconds)
        self.budget = WaitBudget(float(spent_seconds))
        self.stop, self.identity = stop_requested, identity_unchanged
        self.clock, self.sleep, self.emit = clock, sleep, on_event
        self.blocked = False
        self.calls = 0
        self.last = None

    def check(self, *, boundary, optimizer_updates, save_boundary):
        if self.blocked:
            raise RuntimeError("A denied guard session cannot be resumed automatically")
        if type(boundary) is not str or not boundary or type(optimizer_updates) is not int or optimizer_updates < 0:
            raise ValueError("Explicit complete boundary identity required")
        self.calls += 1
        context = {"kind": KIND, "boundary": boundary, "optimizer_updates": optimizer_updates,
                   "call": self.calls, "deadline_epoch": self.deadline}
        saved = None
        ancestry_ok = True

        def read_snapshot():
            nonlocal ancestry_ok
            raw = self.reader()
            try:
                snapshot = tuple(strict.ProcessIdentity.from_record(row) for row in raw)
            except (ValueError, TypeError, KeyError):
                return raw  # The frozen classifier rejects an invalid snapshot.
            ancestry_ok = ancestry_ok and _allowed_ancestry_unchanged(self.allowed, snapshot)
            return raw

        def save():
            nonlocal saved
            saved = save_boundary()
            verify_boundary_receipt(saved, optimizer_updates)
            self.emit({**context, "event": "wait_start", "at_epoch": self.clock(), "saved_boundary": saved})

        def observe(observation):
            self.emit({**context, "event": "observation", **observation})

        def identity():
            if not ancestry_ok or self.identity() is not True:
                return False
            if saved is not None:
                verify_boundary_receipt(saved, optimizer_updates)
            return True

        result = wait_for_clearance(read_snapshot, owned_identities=self.owned,
            allowed_identities=self.allowed, deadline_epoch=self.deadline,
            budget=self.budget, policy=self.policy, stop_requested=self.stop,
            identity_unchanged=identity, clock=self.clock, sleep=self.sleep,
            on_wait_start=save, on_observation=observe).to_dict()
        self.last = {**context, **result, "event": "result", "saved_boundary": saved,
            "inherited_waited_seconds": self.initial_spent_seconds, "at_epoch": self.clock()}
        self.blocked = not result["admission_allowed"]
        self.emit(self.last)
        return self.last

def verify_guard_log(path, *, request, progress):
    """Read-only independent replay of a completed stage's boundary evidence."""
    import json
    validate_policy(request["process_wait_policy"])
    path = Path(path)
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if not rows:
        raise ValueError("No actual process-boundary observations")
    owned = progress["process_guard"]["owned_identities"]
    validate_background(owned)
    expected_owned = tuple(strict.ProcessIdentity.from_record(p) for p in owned)
    allowed = [*owned, *request["supervisor_processes"], *request["allowed_background_processes"]]
    inherited = request["process_wait_spent_seconds"]
    previous_total, deadline = inherited, request["deadline_epoch"]
    results, observations, saved, call = [], [], None, 1
    previous_at, group_boundary, group_update = progress["started_at_epoch"], None, None
    for row in rows:
        if (row.get("call") != call or row.get("deadline_epoch") != deadline
                or type(row.get("optimizer_updates")) is not int or row["optimizer_updates"] < 0
                or type(row.get("at_epoch")) not in (int, float)
                or not previous_at <= row["at_epoch"] < deadline):
            raise ValueError("Boundary sequence, deadline or observation time changed")
        previous_at = row["at_epoch"]
        if group_boundary is None:
            group_boundary, group_update = row.get("boundary"), row["optimizer_updates"]
        elif row.get("boundary") != group_boundary or row["optimizer_updates"] != group_update:
            raise ValueError("A boundary call changed its phase or optimizer cursor")
        event = row.get("event")
        if event == "observation":
            evidence = row["guard"]
            from .recovered_parent_acceptance_r5 import _guard_record
            clear = _guard_record(evidence, owned, allowed,
                {"started_at_epoch": progress["started_at_epoch"], "deadline_epoch": deadline})
            if any(not _allowed_ancestry_unchanged(allowed,
                    tuple(strict.ProcessIdentity.from_record(item) for item in observation["snapshot"]))
                    for observation in evidence["observations"] if observation["snapshot_valid"]):
                raise ValueError("Pinned allowed process ancestry changed")
            if (row.get("cycle") != len(observations) + 1
                    or observations and observations[-1]["guard"]["admission_allowed"]
                    or evidence["observations"][-1]["finished_at_epoch"] > row["at_epoch"]):
                raise ValueError("Unbounded, reordered or post-clear process observations")
            observations.append({key: row[key] for key in
                ("cycle", "at_epoch", "guard", "waited_seconds", "total_waited_seconds")})
        elif event == "wait_start":
            if saved is not None or not observations or observations[-1]["guard"]["admission_allowed"]:
                raise ValueError("Waiting must follow a classified unknown boundary exactly once")
            saved = verify_boundary_receipt(row["saved_boundary"], row["optimizer_updates"])
            if saved["kind"] == "optimizer_checkpoint":
                candidates = [c for c in progress["checkpoints"]
                    if str((Path(c["checkpoint_path"]) / "manifest.json").resolve()) == saved["path"]
                    and c["manifest_sha256"] == saved["sha256"] and c["optimizer_updates"] == row["optimizer_updates"]]
                manifest = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
                if not candidates or manifest.get("status") != "complete" or manifest.get("identity") != progress["identity"]:
                    raise ValueError("Waiting did not pin the current actual complete optimizer checkpoint")
            elif saved["kind"] == "before_cuda":
                if Path(saved["path"]).resolve() != Path(request["output"]).resolve() / "request.json":
                    raise ValueError("Initial idle-boundary receipt differs from the frozen request")
            else:
                cache = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
                journal = Path(request["output"]) / "reference-cache-records.jsonl"
                committed = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
                matched_boundary = re.fullmatch(r"cache_record:(\d+)", group_boundary or "")
                expected_index = (int(matched_boundary.group(1)) if matched_boundary else -1)
                if request["reference_cache"]["mode"] == "reuse":
                    expected_index = sum(item["boundary"].startswith("cache_record:") for item in results)
                if (cache.get("kind") != "alignlab-safe-cache-record-boundary-v1" or cache.get("complete") is not True
                        or cache.get("optimizer_updates") != 0 or Path(cache["cache_journal"]).resolve() != journal.resolve()
                        or matched_boundary is None or not 0 <= expected_index < len(committed)
                        or cache["committed_record"] != committed[expected_index]
                        or cache["committed_record"].get("scored_sequences") != int(matched_boundary.group(1))):
                    raise ValueError("Cache wait lacks its committed complete scoring record")
        elif event == "result":
            if (row.get("schema_version") != 1 or row.get("kind") != "alignlab-process-guard-v3-boundary-wait"
                    or row.get("admission_allowed") is not True or row.get("reason") != "cleared"
                    or row.get("unknown_processes_allowed") is not False or row.get("training_during_wait") is not False
                    or row.get("inherited_waited_seconds") != inherited or row.get("saved_boundary") != saved
                    or row.get("observations") != observations or not observations
                    or row.get("last_guard") != observations[-1]["guard"]):
                raise ValueError("A denied, incomplete or changed boundary cannot accept a stage")
            last = row["last_guard"]
            if (last.get("blockers") or not last["observations"][-1]["snapshot_valid"]
                    or not _owned_present(expected_owned, tuple(strict.ProcessIdentity.from_record(p) for p in last["final_snapshot"]))):
                raise ValueError("Only a fresh complete strict clear snapshot can release a boundary")
            wait, total = row["waited_seconds"], row["total_waited_seconds"]
            if (type(wait) not in (int, float) or type(total) not in (int, float)
                    or not math.isfinite(wait) or not math.isfinite(total)
                    or not 0 <= wait < 300 or not previous_total <= total < 900
                    or not math.isclose(total - previous_total, wait, abs_tol=1e-6, rel_tol=1e-12)
                    or wait > 0 and saved is None):
                raise ValueError("Wait budget was reset, under-accounted, or exceeded")
            waiting = observations[0]["guard"]["admission_allowed"] is False
            if (waiting and (saved is None or len(observations) < 2 or wait <= 0)
                    or not waiting and (saved is not None or len(observations) != 1 or wait != 0)):
                raise ValueError("Unknown boundary needs an actual saved hold and later fresh clearance")
            last_waited = 0.0
            for index, item in enumerate(observations):
                waited, cumulative = item["waited_seconds"], item["total_waited_seconds"]
                if (type(waited) not in (int, float) or type(cumulative) not in (int, float)
                        or not math.isfinite(waited) or not math.isfinite(cumulative)
                        or index == 0 and (waited != 0 or cumulative != previous_total)
                        or waited < last_waited - 1e-6 or waited > wait + 1e-6
                        or not math.isclose(cumulative, previous_total + waited, rel_tol=1e-12, abs_tol=1e-6)
                        or item["at_epoch"] - observations[0]["at_epoch"] > waited + 1e-6):
                    raise ValueError("Unknown waiting time was reset or undercounted")
                last_waited = waited
            previous_total = total
            results.append(row)
            observations, saved, call = [], None, call + 1
            group_boundary, group_update = None, None
        else:
            raise ValueError("Unknown boundary evidence event")
    if observations or saved is not None or not results:
        raise ValueError("Unfinished process boundary")
    names = [r["boundary"] for r in results]
    updates = [r["optimizer_updates"] for r in results if r["boundary"] == "optimizer_complete"]
    target = request["workload"]["optimizer_updates"]
    if (names[0] != "before_cuda" or names[-1] != "engine_final" or names.count("before_cuda") != 1
            or names.count("engine_origin") != 1 or names.count("engine_final") != 1
            or updates != list(range(1, target + 1))
            or any(r["optimizer_updates"] != 0 for r in results if r["boundary"] in {"before_cuda", "engine_origin"} or r["boundary"].startswith("cache_record:"))
            or results[-1]["optimizer_updates"] != target
            or {n for n in names if not n.startswith("cache_record:")} - {"before_cuda", "engine_origin", "optimizer_complete", "engine_final"}
            or not _json_equal(progress["process_guard"].get("last"), results[-1])
            or progress["process_guard"].get("total_waited_seconds") != previous_total):
        raise ValueError("Required complete optimizer boundaries or final wait ledger are missing")
    origin_index = names.index("engine_origin")
    if any(not name.startswith("cache_record:") for name in names[1:origin_index]) or any(name != "optimizer_complete" for name in names[origin_index + 1:-1]):
        raise ValueError("Cache/optimizer boundary ordering changed")
    if request["stage"] == "dpo":
        journal = Path(request["output"]) / "reference-cache-records.jsonl"
        cached = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        expected = progress["reference_cache"]["unique_sequences"] if request["reference_cache"]["mode"] == "create" else 0
        expected_groups = (list(dict.fromkeys([0, *range(CACHE_RECORD_INTERVAL, expected, CACHE_RECORD_INTERVAL), expected]))
                           if expected else [0, 0])
        observed_groups = [int(name.split(":")[1]) for name in names if name.startswith("cache_record:")]
        if len(cached) != (expected + 1 if expected else 2) or observed_groups != expected_groups:
            raise ValueError("Cache scoring boundaries omit a complete frozen group of records or endpoint")
        scores = None
        if expected:
            from .reference_cache import load_cache
            pin = progress["reference_cache"]
            directory = Path(pin["path"])
            identity = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
            scores = load_cache(directory, identity, expected_manifest_sha256=pin["manifest_sha256"])
        chain = "0" * 64
        for index, record in enumerate(cached):
            payload = {k: v for k, v in record.items() if k != "record_sha256"}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
            digest = hashlib.sha256(canonical).hexdigest()
            if (record.get("previous_sha256") != chain or record.get("record_sha256") != digest
                    or record.get("scored_sequences") != (index if expected else 0) or record.get("expected_unique_sequences") != expected
                    or (index == 0 or not expected) != (record.get("record") is None)):
                raise ValueError("Complete cache-record journal was changed or reordered")
            if expected and index and (record["record"].get("key") not in scores.entries
                    or not _json_equal(scores.entries[record["record"]["key"]].to_dict(), record["record"])):
                raise ValueError("Committed cache boundary differs from actually consumed reference scores")
            chain = digest
        if expected and len({record["record"]["key"] for record in cached[1:]}) != expected:
            raise ValueError("Cache journal duplicates or omits a scored sequence")
    elif any(name.startswith("cache_record:") for name in names):
        raise ValueError("PPO must not claim offline cache-scoring boundaries")
    return {"verified": True, "inherited_waited_seconds": inherited, "total_waited_seconds": previous_total,
        "segment_waited_seconds": previous_total - inherited, "boundary_calls": len(results),
        "sha256": hashlib.sha256(raw).hexdigest()}
