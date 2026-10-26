"""Read-only, CPU acceptance of all completed models before final evaluation.

The initial audit reads actual checkpoint tensors with weights_only=True on CPU
through the frozen parent/alignment validators. The reusable roster verifier is
stdlib-only: it authenticates that audit and rehashes its explicit evidence, but
does not deserialize tensors or assert that a new training run was performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-independent-evaluation-model-acceptance-v1"
ROSTER_KIND = "alignlab-accepted-evaluation-roster-v1"
MODELS = {"base": "base", "sft": "sft", "reward": "reward",
          **{f"{stage}{seed}": stage for seed in (42, 43, 44) for stage in ("dpo", "ppo")}}
SOURCES = ("alignment_lab/evaluation_acceptance.py",
           "scripts/audit_alignlab_evaluation_models.py", "tests_v2/test_evaluation_acceptance.py",
           "alignment_lab/parent_acceptance.py", "alignment_lab/alignment_supervisor.py",
           "alignment_lab/study_supervisor.py", "alignment_lab/capacity_supervisor.py",
           "alignment_lab/model_snapshot.py", "alignment_lab/probe_dispatch.py",
           "alignment_lab/training_telemetry.py", "alignment_lab/recovered_parent_acceptance.py",
           "scripts/audit_alignlab_recovered_parents.py", "tests_v2/test_recovered_parent_acceptance.py",
           "alignment_lab/parent_recovery.py", "alignment_lab/process_guard_v2.py",
           "alignment_lab/recovered_parent_acceptance_r5.py",
           "tests_v2/test_recovered_parent_acceptance_r5.py",
           "tests_v2/test_recovered_parent_r5_dispatch.py",
           "alignment_lab/alignment_process_guard.py", "alignment_lab/process_guard_v3.py",
           "tests_v2/test_alignment_process_guard.py")


def _recovered_auditor(kind):
    """Explicit protocol routing; an unknown recovery kind is never accepted."""
    if kind == "alignlab-independent-recovered-parents-acceptance-v1":
        from . import recovered_parent_acceptance
        return recovered_parent_acceptance
    if kind == "alignlab-independent-recovered-parents-r5-acceptance-v1":
        from . import recovered_parent_acceptance_r5
        return recovered_parent_acceptance_r5
    return None

def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Expected an ordinary evidence file without symlinks: " + str(path))
    return path.resolve()


def _file(path, expected=None):
    path = _regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError("Evidence changed during hashing: " + str(path))
    result = {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}
    if expected is not None and result["sha256"] != expected:
        raise ValueError("Evidence SHA256 mismatch: " + str(path))
    return result


def _read(path, expected=None):
    item = _file(path, expected)
    raw = Path(item["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Evidence changed while reading JSON")
    return _parse(raw), item


def _new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    return _file(path)


def _record(evidence, path, expected=None):
    item = _file(path, expected)
    previous = evidence.setdefault(item["path"], item)
    if previous != item:
        raise ValueError("One evidence path acquired conflicting identities")
    return item


def _cuda_uninitialized():
    return "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized()


class _Incomplete(Exception):
    pass


def _live(system):
    """Reject unknown identities; a known unrelated API process is harmless."""
    from .probe_dispatch import _stamp
    values = system.processes()
    seen = set()
    for item in values:
        if (type(item.get("pid")) is not int or item["pid"] <= 0 or item["pid"] in seen
                or not isinstance(item.get("command_line"), str) or not item["command_line"].strip()):
            raise ValueError("Unknown/duplicate live process identity")
        _stamp(item["created_at_utc"])
        seen.add(item["pid"])
    return values


def _branch_processes(campaign, plan_path, plan_sha, current):
    from .alignment_supervisor import _command
    from .probe_dispatch import _same_process, _stamp
    supervisors = campaign.get("supervisor_processes")
    if not isinstance(supervisors, list) or not supervisors:
        raise ValueError("Alignment supervisor process identities missing")
    for item in supervisors:
        if (not _command(item, "run_alignlab_alignment.py", plan_path)
                or plan_sha not in item["command_line"]):
            raise ValueError("Wrong alignment supervisor command")
    expected = list(supervisors)
    for stage in campaign.get("stages", []):
        known = stage.get("known_processes")
        if not isinstance(known, list) or not known:
            raise ValueError("Missing observed alignment child identities")
        by_pid = {p["pid"]: p for p in known}
        if len(by_pid) != len(known):
            raise ValueError("Duplicate recorded child PID")
        request_path = Path(stage["request"]["output"]).parent / (stage["name"] + ".request.json")
        for item in known:
            if (not _command(item, "run_alignlab_alignment_stage.py", request_path)
                    or stage["request_sha256"] not in item["command_line"]
                    or _stamp(item["created_at_utc"]).timestamp() < stage["launch_epoch"]):
                raise ValueError("Wrong child creation/command identity")
            ancestor, visited = item, set()
            while ancestor["pid"] != stage["wrapper_pid"]:
                if ancestor["pid"] in visited or ancestor.get("parent_pid") not in by_pid:
                    raise ValueError("Unknown child ancestry")
                visited.add(ancestor["pid"])
                ancestor = by_pid[ancestor["parent_pid"]]
            if ancestor.get("parent_pid") not in {p["pid"] for p in supervisors}:
                raise ValueError("Wrapper does not descend from recorded supervisor")
            expected.append(item)
    observed = {p["pid"]: p for p in current}
    results = []
    for item in expected:
        created = _stamp(item["created_at_utc"])
        actual = observed.get(item["pid"])
        if actual is None:
            state = "exited"
        elif _same_process(actual, item):
            state = "still_alive"
        elif _stamp(actual["created_at_utc"]) == created:
            raise ValueError("Recorded PID/creation now has an unknown command")
        else:
            state = "pid_reused_original_exited"
        results.append({"expected": item, "observed": actual, "state": state})
    return results


def _no_active(current, histories):
    from .alignment_supervisor import BUSY
    if any(p["state"] == "still_alive" for p in histories):
        raise _Incomplete("A recorded parent/alignment process is still alive")
    for process in current:
        if any(marker in process["command_line"].lower() for marker in BUSY):
            raise _Incomplete("Another training/campaign process is active")


def _quiet_paths(plans, *, check_global_locks=True):
    for plan_path, plan in plans:
        output = Path(plan["output"])
        if Path(str(plan_path) + ".stop").exists() or (output / "STOP").exists():
            raise ValueError("STOP evidence prevents completed-model acceptance")
        if any((output / name / "STOP").exists() for name in MODELS if name != "base"):
            raise ValueError("A stage STOP marker remains")
    locks = (".alignlab-gpu.lock", ".alignlab-parents.lock",
             ".alignlab-alignment.lock", ".alignlab-capacity.lock")
    if check_global_locks and any((ROOT / "reports" / name).exists() for name in locks):
        raise _Incomplete("Training lock remains; unknown ownership is preserved")


def _historical_processes_quiet(current, histories, plans):
    """Read-only reuse requires this run's writers gone, not other GPU users."""
    if any(p["state"] == "still_alive" for p in histories):
        raise ValueError("A recorded parent/alignment process is still alive")
    protected = [str(Path(value).resolve()).replace("\\", "/").lower()
                 for path, plan in plans for value in (path, plan["output"])]
    for process in current:
        # The read-only calling process may itself receive plan paths in argv.
        # Its observed ancestors/other processes never receive that exception.
        if process["pid"] == os.getpid():
            continue
        command = process["command_line"].replace("\\", "/").lower()
        if any(path in command for path in protected):
            raise ValueError("A live potential writer points to this completed plan/output")


def _historical_prerequisites(plans, directory, prior_snapshots=None):
    """Preserve admission facts that may legitimately change after a run."""
    directory.mkdir()
    result = {}
    for plan in plans:
        for prerequisite in plan["prerequisites"]:
            original_key = str(Path(prerequisite["path"]).resolve())
            prior = (prior_snapshots or {}).get(original_key)
            if prior is not None:
                if (prior.get("original_path") != original_key
                        or prior.get("original_sha256") != prerequisite["sha256"]
                        or prior.get("semantics") != "historical_admission_fact_not_current_resource_state"):
                    raise ValueError("Preserved original admission snapshot identity differs")
                item = _file(prior["snapshot"]["path"], prerequisite["sha256"])
                source_path = item["path"]
                item = {**item, "path": original_key}
            else:
                item = _file(prerequisite["path"], prerequisite["sha256"])
                source_path = item["path"]
            key = item["path"]
            if key in result:
                if result[key]["original_sha256"] != item["sha256"]:
                    raise ValueError("Conflicting historical prerequisite identities")
                continue
            raw = Path(source_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Prerequisite changed before historical snapshot")
            name = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] + ".snapshot"
            path = directory / name
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            result[key] = {"original_path": key, "original_sha256": item["sha256"],
                           "snapshot": _file(path, item["sha256"]),
                           "semantics": "historical_admission_fact_not_current_resource_state"}
    return result


def _plan_files(evidence, plan_path, plan_sha, plan, historical):
    _record(evidence, plan_path, plan_sha)
    for name, digest in plan["source_sha256"].items():
        _record(evidence, ROOT / name, digest)
    for name in ("cpu_check_report", "capacity_acceptance"):
        path_key = name if name == "cpu_check_report" else name + "_path"
        _record(evidence, plan[path_key], plan[name + "_sha256"])
    for prerequisite in plan["prerequisites"]:
        key = str(Path(prerequisite["path"]).resolve())
        item = historical.get(key, {})
        snapshot = item.get("snapshot", {})
        if (item.get("original_path") != key or item.get("original_sha256") != prerequisite["sha256"]
                or snapshot.get("sha256") != prerequisite["sha256"]
                or item.get("semantics") != "historical_admission_fact_not_current_resource_state"):
            raise ValueError("Historical prerequisite snapshot identity missing")
        _record(evidence, snapshot["path"], snapshot["sha256"])
    snapshot = Path(plan["output"]) / "plan.snapshot.json"
    saved, _ = _read(snapshot)
    if saved != plan:
        raise ValueError("Campaign plan snapshot differs from frozen plan")
    _record(evidence, snapshot)


def _model_data_files(evidence, plan):
    snapshot_path = Path(plan["model_snapshot"]) / "snapshot-manifest.json"
    snapshot, _ = _read(snapshot_path, plan["model_manifest_sha256"])
    _record(evidence, snapshot_path, plan["model_manifest_sha256"])
    for name, item in snapshot["files"].items():
        _record(evidence, snapshot_path.parent / name, item["sha256"])
    data_path = Path(plan["audit_dir"]) / "manifest.json"
    data, _ = _read(data_path, plan["audit_manifest_sha256"])
    _record(evidence, data_path, plan["audit_manifest_sha256"])
    for name, digest in data["artifacts"].items():
        _record(evidence, data_path.parent / name, digest)
    identity = data.get("identity", {})
    for item in identity.get("inputs", {}).values():
        _record(evidence, item["path"], item["sha256"])
    for name, digest in identity.get("source_sha256", {}).items():
        _record(evidence, ROOT / name, digest)
    if "protocol_sha256" in identity:
        _record(evidence, ROOT / "docs/ALIGNLAB_V2.md", identity["protocol_sha256"])
    tokenizer = identity.get("tokenizer", {})
    for name, digest in tokenizer.get("files", {}).items():
        _record(evidence, Path(tokenizer["path"]) / name, digest)


def _telemetry_files(evidence, directory, progress):
    """Rebuild mandatory observed branch evidence without deserializing tensors."""
    from .training_telemetry import PROTOCOL, REVISION, verify_training_telemetry
    config = progress["identity"]["config"]
    if config.get("execution_revision") != REVISION or config.get("telemetry_protocol") != PROTOCOL:
        raise ValueError("Alignment evidence requires the observed-v2 execution identity")
    pin = progress.get("telemetry")
    expected = Path(directory).resolve() / "telemetry" / "manifest.json"
    if (not isinstance(pin, dict) or set(pin) != {"path", "sha256"}
            or _regular(pin["path"]) != expected):
        raise ValueError("Alignment telemetry must name its own stage manifest")
    manifest = verify_training_telemetry(expected.parent,
        expected_manifest_sha256=pin["sha256"],
        expected_identity_sha256=hashlib.sha256(_canonical(progress["identity"])).hexdigest(),
        expected_completion=progress["completion"], checkpoints=progress["checkpoints"])
    if manifest.get("kind") != PROTOCOL or manifest.get("status") != "complete":
        raise ValueError("Complete verified alignment telemetry required")
    _record(evidence, expected, pin["sha256"])
    for name, item in manifest["files"].items():
        path = expected.parent / name
        if not _regular(path).is_relative_to(expected.parent):
            raise ValueError("Telemetry payload escaped its stage")
        actual = _record(evidence, path, item["sha256"])
        if actual["size_bytes"] != item["size_bytes"]:
            raise ValueError("Telemetry payload size mismatch")
    for receipt in progress["checkpoints"]:
        for field in ("telemetry_sidecar", "telemetry_binding"):
            item = receipt[field]
            path = _regular(item["path"])
            if not path.is_relative_to(expected.parent):
                raise ValueError("Checkpoint telemetry escaped its stage")
            key = path.relative_to(expected.parent).as_posix()
            if manifest["files"].get(key, {}).get("sha256") != item["sha256"]:
                raise ValueError("Checkpoint telemetry missing from complete inventory")
            _record(evidence, path, item["sha256"])
    return {"manifest": _file(expected, pin["sha256"]), "costs": manifest["costs"],
            "rollout_count": len(manifest["rollouts"]),
            "checkpoint_binding_count": len(manifest["checkpoint_bindings"]),
            "interpretation": "Training rewards and cost evidence; not independent model quality"}


def _stage_guard_files(evidence, directory, progress):
    """Rebuild and pin every guard file actually used by a completed child."""
    from .alignment_process_guard import verify_guard_log, verify_boundary_receipt
    directory = Path(directory).resolve()
    request = progress["request"]
    if Path(request["output"]).resolve() != directory:
        raise ValueError("Child guard output differs from its accepted stage")
    log = directory / "process-guard.jsonl"
    _record(evidence, log)
    if request["stage"] == "dpo":
        _record(evidence, directory / "reference-cache-records.jsonl")
    proof = verify_guard_log(log, request=request, progress=progress)
    _record(evidence, log, proof["sha256"])
    for row in (json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()):
        if row.get("event") == "wait_start":
            receipt = verify_boundary_receipt(row["saved_boundary"], row["optimizer_updates"])
            if not _regular(receipt["path"]).is_relative_to(directory):
                raise ValueError("Actual saved guard receipt escaped its own stage")
            _record(evidence, receipt["path"], receipt["sha256"])
    return proof


def _supervisor_wait_session(rows, *, plan, campaign, total, earliest, latest, evidence):
    """Replay one existing GuardSession using its strict snapshot/receipt rules."""
    from . import alignment_process_guard as guard
    from .recovered_parent_acceptance_r5 import _guard_record
    owned = campaign["supervisor_processes"]
    allowed = [*owned, *plan["allowed_background_processes"]]
    guard.validate_background(owned)
    guard.validate_background(plan["allowed_background_processes"])
    boundary = "supervisor_before_" + rows[0]["stage_name"]
    observations, saved, result = [], None, None
    previous_at = earliest
    for row in rows:
        at = row.get("at_epoch")
        if (row.get("stage_name") != rows[0]["stage_name"] or row.get("call") != 1
                or row.get("boundary") != boundary or row.get("optimizer_updates") != 0
                or row.get("deadline_epoch") != campaign["deadline_epoch"]
                or type(at) not in (int, float) or not math.isfinite(at)
                or not previous_at <= at <= latest or at >= campaign["deadline_epoch"]
                or result is not None):
            raise ValueError("Supervisor wait boundary, ordering or absolute deadline changed")
        previous_at = at
        if row.get("event") == "observation":
            decision = row["guard"]
            _guard_record(decision, owned, allowed,
                {"started_at_epoch": campaign["started_at_epoch"], "deadline_epoch": campaign["deadline_epoch"]})
            if (row.get("cycle") != len(observations) + 1
                    or observations and observations[-1]["guard"]["admission_allowed"]
                    or decision["observations"][-1]["finished_at_epoch"] > at
                    or any(not guard._allowed_ancestry_unchanged(allowed,
                        tuple(guard.strict.ProcessIdentity.from_record(p) for p in item["snapshot"]))
                        for item in decision["observations"] if item["snapshot_valid"])):
                raise ValueError("Supervisor strict observations or process ancestry changed")
            observations.append({k: row[k] for k in
                ("cycle", "at_epoch", "guard", "waited_seconds", "total_waited_seconds")})
        elif row.get("event") == "wait_start":
            if saved is not None or not observations or observations[-1]["guard"]["admission_allowed"]:
                raise ValueError("Supervisor waiting lacks a preceding unknown observation")
            saved = guard.verify_boundary_receipt(row["saved_boundary"], 0)
            expected = Path(plan["output"]).resolve() / "plan.snapshot.json"
            if saved["kind"] != "before_cuda" or _regular(saved["path"]) != expected:
                raise ValueError("Supervisor wait must pin its actual idle plan boundary")
            snapshot, _ = _read(expected, saved["sha256"])
            if snapshot != plan:
                raise ValueError("Supervisor safe boundary differs from its frozen plan")
            _record(evidence, expected, saved["sha256"])
        elif row.get("event") == "result":
            result = {k: v for k, v in row.items() if k != "stage_name"}
            if (result.get("schema_version") != 1
                    or result.get("kind") != "alignlab-process-guard-v3-boundary-wait"
                    or result.get("admission_allowed") is not True or result.get("reason") != "cleared"
                    or result.get("unknown_processes_allowed") is not False
                    or result.get("training_during_wait") is not False
                    or result.get("inherited_waited_seconds") != total
                    or result.get("saved_boundary") != saved or result.get("observations") != observations
                    or not observations or result.get("last_guard") != observations[-1]["guard"]
                    or observations[-1]["guard"]["admission_allowed"] is not True):
                raise ValueError("Supervisor session did not independently clear its inherited wait ledger")
            waited, final = result.get("waited_seconds"), result.get("total_waited_seconds")
            waiting = observations[0]["guard"]["admission_allowed"] is False
            if (type(waited) not in (int, float) or type(final) not in (int, float)
                    or not math.isfinite(waited) or not math.isfinite(final)
                    or not 0 <= waited < 300 or not total <= final < 900
                    or not math.isclose(final - total, waited, rel_tol=1e-12, abs_tol=1e-6)
                    or waiting and (saved is None or len(observations) < 2 or waited <= 0)
                    or not waiting and (saved is not None or len(observations) != 1 or waited != 0)):
                raise ValueError("Supervisor cumulative waiting was reset, hidden or exceeded")
            previous_wait = 0.0
            for index, item in enumerate(observations):
                delay, cumulative = item["waited_seconds"], item["total_waited_seconds"]
                if (type(delay) not in (int, float) or type(cumulative) not in (int, float)
                        or not math.isfinite(delay) or not math.isfinite(cumulative)
                        or index == 0 and (delay != 0 or cumulative != total)
                        or delay < previous_wait - 1e-6 or delay > waited + 1e-6
                        or not math.isclose(cumulative, total + delay, rel_tol=1e-12, abs_tol=1e-6)
                        or item["at_epoch"] - observations[0]["at_epoch"] > delay + 1e-6):
                    raise ValueError("Supervisor observed elapsed waiting is not fully charged")
                previous_wait = delay
        else:
            raise ValueError("Unknown supervisor process-boundary event")
    if result is None:
        raise ValueError("Incomplete supervisor wait session")
    return result


def _supervisor_wait_ledger(plan, campaign, evidence):
    """Join actual supervisor and child waits; no request can declare its budget."""
    from . import alignment_process_guard as guard
    from .alignment_supervisor import SEQUENCE
    guard.validate_policy(plan["process_wait_policy"])
    if campaign.get("process_wait_policy") != plan["process_wait_policy"]:
        raise ValueError("Campaign wait policy differs from its frozen plan")
    log = Path(plan["output"]) / "supervisor-process-guard.jsonl"
    _record(evidence, log)
    groups, group = [], []
    for line in log.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        group.append(row)
        if row.get("event") == "result":
            groups.append(group)
            group = []
    if group or not groups:
        raise ValueError("Supervisor guard log contains an unfinished or missing session")
    names = [f"{stage}{seed}" for stage, seed in SEQUENCE]
    if [record.get("name") for record in campaign["stages"]] != names:
        raise ValueError("Supervisor wait ledger requires all six fixed branch records")
    records = {record["name"]: record for record in campaign["stages"]}
    total, index, previous_end, last = 0.0, 0, campaign["started_at_epoch"], None
    dispatch_spent = {}
    for name in [*names, "final_publication"]:
        record = records.get(name)
        latest = record["launch_epoch"] if record else campaign["finished_at_epoch"]
        count = 0
        while index < len(groups) and groups[index][0].get("stage_name") == name:
            last = _supervisor_wait_session(groups[index], plan=plan, campaign=campaign,
                total=total, earliest=previous_end, latest=latest, evidence=evidence)
            total, previous_end = last["total_waited_seconds"], last["at_epoch"]
            index, count = index + 1, count + 1
        if not count:
            raise ValueError("Supervisor stage admission or final publication proof is missing")
        if record is not None:
            dispatch_spent[name] = total
            request, _ = _read(Path(plan["output"]) / (name + ".request.json"), record["request_sha256"])
            if (request.get("process_wait_policy") != plan["process_wait_policy"]
                    or request.get("allowed_background_processes") != plan["allowed_background_processes"]
                    or request.get("supervisor_processes") != campaign["supervisor_processes"]
                    or request.get("process_wait_spent_seconds") != total
                    or record.get("inherited_process_wait_seconds") != total or record.get("request") != request):
                raise ValueError("Child guard request differs from the independent campaign wait ledger")
            progress, _ = _read(record["progress_path"], record["progress_sha256"])
            if progress.get("request") != request:
                raise ValueError("Child guard progress/request identity differs")
            proof = _stage_guard_files(evidence, Path(plan["output"]) / name, progress)
            if (proof["inherited_waited_seconds"] != total
                    or record.get("total_process_wait_seconds") != proof["total_waited_seconds"]
                    or not total <= proof["total_waited_seconds"] < 900):
                raise ValueError("Child did not inherit and charge the same campaign wait budget")
            total, previous_end = proof["total_waited_seconds"], record["finished_at_epoch"]
    if (index != len(groups) or campaign.get("total_process_wait_seconds") != total
            or not guard._json_equal(campaign.get("last_process_guard"), last)):
        raise ValueError("Final campaign ledger differs from actual supervisor and child waits")
    return dispatch_spent


def _stage_files(evidence, directory, progress, historical=None):
    for name in ("progress.json", "request.json", "identity.json", "selection.json", "metrics.jsonl"):
        _record(evidence, directory / name)
    identity = progress["identity"]
    for name, item in identity["inputs"].items():
        fact = (historical or {}).get(str(Path(item["path"]).resolve()))
        protected = name.startswith(("source/", "data/", "base/", "tokenizer/")) or Path(item["path"]).suffix == ".py"
        if fact and not protected:
            if item["sha256"] != fact["original_sha256"]:
                raise ValueError("Stage prerequisite differs from historical admission")
            _record(evidence, fact["snapshot"]["path"], item["sha256"])
        else:
            _record(evidence, item["path"], item["sha256"])
    for item in identity["parents"].values():
        _record(evidence, item["path"], item["sha256"])
    if "implementation" in identity:
        item = identity["implementation"]
        _record(evidence, item["path"], item["sha256"])
    details = []
    for receipt in progress["checkpoints"]:
        path = Path(receipt["checkpoint_path"])
        manifest, _ = _read(path / "manifest.json", receipt["manifest_sha256"])
        _record(evidence, path / "manifest.json", receipt["manifest_sha256"])
        payload = _record(evidence, path / "state.pt", manifest["artifacts"]["state.pt"]["sha256"])
        details.append({"checkpoint_path": str(path), "manifest_sha256": receipt["manifest_sha256"],
                        "payload_sha256": payload["sha256"], "optimizer_updates": receipt["optimizer_updates"],
                        "actual_cpu_payload_verified": True})
    final = progress["artifact"]
    path = Path(final["artifact_dir"])
    manifest, _ = _read(path / "manifest.json", final["manifest_sha256"])
    _record(evidence, path / "manifest.json", final["manifest_sha256"])
    _record(evidence, path / "weights.pt", manifest["artifacts"]["weights.pt"]["sha256"])
    cache = progress.get("reference_cache")
    if cache:
        path = Path(cache["path"])
        manifest, _ = _read(path / "manifest.json", cache["manifest_sha256"])
        _record(evidence, path / "manifest.json", cache["manifest_sha256"])
        for name, item in manifest["artifacts"].items():
            _record(evidence, path / name, item["sha256"])
    if progress["identity"]["config"].get("stage") in ("dpo", "ppo"):
        _stage_guard_files(evidence, directory, progress)
        _telemetry_files(evidence, directory, progress)
    return details


def _alignment(plan_path, plan_sha, plan, campaign, system, evidence, historical):
    from . import alignment_supervisor as alignment
    directory = Path(plan["output"])
    if (campaign.get("kind") != alignment.KIND or campaign.get("status") != "primary_alignment_passed"
            or any(campaign.get(k) is not True for k in ("primary_alignment_verified", "parents_verified",
                                                       "capacity_verified", "formal_training"))
            or campaign.get("research_complete") is not False or campaign.get("error")
            or campaign.get("child_may_still_be_running")
            or campaign.get("scope") != "primary_six_alignment_branches_only"
            or campaign.get("plan_sha256") != plan_sha
            or Path(campaign.get("plan_path", "")).resolve() != plan_path):
        raise ValueError("Only a completed clean six-branch alignment campaign is acceptable")
    times = [campaign[k] for k in ("started_at_epoch", "finished_at_epoch", "deadline_epoch")]
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
            or not times[0] <= times[1] <= times[2]
            or times[2] != times[0] + plan["wall_seconds"]):
        raise ValueError("Alignment campaign budget evidence is invalid")
    sequence = [(s.get("stage"), s.get("seed")) for s in campaign.get("stages", [])]
    if sequence != list(alignment.SEQUENCE):
        raise ValueError("All six fixed seeds/stages must complete in order")
    histories = _branch_processes(campaign, plan_path, plan_sha, _live(system))
    _no_active(_live(system), histories)
    dispatch_spent = _supervisor_wait_ledger(plan, campaign, evidence)
    models, details, cache = {}, {}, None
    previous_end = times[0]
    for record in campaign["stages"]:
        stage, seed = record["stage"], record["seed"]
        name = f"{stage}{seed}"
        output = directory / name
        request_path = directory / (name + ".request.json")
        request, request_file = _read(request_path, record["request_sha256"])
        reference = None if stage == "ppo" else (
            {"mode": "create", "path": str(output / "reference-cache")} if cache is None
            else {"mode": "reuse", **cache})
        expected = {key: plan[key] for key in (
            "source_sha256", "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
            "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256",
            "parent_plan_path", "parent_plan_sha256", "parent_acceptance_path", "parent_acceptance_sha256",
            "allowed_background_processes", "process_wait_policy")}
        expected.update(schema_version=1, kind=alignment.STAGE_KIND, stage=stage, seed=seed,
            output=str(output), deadline_epoch=campaign["deadline_epoch"],
            alignment_plan_path=str(plan_path), alignment_plan_sha256=plan_sha,
            supervisor_processes=campaign["supervisor_processes"],
            process_wait_spent_seconds=dispatch_spent[name],
            parents={k: plan["parents"][k] for k in (("sft",) if stage == "dpo" else ("sft", "reward"))},
            workload=alignment.workload(stage, plan["ppo_rows"]), reference_cache=reference)
        if (record.get("name") != name or record.get("status") != "accepted"
                or record.get("exit_code") != 0 or record.get("active_processes") != []
                or record.get("monitoring_error") or record.get("stop_reason") is not None
                or request != expected or record["request"] != expected
                or not previous_end <= record["launch_epoch"] <= record["finished_at_epoch"] <= times[1]
                or Path(record["progress_path"]).resolve() != output / "progress.json"):
            raise ValueError("Branch dispatch/exit/request/serial lineage is incomplete")
        previous_end = record["finished_at_epoch"]
        _record(evidence, request_path, request_file["sha256"])
        _record(evidence, output / "progress.json", record["progress_sha256"])
        # This reopens every actual checkpoint with weights_only=True on CPU,
        # verifies Adam steps/finite parameters/cursors, and compares the final
        # checkpoint tensors exactly with the exported policy/value tensors.
        progress = alignment._verify_stage(output, request,
            {p["pid"]: p for p in record["known_processes"]}, record["exit_code"])
        if progress["artifact"] != record.get("artifact") or progress.get("reference_cache") != record.get("reference_cache"):
            raise ValueError("Campaign artifacts/cache differ from actual accepted progress")
        if stage == "dpo":
            produced = {k: progress["reference_cache"][k] for k in ("path", "manifest_sha256", "identity_sha256")}
            if cache is None:
                cache = produced
            elif cache != produced:
                raise ValueError("Cross-seed reference-cache identity changed")
        checkpoints = _stage_files(evidence, output, progress, historical)
        models[name] = {"stage": stage, "artifact_dir": progress["artifact"]["artifact_dir"],
            "manifest_sha256": progress["artifact"]["manifest_sha256"],
            "progress_path": str(output / "progress.json"), "progress_sha256": record["progress_sha256"]}
        details[name] = {"rows": request["workload"]["rows"],
            "optimizer_updates": progress["actual_optimizer_updates"],
            "response_tokens": progress["cursor"]["response_tokens"],
            "completed_rollouts": progress["cursor"]["completed_rollouts"],
            "checkpoints": checkpoints, "final_checkpoint_export_exact": True,
            "metrics": _file(output / "metrics.jsonl"), "gpu_samples": progress["gpu_samples"],
            "reference_cache": progress.get("reference_cache"),
            "telemetry": _telemetry_files(evidence, output, progress)}
    return models, details, histories


def audit_evaluation_models(parent_plan_path, parent_plan_sha256, alignment_plan_path,
                            alignment_plan_sha256, *, output_dir, system=None):
    """Create a fresh diagnostic directory; accepted manifests require real completion."""
    from . import parent_acceptance as parents
    from . import alignment_supervisor as alignment
    from .probe_dispatch import _System
    output = Path(output_dir).absolute()
    if (output.exists() or any(p.is_symlink() for p in (output, *output.parents))
            or output.resolve().parent != (ROOT / "reports").resolve()):
        raise ValueError("Use a new direct directory in project reports; never overwrite")
    output.mkdir()
    started = time.time()
    report = {"schema_version": 1, "kind": "alignlab-evaluation-model-audit-v1",
        "status": "incomplete", "all_checks_passed": False, "models_verified": False,
        "research_complete": False, "device": "cpu", "gpu_operations_performed": False,
        "model_loaded": False, "auditor_pid": os.getpid(), "started_at_epoch": started,
        "accepted_models": None, "roster": None, "blockers": []}
    system = system or _System()
    evidence = {}
    try:
        if not _cuda_uninitialized():
            raise ValueError("Acceptance process has initialized CUDA")
        for name in SOURCES:
            _record(evidence, ROOT / name)
        auditor_sources = {name: evidence[str((ROOT / name).resolve())]["sha256"] for name in SOURCES}
        _live(system)
        parent_path = Path(parent_plan_path).resolve()
        plan_path = Path(alignment_plan_path).resolve()
        candidate = _read(plan_path, alignment_plan_sha256)[0] if plan_path.is_file() else None
        candidate_parent = (_read(candidate["parent_acceptance_path"], candidate["parent_acceptance_sha256"])[0]
                            if candidate is not None else None)
        recovery_audit = _recovered_auditor((candidate_parent or {}).get("kind"))
        recovered = recovery_audit is not None
        parent_audit_path = output / "parent-audit.json"
        recovery_plan = None
        if recovered:
            parent_plan, _ = _read(parent_path, parent_plan_sha256)
            recovery_pin = candidate_parent["recovery_plan"]
            recovery_plan, _ = _read(recovery_pin["path"], recovery_pin["sha256"])
            parent_report = recovery_audit.audit_recovered_parents(parent_path, parent_plan_sha256,
                recovery_pin["path"], recovery_pin["sha256"], output=parent_audit_path, system=system)
            parent_accepted_status = recovery_audit.ACCEPTED
        else:
            parent_plan = parents.parents.validate_plan(parent_path, parent_plan_sha256)
            parent_report = parents.audit_parents(parent_path, parent_plan_sha256,
                                                   output=parent_audit_path, system=system)
            parent_accepted_status = parents.ACCEPTED
        report["parent_audit"] = _record(evidence, parent_audit_path)
        if parent_report["status"] != parent_accepted_status:
            if parent_report["status"] == "failed":
                raise ValueError("Independent actual parent audit failed: " + parent_report.get("error", "unknown"))
            raise _Incomplete("Formal parents are not completed and exited")
        campaign_pin = parent_report["original_campaign" if recovered else "campaign"]
        parent_campaign, _ = _read(campaign_pin["path"], campaign_pin["sha256"])
        if not recovered and (parent_campaign.get("error") or parent_campaign.get("child_may_still_be_running")):
            raise ValueError("Parent campaign retains failed/uncertain process evidence")
        if not plan_path.is_file():
            raise _Incomplete("No frozen alignment plan exists yet")
        plan = alignment.validate_plan(plan_path, alignment_plan_sha256)
        if (Path(plan["parent_plan_path"]).resolve() != parent_path
                or plan["parent_plan_sha256"] != parent_plan_sha256):
            raise ValueError("Alignment does not descend from the supplied formal parent plan")
        quiet_plans = (((Path(recovery_pin["path"]), recovery_plan), (plan_path, plan)) if recovered
                       else ((parent_path, parent_plan), (plan_path, plan)))
        _quiet_paths(quiet_plans)
        campaign_path = Path(plan["output"]) / "campaign.json"
        if not campaign_path.is_file():
            raise _Incomplete("Alignment campaign has not started")
        campaign, campaign_file = _read(campaign_path)
        report["alignment_campaign"] = campaign_file
        if campaign.get("status") != "primary_alignment_passed":
            if campaign.get("status") in {"failed", "paused", "budget_exhausted", "stopped_before_start"}:
                raise ValueError("Alignment campaign did not complete: " + campaign["status"])
            raise _Incomplete("Alignment campaign has not completed all six branches")
        _record(evidence, campaign_path, campaign_file["sha256"])
        _record(evidence, campaign_pin["path"], campaign_pin["sha256"])
        historical = _historical_prerequisites((parent_plan, plan), output / "historical-prerequisites",
            recovery_plan.get("historical_prerequisites") if recovered else None)
        if recovered:
            for item in parent_report["evidence_files"].values():
                _record(evidence, item["path"], item["sha256"])
        _plan_files(evidence, parent_path, parent_plan_sha256, parent_plan, historical)
        _plan_files(evidence, plan_path, alignment_plan_sha256, plan, historical)
        _record(evidence, plan["parent_acceptance_path"], plan["parent_acceptance_sha256"])
        models = {"base": {"stage": "base", "artifact_dir": None, "manifest_sha256": None,
                           "progress_path": None, "progress_sha256": None}}
        for name in ("sft", "reward"):
            row = parent_report["stages"][name]
            if any(row[k] != plan["parents"][name][k] for k in ("artifact_dir", "manifest_sha256")):
                raise ValueError("Actual independently audited parents differ from branch parents")
            models[name] = {"stage": name, **{k: row[k] for k in
                ("artifact_dir", "manifest_sha256", "progress_path", "progress_sha256")}}
            progress, _ = _read(row["progress_path"], row["progress_sha256"])
            _stage_files(evidence, Path(row["progress_path"]).parent, progress, historical)
            if recovered and name == "reward":
                _record(evidence, progress["request_file"]["path"], progress["request_file"]["sha256"])
            else:
                parent_request = next(r for r in parent_campaign["stages"] if r["stage"] == name)
                _record(evidence, Path(parent_plan["output"]) / (name + ".request.json"),
                        parent_request["request_sha256"])
        branches, detail, histories = _alignment(plan_path, alignment_plan_sha256, plan, campaign, system, evidence, historical)
        models.update(branches)
        # Include the full fixed snapshot and tokenizer/data audit, not only an
        # inventory's self-reported "complete" bit. Parent audit has already
        # performed full snapshot and raw-data/source/protocol verification.
        _model_data_files(evidence, plan)
        report["alignment_stages"] = detail
        _quiet_paths(quiet_plans)
        alignment.validate_plan(plan_path, alignment_plan_sha256)
        live = _live(system)
        parent_history = (recovery_audit.verify_recovered_parents(parent_audit_path,
            report["parent_audit"]["sha256"], system=system)["processes"] if recovered else
            parents._processes(parent_campaign, parent_path, parent_plan_sha256, live))
        histories = _branch_processes(campaign, plan_path, alignment_plan_sha256, live)
        _no_active(live, parent_history + histories)
        for item in list(evidence.values()):
            if _file(item["path"], item["sha256"]) != item:
                raise ValueError("Bound evidence changed before publishing")
        if not _cuda_uninitialized():
            raise ValueError("CPU acceptance unexpectedly initialized CUDA")
        accepted = {"schema_version": 1, "kind": KIND, "status": "accepted_for_evaluation",
            "all_checks_passed": True, "processes_exited": True, "models_verified": True,
            "formal_training": True, "research_complete": False,
            "snapshot_manifest_sha256": plan["model_manifest_sha256"],
            "models": models, "evidence_files": evidence,
            "auditor_source_sha256": auditor_sources, "historical_prerequisites": historical,
            "parent_plan": {"path": str(parent_path), "sha256": parent_plan_sha256},
            "alignment_plan": {"path": str(plan_path), "sha256": alignment_plan_sha256},
            "parent_audit": report["parent_audit"], "alignment_campaign": campaign_file,
            "alignment_stages": detail, "processes": parent_history + histories,
            "cuda_initialized": False, "model_loaded": False, "gpu_operations_performed": False,
            "auditor_pid": os.getpid(), "finished_at_epoch": time.time()}
        accepted_path = output / "accepted-models.json"
        accepted_sha = hashlib.sha256(_canonical(accepted)).hexdigest()
        roster = {"schema_version": 1, "kind": ROSTER_KIND,
            "model_snapshot": plan["model_snapshot"], "model_manifest_sha256": plan["model_manifest_sha256"],
            "tokenizer_dir": plan["tokenizer_dir"],
            "models": {name: {"model_id": name, "stage": row["stage"],
                "artifact_dir": row["artifact_dir"], "manifest_sha256": row["manifest_sha256"],
                "acceptance_path": str(accepted_path), "acceptance_sha256": accepted_sha}
                for name, row in models.items()}}
        # Acceptance is published last, after every prerequisite and roster
        # byte is ready; a partial directory cannot contain a usable roster.
        roster_file = _new(output / "roster.json", roster)
        accepted_file = _new(accepted_path, accepted)
        report.update(status="evaluation_models_accepted", all_checks_passed=True, models_verified=True,
                      accepted_models=accepted_file, roster=roster_file, blockers=[])
    except _Incomplete as error:
        report["blockers"].append(str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        report["blockers"].append("verification_failed")
    report.update(finished_at_epoch=time.time(), elapsed_seconds=time.time() - started,
                  cuda_initialized=not _cuda_uninitialized())
    _new(output / "report.json", report)
    return report


def verify_evaluation_roster(roster_path, expected_sha256, *, verify_bound_files=True, system=None):
    """Authenticate the completed audit; stdlib-only, no tensor/model loading.

    Initial tensor-content checks are authenticated by the immutable audit and
    the exact bytes it inspected. Live process identities are checked again.
    """
    if verify_bound_files is not True:
        raise ValueError("Full evidence verification cannot be disabled")
    roster, roster_file = _read(roster_path, expected_sha256)
    if (set(roster) != {"schema_version", "kind", "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "models"}
            or type(roster["schema_version"]) is not int or roster["schema_version"] != 1
            or roster["kind"] != ROSTER_KIND or set(roster["models"]) != set(MODELS)):
        raise ValueError("Complete fixed nine-model roster required")
    pins = {(r["acceptance_path"], r["acceptance_sha256"]) for r in roster["models"].values()}
    if len(pins) != 1:
        raise ValueError("All models must belong to the same independent audit")
    accepted_path, accepted_sha = next(iter(pins))
    accepted, accepted_file = _read(accepted_path, accepted_sha)
    if (accepted.get("schema_version") != 1 or accepted.get("kind") != KIND
            or accepted.get("status") != "accepted_for_evaluation"
            or any(accepted.get(k) is not True for k in ("all_checks_passed", "processes_exited", "models_verified"))
            or accepted.get("research_complete") is not False or accepted.get("cuda_initialized") is not False
            or accepted.get("snapshot_manifest_sha256") != roster["model_manifest_sha256"]
            or set(accepted.get("models", {})) != set(MODELS)):
        raise ValueError("Complete independent accepted-model report required")
    evidence = accepted.get("evidence_files")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("Actual audit evidence inventory missing")
    required = []
    for name, stage in MODELS.items():
        record, row = roster["models"][name], accepted["models"][name]
        if (set(record) != {"model_id", "stage", "artifact_dir", "manifest_sha256", "acceptance_path", "acceptance_sha256"}
                or set(row) != {"stage", "artifact_dir", "manifest_sha256", "progress_path", "progress_sha256"}
                or record["model_id"] != name or record["stage"] != stage or row["stage"] != stage
                or any(record[k] != row[k] for k in ("artifact_dir", "manifest_sha256"))):
            raise ValueError("Roster differs from its independently audited model")
        if name == "base":
            if any(row[k] is not None for k in ("artifact_dir", "manifest_sha256", "progress_path", "progress_sha256")):
                raise ValueError("Base cannot substitute a trained adapter")
            required.append((Path(roster["model_snapshot"]) / "snapshot-manifest.json", roster["model_manifest_sha256"]))
        else:
            required.extend(((Path(row["artifact_dir"]) / "manifest.json", row["manifest_sha256"]),
                             (row["progress_path"], row["progress_sha256"])))
    for path, digest in required:
        item = _file(path, digest)
        if evidence.get(item["path"]) != item:
            raise ValueError("Model/progress lacks exact independently audited evidence")
    # These helpers import only stdlib-backed admission code; they never call
    # resources(), instantiate a model, or deserialize a checkpoint.
    from . import parent_acceptance as parents
    from .probe_dispatch import _System
    parent_pin, alignment_pin = accepted["parent_plan"], accepted["alignment_plan"]
    parent_plan, _ = _read(parent_pin["path"], parent_pin["sha256"])
    alignment_plan, _ = _read(alignment_pin["path"], alignment_pin["sha256"])
    parent_audit, _ = _read(accepted["parent_audit"]["path"], accepted["parent_audit"]["sha256"])
    recovery_audit = _recovered_auditor(parent_audit.get("kind"))
    recovered = recovery_audit is not None
    recovery_plan = None
    if recovered:
        recovered_verified = recovery_audit.verify_recovered_parents(accepted["parent_audit"]["path"],
            accepted["parent_audit"]["sha256"], system=system)
        recovery_pin = parent_audit["recovery_plan"]
        recovery_plan, _ = _read(recovery_pin["path"], recovery_pin["sha256"])
    campaign_pin = parent_audit["original_campaign" if recovered else "campaign"]
    parent_campaign, _ = _read(campaign_pin["path"], campaign_pin["sha256"])
    alignment_campaign, _ = _read(accepted["alignment_campaign"]["path"],
                                   accepted["alignment_campaign"]["sha256"])
    if (parent_campaign.get("status") != ("failed" if recovered else "parents_passed")
            or alignment_campaign.get("status") != "primary_alignment_passed"):
        raise ValueError("Accepted campaign status changed")
    # Reconstruct mandatory evidence from authenticated inventories. Removing
    # state.pt/weights.pt/source rows cannot turn this into a metadata-only audit.
    required_files = {}
    source_pins = accepted.get("auditor_source_sha256", {})
    if set(source_pins) != set(SOURCES):
        raise ValueError("Auditor source inventory is incomplete")
    for name, digest in source_pins.items():
        _record(required_files, ROOT / name, digest)
    for item in (accepted["parent_audit"], accepted["alignment_campaign"], campaign_pin):
        _record(required_files, item["path"], item["sha256"])
    if recovered:
        for item in recovered_verified["evidence_files"].values():
            _record(required_files, item["path"], item["sha256"])
    historical = accepted.get("historical_prerequisites", {})
    original_keys = {str(Path(p["path"]).resolve()) for plan in (parent_plan, alignment_plan)
                     for p in plan["prerequisites"]}
    if set(historical) != original_keys:
        raise ValueError("Historical prerequisite inventory differs from the frozen plans")
    _plan_files(required_files, parent_pin["path"], parent_pin["sha256"], parent_plan, historical)
    _plan_files(required_files, alignment_pin["path"], alignment_pin["sha256"], alignment_plan, historical)
    _record(required_files, alignment_plan["parent_acceptance_path"], alignment_plan["parent_acceptance_sha256"])
    _model_data_files(required_files, alignment_plan)
    _supervisor_wait_ledger(alignment_plan, alignment_campaign, required_files)
    stage_campaigns = ((alignment_campaign, alignment_plan),) if recovered else (
        (parent_campaign, parent_plan), (alignment_campaign, alignment_plan))
    for campaign, plan in stage_campaigns:
        for record in campaign["stages"]:
            name = record.get("name", record["stage"])
            directory = Path(plan["output"]) / name
            _record(required_files, Path(plan["output"]) / (name + ".request.json"), record["request_sha256"])
            progress, _ = _read(directory / "progress.json", record["progress_sha256"])
            _stage_files(required_files, directory, progress, historical)
    if any(evidence.get(path) != item for path, item in required_files.items()):
        raise ValueError("Required payload/source evidence was removed or changed")
    for path, item in evidence.items():
        if (set(item) != {"path", "sha256", "size_bytes"} or item["path"] != path
                or not Path(path).is_absolute()):
            raise ValueError("Malformed evidence inventory")
        if _file(path, item["sha256"]) != item:
            raise ValueError("Stored accepted evidence changed")
    plans = ((Path(parent_pin["path"]), parent_plan), (Path(alignment_pin["path"]), alignment_plan))
    if recovered:
        plans += ((Path(recovery_pin["path"]), recovery_plan),)
    _quiet_paths(plans[1:] if recovered else plans, check_global_locks=False)
    live = _live(system or _System())
    history = (recovered_verified["processes"] if recovered else
        parents._processes(parent_campaign, Path(parent_pin["path"]), parent_pin["sha256"], live))
    history += _branch_processes(alignment_campaign, Path(alignment_pin["path"]),
                                 alignment_pin["sha256"], live)
    _historical_processes_quiet(live, history, plans)
    _file(roster_file["path"], roster_file["sha256"])
    _file(accepted_file["path"], accepted_file["sha256"])
    return roster
