"""One admitted, serial six-branch alignment campaign; never auto-resume.

This coordinator performs file/process and safe CPU payload checks only. GPU
work belongs to the separately pinned child. Observed-v2 evidence is mandatory;
earlier prototype proofs cannot accept a newly executed branch.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid

from . import study_supervisor as parent_supervisor
from .probe_dispatch import _System, _same_process, _stamp, _parse, _new, _status
from .capacity_supervisor import sha, _pin, _canonical, _prerequisites, _bundle, NF4
from .model_snapshot import verify_snapshot, REPO_ID, REVISION
from . import alignment_process_guard as process_guard

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-formal-alignment-campaign-v1"
STAGE_KIND = "alignlab-formal-alignment-stage-v1"
EXECUTION_REVISION = "observed-v2"
TELEMETRY_PROTOCOL = "alignlab-training-telemetry-observed-v2"
SEQUENCE = (("dpo", 42), ("ppo", 42), ("dpo", 43), ("ppo", 43), ("dpo", 44), ("ppo", 44))
NEW_SOURCE_FILES = {
    "alignment_lab/alignment_supervisor.py", "alignment_lab/alignment_stage.py",
    "scripts/run_alignlab_alignment.py", "scripts/run_alignlab_alignment_stage.py",
    "tests_v2/test_alignment_supervisor.py", "tests_v2/test_alignment_stage.py",
    "alignment_lab/stage_engine_observed.py", "alignment_lab/training_telemetry.py",
    "tests_v2/test_stage_engine_observed.py", "tests_v2/test_training_telemetry.py",
    "tests_v2/test_recovered_parent_r5_dispatch.py",
} | process_guard.SOURCE_FILES
BUSY = parent_supervisor.BUSY + ("run_alignlab_alignment",)
PARENT_ACCEPTANCE_KIND = "alignlab-independent-formal-parents-acceptance-v1"


def _source(name):
    if (not isinstance(name, str) or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or not name.endswith(".py")):
        raise ValueError("Canonical project-relative frozen Python source required")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen source escaped project")
    return path


def _parent_acceptance(plan):
    """Authenticate both original completed parent artifacts and their audit."""
    _pin(plan["parent_plan_path"], plan["parent_plan_sha256"])
    _pin(plan["parent_acceptance_path"], plan["parent_acceptance_sha256"])
    accepted = _parse(plan["parent_acceptance_path"])
    if accepted.get("kind") == "alignlab-independent-recovered-parents-r5-acceptance-v1":
        from .recovered_parent_acceptance_r5 import validate_recovered_for_request
        validated = validate_recovered_for_request(plan)
        return _parse(plan["parent_plan_path"]), _parse(validated["original_campaign"]["path"])
    if accepted.get("kind") == "alignlab-independent-recovered-parents-acceptance-v1":
        from .recovered_parent_acceptance import validate_recovered_for_request
        validated = validate_recovered_for_request(plan)
        return _parse(plan["parent_plan_path"]), _parse(validated["original_campaign"]["path"])
    original = parent_supervisor.validate_plan(plan["parent_plan_path"], plan["parent_plan_sha256"])
    for key in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
                "audit_manifest_sha256", "capacity_acceptance_path", "capacity_acceptance_sha256"):
        if plan[key] != original[key]:
            raise ValueError("Alignment model/data/capacity differ from formal parents")
    if len(original["source_sha256"]) != 59:
        raise ValueError("The original 59 formal-parent dependencies must be pinned")
    _pin(plan["parent_acceptance_path"], plan["parent_acceptance_sha256"])
    report = _parse(plan["parent_acceptance_path"])
    if (report.get("schema_version") != 1 or report.get("kind") != PARENT_ACCEPTANCE_KIND
            or report.get("status") != "accepted_full_real_formal_parents"
            or any(report.get(k) is not True for k in
                   ("all_checks_passed", "parents_verified", "formal_training", "processes_exited"))
            or report.get("research_complete") is not False
            or report.get("parent_plan") != {"path": plan["parent_plan_path"],
                                             "sha256": plan["parent_plan_sha256"]}):
        raise ValueError("Independent completed real-parent acceptance required")
    campaign_item = report["campaign"]
    expected_campaign = Path(original["output"]).resolve() / "campaign.json"
    if (Path(campaign_item["path"]).resolve() != expected_campaign
            or campaign_item.get("status") != "parents_passed"):
        raise ValueError("Parent audit points to a different campaign")
    _pin(expected_campaign, campaign_item["sha256"])
    campaign = _parse(expected_campaign)
    if (campaign.get("kind") != parent_supervisor.KIND
            or campaign.get("status") != "parents_passed"
            or campaign.get("parents_verified") is not True
            or campaign.get("formal_training") is not True
            or campaign.get("research_complete") is not False
            or campaign.get("plan_sha256") != plan["parent_plan_sha256"]
            or [s.get("stage") for s in campaign.get("stages", [])] != ["sft", "reward"]):
        raise ValueError("Parent campaign is incomplete or has different lineage")
    if set(plan["parents"]) != {"sft", "reward"} or set(report["stages"]) != {"sft", "reward"}:
        raise ValueError("Exactly the genuine completed SFT and reward parents required")
    parent_manifests = {}
    for record in campaign["stages"]:
        stage = record["stage"]
        pin = plan["parents"][stage]
        audited = report["stages"][stage]
        directory = Path(original["output"]).resolve() / stage
        if (set(pin) != {"artifact_dir", "manifest_sha256"}
                or Path(pin["artifact_dir"]).resolve() != directory / "final-artifact"
                or any(audited.get(k) != pin[k] for k in pin)
                or audited.get("rows") != parent_supervisor.COUNTS[stage]
                or audited.get("optimizer_updates") != parent_supervisor.UPDATES[stage]
                or record.get("status") != "accepted" or record.get("exit_code") != 0
                or record.get("active_processes") != [] or not record.get("known_processes")
                or Path(audited["progress_path"]).resolve() != directory / "progress.json"):
            raise ValueError("Parent stage lacks accepted exited full real work")
        _pin(audited["progress_path"], audited["progress_sha256"])
        progress = _parse(audited["progress_path"])
        if (progress.get("status") != "formal_parent_stage_passed"
                or progress.get("formal_training") is not True
                or progress.get("research_complete") is not False
                or progress["artifact"]["manifest_sha256"] != pin["manifest_sha256"]
                or progress["request"]["parent_plan_sha256"] != plan["parent_plan_sha256"]
                or record.get("progress_sha256") != audited["progress_sha256"]):
            raise ValueError("Parent accepted progress identity changed")
        manifest = _bundle(pin["artifact_dir"], pin["manifest_sha256"],
                           {"weights.pt"}, "alignlab-completed-stage-v1")
        parent_supervisor._formal_completion(manifest, stage)
        if (manifest["identity"] != progress["identity"]
                or manifest["identity"]["config"].get("purpose") != parent_supervisor.STAGE_KIND
                or manifest["identity"]["inputs"]["parent_plan"]["sha256"] != plan["parent_plan_sha256"]):
            raise ValueError("A capacity or unrelated artifact cannot be a formal parent")
        parent_manifests[stage] = manifest
    reward_sft = parent_manifests["reward"]["identity"]["parents"].get("sft", {})
    if (reward_sft.get("sha256") != plan["parents"]["sft"]["manifest_sha256"]
            or Path(reward_sft.get("path", "")).resolve()
            != Path(plan["parents"]["sft"]["artifact_dir"]).resolve() / "manifest.json"):
        raise ValueError("Reward and policy must descend from the same completed SFT")
    return original, campaign


def validate_plan(path, expected_sha):
    _pin(path, expected_sha)
    plan = _parse(path)
    fields = {"schema_version", "kind", "output", "wall_seconds", "poll_seconds", "source_sha256",
              "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir",
              "audit_manifest_sha256", "cpu_check_report", "cpu_check_report_sha256",
              "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path",
              "parent_plan_sha256", "parent_acceptance_path", "parent_acceptance_sha256",
              "parents", "ppo_rows", "prerequisites", "process_wait_policy",
              "allowed_background_processes"}
    if (set(plan) != fields or type(plan["schema_version"]) is not int
            or plan["schema_version"] != 1 or plan["kind"] != KIND):
        raise ValueError("Exact versioned primary-alignment plan required")
    if (type(plan["wall_seconds"]) is not int or plan["wall_seconds"] <= 0
            or type(plan["poll_seconds"]) is not int or plan["poll_seconds"] != 10
            or type(plan["ppo_rows"]) is not int or plan["ppo_rows"] != 1024):
        raise ValueError("Explicit positive shared budget, ten-second polling and user-selected PPO1024 required")
    process_guard.validate_policy(plan["process_wait_policy"])
    process_guard.validate_background(plan["allowed_background_processes"])
    output = Path(plan["output"])
    if (not output.is_absolute() or output.is_symlink()
            or output.resolve().parent != (ROOT / "runs").resolve()):
        raise ValueError("Output must be a fresh direct child of project runs")
    original, _ = _parent_acceptance(plan)
    sources = plan["source_sha256"]
    if (not isinstance(sources, dict)
            or not (set(original["source_sha256"]) | NEW_SOURCE_FILES) <= set(sources)
            or any(sources.get(k) != v for k, v in original["source_sha256"].items())):
        raise ValueError("All 59 parent dependencies and observed-v2 implementation/test files must be frozen")
    for name, digest in sources.items():
        _pin(_source(name), digest)
    _pin(plan["cpu_check_report"], plan["cpu_check_report_sha256"])
    cpu = _parse(plan["cpu_check_report"])
    if (cpu.get("status") != "cpu_software_checks_passed" or cpu.get("source_unchanged") is not True
            or cpu.get("cuda_initialized") is not False or cpu.get("device") != "cpu"
            or type(cpu.get("tests_run")) is not int or cpu["tests_run"] < 253
            or any(type(cpu.get(k)) is not int or cpu[k] != 0 for k in ("errors", "failures", "skipped"))):
        raise ValueError("Complete fresh CPU proof required without failures/skips/CUDA")
    checked = {k.replace("\\", "/"): v for k, v in cpu.get("source_before", cpu.get("source_sha256", {})).items()}
    required = set(sources)
    if not required <= set(checked) or any(checked[k] != sources[k] for k in required):
        raise ValueError("Fresh CPU proof does not cover frozen alignment dependencies")
    _prerequisites(plan["prerequisites"])
    _pin(path, expected_sha)
    return plan


class _LiveIdentity:
    """Cheap active-period drift sentinel, not cryptographic content acceptance.

    Complete input hashing still happens before each dispatch and after each
    child exits. During training, small identity files are rehashed while large
    immutable artifacts retain file ID, size and nanosecond timestamp pins.
    Unchanged stat fields alone cannot prove unchanged content.
    """
    HASH_LIMIT = 2 * 1024 * 1024

    @staticmethod
    def _metadata(path):
        path = Path(path).absolute()
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
                or getattr(info, "st_file_attributes", 0) & 0x400
                or any(p.is_symlink() or getattr(p.lstat(), "st_file_attributes", 0) & 0x400
                       for p in path.parents)):
            raise ValueError(f"Live identity path is not an ordinary file: {path}")
        return {"device": info.st_dev, "file_id": info.st_ino, "size_bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
                "mode": info.st_mode, "file_attributes": getattr(info, "st_file_attributes", 0),
                "symlink": False}

    @staticmethod
    def _members(directory):
        directory = Path(directory).absolute()
        found = []
        for parent, folders, files in os.walk(directory, followlinks=False):
            for name in ["", *folders, *files]:
                path = Path(parent) / name if name else Path(parent)
                info = path.lstat()
                if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ValueError(f"Reparse/link in immutable input inventory: {path}")
            found.extend(str((Path(parent) / name).absolute()) for name in files)
        if not directory.is_dir():
            raise ValueError(f"Immutable input directory disappeared: {directory}")
        return tuple(sorted(found))

    def __init__(self, files, *, trees=()):
        self.files, self.digests, self.trees = {}, {}, {}
        for path, digest in files.items():
            self.add_file(path, digest)
        for directory in trees:
            self.add_tree(directory)

    def add_file(self, path, digest=None, *, always_hash=False):
        path = str(Path(path).absolute())
        before = self._metadata(path)
        if path in self.files:
            if before != self.files[path]:
                raise ValueError(f"Cannot refresh a changed live identity: {path}")
            if digest is not None and path in self.digests and self.digests[path] != digest:
                raise ValueError(f"Conflicting live identity hash: {path}")
            return
        if always_hash or before["size_bytes"] <= self.HASH_LIMIT or Path(path).suffix == ".py":
            actual = sha(path)
            if digest is not None and actual != digest:
                raise ValueError(f"Small frozen identity changed: {path}")
            self.digests[path] = actual
        if self._metadata(path) != before:
            raise ValueError(f"Identity file changed while registering: {path}")
        self.files[path] = before

    def add_tree(self, directory):
        directory = str(Path(directory).absolute())
        members = self._members(directory)
        if directory in self.trees and self.trees[directory] != members:
            raise ValueError(f"Cannot refresh a changed immutable inventory: {directory}")
        for path in members:
            self.add_file(path)
        self.trees[directory] = members

    @classmethod
    def from_plan(cls, path, digest, plan):
        files = {str(Path(path).absolute()): digest}
        files.update({str(_source(name)): value for name, value in plan["source_sha256"].items()})
        for key, hash_key in (("cpu_check_report", "cpu_check_report_sha256"),
                              ("parent_plan_path", "parent_plan_sha256"),
                              ("parent_acceptance_path", "parent_acceptance_sha256"),
                              ("capacity_acceptance_path", "capacity_acceptance_sha256")):
            files[plan[key]] = plan[hash_key]
        for pin in plan["prerequisites"]:
            files[pin["path"]] = pin["sha256"]
        accepted = _parse(plan["parent_acceptance_path"])
        for pin in accepted.get("evidence_files", {}).values():
            files[pin["path"]] = pin["sha256"]
        trees = [plan[k] for k in ("model_snapshot", "tokenizer_dir", "audit_dir")]
        trees.extend(pin["artifact_dir"] for pin in plan["parents"].values())
        trees.append(_parse(plan["parent_plan_path"])["output"])
        for protected in accepted.get("protected_paths", []):
            if Path(protected).is_dir():
                trees.append(protected)
        sentinel = cls(files, trees=sorted(set(trees)))
        # Plan/request/CPU/audit authorization files remain hashed even if an
        # unusually verbose report happens to exceed the small-file threshold.
        for filename in files:
            if Path(filename).suffix.lower() in {".json", ".py"}:
                sentinel.digests[str(Path(filename).absolute())] = files[filename]
        return sentinel

    def check(self):
        for path, expected in self.files.items():
            if self._metadata(path) != expected:
                raise ValueError(f"Active-period input metadata drift: {path}")
        for path, expected in self.digests.items():
            if sha(path) != expected or self._metadata(path) != self.files[path]:
                raise ValueError(f"Active-period small identity content drift: {path}")
        for directory, expected in self.trees.items():
            if self._members(directory) != expected:
                raise ValueError(f"Active-period input inventory drift: {directory}")
        return True

    def record(self):
        return {"kind": "alignlab-live-identity-drift-sentinel-v1",
                "large_file_content_proven_by_stat": False,
                "full_hash_required": ["before_each_dispatch", "after_each_child_exit", "final_publication"],
                "files": self.files, "live_content_sha256": self.digests,
                "immutable_tree_members": self.trees}


def _command(record, script, path):
    return parent_supervisor._command(record, script, path)


def _own_processes(processes, plan_path):
    by_pid = {p["pid"]: p for p in processes}
    current, own = by_pid.get(os.getpid()), {}
    if current is None or not _command(current, "run_alignlab_alignment.py", plan_path):
        raise RuntimeError("Cannot identify current supervisor PID/create-time/command")
    while current and _command(current, "run_alignlab_alignment.py", plan_path):
        if current["pid"] in own:
            raise RuntimeError("Cyclic supervisor ancestry")
        _stamp(current["created_at_utc"])
        own[current["pid"]] = current
        current = by_pid.get(current.get("parent_pid"))
    return own


def _same_identity(actual, expected):
    """Keep Windows' seventh creation-time digit and parent identity intact."""
    return (process_guard.strict.ProcessIdentity.from_record(actual)
            == process_guard.strict.ProcessIdentity.from_record(expected))


def _peer_decision(processes, own, known=(), background=()):
    # v2's allow matcher binds PID/create/command, but ancestry is also part of
    # this supervisor's external child proof. Never accept a changed parent.
    known = dict(known)
    actual = {p["pid"]: p for p in processes}
    for pid, expected in known.items():
        observed = actual.get(pid)
        if (observed is not None
                and process_guard.strict.ProcessIdentity.from_record(observed).instance
                == process_guard.strict.ProcessIdentity.from_record(expected).instance
                and not _same_identity(observed, expected)):
            raise RuntimeError("Known child full process identity changed")
    return process_guard.strict_admission(lambda: processes,
        owned_identities=list(own.values()),
        allowed_identities=[*background, *known.values()], deadline_epoch=None)


def _peers(processes, own, known=(), background=()):
    """Compatibility assertion: no arbitrary nonempty Python is grandfathered."""
    decision = _peer_decision(processes, own, known, background)
    if not decision["caller_admission_allowed"]:
        raise RuntimeError("Strict process admission blocked an unknown, competing or changed identity")
    return decision


def _waitable_peer_decision(decision):
    """A live child handles uncertainty at its own committed safe boundary."""
    if decision.get("allowed_ancestry_verified") is False:
        return False
    observations = decision.get("observations", [])
    if (observations and observations[-1].get("snapshot_valid") is True
            and decision.get("owned_identity_verified") is not True):
        return False
    codes = {item["code"] for item in decision.get("blockers", [])}
    return bool(codes) and codes <= {"unknown_command", "unclassified_process", "snapshot_unavailable"}


def _parents_gone(plan, processes):
    report = _parse(plan["parent_acceptance_path"])
    if report.get("kind") in {"alignlab-independent-recovered-parents-acceptance-v1",
                              "alignlab-independent-recovered-parents-r5-acceptance-v1"}:
        historical = [row["expected"] for row in report["processes"]]
    else:
        campaign = _parse(report["campaign"]["path"])
        historical = campaign["supervisor_processes"] + [
            p for stage in campaign["stages"] for p in stage["known_processes"]]
    by_pid = {p["pid"]: p for p in processes}
    if any(p["pid"] in by_pid and _same_identity(by_pid[p["pid"]], p) for p in historical):
        raise RuntimeError("Accepted parent process is still alive; do not start alignment")


def workload(stage, ppo_rows):
    if stage == "dpo":
        return {"scope": "full_accepted_real_training_partition", "rows": 13139,
                "optimizer_updates": 822, "beta": 0.1}
    return {"scope": "registered_unique_real_prompt_subset", "rows": ppo_rows,
            "optimizer_updates": 2 * ppo_rows, "selection_seed": 42,
            "selection_algorithm": "sha256_prompt_id_v1", "kl_coef": 0.05}


def _cache(pin):
    """Verify immutable stored cache contents, not a claim that CPU scored it."""
    from .reference_cache import load_cache
    directory = Path(pin["path"])
    _pin(directory / "identity.json", pin["identity_sha256"])
    identity = _parse(directory / "identity.json")
    cached = load_cache(directory, identity, expected_manifest_sha256=pin["manifest_sha256"])
    return cached, identity


def _cpu_payload(path, expected_sha):
    """Safe CPU deserialization of the very bytes whose external hash is pinned."""
    import torch
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Checkpoint/artifact payload changed before CPU decoding")
    return torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")


def _checkpoint_observation(directory, manifest, receipt):
    """Authenticate the cursor's embedded pin before stripping observation data."""
    from .training_telemetry import verify_checkpoint_telemetry
    sidecar, binding = receipt.get("telemetry_sidecar"), receipt.get("telemetry_binding")
    root = Path(directory).resolve().parent / "telemetry"
    for pin, folder in ((sidecar, "checkpoints"), (binding, "bindings")):
        if (not isinstance(pin, dict) or set(pin) != {"path", "sha256"}
                or Path(pin["path"]).resolve().parent != root / folder):
            raise ValueError("Checkpoint requires its own telemetry sidecar and binding")
        _pin(pin["path"], pin["sha256"])
    bound = _parse(binding["path"])
    if (Path(bound.get("checkpoint_path", "")).resolve() != Path(directory).resolve()
            or bound.get("manifest_sha256") != receipt["manifest_sha256"]
            or bound.get("sidecar") != sidecar):
        raise ValueError("Actual checkpoint payload cursor/telemetry binding differs from the receipt")
    return verify_checkpoint_telemetry(directory,
        expected_manifest_sha256=receipt["manifest_sha256"],
        expected_sidecar_sha256=sidecar["sha256"])


def _verify_observation_artifacts(output, request, result):
    """Require complete independently verified observations, never status alone."""
    from .training_telemetry import verify_training_telemetry
    identity, pin = result["identity"], result.get("telemetry")
    if (identity["config"].get("execution_revision") != EXECUTION_REVISION
            or identity["config"].get("telemetry_protocol") != TELEMETRY_PROTOCOL
            or not isinstance(pin, dict) or set(pin) != {"path", "sha256"}
            or Path(pin["path"]).resolve() != Path(output).resolve() / "telemetry/manifest.json"):
        raise ValueError("Complete observed-v2 telemetry identity and manifest required")
    observed = verify_training_telemetry(Path(pin["path"]).parent,
        expected_manifest_sha256=pin["sha256"],
        expected_identity_sha256=hashlib.sha256(_canonical(identity)).hexdigest(),
        expected_completion=result["completion"], checkpoints=result["checkpoints"])
    if (observed.get("stage") != request["stage"]
            or observed.get("final_cursor") != result["cursor"]
            or observed.get("started_at_epoch") != result["started_at_epoch"]
            or type(observed.get("finished_at_epoch")) not in (int, float)
            or not result["started_at_epoch"] <= observed["finished_at_epoch"] <= result["finished_at_epoch"]):
        raise ValueError("Telemetry does not describe this completed stage/cursor/time")
    _verify_cost_checkpoints(result, observed)
    return observed


def _verify_cost_checkpoints(result, observed):
    """Verify real scheduling/IO clocks, including repeated observations of one state."""
    start, end = result["started_at_epoch"], result["finished_at_epoch"]
    warm = observed.get("warm_training_started_at_epoch")
    number = lambda value: type(value) in (int, float) and math.isfinite(value)
    if not number(warm) or not start <= warm <= end:
        raise ValueError("Telemetry lacks its original warm-training clock")
    crossed, eligible = set(), set()
    for receipt in result["checkpoints"]:
        before, after = receipt.get("save_started_at_epoch"), receipt.get("save_finished_at_epoch")
        points = receipt.get("cost_observations")
        if (not number(before) or not number(after) or not start <= before <= after <= end
                or not isinstance(points, list) or not points
                or receipt["active_elapsed_seconds"] != points[0].get("stage_elapsed_seconds")):
            raise ValueError("Checkpoint lacks complete measured save/observation times")
        binding = _parse(receipt["telemetry_binding"]["path"])
        if not (before <= binding["save_started_at_epoch"] <= binding["save_finished_at_epoch"] <= after):
            raise ValueError("Checkpoint I/O timing differs from its immutable binding")
        all_reasons, previous_time = set(), start
        for index, point in enumerate(points):
            at = point.get("observed_at_epoch")
            stage_time, warm_time = point.get("stage_elapsed_seconds"), point.get("warm_training_elapsed_seconds")
            reasons = point.get("reasons")
            if (not all(number(v) for v in (at, stage_time, warm_time))
                    or not previous_time <= at <= end or stage_time < 0 or warm_time < 0
                    or stage_time > at - start + 1e-6 or warm_time > at - warm + 1e-6
                    or point.get("reused_same_cursor") is not (index > 0)
                    or not isinstance(reasons, list) or not reasons
                    or len(set(reasons)) != len(reasons)):
                raise ValueError("Invalid actual checkpoint cost observation")
            previous_time = at
            all_reasons.update(reasons)
            expected = []
            for axis, prefix, elapsed in (("stage_wall", "cost_", stage_time),
                                          ("warm_training_wall", "warm_cost_", warm_time)):
                for hour in (1, 2, 4, 8):
                    key, target = (axis, hour), hour * 3600
                    if elapsed >= target:
                        eligible.add(key)
                    if f"{prefix}{hour}h" in reasons:
                        if elapsed < target or key in crossed:
                            raise ValueError("Premature or duplicate cost checkpoint label")
                        crossed.add(key)
                        expected.append({"axis": axis, "target_seconds": target,
                            "actual_seconds": elapsed, "overshoot_seconds": elapsed - target})
            allowed = {"first_update", "every_50_updates", "full_data_pass", "safe_engine_return"}
            allowed.update(f"{prefix}{hour}h" for prefix in ("cost_", "warm_cost_") for hour in (1, 2, 4, 8))
            if not set(reasons) <= allowed or point.get("thresholds") != expected:
                raise ValueError("Cost target/actual/overshoot evidence differs")
        if set(receipt["reasons"]) != all_reasons:
            raise ValueError("Checkpoint labels omit or invent cost observations")
    if crossed != eligible:
        raise ValueError("An actually reached checkpoint cost threshold was omitted")


def _checkpoint_content(directory, manifest, receipt, *, final=False, artifact=None,
                        require_observation=False):
    """Verify raw historical parent cursors or authenticated observed cursors.

    The frozen parent auditor also calls this helper. Its original SFT/RM
    schema has no telemetry; only that exact unobserved parent identity may
    use the original cursor. Alignment branches explicitly require telemetry,
    even if a caller were to supply a parent-shaped identity.
    """
    import torch
    payload = _cpu_payload(Path(directory) / "state.pt", manifest["artifacts"]["state.pt"]["sha256"])
    if (not isinstance(payload, dict)
            or set(payload) != {"parameters", "optimizer", "cursor", "rng", "extra_state"}
            or not isinstance(payload["cursor"], dict)):
        raise ValueError("Actual checkpoint payload cursor differs from its receipt")
    config = manifest.get("identity", {}).get("config", {})
    legacy_parent = (
        not require_observation
        and config.get("stage") in parent_supervisor.STAGES
        and config.get("purpose") == parent_supervisor.STAGE_KIND
        and "execution_revision" not in config and "telemetry_protocol" not in config
        and "telemetry" not in payload["cursor"]
        and not {"telemetry_sidecar", "telemetry_binding"}.intersection(receipt))
    cursor = (payload["cursor"] if legacy_parent else
              _checkpoint_observation(directory, manifest, receipt)["cursor"])
    if cursor != receipt["cursor"]:
        raise ValueError("Actual checkpoint payload cursor differs from its receipt")
    parameters, optimizer = payload["parameters"], payload["optimizer"]
    if (not isinstance(parameters, dict) or not parameters or not isinstance(optimizer, dict)
            or set(optimizer) != {"state", "param_groups"} or not optimizer["param_groups"]
            or not isinstance(payload["rng"], dict) or not payload["rng"]):
        raise ValueError("Checkpoint lacks actual parameters/optimizer/RNG state")
    ids = [value for group in optimizer["param_groups"] for value in group["params"]]
    if (len(ids) != len(parameters) or len(set(ids)) != len(ids)
            or set(optimizer["state"]) != set(ids)):
        raise ValueError("Checkpoint optimizer does not own every saved training parameter")
    for name, parameter in parameters.items():
        if (not isinstance(name, str) or not isinstance(parameter, torch.Tensor)
                or parameter.device.type != "cpu" or not parameter.is_floating_point()
                or not bool(torch.isfinite(parameter).all())):
            raise ValueError("Invalid or nonfinite CPU checkpoint parameter")
    for parameter, index in zip(parameters.values(), ids):
        state = optimizer["state"][index]
        step = state.get("step")
        if (not isinstance(step, torch.Tensor) or step.device.type != "cpu" or step.numel() != 1
                or step.item() != receipt["optimizer_updates"]):
            raise ValueError("Actual Adam optimizer step differs from claimed checkpoint update")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.shape != parameter.shape or not bool(torch.isfinite(value).all())):
                raise ValueError("Missing/nonfinite/malformed Adam optimizer moments")
    active = cursor.get("active_rollout")
    if (type(active) is not bool or active != (payload["extra_state"] is not None)
            or (final and (active or payload["extra_state"] is not None))):
        raise ValueError("Checkpoint active-rollout payload disagrees with its cursor")
    if final:
        if artifact is None:
            raise ValueError("Final checkpoint requires the actually exported weights")
        normalized = {}
        for name, value in parameters.items():
            if name.startswith("scalar_heads.value."):
                key = "heads/value/" + name[len("scalar_heads.value."):]
            elif ".lora_A.policy." in name or ".lora_B.policy." in name:
                key = "adapter/" + name.replace(".lora_A.policy.", ".lora_A.").replace(".lora_B.policy.", ".lora_B.")
            else:
                raise ValueError("Unexpected final policy/value parameter name")
            if key in normalized:
                raise ValueError("Final parameter names collide after role normalization")
            normalized[key] = value
        if (not isinstance(artifact, dict) or set(artifact) != set(normalized)
                or any(not isinstance(artifact[name], torch.Tensor)
                       or artifact[name].device.type != "cpu"
                       or artifact[name].dtype != value.dtype or artifact[name].shape != value.shape
                       or not torch.equal(artifact[name], value) for name, value in normalized.items())):
            raise ValueError("Final exported weights differ from the actual final checkpoint")


def _verify_stage(output, request, known, code):
    """Accept an exited child only after real files and exact work are checked."""
    if code != 0:
        raise RuntimeError(f"Alignment child exited {code}; no automatic retry")
    output = Path(output).resolve()
    result = _parse(output / "progress.json")
    stage, work = request["stage"], request["workload"]
    rows, updates = work["rows"], work["optimizer_updates"]
    if (result.get("status") != "formal_alignment_stage_passed"
            or result.get("formal_training") is not True or result.get("research_complete") is not False
            or result.get("stage") != stage or result.get("request") != request
            or _parse(output / "request.json") != request or result.get("pid") not in known
            or result.get("actual_optimizer_updates") != updates
            or result.get("deadline_epoch") != request["deadline_epoch"]
            or type(result.get("started_at_epoch")) not in (int, float)
            or type(result.get("finished_at_epoch")) not in (int, float)
            or not result["started_at_epoch"] <= result["finished_at_epoch"] <= request["deadline_epoch"]):
        raise RuntimeError("Missing exact completed request/process/deadline/update evidence")
    model = result.get("model", {})
    if (model.get("device") != "cuda:0" or model.get("quantization") != NF4
            or model.get("quantized_linear_count") != 196
            or type(model.get("parameters_selected")) is not int or model["parameters_selected"] <= 0
            or model.get("snapshot_manifest_sha256") != request["model_manifest_sha256"]):
        raise RuntimeError("Child lacks actual initialized CUDA NF4 model evidence")
    samples = result.get("gpu_samples")
    if (not isinstance(samples, list) or not samples
            or any(not any(g.get("index") == 0 for g in sample.get("gpus", []))
                   or not all(type(sample.get(k)) in (int, float) and math.isfinite(sample[k])
                              and sample[k] > 0 for k in ("max_allocated_bytes", "max_reserved_bytes"))
                   for sample in samples)):
        raise RuntimeError("Child lacks positive actual GPU memory measurements")
    selection = result["selection"]
    partition = "preference_train" if stage == "dpo" else "ppo_prompts"
    if (selection.get("scope") != work["scope"] or selection.get("partition") != partition
            or any(selection.get(k) != rows for k in ("selected_rows", "real_rows"))
            or selection.get("accepted_partition_rows") != (13139 if stage == "dpo" else 12748)
            or selection.get("selection_algorithm") != ("all_accepted_rows" if stage == "dpo" else work["selection_algorithm"])
            or selection.get("selection_seed") != (None if stage == "dpo" else 42)
            or selection.get("synthetic_rows") != 0
            or selection.get("accepted_manifest_sha256") != request["audit_manifest_sha256"]):
        raise ValueError("Planned selection must contain the exact registered real work")
    selection_path = Path(selection["selection_manifest_path"]).resolve()
    if selection_path != output / "selection.json":
        raise ValueError("Selection manifest escaped child output")
    _pin(selection_path, selection["selection_manifest_sha256"])
    saved_selection = _parse(selection_path)
    summary = {k: v for k, v in selection.items()
               if k not in {"selection_manifest_path", "selection_manifest_sha256"}}
    records = saved_selection["records"]
    if (saved_selection["summary"] != summary or len(records) != rows
            or len({r["record_id"] for r in records}) != rows
            or hashlib.sha256(_canonical(records)).hexdigest() != selection["records_sha256"]
            or any(not isinstance(r["record_id"], str) or r["record_id"].startswith("synthetic-")
                   or not 1 <= r["prompt_tokens"] <= 1024
                   or not re.fullmatch("[0-9a-f]{64}", r["token_sha256"])
                   or (stage == "dpo" and not (1 <= r["chosen_tokens"] <= 768
                                               and 1 <= r["rejected_tokens"] <= 768))
                   or (stage == "ppo" and (r["chosen_tokens"] is not None or r["rejected_tokens"] is not None))
                   for r in records)):
        raise ValueError("Invalid selection identity, lengths or token summary")
    manifest = _parse(Path(request["audit_dir"]) / "manifest.json")
    partition_path = Path(request["audit_dir"]) / (partition + ".jsonl")
    partition_sha = manifest["artifacts"][partition_path.name]
    _pin(partition_path, partition_sha)
    with partition_path.open(encoding="utf-8") as stream:
        accepted = [json.loads(line) for line in stream if line.strip()]
    if stage == "ppo":
        accepted.sort(key=lambda r: (hashlib.sha256(
            f'{work["selection_seed"]}:{r["prompt_id"]}'.encode()).hexdigest(), r["prompt_id"]))
        accepted = accepted[:rows]
        if len({r["prompt_id"] for r in records}) != rows:
            raise ValueError("PPO registered prompts must be unique")
    if [(r["record_id"], r["cluster_id"], r["prompt_id"]) for r in records] != [
            (r["record_id"], r["cluster_id"], r["prompt_id"]) for r in accepted]:
        raise ValueError("Selection does not match the exact registered train IDs/order")
    if (selection.get("unique_prompt_ids") != len({r["prompt_id"] for r in records})
            or selection.get("independent_prompt_clusters") != len({r["cluster_id"] for r in records})):
        raise ValueError("Selection cluster/prompt counts differ from its actual records")
    identity = result["identity"]
    if _parse(output / "identity.json") != identity:
        raise ValueError("Saved identity differs from progress")
    config = identity["config"]
    if (config.get("stage") != stage or config.get("purpose") != STAGE_KIND
            or config.get("execution_revision") != EXECUTION_REVISION
            or config.get("telemetry_protocol") != TELEMETRY_PROTOCOL
            or config.get("scope") != work["scope"] or config.get("formal_training") is not True
            or config.get("synthetic_rows") != 0 or config.get("registered_workload") != work
            or config.get("work", {}).get("seed") != request["seed"]):
        raise ValueError("Artifact identity does not declare this registered formal branch")
    if not NEW_SOURCE_FILES <= set(request["source_sha256"]):
        raise ValueError("Completed branch lacks mandatory observed-v2 source identity")
    if set(identity["parents"]) != set(request["parents"]):
        raise ValueError("Branch parent identity differs")
    for name, pin in request["parents"].items():
        item = identity["parents"][name]
        if (item["sha256"] != pin["manifest_sha256"]
                or Path(item["path"]).resolve() != Path(pin["artifact_dir"]).resolve() / "manifest.json"):
            raise ValueError("Branch parent is not the admitted completed formal model")
    expected_imports = {"policy": ("sft", []), "reference": ("sft", [])}
    if stage == "ppo":
        expected_imports["reward"] = ("reward", ["reward"])
    imports = result.get("imports", {})
    identity_sha = hashlib.sha256(_canonical(identity)).hexdigest()
    if set(imports) != set(expected_imports):
        raise ValueError("Missing exact policy/reference/reward initialization receipts")
    for role, (parent, heads) in expected_imports.items():
        receipt = imports[role]
        if (receipt.get("stage") != parent or receipt.get("target_stage") != stage
                or receipt.get("target_role") != role or receipt.get("heads") != heads
                or receipt.get("target_identity_sha256") != identity_sha
                or receipt.get("manifest_sha256") != request["parents"][parent]["manifest_sha256"]):
            raise ValueError("Initialization receipt differs from final branch/cache identity")
    required = {"source/" + name: (ROOT / name, digest) for name, digest in request["source_sha256"].items()}
    required.update({name: (request[name + "_path"], request[name + "_sha256"])
                     for name in ("alignment_plan", "parent_plan", "parent_acceptance", "capacity_acceptance")})
    required.update({
        "data/manifest": (Path(request["audit_dir"]) / "manifest.json", request["audit_manifest_sha256"]),
        "data/partition": (partition_path, partition_sha),
        "data/selection": (selection_path, selection["selection_manifest_sha256"]),
    })
    snapshot = _parse(Path(request["model_snapshot"]) / "snapshot-manifest.json")
    required.update({"base/" + name: (Path(request["model_snapshot"]) / name, item["sha256"])
                     for name, item in snapshot["files"].items()})
    token_dir = Path(request["tokenizer_dir"])
    token_manifest = _parse(token_dir / "source-manifest.json")
    required.update({"tokenizer/" + name: (token_dir / name, item["sha256"])
                     for name, item in token_manifest["artifacts"].items()})
    required["tokenizer/source-manifest.json"] = (token_dir / "source-manifest.json", sha(token_dir / "source-manifest.json"))
    for name, (path, digest) in required.items():
        item = identity["inputs"].get(name, {})
        if item.get("sha256") != digest or Path(item.get("path", "")).resolve() != Path(path).resolve():
            raise ValueError("Frozen branch input identity mismatch: " + name)
    for name, item in identity["inputs"].items():
        if not name.startswith("base/"):
            _pin(item["path"], item["sha256"])
            if Path(item["path"]).stat().st_size != item["size_bytes"]:
                raise ValueError("Frozen branch input size changed")
    cursor = result["cursor"]
    if (result.get("outcome_stop_reason") != "data_exhausted" or cursor.get("offset") != rows
            or cursor.get("optimizer_updates") != updates or cursor.get("active_rollout") is not False
            or cursor.get("completed_rollouts") != (rows // 16 if stage == "ppo" else 0)
            or result.get("adapter_weights_changed") is not True
            or result.get("scalar_heads_changed") != ({"value": True} if stage == "ppo" else {})):
        raise RuntimeError("Incomplete registered optimizer work")
    metrics = [json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
               for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    boundaries = [m for m in metrics if m.get("event") == "engine_boundary"]
    objectives = [m for m in metrics if m.get("event") == "objective_after_engine_return"]
    if (result.get("objective_logging") != "after_engine_returns" or result.get("objective_records") != updates
            or [m["cursor"]["optimizer_updates"] for m in boundaries] != list(range(1, updates + 1))
            or [m.get("update") for m in objectives] != list(range(1, updates + 1))
            or any(not all(type(m.get(k)) in (int, float) and math.isfinite(m[k])
                           for k in ("objective", "gradient_norm")) for m in objectives)):
        raise RuntimeError("Missing continuous finite objective/update evidence")
    if stage == "dpo" and any(m.get("rows") != min(16, rows - i * 16) for i, m in enumerate(objectives)):
        raise ValueError("DPO final partial batch/work accounting changed")
    if stage == "ppo" and any(m.get("responses") != 1 or type(m.get("action_tokens")) is not int
                             or not 1 <= m["action_tokens"] <= 256 for m in objectives):
        raise ValueError("PPO exact per-response optimizer work changed")
    completion = result["completion"]
    if (completion.get("scope") != work["scope"] or completion.get("synthetic_rows") != 0
            or completion.get("stop_reason") != "data_exhausted"
            or any(completion.get(k) is not True for k in
                   ("data_pass_complete", "protocol_complete", "full_selected_dataset_pass"))
            or completion.get("full_accepted_dataset_pass") is not (stage == "dpo")
            or completion.get("rows_processed") != rows or completion.get("optimizer_updates") != updates
            or completion.get("response_tokens") != cursor.get("response_tokens")
            or type(cursor.get("response_tokens")) is not int or cursor["response_tokens"] <= 0):
        raise ValueError("Incomplete registered work must not claim full-partition completion")
    if (completion.get("completed_rollouts") != cursor["completed_rollouts"]
            or (stage == "dpo" and cursor["response_tokens"] != sum(r["chosen_tokens"] + r["rejected_tokens"] for r in records))
            or (stage == "ppo" and (not rows <= cursor["response_tokens"] <= rows * 256
                                   or sum(m["action_tokens"] for m in objectives) != 2 * cursor["response_tokens"]))):
        raise ValueError("Actual token/rollout counts differ from registered work")
    artifact = result["artifact"]
    if (Path(artifact["artifact_dir"]).resolve() != output / "final-artifact"
            or Path(artifact["manifest_path"]).resolve() != output / "final-artifact/manifest.json"):
        raise ValueError("Final artifact escaped child output")
    saved = _bundle(output / "final-artifact", artifact["manifest_sha256"],
                    {"weights.pt"}, "alignlab-completed-stage-v1")
    if saved.get("stage") != stage or saved["identity"] != identity or saved.get("completion") != completion:
        raise ValueError("Final artifact identity/completion differs")
    exported_weights = _cpu_payload(output / "final-artifact/weights.pt",
                                    saved["artifacts"]["weights.pt"]["sha256"])
    checkpoints = result["checkpoints"]
    indices = [c["optimizer_updates"] for c in checkpoints]
    keys = [c["cursor_sha256"] for c in checkpoints]
    if (len(keys) != len(set(keys)) or not ({1, updates} | set(range(50, updates + 1, 50))) <= set(indices)
            or result["latest_safe_checkpoint"] not in checkpoints
            or result["latest_safe_checkpoint"]["optimizer_updates"] != updates
            or result["latest_safe_checkpoint"]["cursor"] != cursor):
        raise ValueError("First/periodic/final safe checkpoints missing")
    for receipt in checkpoints:
        directory, update = Path(receipt["checkpoint_path"]).resolve(), receipt["optimizer_updates"]
        cursor_sha = hashlib.sha256(_canonical(receipt["cursor"])).hexdigest()
        if (directory.parent != output or type(update) is not int or not 0 <= update <= updates
                or receipt["cursor_sha256"] != cursor_sha
                or receipt["cursor"].get("optimizer_updates") != update
                or directory.name != f"update-{update:06d}-{cursor_sha[:16]}" or not receipt["reasons"]
                or type(receipt.get("active_elapsed_seconds")) not in (int, float)
                or not 0 <= receipt["active_elapsed_seconds"] <= result["finished_at_epoch"] - result["started_at_epoch"]):
            raise ValueError("Invalid checkpoint receipt")
        checkpoint = _bundle(directory, receipt["manifest_sha256"], {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint branch identity changed")
        _checkpoint_content(directory, checkpoint, receipt,
                            final=receipt == result["latest_safe_checkpoint"], artifact=exported_weights,
                            require_observation=True)
    # A CPU or metadata-only cache cannot substitute for the child's measured
    # frozen-reference guards. The recorded actual loaded tensor anchor must
    # agree with the immutable cache scoring anchor.
    roles = ["reference", "reward"] if stage == "ppo" else ["reference"]
    for role in roles:
        anchor = result.get(role + "_anchor", {})
        metadata = anchor.get("metadata", {})
        if (not anchor.get("tensors") or metadata.get("device") != "cuda:0"
                or metadata.get("precision") != "bfloat16" or metadata.get("quantization") != NF4):
            raise RuntimeError("Missing actual frozen CUDA reference/reward tensor anchor")
    audits = result.get("frozen_full_audits", [])
    if ([a.get("phase") for a in audits] != ["before_training", "after_training", "after_export"]
            or any(a.get("roles") != roles or a.get("reward_head_content_verified") is not (stage == "ppo")
                   or type(a.get("time_epoch")) not in (int, float)
                   or not result["started_at_epoch"] <= a["time_epoch"] <= result["finished_at_epoch"]
                   for a in audits)
            or [a["time_epoch"] for a in audits] != sorted(a["time_epoch"] for a in audits)):
        raise RuntimeError("Required before/after/export frozen content audits missing")
    if stage == "ppo":
        if result.get("reference_cache") is not None or not result.get("reward_head_anchor"):
            raise ValueError("PPO requires online frozen scoring and an independently frozen reward head")
        if any(a.get("cache_manifest_sha256") is not None for a in audits):
            raise ValueError("PPO must not claim an offline DPO reference cache")
    else:
        cache_pin = result["reference_cache"]
        requested = request["reference_cache"]
        if (cache_pin.get("mode") != requested["mode"]
                or Path(cache_pin["path"]).resolve() != Path(requested["path"]).resolve()
                or cache_pin.get("exact_key_coverage_verified") is not True
                or any(a.get("cache_manifest_sha256") != cache_pin["manifest_sha256"] for a in audits)
                or (requested["mode"] == "reuse"
                    and any(cache_pin[k] != requested[k] for k in ("manifest_sha256", "identity_sha256")))):
            raise ValueError("Reference cache mode/path/pins/coverage differ from the admitted cache")
        cached, cache_identity = _cache(cache_pin)
        if (cache_pin.get("unique_sequences") != len(cached.entries)
                or cache_pin.get("expected_unique_sequences") != len(cached.entries)
                or not 1 <= len(cached.entries) <= 2 * rows
                or cache_identity["accepted_dataset_manifest"]["sha256"] != request["audit_manifest_sha256"]
                or cache_identity["scoring_config"].get("frozen_reference_receipt") != result["reference_anchor"]):
            raise ValueError("Stored reference cache differs from actual loaded frozen-reference identity")
        for group in ("base", "tokenizer", "source"):
            expected_files = {k: v for k, v in identity["inputs"].items() if k.startswith(group + "/")}
            if cache_identity["files"][group] != expected_files:
                raise ValueError("Reference cache uses a different frozen base/tokenizer/scoring source")
        sft_dir = Path(request["parents"]["sft"]["artifact_dir"])
        sft_manifest = _parse(sft_dir / "manifest.json")
        for label, filename, digest in (
                ("manifest", "manifest.json", request["parents"]["sft"]["manifest_sha256"]),
                ("weights", "weights.pt", sft_manifest["artifacts"]["weights.pt"]["sha256"])):
            item = cache_identity["files"]["adapter"].get(label, {})
            if item.get("sha256") != digest or Path(item.get("path", "")).resolve() != (sft_dir / filename).resolve():
                raise ValueError("Reference cache does not belong to the common completed SFT")
        for key, filename, digest in (("reference_cache/manifest", "manifest.json", cache_pin["manifest_sha256"]),
                                      ("reference_cache/identity", "identity.json", cache_pin["identity_sha256"])):
            item = identity["inputs"].get(key, {})
            if (item.get("sha256") != digest
                    or Path(item.get("path", "")).resolve() != Path(cache_pin["path"]).resolve() / filename):
                raise ValueError("Final branch identity does not pin the consumed reference cache")
    _verify_observation_artifacts(output, request, result)
    guard = result.get("process_guard", {})
    owned = guard.get("owned_identities", [])
    if (not owned or any(p["pid"] not in known or not _same_identity(p, known[p["pid"]])
                         for p in owned)
            or {p["pid"] for p in owned} != set(known)):
        raise ValueError("Child guard identities differ from the externally observed process chain")
    verified_guard = process_guard.verify_guard_log(output / "process-guard.jsonl",
                                                  request=request, progress=result)
    if (verified_guard.get("verified") is not True
            or verified_guard.get("inherited_waited_seconds") != request["process_wait_spent_seconds"]
            or verified_guard.get("total_waited_seconds") != guard.get("total_waited_seconds")):
        raise ValueError("Complete strict boundary-guard and inherited campaign wait proof required")
    return result


def run_alignment(plan_path, expected_plan_sha256, *, execute=False, system=None):
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(plan_path, expected_plan_sha256)
    flags = {"primary_alignment_verified": False, "parents_verified": True,
             "capacity_verified": True, "formal_training": False, "research_complete": False}
    if not execute:
        return {"status": "plan_only", "plan": plan, **flags}
    live_identity = _LiveIdentity.from_plan(plan_path, expected_plan_sha256, plan)
    # Bracket the initial full validation with metadata observations rather
    # than silently blessing a file replacement that occurred while hashing.
    validate_plan(plan_path, expected_plan_sha256)
    live_identity.check()
    system, output = system or _System(), Path(plan["output"]).resolve()
    lock = ROOT / "reports/.alignlab-alignment.lock"
    foreign_locks = [ROOT / ("reports/" + name) for name in
                     (".alignlab-gpu.lock", ".alignlab-parents.lock", ".alignlab-capacity.lock")]
    if output.exists() or lock.exists() or any(p.exists() for p in foreign_locks):
        raise FileExistsError("Existing output or lock; never reuse outputs or delete unknown locks")
    processes = system.processes()
    _parents_gone(plan, processes)
    own = _own_processes(processes, plan_path)
    initial_guard = process_guard.strict_admission(system.processes,
        owned_identities=list(own.values()), allowed_identities=plan["allowed_background_processes"],
        deadline_epoch=None, clock=system.now, sleep=system.sleep)
    if not initial_guard["caller_admission_allowed"]:
        return {"status": "blocked_before_start", "process_guard": initial_guard,
                "started_at_epoch": None, **flags}
    admission = system.resources()
    if admission["blockers"]:
        return {"status": "blocked_before_start", "resources": admission, "started_at_epoch": None, **flags}
    if Path(str(plan_path) + ".stop").exists():
        return {"status": "stopped_before_start", **flags}
    token = {"owner": KIND, "pid": os.getpid(), "plan_sha256": expected_plan_sha256, "token": uuid.uuid4().hex}
    _new(lock, token)
    started = system.now()
    state = {"kind": KIND, "status": "starting", **flags, "formal_training": True,
             "scope": "primary_six_alignment_branches_only", "plan_path": str(plan_path),
             "plan_sha256": expected_plan_sha256, "started_at_epoch": started,
             "deadline_epoch": started + plan["wall_seconds"], "supervisor_processes": list(own.values()),
              "resources_before": admission, "initial_process_guard": initial_guard,
              "total_process_wait_seconds": 0.0, "process_wait_policy": plan["process_wait_policy"], "stages": []}
    child, child_finished = None, True

    def write(status=None, **values):
        if status:
            state["status"] = status
        state.update(values, updated_at_epoch=system.now())
        _status(output / "campaign.json", state)

    def stop_reason():
        if (output / "STOP").exists() or Path(str(plan_path) + ".stop").exists():
            return "paused"
        return "budget_exhausted" if system.now() >= state["deadline_epoch"] else None

    def signal(reason):
        if not (output / "STOP").exists():
            _new(output / "STOP", {"owner": KIND, "plan_sha256": expected_plan_sha256, "reason": reason})

    def frozen_identity():
        try:
            live_identity.check()
            return True
        except Exception as exc:
            state["identity_guard_error"] = f"{type(exc).__name__}: {exc}"
            return False

    def complete_identity():
        live_identity.check()
        validate_plan(plan_path, expected_plan_sha256)
        live_identity.check()

    def stage_admission(name):
        """No child/GPU work is active here; all waiting keeps the same ledger."""
        def event(value):
            with (output / "supervisor-process-guard.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage_name": name, **value}, allow_nan=False) + "\n")
            if value.get("event") == "wait_start":
                write("waiting_process_identity", waiting_stage=name)

        session = process_guard.GuardSession(system.processes, owned_identities=list(own.values()),
            allowed_identities=plan["allowed_background_processes"], deadline_epoch=state["deadline_epoch"],
            policy=plan["process_wait_policy"], spent_seconds=state["total_process_wait_seconds"],
            stop_requested=lambda: stop_reason() == "paused", identity_unchanged=frozen_identity,
            clock=system.now, sleep=system.sleep, on_event=event)
        receipt = lambda: {"complete": True, "optimizer_updates": 0, "kind": "before_cuda",
            "path": str(output / "plan.snapshot.json"), "sha256": sha(output / "plan.snapshot.json")}
        decision = session.check(boundary="supervisor_before_" + name, optimizer_updates=0,
                                 save_boundary=receipt)
        state["total_process_wait_seconds"] = decision["total_waited_seconds"]
        state["last_process_guard"] = decision
        if not decision["admission_allowed"]:
            reason = decision["reason"]
            status = "budget_exhausted" if reason == "deadline_exhausted" else (
                "paused" if reason == "stop_requested" else "failed")
            write(status, process_guard_failure=reason)
            return False
        return True

    try:
        output.mkdir(exist_ok=False)
        _new(output / "plan.snapshot.json", plan)
        live_identity.add_file(output / "plan.snapshot.json", sha(output / "plan.snapshot.json"), always_hash=True)
        _status(output / "live-identity-sentinel.json", live_identity.record())
        write()
        snapshot = verify_snapshot(plan["model_snapshot"], plan["model_manifest_sha256"])
        if snapshot["repo_id"] != REPO_ID or snapshot["revision"] != REVISION:
            raise ValueError("Actual model snapshot differs from registered revision")
        cache_pin = None
        for stage, seed in SEQUENCE:
            name = f"{stage}{seed}"
            while True:
                live_identity.check()
                reason = stop_reason()
                if reason:
                    write(reason)
                    return state
                if not stage_admission(name):
                    return state
                if any(p.exists() for p in foreign_locks):
                    raise FileExistsError("Foreign or retained training/GPU lock")
                admission = system.resources()
                if not admission["blockers"]:
                    break
                write("waiting_resources", resources=admission)
                remaining = state["deadline_epoch"] - system.now()
                if remaining > 0:
                    system.sleep(min(10, remaining))
            reference = None
            if stage == "dpo":
                if cache_pin is None:
                    reference = {"mode": "create", "path": str(output / name / "reference-cache")}
                else:
                    _cache(cache_pin)
                    reference = {"mode": "reuse", **cache_pin}
            request = {k: plan[k] for k in ("source_sha256", "model_snapshot", "model_manifest_sha256",
                "tokenizer_dir", "audit_dir", "audit_manifest_sha256", "capacity_acceptance_path",
                "capacity_acceptance_sha256", "parent_plan_path", "parent_plan_sha256",
                "parent_acceptance_path", "parent_acceptance_sha256", "allowed_background_processes",
                "process_wait_policy")}
            request.update(schema_version=1, kind=STAGE_KIND, stage=stage, seed=seed,
                output=str(output / name), deadline_epoch=state["deadline_epoch"],
                alignment_plan_path=str(plan_path), alignment_plan_sha256=expected_plan_sha256,
                supervisor_processes=list(own.values()),
                process_wait_spent_seconds=state["total_process_wait_seconds"],
                parents={k: plan["parents"][k] for k in (("sft",) if stage == "dpo" else ("sft", "reward"))},
                workload=workload(stage, plan["ppo_rows"]), reference_cache=reference)
            path = output / f"{name}.request.json"
            _new(path, request)
            live_identity.add_file(path, sha(path), always_hash=True)
            _status(output / "live-identity-sentinel.json", live_identity.record())
            args = [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B", "-u",
                    "scripts/run_alignlab_alignment_stage.py", "--request", str(path),
                    "--request-sha256", sha(path), "--execute"]
            record = {"name": name, "stage": stage, "seed": seed, "request": request,
                      "request_sha256": sha(path), "command": args, "launch_epoch": system.now(),
                      "known_processes": [], "status": "dispatching"}
            state["stages"].append(record)
            write("dispatching")
            with (output / f"{name}.stdout.log").open("xb") as stdout, (output / f"{name}.stderr.log").open("xb") as stderr:
                complete_identity()
                reason = stop_reason()
                if reason:
                    write(reason)
                    return state
                # The child independently guards its before-CUDA boundary. An
                # unknown process appearing after stage_admission must use that
                # bounded wait, not force a fresh request/ledger or early STOP.
                child = system.spawn(args, stdout, stderr)
                child_finished = False
                record["wrapper_pid"] = child.pid
                known, monitor_error, stopped = {}, None, None
                snapshot_uncertain_since = None
                while True:
                    try:
                        live_identity.check()
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    stopped = stopped or stop_reason()
                    if stopped or monitor_error:
                        signal(stopped or "identity_failure")
                    try:
                        live = system.processes()
                    except Exception as exc:
                        # Unknown whole-system observations are handled by the
                        # live child's bounded safe-boundary wait. A supervisor
                        # query failure cannot itself prove child exit.
                        if snapshot_uncertain_since is None:
                            snapshot_uncertain_since = system.now()
                        observation_deadline = min(state["deadline_epoch"], snapshot_uncertain_since + 300)
                        record.update(pending_boundary_wait=True,
                            process_snapshot_error=f"{type(exc).__name__}: {exc}",
                            process_snapshot_uncertain_since=snapshot_uncertain_since,
                            process_snapshot_observation_deadline=observation_deadline,
                            exit_code=child.poll())
                        if system.now() >= observation_deadline:
                            failed_status = stop_reason() or "failed"
                            signal(stop_reason() or "process_snapshot_unavailable")
                            # Keep the lock and make the remaining uncertainty
                            # explicit; a wrapper exit alone is not child exit.
                            state["child_may_still_be_running"] = True
                            record["exit_identity_verified"] = False
                            write(failed_status,
                                  process_guard_failure="process_snapshot_unavailable_exit_unverified")
                            return state
                        write("stopping" if stopped or monitor_error else "pending_boundary_wait")
                        system.sleep(max(0, min(10, observation_deadline - system.now())))
                        continue
                    snapshot_uncertain_since = None
                    by_pid = {p["pid"]: p for p in live}
                    for _ in range(len(live) + 1):
                        added = False
                        for p in live:
                            if p["pid"] not in known and (p["pid"] == child.pid or p.get("parent_pid") in known):
                                if (not _command(p, "run_alignlab_alignment_stage.py", path)
                                        or _stamp(p["created_at_utc"]).timestamp() < record["launch_epoch"]):
                                    monitor_error = "Child command/creation/ancestry mismatch"
                                else:
                                    known[p["pid"]] = p
                                    added = True
                        if not added:
                            break
                    try:
                        _parents_gone(plan, live)
                        peer_guard = _peer_decision(live, own, known, plan["allowed_background_processes"])
                        record["process_guard"] = peer_guard
                        pending = not peer_guard["caller_admission_allowed"]
                        record["pending_boundary_wait"] = pending and _waitable_peer_decision(peer_guard)
                        if pending and not record["pending_boundary_wait"]:
                            raise RuntimeError("Known competing training or process identity changed")
                    except Exception as exc:
                        monitor_error = f"{type(exc).__name__}: {exc}"
                    if monitor_error:
                        signal("identity_failure")
                    active = [p for pid, p in known.items() if pid in by_pid and _same_identity(by_pid[pid], p)]
                    code = child.poll()
                    record.update(known_processes=list(known.values()), active_processes=active,
                                  monitoring_error=monitor_error, stop_reason=stopped, exit_code=code)
                    write("stopping" if stopped or monitor_error else (
                        "pending_boundary_wait" if record.get("pending_boundary_wait") else "running"))
                    if code is not None and not active:
                        unverified = [p for p in live if
                            p["pid"] not in known and (p["pid"] == child.pid or p.get("parent_pid") in known)
                            or p["pid"] in known
                            and process_guard.strict.ProcessIdentity.from_record(p).instance
                                == process_guard.strict.ProcessIdentity.from_record(known[p["pid"]]).instance
                            and not _same_identity(p, known[p["pid"]])]
                        if not known or unverified:
                            state["child_may_still_be_running"] = True
                            record["exit_identity_verified"] = False
                            record["unverified_descendants"] = unverified
                            raise RuntimeError("Child exit identity is unverified; wrapper exit is insufficient")
                        child_finished = True
                        if monitor_error:
                            raise RuntimeError(monitor_error)
                        if stopped:
                            write(stopped)
                            return state
                        if any(p.exists() for p in foreign_locks):
                            raise FileExistsError("Exited child retained a lock; do not advance")
                        complete_identity()
                        result = _verify_stage(output / name, request, known, code)
                        verified_wait = result["process_guard"]["total_waited_seconds"]
                        if (request["process_wait_spent_seconds"] != state["total_process_wait_seconds"]
                                or not state["total_process_wait_seconds"] <= verified_wait < 900):
                            raise ValueError("Child reset or exceeded the shared campaign waiting ledger")
                        state["total_process_wait_seconds"] = verified_wait
                        if stage == "dpo":
                            produced = {k: result["reference_cache"][k]
                                        for k in ("path", "manifest_sha256", "identity_sha256")}
                            if cache_pin is None:
                                if Path(produced["path"]).resolve() != output / name / "reference-cache":
                                    raise ValueError("Initial DPO cache escaped its owning stage")
                                cache_pin = produced
                                # Cache becomes immutable only after full
                                # completed-stage acceptance. Later branches
                                # reuse its exact files; no live rebaselining.
                                live_identity.add_tree(cache_pin["path"])
                            elif produced != cache_pin:
                                raise ValueError("Later seeds did not reuse the original frozen reference cache")
                        record.update(status="accepted", progress_path=str(output / name / "progress.json"),
                            progress_sha256=sha(output / name / "progress.json"), artifact=result["artifact"],
                            reference_cache=result.get("reference_cache"),
                            inherited_process_wait_seconds=request["process_wait_spent_seconds"],
                            total_process_wait_seconds=verified_wait, finished_at_epoch=system.now())
                        write("stage_accepted")
                        break
                    system.sleep(10)
        complete_identity()
        if not stage_admission("final_publication"):
            return state
        if any(p.exists() for p in foreign_locks):
            raise FileExistsError("A training/GPU lock remains after child exit")
        reason = stop_reason()
        if reason:
            write(reason)
            return state
        write("primary_alignment_passed", primary_alignment_verified=True, research_complete=False)
        return state
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        if output.exists():
            if child is not None and not child_finished:
                signal("supervisor_error")
                state["child_may_still_be_running"] = True
            write("failed")
        raise
    finally:
        if output.exists():
            state.update(finished_at_epoch=system.now(), elapsed_seconds=system.now() - started)
            _status(output / "campaign.json", state)
        if child_finished and lock.exists() and _parse(lock) == token:
            lock.unlink()
