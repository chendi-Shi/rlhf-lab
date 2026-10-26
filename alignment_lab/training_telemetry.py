"""Immutable observed-v2 training evidence, independent of numerical objectives.

Times are measured host wall intervals with CUDA synchronization supplied by the
observed engine. They are not CUDA kernel time. Rollouts commit before their first
optimizer update. Checkpoint sidecars are outside the frozen two-file checkpoint
bundle; their SHA is carried in its saved cursor and therefore in state.pt's SHA.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = "alignlab-training-telemetry-observed-v2"
REVISION = "observed-v2"
COST_SPANS = {"cold_cache_prepare_seconds": ("cold_cache_start", "cold_cache_prepared"),
              "cold_cache_scoring_seconds": ("cold_cache_prepared", "cold_cache_scoring_complete"),
              "cold_cache_commit_bind_seconds": ("cold_cache_scoring_complete", "cold_cache_bind_complete"),
              "cold_cache_total_seconds": ("cold_cache_start", "cold_cache_bind_complete"),
              "warm_cache_bind_seconds": ("warm_cache_bind_start", "warm_cache_bind_complete"),
              "engine_wall_seconds": ("warm_training_start", "training_engine_return")}


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: plain(v) for k, v in value.items()}
    raise ValueError("Telemetry requires finite plain values, detached tensors or bytes")


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Telemetry needs ordinary existing files")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        after = os.fstat(stream.fileno())
    sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if sig(before) != sig(after) or sig(after) != sig(path.stat()):
        raise ValueError("File changed during telemetry verification")
    return h.hexdigest()


def read(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("Duplicate telemetry JSON key")
            out[k] = v
        return out
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite telemetry JSON")))
    plain(result)
    return result


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "sha256": sha(path)}


def _pin(record):
    if set(record) != {"path", "sha256"} or sha(record["path"]) != record["sha256"]:
        raise ValueError("Telemetry SHA pin mismatch")
    return Path(record["path"])


def _owned_pin(root, record, files):
    path = _pin(record)
    if not path.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Telemetry pin escapes its evidence directory")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Symlink ancestors forbidden in telemetry evidence")
        if parent == root:
            break
    name = path.relative_to(root).as_posix()
    if name not in files or files[name]["sha256"] != record["sha256"]:
        raise ValueError("Telemetry pin is outside its tracked inventory")
    return path


def policy_anchor(backend, role, *, update, identity_sha256):
    """Hash actual trainable policy adapter tensors, never the post-step proxy."""
    from .stage_engine import parameter_mapping
    from .gpu_reference import _tensor_hash
    if any(p.grad is not None for p in backend._all_parameters()):
        raise RuntimeError("Generation identity requires a clean pre-sampling boundary")
    mapping = parameter_mapping(backend, role)
    tensors = {name: _tensor_hash(p, 1024 * 1024) for name, p in mapping.items()}
    body = {"optimizer_updates": update, "identity_sha256": identity_sha256,
            "role": role, "tensors": tensors}
    return {**body, "adapter_sha256": digest(tensors)}


def rollout_summary(record):
    responses, active = record["responses"], record["active"]
    n = len(responses)
    if n < 1 or len(record["record_ids"]) != n or len(set(record["record_ids"])) != n:
        raise ValueError("Missing or duplicate rollout records")
    if len(active["reward_scores"]) != n or len(active["reference_logprobs"]) != n:
        raise ValueError("Rollout reward/reference denominator differs")
    lengths, eos, log_ratios, rewards = [], 0, [], []
    for index, response in enumerate(responses):
        ids, lp, values = response["action_ids"], response["old_logprobs"], response["old_values"]
        if (not response["prompt_ids"] or not ids or len(ids) != len(lp) or len(ids) != len(values)
                or any(type(t) is not int or not 0 <= t < record["vocab_size"] for t in response["prompt_ids"] + ids)
                or response["termination"] not in ("eos", "max_new_tokens")
                or len(ids) > record["max_new_tokens"]):
            raise ValueError("Invalid exact rollout IDs/statistics")
        is_eos = ids[-1] == record["eos_id"]
        if (record["eos_id"] in ids[:-1] or is_eos != (response["termination"] == "eos")
                or not is_eos and len(ids) != record["max_new_tokens"]):
            raise ValueError("Rollout EOS/horizon mismatch")
        refs = active["reference_logprobs"][index]
        if (len(refs) < len(ids) or any(type(v) not in (int, float) for v in lp + values + refs)
                or any(v > 0 for v in lp + refs[:len(ids)])):
            raise ValueError("Invalid rollout log probabilities")
        lengths.append(len(ids))
        eos += is_eos
        log_ratios.extend(a - b for a, b in zip(lp, refs[:len(ids)]))
        rewards.append(active["reward_scores"][index])
    plain(record)
    if any(type(v) not in (int, float) or isinstance(v, bool) for v in rewards + log_ratios):
        raise ValueError("Rewards/log probabilities must be finite numeric scalars")
    return {"responses": n, "action_tokens": sum(lengths), "lengths": lengths,
            "mean_length": sum(lengths) / n, "eos_count": eos, "eos_rate": eos / n,
            "length_limit_count": n - eos, "reward_mean": sum(rewards) / n,
            "reward_min": min(rewards), "reward_max": max(rewards),
            "sampled_log_ratio_sum": sum(log_ratios),
            "sampled_log_ratio_token_mean": sum(log_ratios) / len(log_ratios),
            "interpretation": "training RM rewards and sampled log ratios; not independent quality or exact KL"}


def verify_rollout_origin(record, anchor):
    origin = record["active"]["observation_origin"]
    expected = hashlib.sha256(json.dumps({k: v for k, v in origin.items() if k != "origin_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if origin.get("origin_sha256") != expected:
        raise ValueError("Original collection identity changed")
    for phase in ("before", "after"):
        raw = bytes.fromhex(origin[f"rng_{phase}_hex"])
        if (not raw or hashlib.sha256(raw).hexdigest() != origin[f"rng_{phase}_sha256"]
                or record[f"rng_{phase}"] != {"encoding": "hex", "value": raw.hex()}):
            raise ValueError("Original sampling RNG changed")
    if (any(origin.get(k) != v for k, v in anchor["origin"].items())
            or origin["model_versions"] != record["model_versions"]
            or origin["rollout_index"] != record["rollout_index"]
            or origin["record_ids"] != record["record_ids"]
            or record["active"]["record_ids"] != record["record_ids"]
            or record["active"]["responses"] != record["responses"]
            or origin["cursor_before"]["active_rollout"] is not False
            or origin["cursor_before"]["optimizer_updates"] != record["generated_at_optimizer_updates"]
            or record["cursor_before_updates"]["optimizer_updates"] != record["generated_at_optimizer_updates"]
            or not origin["generation_model_versions"] or not origin["model_versions"]):
        raise ValueError("Original policy generation anchor differs")
    widths = [len(r["action_ids"]) for r in record["responses"]]
    for key in ("advantages", "returns", "reference_logprobs"):
        matrix = record["active"][key]
        if len(matrix) != len(widths) or any(len(row) != max(widths) for row in matrix):
            raise ValueError("Frozen rollout statistic shape differs")


def _same_fixed_active(active, recorded):
    return all(plain(active.get(k)) == recorded[k] for k in (
        "responses", "record_ids", "advantages", "returns", "reference_logprobs",
        "reward_scores", "observation_origin"))


def _duration(record):
    start, end, seconds = (record[k] for k in ("started_at_epoch", "finished_at_epoch", "seconds"))
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end, seconds))
            or end < start or seconds != end - start):
        raise ValueError("Invalid measured interval")
    return seconds


def cost_summary(points, bindings, *, started, finished, warm_started):
    """Recompute costs from original measured boundaries, without tensor imports."""
    names = [p["event"] for p in points]
    epochs = [p["time_epoch"] for p in points]
    if any(a > b for a, b in zip(epochs, epochs[1:])):
        raise ValueError("Cost event wall clock moved backwards")
    result = {"observer_seconds": 0.0, "checkpoint_seconds": 0.0,
              "observer_cuda_sync_seconds": 0.0}
    for label, (begin, end) in COST_SPANS.items():
        if begin in names or end in names:
            if names.count(begin) != 1 or names.count(end) != 1:
                raise ValueError("Measured cost interval is incomplete or duplicated")
            duration = points[names.index(end)]["time_epoch"] - points[names.index(begin)]["time_epoch"]
            if duration < 0:
                raise ValueError("Measured cost interval went backwards")
            result[label] = duration
    active, ended = {}, {}
    phase_ends = {"sampling_end": "sampling_start", "frozen_scoring_end": "frozen_scoring_start",
                  "optimizer_end": "optimizer_start"}
    for point in points:
        name = point["event"]
        if name in phase_ends.values():
            if name in active or name in ended:
                raise ValueError("Duplicate observed phase start")
            phase_id = point["target_update"] if name == "optimizer_start" else point["rollout_index"]
            active[name] = {"phase_id": phase_id, "boundary": point["boundary_epoch"], "start": None}
        elif name == "phase_interval_start":
            phase = point["phase"]
            if (phase not in active or active[phase]["start"] is not None
                    or point["phase_id"] != active[phase]["phase_id"]
                    or not active[phase]["boundary"] <= point["started_at_epoch"] <= point["time_epoch"]):
                raise ValueError("Unpaired measured phase start")
            active[phase]["start"] = point["started_at_epoch"]
        elif name in phase_ends:
            phase = phase_ends[name]
            phase_id = point["target_update"] if name == "optimizer_end" else point["rollout_index"]
            if (phase not in active or active[phase]["start"] is None
                    or phase_id != active[phase]["phase_id"]):
                raise ValueError("Unpaired observed phase end")
            ended[phase] = {**active.pop(phase), "end": point["boundary_epoch"]}
        if name == "measured_phase":
            phase = point["phase"]
            if (phase not in ended or point["phase_id"] != ended[phase]["phase_id"]
                    or point["started_at_epoch"] != ended[phase]["start"]
                    or point["finished_at_epoch"] != ended[phase]["end"]):
                raise ValueError("Measured phase is missing, duplicated or detached from its actual boundary")
            ended.pop(phase)
            axis = {"sampling_start": "sampling_seconds", "frozen_scoring_start": "frozen_scoring_seconds",
                    "optimizer_start": "optimizer_active_seconds"}[point["phase"]]
            result[axis] = result.get(axis, 0.0) + _duration(point)
        elif name in ("observer_interval", "cuda_synchronization"):
            axis = "observer_seconds" if name == "observer_interval" else "observer_cuda_sync_seconds"
            result[axis] += _duration(point)
    if active or ended:
        raise ValueError("Measured phase interval was omitted")
    if (names.count("telemetry_complete") != 1 or names[-1] != "telemetry_complete"
            or not all(started <= epoch <= finished for epoch in epochs)):
        raise ValueError("Incomplete or out-of-budget-boundary telemetry event timeline")
    for binding in bindings:
        start, end = binding["save_started_at_epoch"], binding["save_finished_at_epoch"]
        if end < start:
            raise ValueError("Checkpoint wall clock moved backwards")
        result["checkpoint_seconds"] += end - start
    if not started <= warm_started <= finished:
        raise ValueError("Invalid stage/warm-training measured boundaries")
    result["stage_wall_seconds_to_telemetry_commit"] = finished - started
    result["warm_training_wall_seconds_to_telemetry_commit"] = finished - warm_started
    return result


class TrainingTelemetry:
    """One fresh stage's evidence writer; no implicit resume or partial reuse."""
    def __init__(self, output, *, stage, started_at_epoch, clock=time.time):
        self.path, self.stage, self.started, self.clock = Path(output), stage, started_at_epoch, clock
        self.path.mkdir(parents=True, exist_ok=False)
        self.identity_sha256 = None
        self.rollouts, self.anchors, self.checkpoints = [], {}, []
        self.points, self.cost = [], {"observer_seconds": 0.0, "checkpoint_seconds": 0.0}
        self.warm_started = None
        self._pending = None
        self._closed = False

    def set_identity(self, identity):
        if self.identity_sha256 is not None:
            raise ValueError("Telemetry identity already bound")
        self.identity_sha256 = digest(identity)

    def point(self, name, **fields):
        if self._closed:
            raise RuntimeError("Telemetry already committed")
        record = plain({"event": name, "time_epoch": self.clock(), **fields})
        with (self.path / "cost-events.jsonl").open("ab") as stream:
            stream.write(canonical(record) + b"\n")
            stream.flush()
        self.points.append(record)
        return record

    def warm_start(self):
        if self.warm_started is not None:
            raise ValueError("Warm training clock cannot reset")
        self.warm_started = self.clock()
        self.point("warm_training_start")

    def warm_elapsed(self):
        return 0.0 if self.warm_started is None else self.clock() - self.warm_started

    def begin_rollout(self, backend, role, cursor, *, origin=None):
        if self._pending is not None or not self.identity_sha256 or cursor["active_rollout"]:
            raise ValueError("Fresh pre-generation boundary required")
        index = cursor["completed_rollouts"]
        if index != len(self.rollouts):
            raise ValueError("Rollout generation order changed")
        anchor = policy_anchor(backend, role, update=cursor["optimizer_updates"],
                               identity_sha256=self.identity_sha256)
        pin = write_new(self.path / "anchors" / f"rollout-{index:06d}.json",
                        {"protocol": PROTOCOL, "rollout_index": index, "origin": origin, **anchor})
        self.anchors[index] = pin
        self._pending = index
        return pin

    def commit_rollout(self, *, cursor, active, rollout):
        if self._pending is None:
            raise ValueError("Missing original pre-generation policy anchor")
        index = self._pending
        anchor_pin = self.anchors[index]
        anchor = read(_pin(anchor_pin))
        if cursor["optimizer_updates"] != anchor["optimizer_updates"]:
            raise ValueError("Rollout was not persisted before its first optimizer update")
        body = {"protocol": PROTOCOL, "identity_sha256": self.identity_sha256,
            "rollout_index": index, "generation_anchor": anchor_pin,
            "generated_at_optimizer_updates": anchor["optimizer_updates"],
            "cursor_before_updates": cursor, "record_ids": active["record_ids"],
            "responses": [asdict(r) if is_dataclass(r) else vars(r) for r in rollout.responses], "active": active,
            "rng_before": rollout.rng_before, "rng_after": rollout.rng_after,
            "model_versions": rollout.model_versions, "policy_role": rollout.policy_role,
            "eos_id": rollout.eos_id, "pad_id": rollout.pad_id,
            "vocab_size": rollout.vocab_size, "max_new_tokens": rollout.max_new_tokens,
            "committed_at_epoch": self.clock()}
        body = plain(body)
        if not body["rng_before"]["value"] or not body["rng_after"]["value"] or not body["model_versions"]:
            raise ValueError("Original RNG and sampled model versions must be retained")
        body["summary"] = rollout_summary(body)
        verify_rollout_origin(body, anchor)
        pin = write_new(self.path / "rollouts" / f"rollout-{index:06d}.json", body)
        self.rollouts.append(pin)
        self._pending = None
        return pin

    def observe(self, event, *, backend, role):
        """Consume the observed engine's detached immutable event contract."""
        from .stage_engine_observed import event_to_plain
        e = event_to_plain(event)
        started = self.clock()
        name, cursor = e["event"], e["cursor"]
        if not hasattr(self, "_phase_starts"):
            self._phase_starts = {}
        if name == "rollout_start":
            origin = e["observation_origin"]
            self.begin_rollout(backend, role, cursor, origin=origin)
        elif name == "rollout_collected":
            # This event precedes every optimizer step; source RNG/version fields
            # survive checkpointing even though reconstructed RolloutBatch does not.
            from types import SimpleNamespace
            responses = tuple(SimpleNamespace(**r) for r in e["responses"])
            origin = e.get("observation_origin", e.get("generation_anchor", e))
            metadata = e["rollout_metadata"]
            active = {k: e[k] for k in ("record_ids", "advantages", "returns",
                                       "reference_logprobs", "reward_scores")}
            active.update(responses=e["responses"], epoch=0, offset=0, observation_origin=origin)
            rollout = SimpleNamespace(responses=responses, policy_role=role,
                eos_id=metadata["eos_id"], pad_id=metadata["pad_id"], vocab_size=metadata["vocab_size"],
                max_new_tokens=metadata["max_new_tokens"],
                rng_before=bytes.fromhex(origin["rng_before_hex"]),
                rng_after=bytes.fromhex(origin["rng_after_hex"]),
                model_versions=origin["model_versions"])
            self.commit_rollout(cursor=cursor, active=active, rollout=rollout)
        fields = {k: v for k, v in e.items() if k not in ("event", "responses", "reference_logprobs",
                   "reward_scores", "advantages", "returns", "observation_origin", "generation_anchor",
                   "old_logprobs", "old_values", "action_mask")}
        fields["cursor"] = {k: v for k, v in cursor.items() if k != "order"}
        fields["cursor_sha256"] = digest(cursor)
        fields["order_sha256"] = digest(cursor["order"]) if "order" in cursor else None
        self.point(name, boundary_epoch=started, **fields)
        pairs = {"sampling_end": "sampling_start", "frozen_scoring_end": "frozen_scoring_start",
                 "optimizer_end": "optimizer_start"}
        if name in pairs:
            key = pairs[name]
            if key not in self._phase_starts:
                raise ValueError("Observed phase end lacks its original start")
            phase_started, phase_id = self._phase_starts.pop(key)
            duration = started - phase_started
            if duration < 0:
                raise ValueError("Wall clock moved backwards during training observation")
            axis = {"sampling_end": "sampling_seconds", "frozen_scoring_end": "frozen_scoring_seconds",
                    "optimizer_end": "optimizer_active_seconds"}[name]
            self.cost[axis] = self.cost.get(axis, 0.0) + duration
            self.point("measured_phase", phase=key, seconds=duration, started_at_epoch=phase_started,
                       finished_at_epoch=started, phase_id=phase_id,
                       optimizer_updates=cursor["optimizer_updates"])
        finished = self.clock()
        self.cost["observer_seconds"] += finished - started
        self.point("observer_interval", observed_event=name, started_at_epoch=started,
                   finished_at_epoch=finished, seconds=finished - started)
        if name in ("sampling_start", "frozen_scoring_start", "optimizer_start"):
            phase_id = e["target_update"] if name == "optimizer_start" else e["rollout_index"]
            phase_start = self.clock()
            self.point("phase_interval_start", phase=name, phase_id=phase_id, started_at_epoch=phase_start)
            self._phase_starts[name] = (phase_start, phase_id)

    def safe_engine_return(self, cursor, extra, reason):
        """Only a known clean engine return can close an undispatched sample."""
        if self._pending is not None:
            if (reason not in ("paused", "budget_exhausted") or cursor["active_rollout"] or extra is not None
                    or self._pending != cursor["completed_rollouts"]
                    or any(p["event"] == "sampling_end" and p.get("rollout_index") == self._pending
                           for p in self.points)):
                raise ValueError("A partially observed executed rollout is not a safe unsampled return")
            self.point("rollout_aborted_before_sampling", rollout_index=self._pending,
                       anchor=self.anchors[self._pending], cursor=cursor, stop_reason=reason)
            self._pending = None
        self.point("training_engine_return", cursor=cursor, stop_reason=reason)

    def checkpoint_sidecar(self, name, cursor, extra):
        if self._pending is not None:
            raise ValueError("Cannot checkpoint a partially observed rollout")
        active_pin = None
        if cursor["active_rollout"]:
            index = cursor["completed_rollouts"]
            if extra is None or index >= len(self.rollouts):
                raise ValueError("Active recovery requires its original generation anchor and rollout")
            active_pin = self.rollouts[index]
            stored = read(_pin(active_pin))
            if not _same_fixed_active(extra, stored["active"]):
                raise ValueError("Active rollout no longer matches its immutable generation")
        return write_new(self.path / "checkpoints" / f"{name}.json", {
            "protocol": PROTOCOL, "identity_sha256": self.identity_sha256,
            "cursor_sha256": digest(cursor), "cursor": cursor, "rollouts": self.rollouts,
            "active_rollout": active_pin, "generation_anchors": [self.anchors[i] for i in sorted(self.anchors)],
            "created_at_epoch": self.clock(), "warm_training_elapsed_seconds": self.warm_elapsed()})

    def bind_checkpoint(self, receipt, sidecar, *, started_at_epoch, finished_at_epoch):
        body = {"checkpoint_path": receipt["checkpoint_path"], "manifest_sha256": receipt["manifest_sha256"],
                "sidecar": sidecar, "save_started_at_epoch": started_at_epoch,
                "save_finished_at_epoch": finished_at_epoch}
        pin = write_new(self.path / "bindings" / (Path(receipt["checkpoint_path"]).name + ".json"), body)
        self.checkpoints.append(pin)
        self.cost["checkpoint_seconds"] += finished_at_epoch - started_at_epoch
        return pin

    def finish(self, *, cursor, completion, checkpoints=None):
        if self._pending is not None or self._closed or self.identity_sha256 is None:
            raise ValueError("Cannot commit incomplete telemetry")
        if len(self.rollouts) != cursor["completed_rollouts"] or cursor["active_rollout"]:
            raise ValueError("Completed telemetry needs every full rollout")
        if getattr(self, "_phase_starts", {}):
            raise ValueError("Observed training phase never completed")
        self.point("telemetry_complete", cursor=cursor)
        finished = self.clock()
        self.cost = cost_summary(self.points, [read(_pin(p)) for p in self.checkpoints],
            started=self.started, finished=finished, warm_started=self.warm_started)
        files = {str(p.relative_to(self.path)).replace("\\", "/"): {
            "sha256": sha(p), "size_bytes": p.stat().st_size}
            for p in sorted(self.path.rglob("*")) if p.is_file()}
        manifest = {"schema_version": 1, "kind": PROTOCOL, "status": "complete",
            "execution_revision": REVISION, "identity_sha256": self.identity_sha256, "stage": self.stage,
            "started_at_epoch": self.started, "warm_training_started_at_epoch": self.warm_started,
            "finished_at_epoch": finished, "final_cursor": cursor, "completion": completion,
            "rollouts": self.rollouts, "checkpoint_bindings": self.checkpoints, "files": files,
            "checkpoint_receipts": [] if checkpoints is None else plain(checkpoints),
            "costs": self.cost, "clock_semantics": {
                "stage_wall": "all time since stage admission, including model/cache/checkpoints/observation",
                "warm_training_wall": "since pre-engine boundary, includes sampling/scoring/updates/checkpoint and observer",
                "phase_intervals": "synchronized host wall intervals; not CUDA kernel time",
                "optimizer_active": "update body only; excludes before/after observer and checkpoint callbacks",
                "cache_scoring": "per-sequence scoring interval includes guards and Python overhead",
                "observer_checkpoint": "reported separately; no subtraction from absolute deadline",
                "overlap": "diagnostic views overlap and MUST NOT be summed: end CUDA sync lies inside phase wall; cold total contains subphases",
                "observer_measurement": "callback body excludes writing its own timing record; engine event-copy overhead remains in wall intervals"}}
        pin = write_new(self.path / "manifest.json", manifest)
        self._closed = True
        return pin


def verify_checkpoint_telemetry(checkpoint_path, *, expected_manifest_sha256, expected_sidecar_sha256):
    """CPU-only payload inspection; caller must separately restore via checkpointing."""
    import torch
    directory = Path(checkpoint_path)
    if sha(directory / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Checkpoint manifest pin mismatch")
    manifest = read(directory / "manifest.json")
    if sha(directory / "state.pt") != manifest["artifacts"]["state.pt"]["sha256"]:
        raise ValueError("Checkpoint payload pin mismatch")
    payload = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    cursor = dict(payload["cursor"])
    pin = cursor.pop("telemetry", None)
    if pin is None or pin.get("sha256") != expected_sidecar_sha256:
        raise ValueError("Checkpoint lacks its original telemetry binding")
    sidecar = read(_pin(pin))
    if not Path(pin["path"]).resolve().is_relative_to((directory.parent / "telemetry").resolve()):
        raise ValueError("Checkpoint telemetry sidecar escapes the stage evidence")
    if sidecar["cursor"] != cursor or sidecar["identity_sha256"] != digest(manifest["identity"]):
        raise ValueError("Checkpoint telemetry cursor/identity mismatch")
    for record in sidecar["rollouts"] + sidecar["generation_anchors"]:
        _pin(record)
    if cursor["active_rollout"]:
        if sidecar["active_rollout"] is None or payload["extra_state"] is None:
            raise ValueError("Active recovery lacks original generation anchor")
        rollout = read(_pin(sidecar["active_rollout"]))
        _pin(rollout["generation_anchor"])
        if not _same_fixed_active(payload["extra_state"], rollout["active"]):
            raise ValueError("Checkpoint active trajectory differs")
    return {"cursor": cursor, "extra_state": payload["extra_state"], "sidecar": sidecar}


def verify_training_telemetry(output, *, expected_manifest_sha256, expected_identity_sha256,
                              expected_completion=None, checkpoints=None):
    """Independent full SHA/denominator/summary verification; stdlib only."""
    path = Path(output)
    if path.is_symlink():
        raise ValueError("Telemetry root cannot be a symlink")
    if sha(path / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Telemetry manifest SHA mismatch")
    m = read(path / "manifest.json")
    if (m["kind"] != PROTOCOL or m["status"] != "complete" or m["execution_revision"] != REVISION
            or m["identity_sha256"] != expected_identity_sha256):
        raise ValueError("Telemetry identity/status mismatch")
    if expected_completion is not None and m["completion"] != expected_completion:
        raise ValueError("Telemetry completion differs")
    actual = set()
    for p in path.rglob("*"):
        if p.is_symlink():
            raise ValueError("Telemetry symlinks forbidden")
        if p.is_file() and p != path / "manifest.json":
            actual.add(p.relative_to(path).as_posix())
    if actual != set(m["files"]):
        raise ValueError("Untracked or missing telemetry files")
    for name, record in m["files"].items():
        p = path / name
        if not p.resolve().is_relative_to(path.resolve()) or sha(p) != record["sha256"] or p.stat().st_size != record["size_bytes"]:
            raise ValueError("Telemetry artifact changed")
    tokens = responses = 0
    record_ids = set()
    for i, pin in enumerate(m["rollouts"]):
        r = read(_owned_pin(path, pin, m["files"]))
        a = read(_owned_pin(path, r["generation_anchor"], m["files"]))
        if (r["rollout_index"] != i or a["rollout_index"] != i or r["identity_sha256"] != expected_identity_sha256
                or a["identity_sha256"] != expected_identity_sha256 or a["adapter_sha256"] != digest(a["tensors"])
                or a["optimizer_updates"] != r["generated_at_optimizer_updates"]
                or r["summary"] != rollout_summary(r)):
            raise ValueError("Rollout generation/summary mismatch")
        verify_rollout_origin(r, a)
        if record_ids.intersection(r["record_ids"]):
            raise ValueError("Registered unique prompts were repeated across rollouts")
        record_ids.update(r["record_ids"])
        tokens += r["summary"]["action_tokens"]
        responses += r["summary"]["responses"]
    if len(m["rollouts"]) != m["final_cursor"]["completed_rollouts"]:
        raise ValueError("Telemetry rollout denominator mismatch")
    if m["stage"] == "ppo" and (tokens != m["completion"]["response_tokens"]
                                or responses != m["completion"]["rows_processed"]):
        raise ValueError("Telemetry response/token denominator mismatch")
    bindings = [read(_owned_pin(path, p, m["files"])) for p in m["checkpoint_bindings"]]
    for b in bindings:
        sidecar = read(_owned_pin(path, b["sidecar"], m["files"]))
        if (sha(Path(b["checkpoint_path"]) / "manifest.json") != b["manifest_sha256"]
                or sidecar["identity_sha256"] != expected_identity_sha256
                or b["save_finished_at_epoch"] < b["save_started_at_epoch"]):
            raise ValueError("Checkpoint binding mismatch")
    if checkpoints is not None:
        if m["checkpoint_receipts"] != plain(checkpoints):
            raise ValueError("Checkpoint time/label evidence differs")
        if {(b["checkpoint_path"], b["manifest_sha256"], b["sidecar"]["sha256"]) for b in bindings} != {
                (r["checkpoint_path"], r["manifest_sha256"], r["telemetry_sidecar"]["sha256"]) for r in checkpoints}:
            raise ValueError("Telemetry omitted or substituted a committed checkpoint")
    points = [json.loads(line) for line in (path / "cost-events.jsonl").read_text(encoding="utf-8").splitlines()]
    plain(points)
    if m["costs"] != cost_summary(points, bindings, started=m["started_at_epoch"],
            finished=m["finished_at_epoch"], warm_started=m["warm_training_started_at_epoch"]):
        raise ValueError("Telemetry cost summary differs from measured boundaries")
    if "optimizer_updates" in m["final_cursor"]:
        updates = m["final_cursor"]["optimizer_updates"]
        if ([p["target_update"] for p in points if p["event"] == "optimizer_start"] != list(range(1, updates + 1))
                or [p["update"] for p in points if p["event"] == "optimizer_end"] != list(range(1, updates + 1))):
            raise ValueError("Optimizer cost/objective events omit or duplicate completed updates")
    if m["stage"] == "ppo":
        for event in ("rollout_start", "rollout_collected", "rollout_finished",
                      "sampling_start", "sampling_end", "frozen_scoring_start", "frozen_scoring_end"):
            if [p["rollout_index"] for p in points if p["event"] == event] != list(range(len(m["rollouts"]))):
                raise ValueError("Rollout lifecycle events omit or duplicate actual trajectories")
    return m
