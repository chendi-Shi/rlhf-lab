"""Read-only, case-bound supplemental acceptance for the completed v2 PPO43.

The frozen telemetry serializer omitted the scalar optimizer ``responses``.
Recover that scalar from authenticated sampled trajectories and deterministic
optimizer selections, then execute every original R3 and fresh-wrapper check.
No historical file or module-global function is replaced.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/alignlab-v2-throughput-resume609-20261023T080559Z/campaign.json"
REQUEST_SHA = "90003a0cd3466e880d1cb95e674e80b2853a4cba174e82df6ee8c6c44e100d5c"
PLAN_SHA = "adea484fc1b052d5c19eda58e8b5aba8dd18e4b13ee988e53034181cbea89086"
DEADLINE = 1792894045.7777236
KIND = "alignlab-ppo43-supplemental-acceptance-v3"
SOURCES = ("alignment_lab/ppo43_supplemental_v3.py",
           "alignment_lab/alignment_supervisor_r3.py", "alignment_lab/throughput_fresh_v1.py",
           "alignment_lab/training_telemetry.py", "alignment_lab/stage_engine_observed.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _check_pin(pin):
    path = Path(pin["path"])
    if (set(pin) != {"path", "sha256", "size_bytes"} or not path.is_absolute()
            or path.is_symlink() or any(p.is_symlink() for p in path.parents)
            or sha(path) != pin["sha256"] or path.stat().st_size != pin["size_bytes"]):
        raise ValueError("Supplemental frozen input changed")
    return path.resolve()


def build_binding(campaign_path=CAMPAIGN):
    """Capture the specific completed case; caller persists this before acceptance."""
    campaign_path = Path(campaign_path).resolve()
    if campaign_path != CAMPAIGN.resolve():
        raise ValueError("Supplemental acceptance is limited to the original v2 campaign")
    output = campaign_path.parent / "ppo43"
    paths = {"campaign": campaign_path, "wrapper": output.parent / "ppo43.request.json"}
    for name in ("progress.json", "request.json", "wrapper.request.json", "identity.json",
                 "selection.json", "metrics.jsonl", "process-guard.jsonl", "telemetry/manifest.json",
                 "telemetry/cost-events.jsonl", "final-artifact/manifest.json"):
        paths["stage/" + name] = output / name
    paths.update({"source/" + name: ROOT / name for name in SOURCES})
    return {"kind": KIND, "output": str(output.resolve()),
            "files": {name: _pin(path) for name, path in paths.items()}}


def _verify_binding(binding):
    if set(binding) != {"kind", "output", "files"} or binding["kind"] != KIND:
        raise ValueError("Exact supplemental binding required")
    output = CAMPAIGN.parent / "ppo43"
    if Path(binding["output"]).resolve() != output.resolve():
        raise ValueError("Supplemental output changed")
    expected = {"campaign": CAMPAIGN, "wrapper": output.parent / "ppo43.request.json"}
    for name in ("progress.json", "request.json", "wrapper.request.json", "identity.json",
                 "selection.json", "metrics.jsonl", "process-guard.jsonl", "telemetry/manifest.json",
                 "telemetry/cost-events.jsonl", "final-artifact/manifest.json"):
        expected["stage/" + name] = output / name
    expected.update({"source/" + name: ROOT / name for name in SOURCES})
    if set(binding["files"]) != set(expected):
        raise ValueError("Supplemental binding inventory differs")
    for name, path in expected.items():
        if _check_pin(binding["files"][name]) != path.resolve():
            raise ValueError("Supplemental input path changed")
    if binding["files"]["wrapper"]["sha256"] != REQUEST_SHA:
        raise ValueError("This exception is bound to the original PPO43 request only")
    return output.resolve()


def reconstruct_objectives(events, rollouts, records, work, final_cursor, objectives):
    """Reconstruct response counts without trusting returned counts or defaults."""
    for key in ("seed", "rollout_batch", "optimizer_batch", "epochs", "max_responses", "max_updates"):
        if type(work.get(key)) is not int or (key != "seed" and work[key] <= 0):
            raise ValueError("Invalid integer PPO workload")
    rows, batch, epochs = len(records), work["optimizer_batch"], work["epochs"]
    order = list(range(rows))
    random.Random(work["seed"]).shuffle(order)
    if rows != work["max_responses"] or len({r["record_id"] for r in records}) != rows:
        raise ValueError("Selected record denominator/identity differs")
    starts_ends = [e for e in events if e.get("event") in ("optimizer_start", "optimizer_end")]
    if len(starts_ends) != 2 * work["max_updates"] or len(objectives) != work["max_updates"]:
        raise ValueError("Optimizer event/objective count differs")
    if not rollouts or len(rollouts) != math.ceil(rows / work["rollout_batch"]):
        raise ValueError("Rollout coverage differs")
    baseline = {k: v for k, v in rollouts[0]["cursor_before_updates"].items()
                if k not in ("offset", "optimizer_updates", "response_tokens", "active_rollout", "completed_rollouts")}
    if baseline.get("order") != order:
        raise ValueError("Seeded data order differs")
    completed_updates = total_responses = total_tokens = 0
    derived = []
    for index, rollout in enumerate(rollouts):
        responses = rollout["responses"]
        count = min(work["rollout_batch"], rows - total_responses)
        expected_ids = [records[i]["record_id"] for i in order[total_responses:total_responses + count]]
        if (rollout["rollout_index"] != index or len(responses) != count
                or rollout["record_ids"] != expected_ids
                or rollout["generated_at_optimizer_updates"] != completed_updates):
            raise ValueError("Rollout record selection or generation update differs")
        lengths = [len(r["action_ids"]) for r in responses]
        if any(not 1 <= n <= work["max_new_tokens"] for n in lengths):
            raise ValueError("Sampled response length outside registered bounds")
        total_responses += count
        total_tokens += sum(lengths)
        cursor = {**baseline, "offset": total_responses, "optimizer_updates": completed_updates,
                  "response_tokens": total_tokens, "active_rollout": True, "completed_rollouts": index}
        if rollout["cursor_before_updates"] != cursor:
            raise ValueError("Collected rollout cursor differs from independent totals")
        for epoch in range(epochs):
            selected_order = list(range(count))
            random.Random(work["seed"] + index * epochs + epoch).shuffle(selected_order)
            for offset in range(0, count, batch):
                indices = selected_order[offset:offset + batch]
                ids = [expected_ids[i] for i in indices]
                tokens = sum(lengths[i] for i in indices)
                target = completed_updates + 1
                start, end = starts_ends[2 * completed_updates:2 * completed_updates + 2]
                if start.get("event") != "optimizer_start" or end.get("event") != "optimizer_end":
                    raise ValueError("Optimizer starts/ends are not paired")
                info = {"rollout_index": index, "record_ids": ids, "response_indices": indices,
                        "epoch": epoch, "rollout_offset": offset, "target_update": target}
                for event in (start, end):
                    if any(event.get(k) != value for k, value in info.items()):
                        raise ValueError("Optimizer indices/record IDs/epoch/offset differ")
                    if any(type(i) is not int for i in event["response_indices"]):
                        raise ValueError("Response indices must be exact integers")
                if type(start.get("denominator")) is not int or start["denominator"] != tokens:
                    raise ValueError("Optimizer token denominator differs from sampled tokens")
                if "responses" in end:
                    raise ValueError("Supplemental exception is only for the historical omitted scalar")
                for event, after in ((start, False), (end, True)):
                    expected_cursor = dict(cursor)
                    expected_cursor["optimizer_updates"] = target if after else completed_updates
                    if after and epoch == epochs - 1 and offset + len(indices) == count:
                        expected_cursor.update(active_rollout=False, completed_rollouts=index + 1)
                    if (event.get("cursor") != {k: v for k, v in expected_cursor.items() if k != "order"}
                            or event.get("cursor_sha256") != digest(expected_cursor)
                            or event.get("order_sha256") != digest(order)):
                        raise ValueError("Optimizer cursor or cursor hash differs")
                objective = objectives[completed_updates]
                for key in ("objective", "gradient_norm"):
                    if (type(end.get(key)) not in (int, float) or not math.isfinite(end[key])
                            or type(objective.get(key)) not in (int, float) or not math.isfinite(objective[key])):
                        raise ValueError("Finite numerical optimizer observations required")
                actual = {"update": target, "responses": len(indices), "action_tokens": tokens,
                          "objective": end["objective"], "gradient_norm": end["gradient_norm"]}
                if (type(end.get("update")) is not int or end["update"] != target
                        or type(end.get("action_tokens")) is not int or end["action_tokens"] != tokens
                        or {k: objective.get(k) for k in actual} != actual
                        or type(objective.get("responses")) is not int):
                    raise ValueError("Live objectives differ from independently reconstructed work")
                derived.append({**info, **actual})
                completed_updates = target
                cursor["optimizer_updates"] = target
    expected_final = {**baseline, "offset": total_responses, "optimizer_updates": completed_updates,
                      "response_tokens": total_tokens, "active_rollout": False,
                      "completed_rollouts": len(rollouts)}
    if (final_cursor != expected_final or completed_updates != work["max_updates"]
            or sum(d["responses"] for d in derived) != epochs * rows
            or sum(d["action_tokens"] for d in derived) != epochs * total_tokens):
        raise ValueError("Complete epoch/response/token/final cursor coverage differs")
    return {"optimizer_updates": completed_updates, "rollouts": len(rollouts),
            "unique_responses": rows, "response_optimizer_uses": epochs * rows,
            "response_tokens": total_tokens, "optimizer_action_tokens": epochs * total_tokens,
            "derived_records_sha256": digest(derived), "scalar_responses_reconstructed": completed_updates,
            "other_four_objective_fields_exact": True, "all_cursor_hashes_verified": True}


def _clone(function, **replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def verify_bound(binding):
    """Read original supervisor observations and run complete CPU-only acceptance."""
    output = _verify_binding(binding)
    campaign = read(CAMPAIGN)
    stages = [s for s in campaign["stages"] if s["name"] == "ppo43"]
    if (campaign["status"] != "failed" or campaign["primary_alignment_verified"] is not False
            or campaign["plan_sha256"] != PLAN_SHA or campaign["deadline_epoch"] != DEADLINE
            or len(stages) != 1 or stages[0].get("exit_code") != 0
            or stages[0].get("active_processes") != [] or stages[0].get("monitoring_error") is not None):
        raise ValueError("Original failed campaign and exited PPO43 evidence required")
    record = stages[0]
    wrapper = read(output.parent / "ppo43.request.json")
    if record["request"] != wrapper or record["request_sha256"] != REQUEST_SHA:
        raise ValueError("Original supervisor request differs")
    return verify_ppo43(output, wrapper, {p["pid"]: p for p in record["known_processes"]},
                        record["exit_code"], binding=binding)


def verify_ppo43(output, wrapper, known, code, *, binding):
    from . import alignment_supervisor_r3 as legacy
    from . import throughput_fresh_v1 as fresh
    from .process_guard_v2 import ProcessIdentity
    if Path(output).resolve() != _verify_binding(binding):
        raise ValueError("Supplemental caller output differs")
    output = Path(output).resolve()
    result_before = read(output / "progress.json")
    if (wrapper != read(output.parent / "ppo43.request.json")
            or wrapper["legacy_request"]["stage"] != "ppo"
            or wrapper["legacy_request"]["seed"] != 43 or code != 0):
        raise ValueError("Supplemental caller changed original stage evidence")
    # Equivalent ISO fractional-second spellings must describe the same exact
    # process identity; retain all original evidence and normalize only a copy.
    normalized, spelling_changes = {}, []
    for observed in result_before["process_guard"]["owned_identities"]:
        old = known.get(observed["pid"])
        if old is None or ProcessIdentity.from_record(old) != ProcessIdentity.from_record(observed):
            raise ValueError("Externally observed process identity differs")
        if {k: v for k, v in old.items() if k != "created_at_utc"} != {
                k: v for k, v in observed.items() if k != "created_at_utc"}:
            raise ValueError("Non-time process identity fields differ")
        normalized[observed["pid"]] = copy.deepcopy(observed)
        if old != observed:
            spelling_changes.append({"pid": observed["pid"], "supervisor": old["created_at_utc"],
                                     "child": observed["created_at_utc"], "same_exact_identity": True})
    if set(normalized) != set(known):
        raise ValueError("Externally observed process inventory differs")
    reconstruction = []

    def verified_live(actual_output, stage, objectives):
        if Path(actual_output).resolve() != output or stage != "ppo" or reconstruction:
            raise ValueError("Supplemental comparator escaped its one bound invocation")
        manifest = read(output / "telemetry/manifest.json")
        rollouts = []
        for index, pin in enumerate(manifest["rollouts"]):
            path = output / "telemetry/rollouts" / f"rollout-{index:06d}.json"
            if Path(pin["path"]).resolve() != path.resolve() or sha(path) != pin["sha256"]:
                raise ValueError("Rollout hash/path changed")
            rollouts.append(read(path))
        events = [json.loads(line) for line in (output / "telemetry/cost-events.jsonl").read_text(encoding="utf-8").splitlines()]
        reconstruction.append(reconstruct_objectives(events, rollouts, read(output / "selection.json")["records"],
            result_before["identity"]["config"]["work"], result_before["cursor"], objectives))

    # Function-local namespaces preserve original bytecode and every check
    # before and after the original failing comparison. No global patch exists.
    local_legacy = _clone(legacy._verify_stage, _verify_live_objectives=verified_live)
    local_fresh = _clone(fresh.verify_stage, legacy_supervisor=SimpleNamespace(_verify_stage=local_legacy))
    accepted = local_fresh(output, wrapper, normalized, code)
    if accepted != result_before or len(reconstruction) != 1:
        raise ValueError("Supplemental acceptance did not cover the original result exactly")
    _verify_binding(binding)
    receipt = {"kind": KIND, "status": "ppo43_supplemental_accepted", "binding": binding,
               "original_progress_sha256": binding["files"]["stage/progress.json"]["sha256"],
               "artifact": accepted["artifact"], "checkpoint_count": len(accepted["checkpoints"]),
               "live_objectives": reconstruction[0], "original_full_acceptance_passed": True,
               "original_guard_and_fresh_wrapper_checks_passed": True,
               "process_time_spelling_normalizations": spelling_changes,
               "historical_files_modified": False, "training_updates_executed": 0,
               "device": "cpu", "research_complete": False}
    return accepted, receipt
