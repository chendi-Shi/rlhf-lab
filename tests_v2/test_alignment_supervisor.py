"""CPU lifecycle fixtures exercise real file/hash validation; never launch CUDA."""
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import alignment_supervisor as ss
from alignment_lab import alignment_process_guard as process_guard
from alignment_lab.reference_cache import build_identity, write_or_reuse_cache
from tests_v2 import test_study_supervisor as parent_fixtures
from tests_v2.test_study_supervisor import FakeSystem as ParentFakeSystem, write, file_record, bundle


def stage_bundle(path, identity, *, stage=None, completion=None, cursor=None, extra_state=None):
    """Tiny safe tensor files, including distinct real Adam step/cursor contents."""
    import torch
    bundle(path, identity, stage=stage, completion=completion)
    target_stage = identity["config"]["stage"]
    update = completion["optimizer_updates"] if stage else cursor["optimizer_updates"]
    parameters = {"base.layer.lora_A.policy.weight": torch.tensor([float(update)])}
    if target_stage == "ppo":
        parameters["scalar_heads.value.weight"] = torch.tensor([float(update) / 2])
    if stage:
        payload = {("heads/value/weight" if k.startswith("scalar_heads.") else
                    "adapter/" + k.replace(".lora_A.policy.", ".lora_A.")): v
                   for k, v in parameters.items()}
        name = "weights.pt"
    else:
        payload = {"parameters": parameters, "cursor": cursor,
                   "optimizer": {"state": {i: {"step": torch.tensor(float(update)),
                           "exp_avg": torch.zeros_like(value), "exp_avg_sq": torch.zeros_like(value)}
                       for i, value in enumerate(parameters.values())},
                       "param_groups": [{"params": list(range(len(parameters)))}]},
                   "rng": {"torch_cpu": torch.get_rng_state()},
                   "extra_state": extra_state if cursor["active_rollout"] else None}
        name = "state.pt"
    torch.save(payload, path / name)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["artifacts"][name] = {"sha256": ss.sha(path / name), "size_bytes": (path / name).stat().st_size}
    write(path / "manifest.json", manifest)
    return ss.sha(path / "manifest.json")


def lifecycle_observations(output, identity, request, rows, *, started, finished):
    """Persist real JSON/tensor bindings for deliberately synthetic CPU lifecycles.

    The sampler/optimizer values below are fixtures, not numerically trained
    policies. Actual tiny-model observation integration is tested separately.
    No verifier is mocked, including the downstream subprocess roster checks.
    """
    import torch
    from alignment_lab import alignment_stage as actual_stage
    from alignment_lab import stage_engine_observed as engine
    from alignment_lab.rollout import SampledResponse, RolloutBatch
    from alignment_lab.training_telemetry import TrainingTelemetry, plain
    count, updates = len(rows), request["workload"]["optimizer_updates"]
    stage = request["stage"]
    moment = [started + .1]
    telemetry = TrainingTelemetry(output / "telemetry", stage=stage, started_at_epoch=started,
                                  clock=lambda: moment[0])
    telemetry.set_identity(identity)
    if stage == "dpo":
        names = ("cold_cache_start", "cold_cache_prepared", "cold_cache_scoring_complete",
                 "cold_cache_bind_complete") if request["reference_cache"]["mode"] == "create" else (
                 "warm_cache_bind_start", "warm_cache_bind_complete")
        for name in names:
            telemetry.point(name)
    telemetry.warm_start()
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    backend = SimpleNamespace(
        _model=SimpleNamespace(named_parameters=lambda: [("base.layer.lora_A.policy.weight", parameter)]),
        _heads={}, _all_parameters=lambda: (parameter,),
        parameters_for=lambda role, heads=(): (parameter,))
    generator = torch.Generator().manual_seed(request["seed"])
    order = list(range(count))
    random.Random(request["seed"]).shuffle(order)
    cursor = {"offset": 0, "optimizer_updates": 0, "completed_rollouts": 0,
              "active_rollout": False, "response_tokens": 0, "order": order}
    def event(name, **fields):
        counts = {key: value for key, value in cursor.items() if key != "order"}
        value = engine._immutable({"schema_version": 1, "event": name,
                                   "stage": stage, "cursor": counts, **fields})
        telemetry.observe(value, backend=backend, role="policy")
    schedule = actual_stage.BoundaryCheckpoints(updates)
    active = None
    for update in range(1, updates + 1):
        moment[0] = started + .2 + (finished - started - 1) * (update - 1) / updates
        if stage == "ppo" and active is None:
            chosen = [rows[i] for i in order[cursor["offset"]:cursor["offset"] + 16]]
            before = bytes(generator.get_state().tolist())
            versions = ((id(parameter), parameter._version),)
            origin = engine._origin(cursor, [{"record_id": r.record_id} for r in chosen], backend, generator)
            phase = {"rollout_index": cursor["completed_rollouts"], "record_ids": origin["record_ids"],
                     "observation_origin": origin}
            event("rollout_start", **phase)
            event("sampling_start", **phase)
            torch.rand(1, generator=generator)
            after = bytes(generator.get_state().tolist())
            responses = tuple(SampledResponse(r.prompt_ids, (3, actual_stage.EOS),
                              (-1.0, -1.0), (0.0, 0.0), "eos") for r in chosen)
            rollout = RolloutBatch(responses, "policy", actual_stage.EOS, actual_stage.PAD, 256,
                                   actual_stage.VOCAB_SIZE, versions, before, after, backend)
            origin = engine._complete_origin(origin, rollout)
            phase["observation_origin"] = origin
            event("sampling_end", **phase)
            event("frozen_scoring_start", **phase)
            event("frozen_scoring_end", **phase)
            active = {"responses": [asdict(r) for r in responses], "record_ids": [r.record_id for r in chosen],
                      "reference_logprobs": [[-1.1, -1.1] for _ in chosen],
                      "reward_scores": [.2] * len(chosen), "advantages": [[.1, -.1] for _ in chosen],
                      "returns": [[.2, .2] for _ in chosen], "epoch": 0, "offset": 0,
                      "observation_origin": origin}
            cursor.update(offset=cursor["offset"] + len(chosen), active_rollout=True,
                          response_tokens=cursor["response_tokens"] + 2 * len(chosen))
            event("rollout_collected", rollout_index=origin["rollout_index"], **active,
                  rollout_metadata={"policy_role": "policy", "eos_id": actual_stage.EOS,
                      "pad_id": actual_stage.PAD, "vocab_size": actual_stage.VOCAB_SIZE, "max_new_tokens": 256})
        step_info = {"target_update": update}
        if stage == "ppo":
            step_info["rollout_index"] = active["observation_origin"]["rollout_index"]
        event("optimizer_start", **step_info)
        with torch.no_grad():
            parameter.fill_(float(update))
        cursor["optimizer_updates"] = update
        if stage == "dpo":
            cursor.update(offset=min(count, update * 16), response_tokens=3 * min(count, update * 16))
        elif update % 32 == 0:
            cursor.update(completed_rollouts=cursor["completed_rollouts"] + 1, active_rollout=False)
            active = None
        else:
            active["epoch"], active["offset"] = divmod(update % 32, 16)
        event("optimizer_end", update=update, **step_info)
        if stage == "ppo" and active is None:
            event("rollout_finished", rollout_index=cursor["completed_rollouts"] - 1)
        saved_cursor = json.loads(json.dumps(cursor))
        def writer(name):
            directory = output / name
            sidecar = telemetry.checkpoint_sidecar(name, saved_cursor, active)
            receipt = {"checkpoint_path": str(directory),
                "manifest_sha256": stage_bundle(directory, identity,
                    cursor={**saved_cursor, "telemetry": sidecar}, extra_state=plain(active))}
            binding = telemetry.bind_checkpoint(receipt, sidecar,
                started_at_epoch=moment[0], finished_at_epoch=moment[0])
            return {**receipt, "telemetry_sidecar": sidecar, "telemetry_binding": binding}
        schedule.save(saved_cursor, moment[0] - started, writer=writer,
                      warm_elapsed=telemetry.warm_elapsed(), clock=lambda: moment[0])
    from alignment_lab.stage_engine import StageOutcome
    completion = actual_stage.verify_completion(StageOutcome(dict(cursor), None, "data_exhausted",
                                               tuple({} for _ in range(updates))), rows, request)
    moment[0] = finished - .1
    telemetry.point("training_engine_return", cursor=dict(cursor))
    pin = telemetry.finish(cursor=dict(cursor), completion=completion,
                           checkpoints=list(schedule.records.values()))
    return dict(cursor), completion, list(schedule.records.values()), pin


class FakeSystem(ParentFakeSystem):
    def processes(self):
        records = super().processes()
        for record in records:
            record["command_line"] = record.get("command_line", "").replace(
                "run_alignlab_parents.py", "run_alignlab_alignment.py")
        return records


class AlignmentSupervisorTests(unittest.TestCase):
    def setUp(self):
        parent = parent_fixtures.StudySupervisorTests("test_plan_only_no_output_lock_or_spawn")
        parent.setUp()
        self.addCleanup(parent.doCleanups)
        self.parent, self.root = parent, parent.root
        patcher = patch.object(ss, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Build a completed parent campaign using the already tested lifecycle,
        # with genuine temporary files and the exact independent-audit schema.
        audit_path = Path(parent.plan["audit_dir"]) / "manifest.json"
        audit = json.loads(audit_path.read_text())
        for partition, prefix, count in (("sft_train", "sft", 4000),
                                        ("preference_train", "reward", 13139),
                                        ("ppo_prompts", "ppo", 12748)):
            path = audit_path.parent / (partition + ".jsonl")
            path.write_text("".join(json.dumps({"record_id": f"{prefix}-{i}", "cluster_id": f"c-{i}",
                            "prompt_id": f"p-{i}"}) + "\n" for i in range(count)), encoding="utf-8")
            audit["artifacts"][path.name] = ss.sha(path)
        audit.update(status="audit_complete", data_ready=True, readiness_blockers=[], training_started=False)
        write(audit_path, audit)
        parent.plan["audit_manifest_sha256"] = ss.sha(audit_path)
        cap_path = Path(parent.plan["capacity_acceptance_path"])
        cap = json.loads(cap_path.read_text())
        cap["data"]["manifest"]["sha256"] = ss.sha(audit_path)
        write(cap_path, cap)
        parent.plan["capacity_acceptance_sha256"] = ss.sha(cap_path)
        extra = self.root / "rlhf_lab/__init__.py"
        extra.write_text("# pinned implicit package dependency\n")
        parent.plan["source_sha256"]["rlhf_lab/__init__.py"] = ss.sha(extra)
        pcpu = Path(parent.plan["cpu_check_report"])
        proof = json.loads(pcpu.read_text())
        proof["source_sha256"] = parent.plan["source_sha256"]
        write(pcpu, proof)
        parent.plan["cpu_check_report_sha256"] = ss.sha(pcpu)
        parent.save_plan()
        parent.run_campaign()
        campaign_path = parent.output / "campaign.json"
        campaign = json.loads(campaign_path.read_text())
        parents, accepted = {}, {}
        for record in campaign["stages"]:
            name = record["stage"]
            parents[name] = {k: record["artifact"][k] for k in ("artifact_dir", "manifest_sha256")}
            accepted[name] = {**parents[name], "progress_path": record["progress_path"],
                              "progress_sha256": record["progress_sha256"],
                              "rows": ss.parent_supervisor.COUNTS[name],
                              "optimizer_updates": ss.parent_supervisor.UPDATES[name]}
        acceptance = self.root / "reports/parent-acceptance.json"
        write(acceptance, {"schema_version": 1, "kind": ss.PARENT_ACCEPTANCE_KIND,
            "status": "accepted_full_real_formal_parents", "all_checks_passed": True,
            "parents_verified": True, "formal_training": True, "research_complete": False,
            "processes_exited": True, "parent_plan": {"path": str(parent.path), "sha256": parent.plan_sha},
            "campaign": {"path": str(campaign_path), "sha256": ss.sha(campaign_path), "status": "parents_passed"},
            "stages": accepted})
        for name in ss.NEW_SOURCE_FILES:
            (self.root / name).write_text("# new frozen fixture dependency\n")
        sources = {**parent.plan["source_sha256"],
                   **{n: ss.sha(self.root / n) for n in ss.NEW_SOURCE_FILES}}
        cpu_path = self.root / "reports/alignment-cpu.json"
        write(cpu_path, {"status": "cpu_software_checks_passed", "source_unchanged": True,
              "cuda_initialized": False, "device": "cpu", "tests_run": 400,
              "errors": 0, "failures": 0, "skipped": 0, "source_sha256": sources})
        self.path, self.output = self.root / "reports/alignment-plan.json", self.root / "runs/alignment"
        self.plan = {k: parent.plan[k] for k in ("model_snapshot", "model_manifest_sha256", "tokenizer_dir",
                     "audit_dir", "audit_manifest_sha256", "capacity_acceptance_path",
                     "capacity_acceptance_sha256", "prerequisites")}
        self.plan.update(schema_version=1, kind=ss.KIND, output=str(self.output), wall_seconds=1000,
            poll_seconds=10, source_sha256=sources, cpu_check_report=str(cpu_path),
            cpu_check_report_sha256=ss.sha(cpu_path), parent_plan_path=str(parent.path),
            parent_plan_sha256=parent.plan_sha, parent_acceptance_path=str(acceptance),
            parent_acceptance_sha256=ss.sha(acceptance), parents=parents, ppo_rows=1024,
            allowed_background_processes=[], process_wait_policy=dict(process_guard.POLICY))
        self.save_plan()
        self.system = FakeSystem(self)
        self.result_changes, self.after_publish = {}, None
        self.guard_wait_branches = set()
        snapshot = patch.object(ss, "verify_snapshot", return_value={"repo_id": ss.REPO_ID, "revision": ss.REVISION})
        self.snapshot = snapshot.start()
        self.addCleanup(snapshot.stop)

    def save_plan(self):
        write(self.path, self.plan)
        self.plan_sha = ss.sha(self.path)

    def run_campaign(self, execute=True):
        return ss.run_alignment(self.path, self.plan_sha, execute=execute, system=self.system)

    def guard_evidence(self, output, request, result, pid):
        """Real strict guard records for a deliberately synthetic CPU lifecycle."""
        command = " ".join(self.system.commands[-1])
        wrapper = {"pid": pid - 1, "parent_pid": os.getpid(),
            "created_at_utc": parent_fixtures.stamp(self.system.child_start), "command_line": command}
        child = {"pid": pid, "parent_pid": pid - 1,
            "created_at_utc": parent_fixtures.stamp(self.system.child_start + .001), "command_line": command}
        owned = [child, wrapper]
        allowed = [*request["allowed_background_processes"], *request["supervisor_processes"]]
        snapshot = [*allowed, wrapper, child]
        epoch = [self.system.child_start + .01]
        end = min(self.system.child_end - .1, request["deadline_epoch"] - .1)
        records = []
        unknown_reads = [0]
        def processes():
            if unknown_reads[0]:
                unknown_reads[0] -= 1
                return snapshot + [{"pid": 700001, "parent_pid": 0,
                    "created_at_utc": parent_fixtures.stamp(self.system.child_start),
                    "command_line": "python -m pip install --target .runtime Pillow"}]
            return snapshot
        initial = process_guard.strict_admission(processes, owned_identities=owned,
            allowed_identities=allowed, deadline_epoch=request["deadline_epoch"],
            clock=lambda: epoch[0], sleep=lambda seconds: None)
        guard = process_guard.GuardSession(processes, owned_identities=owned, allowed_identities=allowed,
            deadline_epoch=request["deadline_epoch"], policy=request["process_wait_policy"],
            spent_seconds=request["process_wait_spent_seconds"], clock=lambda: epoch[0],
            sleep=lambda seconds: epoch.__setitem__(0, epoch[0] + seconds), on_event=records.append)
        phases = [("before_cuda", 0)]
        if request["stage"] == "dpo":
            count = result["reference_cache"]["unique_sequences"] if request["reference_cache"]["mode"] == "create" else 0
            chain, journal = "0" * 64, []
            scored_records = []
            if count:
                from alignment_lab.reference_cache import load_cache
                cache_dir = Path(result["reference_cache"]["path"])
                cache_identity = json.loads((cache_dir / "identity.json").read_text())
                cache = load_cache(cache_dir, cache_identity,
                    expected_manifest_sha256=result["reference_cache"]["manifest_sha256"])
                scored_records = [cache.entries[key].to_dict() for key in sorted(cache.entries)]
            for scored in (range(count + 1) if count else (0, 0)):
                payload = {"schema_version": 1, "previous_sha256": chain,
                    "scored_sequences": scored, "expected_unique_sequences": count,
                    "record": None if scored == 0 else scored_records[scored-1]}
                chain = hashlib.sha256(ss._canonical(payload)).hexdigest()
                journal.append({**payload, "record_sha256": chain})
                if scored == 0 or scored == count or scored % 16 == 0:
                    phases.append((f"cache_record:{scored}", 0))
            (output / "reference-cache-records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal))
        phases += [("engine_origin", 0)]
        phases += [("optimizer_complete", n) for n in range(1, result["actual_optimizer_updates"] + 1)]
        phases += [("engine_final", result["actual_optimizer_updates"])]
        spacing = max(0, end - epoch[0] - 3) / (len(phases) + 1)
        for boundary, update in phases:
            epoch[0] += spacing
            if boundary == "optimizer_complete" and update == 1 and (request["stage"], request["seed"]) in self.guard_wait_branches:
                unknown_reads[0] = 2
            def save():
                checkpoint = next(item for item in result["checkpoints"] if item["optimizer_updates"] == update)
                return {"complete": True, "optimizer_updates": update, "kind": "optimizer_checkpoint",
                    "path": str((Path(checkpoint["checkpoint_path"]) / "manifest.json").resolve()),
                    "sha256": checkpoint["manifest_sha256"]}
            decision = guard.check(boundary=boundary, optimizer_updates=update, save_boundary=save)
            self.assertTrue(decision["admission_allowed"])
        (output / "process-guard.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        return {"owned_identities": owned, "initial_admission": initial,
            "inherited_waited_seconds": request["process_wait_spent_seconds"],
            "total_waited_seconds": guard.budget.spent_seconds, "last": decision}

    def publish(self, request, pid):
        from alignment_lab import alignment_stage as actual_stage
        from alignment_lab.training_data import TrainingPartition, TrainingRow
        output, stage, work = Path(request["output"]), request["stage"], request["workload"]
        output.mkdir()
        write(output / "request.json", request)
        count, updates = work["rows"], work["optimizer_updates"]
        partition = "preference_train" if stage == "dpo" else "ppo_prompts"
        partition_path = Path(request["audit_dir"]) / (partition + ".jsonl")
        accepted = [json.loads(line) for line in partition_path.read_text().splitlines()]
        pool = TrainingPartition(tuple(TrainingRow(**r, prompt_ids=(1,),
                 chosen_ids=(actual_stage.EOS,) if stage == "dpo" else None,
                 rejected_ids=(3, actual_stage.EOS) if stage == "dpo" else None) for r in accepted),
                 partition, request["audit_manifest_sha256"], "a" * 64, "fixture-serialization", b"{}")
        rows, summary, records = actual_stage.select_real_rows(pool, request)
        write(output / "selection.json", {"summary": summary, "records": records})
        selection = {**summary, "selection_manifest_path": str(output / "selection.json"),
                     "selection_manifest_sha256": ss.sha(output / "selection.json")}
        inputs = {"source/" + name: file_record(self.root / name) for name in request["source_sha256"]}
        inputs.update({name: file_record(request[name + "_path"])
                       for name in ("alignment_plan", "parent_plan", "parent_acceptance", "capacity_acceptance")})
        inputs.update({"data/manifest": file_record(Path(request["audit_dir"]) / "manifest.json"),
                       "data/partition": file_record(partition_path),
                       "data/selection": file_record(output / "selection.json"),
                       "base/config.json": file_record(Path(request["model_snapshot"]) / "config.json"),
                       "tokenizer/config.json": file_record(Path(request["tokenizer_dir"]) / "config.json"),
                       "tokenizer/source-manifest.json": file_record(Path(request["tokenizer_dir"]) / "source-manifest.json")})
        anchor = {"metadata": {"device": "cuda:0", "precision": "bfloat16", "quantization": ss.NF4},
                  "tensors": {"fixture": {"sha256": "a" * 64}}}
        cache_pin = None
        if stage == "dpo":
            cache_path = Path(request["reference_cache"]["path"])
            if request["reference_cache"]["mode"] == "create":
                ref_identity = build_identity(base_files={k: v["path"] for k, v in inputs.items() if k.startswith("base/")},
                    adapter_files={name: Path(request["parents"]["sft"]["artifact_dir"]) / filename
                                   for name, filename in (("manifest", "manifest.json"), ("weights", "weights.pt"))},
                    tokenizer_files={k: v["path"] for k, v in inputs.items() if k.startswith("tokenizer/")},
                    source_files={k: v["path"] for k, v in inputs.items() if k.startswith("source/")},
                    dataset_manifest=Path(request["audit_dir"]) / "manifest.json",
                    accepted_dataset_manifest_sha256=request["audit_manifest_sha256"],
                    scoring_config={"template": "fixture", "serialization": "fixture", "precision": "bfloat16",
                                    "quantization": ss.NF4, "library_versions": {"fixture": "1"},
                                    "frozen_reference_receipt": anchor})
                cache = write_or_reuse_cache(cache_path, ref_identity, [
                    {"prompt_token_ids": [1], "completion_token_ids": [actual_stage.EOS], "sum_logprob": -1.0, "token_count": 1},
                    {"prompt_token_ids": [1], "completion_token_ids": [3, actual_stage.EOS], "sum_logprob": -2.0, "token_count": 2}])
                cache_pin = {"path": str(cache_path), "manifest_sha256": cache.manifest_sha256,
                             "identity_sha256": ss.sha(cache_path / "identity.json")}
            else:
                cache_pin = {k: request["reference_cache"][k] for k in ("path", "manifest_sha256", "identity_sha256")}
            cache_pin.update(mode=request["reference_cache"]["mode"], unique_sequences=2,
                             expected_unique_sequences=2, exact_key_coverage_verified=True)
            inputs.update({"reference_cache/manifest": file_record(cache_path / "manifest.json"),
                           "reference_cache/identity": file_record(cache_path / "identity.json")})
        identity = {"inputs": inputs, "parents": {k: file_record(Path(v["artifact_dir"]) / "manifest.json")
                                                for k, v in request["parents"].items()},
                    "config": {"stage": stage, "purpose": ss.STAGE_KIND, "scope": work["scope"],
                        "execution_revision": ss.EXECUTION_REVISION,
                        "telemetry_protocol": ss.TELEMETRY_PROTOCOL,
                        "process_guard_cache_record_interval": 16,
                        "formal_training": True, "synthetic_rows": 0, "registered_workload": work,
                        "work": asdict(actual_stage.stage_config(request))}}
        write(output / "identity.json", identity)
        cursor, completion, checkpoints, telemetry = lifecycle_observations(
            output, identity, request, rows,
            started=self.system.child_start, finished=self.system.child_end)
        artifact = output / "final-artifact"
        digest = stage_bundle(artifact, identity, stage=stage, completion=completion)
        boundaries = [{"event": "engine_boundary", "cursor": {"optimizer_updates": u,
                        "active_rollout": stage == "ppo" and u != updates}}
                      for u in range(1, updates + 1)]
        objectives = [{"event": "objective_after_engine_return", "update": u, "objective": .5,
                       "gradient_norm": .2, **({"rows": min(16, count - (u - 1) * 16)} if stage == "dpo"
                                              else {"responses": 1, "action_tokens": 2})}
                      for u in range(1, updates + 1)]
        (output / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in boundaries + objectives))
        result = {"status": "formal_alignment_stage_passed", "formal_training": True, "research_complete": False,
            "stage": stage, "request": request, "pid": pid,
            "started_at_epoch": self.system.child_start, "finished_at_epoch": self.system.child_end,
            "deadline_epoch": request["deadline_epoch"], "actual_optimizer_updates": updates,
            "selection": selection, "identity": identity, "completion": completion,
            "telemetry": telemetry,
            "cursor": cursor,
            "outcome_stop_reason": "data_exhausted", "adapter_weights_changed": True,
            "scalar_heads_changed": {"value": True} if stage == "ppo" else {},
            "model": {"device": "cuda:0", "quantization": ss.NF4, "quantized_linear_count": 196,
                      "parameters_selected": 123, "snapshot_manifest_sha256": request["model_manifest_sha256"]},
            "gpu_samples": [{"gpus": [{"index": 0}], "max_allocated_bytes": 100, "max_reserved_bytes": 200}],
            "objective_logging": "after_engine_returns", "objective_records": updates, "reference_cache": cache_pin,
            "artifact": {"artifact_dir": str(artifact), "manifest_path": str(artifact / "manifest.json"),
                         "manifest_sha256": digest},
            "checkpoints": checkpoints, "latest_safe_checkpoint": checkpoints[-1]}
        result.update(reference_anchor=anchor, reward_anchor=anchor if stage == "ppo" else None,
                      reward_head_anchor={"fixture": {"sha256": "b" * 64}} if stage == "ppo" else None,
                      frozen_full_audits=[{"phase": phase, "time_epoch": self.system.child_start + i,
                           "roles": ["reference", "reward"] if stage == "ppo" else ["reference"],
                           "reward_head_content_verified": stage == "ppo",
                           "cache_manifest_sha256": cache_pin["manifest_sha256"] if cache_pin else None}
                           for i, phase in enumerate(("before_training", "after_training", "after_export"))])
        result["imports"] = {role: {"stage": parent, "target_stage": stage, "target_role": role, "heads": heads,
            "target_identity_sha256": hashlib.sha256(ss._canonical(identity)).hexdigest(),
            "manifest_sha256": request["parents"][parent]["manifest_sha256"]}
            for role, parent, heads in [("policy", "sft", []), ("reference", "sft", [])]
            + ([("reward", "reward", ["reward"])] if stage == "ppo" else [])}
        result["process_guard"] = self.guard_evidence(output, request, result, pid)
        result.update(self.result_changes)
        write(output / "progress.json", result)
        if self.after_publish:
            self.after_publish(output, result)

    def test_plan_only_never_creates_output_or_process(self):
        self.assertEqual(self.run_campaign(False)["status"], "plan_only")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.system.commands)
        self.snapshot.assert_not_called()

    def test_busy_before_admission_does_not_spend_budget(self):
        self.system.initial_busy = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertIsNone(result["started_at_epoch"])
        self.assertFalse(self.system.sleeps)
        self.assertFalse(self.output.exists())

    def test_unknown_python_before_start_does_not_create_or_spend_campaign(self):
        self.system.extra_processes.append({"pid": 700001, "parent_pid": 0,
            "created_at_utc": parent_fixtures.stamp(900), "command_line": "python -m pip install Pillow"})
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertIsNone(result["started_at_epoch"])
        self.assertFalse(self.system.commands)
        self.assertFalse(self.output.exists())

    def test_unknown_live_peer_does_not_prematurely_stop_running_child(self):
        self.plan["wall_seconds"] = 45
        self.save_plan()
        unknown = {"pid": 700001, "parent_pid": 0, "created_at_utc": parent_fixtures.stamp(1001),
                   "command_line": "python -m pip install --target .runtime Pillow"}
        def transient(system):
            if system.clock < system.child_end:
                self.assertFalse((self.output / "STOP").exists())
                system.extra_processes[:] = [unknown]
            else:
                system.extra_processes.clear()
        self.system.on_sleep = transient
        self.guard_wait_branches.add(("dpo", 42))
        self.after_publish = lambda output, result: setattr(self.system, "resources_until", 10000)
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["stages"][0]["status"], "accepted")
        self.assertFalse(result["stages"][0].get("monitoring_error"))
        self.assertEqual(len(self.system.commands), 1)

    def test_known_competing_live_training_signals_safe_stop(self):
        def competing(system):
            system.extra_processes[:] = [{"pid": 700001, "parent_pid": 0,
                "created_at_utc": parent_fixtures.stamp(1001), "command_line": "python windows.train --updates 2000"}]
        self.system.on_sleep = competing
        with self.assertRaisesRegex(RuntimeError, "competing|identity"):
            self.run_campaign()
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(len(self.system.commands), 1)

    def test_persistent_snapshot_failure_is_bounded_and_preserves_uncertain_lock(self):
        original = self.system.processes
        def failed_query():
            if self.system.child and self.system.clock >= self.system.child_start + 10:
                raise OSError("fixture CIM unavailable")
            return original()
        self.system.processes = failed_query
        result = self.run_campaign()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["child_may_still_be_running"])
        self.assertFalse(result["stages"][0]["exit_identity_verified"])
        self.assertLessEqual(self.system.clock, self.system.child_start + 310)
        self.assertTrue((self.root / "reports/.alignlab-alignment.lock").exists())
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(len(self.system.commands), 1)

    def test_wrapper_exit_before_first_snapshot_retains_uncertain_training_lock(self):
        spawn = self.system.spawn
        def fast_wrapper(*args, **kwargs):
            child = spawn(*args, **kwargs)
            self.system.child_end = self.system.child_start
            child.poll = lambda: 0
            return child
        self.system.spawn = fast_wrapper
        with self.assertRaisesRegex(RuntimeError, "exit identity is unverified"):
            self.run_campaign()
        state = json.loads((self.output / "campaign.json").read_text())
        self.assertTrue(state["child_may_still_be_running"])
        self.assertFalse(state["stages"][0]["exit_identity_verified"])
        self.assertTrue((self.root / "reports/.alignlab-alignment.lock").exists())
        self.assertEqual(len(self.system.commands), 1)

    def test_child_wait_ledger_is_inherited_across_all_six_branches(self):
        self.guard_wait_branches.update((("dpo", 42), ("ppo", 42)))
        result = self.run_campaign()
        self.assertEqual(result["status"], "primary_alignment_passed")
        self.assertEqual(result["total_process_wait_seconds"], 4.0)
        self.assertEqual([record["request"]["process_wait_spent_seconds"] for record in result["stages"]],
                         [0.0, 2.0, 4.0, 4.0, 4.0, 4.0])

    def test_missing_guard_final_result_cannot_accept_hash_valid_stage(self):
        def truncate(output, result):
            path = output / "process-guard.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
        self.after_publish = truncate
        with self.assertRaisesRegex(ValueError, "boundary|guard|ledger"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_stop_before_admission_has_no_mutations(self):
        Path(str(self.path) + ".stop").write_text("pause")
        self.assertEqual(self.run_campaign()["status"], "stopped_before_start")
        self.assertFalse(self.output.exists())

    def test_six_serial_branches_reuse_one_pinned_cache(self):
        self.system.descendant_linger = 10
        result = self.run_campaign()
        self.assertEqual(result["status"], "primary_alignment_passed")
        self.assertTrue(result["primary_alignment_verified"])
        self.assertFalse(result["research_complete"])
        self.assertEqual(result["elapsed_seconds"], 180)
        requests = [s["request"] for s in result["stages"]]
        self.assertEqual([(r["stage"], r["seed"]) for r in requests], list(ss.SEQUENCE))
        self.assertEqual(len({r["deadline_epoch"] for r in requests}), 1)
        first = result["stages"][0]["reference_cache"]
        self.assertEqual([r["reference_cache"]["mode"] for r in requests[::2]], ["create", "reuse", "reuse"])
        self.assertTrue(all(r["reference_cache"]["manifest_sha256"] == first["manifest_sha256"] for r in requests[2::2]))
        self.assertTrue(all(r["workload"]["selection_seed"] == 42 for r in requests[1::2]))
        self.assertTrue(all(s["active_processes"] == [] and s["exit_code"] == 0 for s in result["stages"]))
        self.assertFalse((self.root / "reports/.alignlab-alignment.lock").exists())
        with self.assertRaises(FileExistsError):
            self.run_campaign()

    def test_live_parent_process_refused_before_admission(self):
        historical = json.loads((self.parent.output / "campaign.json").read_text())["stages"][0]["known_processes"][0]
        self.system.extra_processes.append(historical)
        with self.assertRaisesRegex(RuntimeError, "parent process"):
            self.run_campaign()
        self.assertFalse(self.output.exists())

    def test_incomplete_or_capacity_parent_refused(self):
        report_path = Path(self.plan["parent_acceptance_path"])
        report = json.loads(report_path.read_text())
        report["status"] = "accepted_registered_full_capacity_workload_only"
        write(report_path, report)
        self.plan["parent_acceptance_sha256"] = ss.sha(report_path)
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "Independent"):
            self.run_campaign()

    def test_registered_parent_dependency_drift_refused(self):
        (self.root / "alignment_lab/study_stage.py").write_text("# drift")
        with self.assertRaises(ValueError):
            self.run_campaign()
        self.assertFalse(self.system.commands)

    def test_new_unimported_eval_file_does_not_invalidate_inventory(self):
        (self.root / "alignment_lab/future_eval.py").write_text("# later unrelated work")
        self.assertEqual(self.run_campaign(False)["status"], "plan_only")

    def test_observed_dependencies_cannot_be_omitted_from_plan(self):
        self.plan["source_sha256"].pop("alignment_lab/stage_engine_observed.py")
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "observed-v2"):
            self.run_campaign(False)
        self.assertFalse(self.system.commands)
        self.assertFalse(self.output.exists())

    def test_cpu_proof_must_cover_current_observation_source_and_cli(self):
        path = Path(self.plan["cpu_check_report"])
        original = json.loads(path.read_text())
        for name in ("alignment_lab/training_telemetry.py",
                     "scripts/run_alignlab_alignment.py"):
            with self.subTest(name=name):
                proof = json.loads(json.dumps(original))
                proof["source_sha256"].pop(name)
                write(path, proof)
                self.plan["cpu_check_report_sha256"] = ss.sha(path)
                self.save_plan()
                with self.assertRaisesRegex(ValueError, "CPU proof"):
                    self.run_campaign(False)
        self.assertFalse(self.system.commands)

    def test_historical_cpu_success_cannot_cover_changed_observation_source(self):
        name = "alignment_lab/stage_engine_observed.py"
        (self.root / name).write_text("# new unverified revision\n")
        self.plan["source_sha256"][name] = ss.sha(self.root / name)
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "CPU proof"):
            self.run_campaign(False)
        self.assertFalse(self.system.commands)

    def test_unknown_locks_preserved(self):
        lock = self.root / "reports/.alignlab-gpu.lock"
        lock.write_text("unknown owner")
        with self.assertRaises(FileExistsError):
            self.run_campaign()
        self.assertEqual(lock.read_text(), "unknown owner")
        self.assertFalse(self.output.exists())

    def test_budget_expiry_signals_stop_without_dispatching_next(self):
        self.plan["wall_seconds"] = 15
        self.save_plan()
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["primary_alignment_verified"])
        self.assertEqual(len(self.system.commands), 1)
        self.assertTrue((self.output / "STOP").exists())

    def test_identity_drift_signals_stop_and_waits_for_exit(self):
        def drift(system):
            (self.root / "alignment_lab/alignment_stage.py").write_text("# changed")
        self.system.on_sleep = drift
        with self.assertRaises(RuntimeError):
            self.run_campaign()
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(len(self.system.commands), 1)
        self.assertGreaterEqual(self.system.clock, self.system.child_end)

    def test_wrong_child_command_cannot_be_accepted(self):
        self.system.process_override[110] = {"command_line": "python wrong.py"}
        with self.assertRaisesRegex(RuntimeError, "command|identity"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)
        self.assertTrue((self.output / "STOP").exists())

    def test_nonzero_exit_refuses_artifact_even_if_present(self):
        self.system.child_code = 2
        with self.assertRaisesRegex(RuntimeError, "exited 2"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_cpu_model_and_false_completion_not_accepted(self):
        self.result_changes["model"] = {"device": "cpu"}
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.run_campaign()

    def test_legacy_stage_identity_cannot_claim_observed_completion(self):
        def legacy(output, result):
            result["identity"]["config"].pop("execution_revision")
            write(output / "identity.json", result["identity"])
            write(output / "progress.json", result)
        self.after_publish = legacy
        with self.assertRaisesRegex(ValueError, "registered formal branch"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_observed_stage_cannot_substitute_another_parent_anchor(self):
        def substitute(output, result):
            result["identity"]["parents"]["sft"]["sha256"] = "f" * 64
            write(output / "identity.json", result["identity"])
            write(output / "progress.json", result)
        self.after_publish = substitute
        with self.assertRaisesRegex(ValueError, "admitted completed formal model"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_checkpoint_payload_tampering_refused(self):
        def tamper(output, result):
            Path(result["latest_safe_checkpoint"]["checkpoint_path"], "state.pt").write_bytes(b"tampered")
        self.after_publish = tamper
        with self.assertRaises(ValueError):
            self.run_campaign()

    def test_old_valid_checkpoint_cannot_impersonate_completed_final_update(self):
        def substitute(output, result):
            first, final = result["checkpoints"][0], result["latest_safe_checkpoint"]
            source, target = Path(first["checkpoint_path"]), Path(final["checkpoint_path"])
            # Both payload and its hash-valid manifest are genuinely committed
            # under the same branch identity; only the claimed cursor is later.
            for name in ("state.pt", "manifest.json"):
                (target / name).write_bytes((source / name).read_bytes())
            final["manifest_sha256"] = first["manifest_sha256"]
            write(output / "progress.json", result)
        self.after_publish = substitute
        with self.assertRaisesRegex(ValueError, "cursor|binding"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_ppo_subset_must_not_claim_all_accepted_prompts(self):
        def change(output, result):
            if result["stage"] == "ppo":
                result["completion"]["full_accepted_dataset_pass"] = True
                write(output / "progress.json", result)
        self.after_publish = change
        with self.assertRaisesRegex(ValueError, "full-partition"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 2)

    def test_ppo_complete_cursor_must_have_no_active_rollout(self):
        def change(output, result):
            if result["stage"] == "ppo":
                result["cursor"]["active_rollout"] = True
                write(output / "progress.json", result)
        self.after_publish = change
        with self.assertRaisesRegex(RuntimeError, "optimizer work"):
            self.run_campaign()

    def test_ppo_extra_checkpoint_requires_observation_binding(self):
        def variant(output, result):
            if result["stage"] != "ppo":
                return
            from alignment_lab.alignment_stage import BoundaryCheckpoints
            source = next(c for c in result["checkpoints"] if c["optimizer_updates"] == 50)
            cursor = {**source["cursor"], "offset": source["cursor"]["offset"] + 16}
            schedule = BoundaryCheckpoints(result["actual_optimizer_updates"])
            def writer(name):
                directory = output / name
                return {"checkpoint_path": str(directory), "manifest_sha256": stage_bundle(directory, result["identity"], cursor=cursor)}
            extra = schedule.save(cursor, 2, writer=writer)
            result["checkpoints"].append(extra)
            write(output / "progress.json", result)
        self.after_publish = variant
        with self.assertRaisesRegex(ValueError, "telemetry"):
            self.run_campaign()

    def test_missing_observation_manifest_refuses_completed_stage(self):
        def omit(output, result):
            result.pop("telemetry")
            write(output / "progress.json", result)
        self.after_publish = omit
        with self.assertRaisesRegex(ValueError, "telemetry"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_forged_rollout_statistics_refuse_completed_stage(self):
        def forge(output, result):
            if result["stage"] == "ppo":
                manifest = json.loads(Path(result["telemetry"]["path"]).read_text())
                path = Path(manifest["rollouts"][0]["path"])
                value = json.loads(path.read_text())
                value["active"]["reward_scores"][0] += 100
                write(path, value)
        self.after_publish = forge
        with self.assertRaisesRegex(ValueError, "Telemetry|telemetry"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 2)

    def test_reference_cache_score_tamper_prevents_acceptance(self):
        def tamper(output, result):
            if result["stage"] == "dpo":
                Path(result["reference_cache"]["path"], "scores.jsonl").write_text("{}\n")
        self.after_publish = tamper
        with self.assertRaises(ValueError):
            self.run_campaign()

    def test_missing_full_frozen_reference_audit_prevents_acceptance(self):
        self.result_changes["frozen_full_audits"] = []
        with self.assertRaisesRegex(RuntimeError, "frozen content"):
            self.run_campaign()

    def test_wait_for_resources_after_start_is_inside_deadline(self):
        def busy_after_first(output, result):
            self.system.resources_until = 10000
        self.after_publish = busy_after_first
        self.plan["wall_seconds"] = 45
        self.save_plan()
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(len(self.system.commands), 1)
        self.assertEqual(result["elapsed_seconds"], 45)


class StrictSupervisorInventoryTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = {"pid": 501, "parent_pid": 0, "created_at_utc": "2026-10-21T10:00:00.1234567Z",
            "command_line": "python scripts/run_alignlab_alignment.py --execute"}
        self.child = {"pid": 502, "parent_pid": 501, "created_at_utc": "2026-10-21T10:00:01.1234567Z",
            "command_line": "python scripts/run_alignlab_alignment_stage.py --execute"}

    def test_normal_child_exit_requires_only_supervisor_to_remain(self):
        decision = ss._peer_decision([self.supervisor], {501: self.supervisor}, {502: self.child})
        self.assertTrue(decision["caller_admission_allowed"])
        self.assertTrue(decision["owned_identity_verified"])

    def test_same_pid_new_creation_or_parent_never_impersonates_known_child(self):
        for change in ({"created_at_utc": "2026-10-21T10:00:01.1234568Z"}, {"parent_pid": 99}):
            with self.subTest(change=change):
                actual = {**self.child, **change}
                self.assertFalse(ss._same_identity(actual, self.child))
                if "parent_pid" in change:
                    with self.assertRaises(RuntimeError):
                        ss._peer_decision([self.supervisor, actual], {501: self.supervisor}, {502: self.child})
                else:
                    decision = ss._peer_decision([self.supervisor, actual], {501: self.supervisor}, {502: self.child})
                    self.assertFalse(decision["caller_admission_allowed"])
                    self.assertFalse(ss._waitable_peer_decision(decision))

    def test_external_unknown_is_waitable_but_lost_supervisor_is_not(self):
        unknown = {**self.child, "pid": 503, "parent_pid": 0, "command_line": "python -m pip install Pillow"}
        decision = ss._peer_decision([self.supervisor, unknown], {501: self.supervisor})
        self.assertFalse(decision["caller_admission_allowed"])
        self.assertTrue(ss._waitable_peer_decision(decision))
        missing = ss._peer_decision([unknown], {501: self.supervisor})
        self.assertFalse(ss._waitable_peer_decision(missing))


class LiveIdentitySentinelTests(unittest.TestCase):
    """Metadata is a drift alarm between full hashes, not content acceptance."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_large_artifact_check_does_not_reread_payload(self):
        path = self.root / "weights.pt"
        path.write_bytes(b"x" * (ss._LiveIdentity.HASH_LIMIT + 1))
        sentinel = ss._LiveIdentity({str(path): ss.sha(path)})
        with patch.object(ss, "sha", side_effect=AssertionError("Large payload must not be rehashed in active sentinel")):
            self.assertTrue(sentinel.check())
        original = path.stat()
        with path.open("r+b") as stream:
            stream.write(b"y")
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
        with self.assertRaises(ValueError):
            sentinel.check()

    def test_same_size_and_restored_mtime_replacement_is_rejected_by_file_id(self):
        path = self.root / "weights.pt"
        path.write_bytes(b"x" * (ss._LiveIdentity.HASH_LIMIT + 1))
        sentinel = ss._LiveIdentity({str(path): ss.sha(path)})
        original = path.stat()
        replacement = self.root / "new.pt"
        replacement.write_bytes(b"y" * original.st_size)
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        replacement.replace(path)
        self.assertEqual(path.stat().st_size, original.st_size)
        self.assertEqual(path.stat().st_mtime_ns, original.st_mtime_ns)
        self.assertNotEqual(path.stat().st_ino, original.st_ino)
        with self.assertRaises(ValueError):
            sentinel.check()

    def test_small_source_is_rehashed_even_when_metadata_is_identical(self):
        path = self.root / "guard.py"
        path.write_bytes(b"old source")
        sentinel = ss._LiveIdentity({str(path): ss.sha(path)})
        saved = dict(sentinel.files[str(path.absolute())])
        path.write_bytes(b"new source")
        with patch.object(sentinel, "_metadata", return_value=saved):
            with self.assertRaises(ValueError):
                sentinel.check()

    def test_protected_tree_member_addition_or_removal_cannot_be_refreshed(self):
        tree = self.root / "parent"
        tree.mkdir()
        path = tree / "manifest.json"
        path.write_text("{}")
        sentinel = ss._LiveIdentity({}, trees=(tree,))
        self.assertTrue(sentinel.check())
        extra = tree / "new.json"
        extra.write_text("{}")
        with self.assertRaises(ValueError):
            sentinel.check()
        with self.assertRaises(ValueError):
            sentinel.add_tree(tree)


class CheckpointSchemaCompatibilityTests(unittest.TestCase):
    """Historical parents stay readable; no branch may downgrade observations."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cursor = {"optimizer_updates": 1, "active_rollout": False}

    def checkpoint(self, config, *, cursor=None, receipt_fields=None):
        directory = self.root / ("checkpoint-" + str(len(list(self.root.iterdir()))))
        digest = stage_bundle(directory, {"config": config}, cursor=cursor or self.cursor)
        manifest = json.loads((directory / "manifest.json").read_text())
        receipt = {"checkpoint_path": str(directory), "manifest_sha256": digest,
                   "optimizer_updates": 1, "cursor": self.cursor, **(receipt_fields or {})}
        return directory, manifest, receipt

    def test_only_original_parent_schema_accepts_raw_cursor(self):
        for stage in ("sft", "reward"):
            args = self.checkpoint({"stage": stage, "purpose": ss.parent_supervisor.STAGE_KIND})
            ss._checkpoint_content(*args)
            with self.assertRaisesRegex(ValueError, "telemetry"):
                ss._checkpoint_content(*args, require_observation=True)

    def test_alignment_identity_without_sidecar_cannot_use_parent_compatibility(self):
        for stage in ("dpo", "ppo"):
            for revision in ({}, {"execution_revision": ss.EXECUTION_REVISION,
                                  "telemetry_protocol": ss.TELEMETRY_PROTOCOL}):
                with self.subTest(stage=stage, revision=revision):
                    args = self.checkpoint({"stage": stage, "purpose": ss.STAGE_KIND, **revision})
                    with self.assertRaisesRegex(ValueError, "telemetry"):
                        ss._checkpoint_content(*args)

    def test_parent_with_any_observation_marker_requires_complete_binding(self):
        parent = {"stage": "sft", "purpose": ss.parent_supervisor.STAGE_KIND}
        variants = [
            self.checkpoint({**parent, "execution_revision": ss.EXECUTION_REVISION}),
            self.checkpoint({**parent, "telemetry_protocol": ss.TELEMETRY_PROTOCOL}),
            self.checkpoint(parent, cursor={**self.cursor, "telemetry": {}}),
            self.checkpoint(parent, receipt_fields={"telemetry_sidecar": None}),
            self.checkpoint(parent, receipt_fields={"telemetry_binding": None}),
            self.checkpoint({**parent, "purpose": "unregistered-parent"}),
        ]
        for args in variants:
            with self.subTest(directory=str(args[0])):
                with self.assertRaisesRegex(ValueError, "telemetry"):
                    ss._checkpoint_content(*args)

    def test_parent_payload_still_cannot_relabel_its_actual_boundary(self):
        args = self.checkpoint({"stage": "sft", "purpose": ss.parent_supervisor.STAGE_KIND})
        args[2]["cursor"] = {**self.cursor, "optimizer_updates": 2}
        with self.assertRaisesRegex(ValueError, "payload cursor"):
            ss._checkpoint_content(*args)


class CostCheckpointEvidenceTests(unittest.TestCase):
    """Pure receipt arithmetic: no elapsed-time or threshold claim is inferred."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        binding = Path(temporary.name) / "binding.json"
        write(binding, {"save_started_at_epoch": 3700.1, "save_finished_at_epoch": 3700.9})
        self.receipt = {"save_started_at_epoch": 3700., "save_finished_at_epoch": 3701.,
            "active_elapsed_seconds": 3600., "telemetry_binding": {"path": str(binding)},
            "reasons": ["cost_1h"], "cost_observations": [
                {"observed_at_epoch": 3700., "stage_elapsed_seconds": 3600.,
                 "warm_training_elapsed_seconds": 3590., "reused_same_cursor": False,
                 "reasons": ["cost_1h"], "thresholds": [{"axis": "stage_wall",
                     "target_seconds": 3600, "actual_seconds": 3600., "overshoot_seconds": 0.}]}]}
        self.result = {"started_at_epoch": 100., "finished_at_epoch": 4000., "checkpoints": [self.receipt]}
        self.observed = {"warm_training_started_at_epoch": 110.}

    def test_same_cursor_later_threshold_retains_actual_later_clock(self):
        self.receipt["reasons"].append("warm_cost_1h")
        self.receipt["cost_observations"].append({
            "observed_at_epoch": 3710., "stage_elapsed_seconds": 3610.,
            "warm_training_elapsed_seconds": 3600., "reused_same_cursor": True,
            "reasons": ["warm_cost_1h"], "thresholds": [{"axis": "warm_training_wall",
                "target_seconds": 3600, "actual_seconds": 3600., "overshoot_seconds": 0.}]})
        ss._verify_cost_checkpoints(self.result, self.observed)
        self.assertEqual(self.receipt["active_elapsed_seconds"], 3600.)

    def test_forged_threshold_or_overshoot_rejected(self):
        self.receipt["cost_observations"][0]["thresholds"][0]["overshoot_seconds"] = 10
        with self.assertRaisesRegex(ValueError, "overshoot"):
            ss._verify_cost_checkpoints(self.result, self.observed)

    def test_checkpoint_io_cannot_finish_after_reported_save(self):
        self.receipt["save_finished_at_epoch"] = 3700.5
        with self.assertRaisesRegex(ValueError, "I/O"):
            ss._verify_cost_checkpoints(self.result, self.observed)

    def test_missing_cost_observation_cannot_be_inferred_from_label(self):
        self.receipt["cost_observations"] = []
        with self.assertRaisesRegex(ValueError, "observation"):
            ss._verify_cost_checkpoints(self.result, self.observed)


class ActualTinyStageInteropTests(unittest.TestCase):
    """Real CPU engine/export output is hash-valid but must fail CUDA admission."""
    def test_actual_tiny_dpo_and_ppo_output_schema_never_claims_gpu_pass(self):
        import torch
        from tests_v2 import test_alignment_stage as fixtures
        fixture = fixtures.AlignmentStageTests()
        fixture.setUpClass()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.tearDown)
        for request in (fixture.request, fixture.ppo_request()):
            with fixture.runtime(request):
                result = fixtures.stage.run_alignment_stage(request, execute=True)
            output = Path(request["output"])
            self.assertEqual(result["status"], "formal_alignment_stage_passed")
            self.assertEqual(result["identity"], json.loads((output / "identity.json").read_text()))
            self.assertEqual(result["selection"]["records_sha256"], hashlib.sha256(
                ss._canonical(json.loads((output / "selection.json").read_text())["records"])).hexdigest())
            observed = ss._verify_observation_artifacts(output, request, result)
            self.assertTrue(process_guard.verify_guard_log(output / "process-guard.jsonl",
                request=request, progress=result)["verified"])
            self.assertEqual(observed["final_cursor"], result["cursor"])
            self.assertEqual(len(observed["rollouts"]), result["cursor"]["completed_rollouts"])
            exported = ss._bundle(result["artifact"]["artifact_dir"], result["artifact"]["manifest_sha256"],
                                  {"weights.pt"}, "alignlab-completed-stage-v1")
            weights = ss._cpu_payload(Path(result["artifact"]["artifact_dir"]) / "weights.pt",
                                      exported["artifacts"]["weights.pt"]["sha256"])
            for checkpoint in result["checkpoints"]:
                manifest = ss._bundle(checkpoint["checkpoint_path"], checkpoint["manifest_sha256"],
                                      {"state.pt"}, "alignlab-optimizer-checkpoint-v1")
                self.assertEqual(checkpoint["cursor_sha256"],
                                 hashlib.sha256(ss._canonical(checkpoint["cursor"])).hexdigest())
                ss._checkpoint_content(checkpoint["checkpoint_path"], manifest, checkpoint,
                    final=checkpoint == result["latest_safe_checkpoint"], artifact=weights)
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                ss._verify_stage(output, request, {result["pid"]: {}}, 0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
