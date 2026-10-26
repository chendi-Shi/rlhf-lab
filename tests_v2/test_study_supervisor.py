"""CPU-only formal-parent lifecycle and real on-disk identity checks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import study_supervisor as ss


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": ss.sha(path), "size_bytes": path.stat().st_size}


def bundle(path, identity, *, stage=None, completion=None):
    path.mkdir()
    payload = "weights.pt" if stage else "state.pt"
    (path / payload).write_bytes(b"CPU fixture payload; no tensor load or CUDA")
    item = file_record(path / payload)
    manifest = {"format": "alignlab-completed-stage-v1" if stage else "alignlab-optimizer-checkpoint-v1",
                "status": "complete", "identity": identity,
                "identity_sha256": hashlib.sha256(ss._canonical(identity)).hexdigest(),
                "artifacts": {payload: {k: item[k] for k in ("sha256", "size_bytes")}}}
    if stage:
        manifest.update(stage=stage, completion=completion)
    write(path / "manifest.json", manifest)
    return ss.sha(path / "manifest.json")


class FakeChild:
    def __init__(self, system, request, pid):
        self.system, self.request, self.pid, self.published = system, request, pid, False

    def poll(self):
        if self.system.clock < self.system.child_end:
            return None
        if not self.published:
            self.system.test.publish(self.request, self.pid + 1)
            self.published = True
        return self.system.child_code


class FakeSystem:
    def __init__(self, test):
        self.test, self.clock, self.commands, self.sleeps = test, 1000.0, [], []
        self.child, self.child_code, self.child_duration, self.descendant_linger = None, 0, 20, 0
        self.initial_busy, self.resources_until, self.extra_processes = False, 0, []
        self.on_sleep, self.process_override = None, {}

    def now(self):
        return self.clock

    def sleep(self, seconds):
        assert 0 < seconds <= 10
        self.sleeps.append(seconds)
        self.clock += seconds
        if self.on_sleep:
            self.on_sleep(self)

    def processes(self):
        command = f"python scripts/run_alignlab_parents.py --plan {self.test.path} --execute"
        records = [
            {"pid": os.getpid(), "parent_pid": 91, "created_at_utc": stamp(900), "command_line": command},
            {"pid": 91, "parent_pid": 1, "created_at_utc": stamp(899), "command_line": command},
        ] + self.extra_processes
        if self.child:
            for pid, parent, end, offset in (
                    (self.child.pid, os.getpid(), self.child_end, 0),
                    (self.child.pid + 1, self.child.pid, self.child_end + self.descendant_linger, .001)):
                if self.clock < end:
                    records.append({"pid": pid, "parent_pid": parent, "created_at_utc": stamp(self.child_start + offset),
                                    "command_line": " ".join(self.commands[-1]), **self.process_override.get(pid, {})})
        return records

    def resources(self):
        return {"blockers": ["busy"] if self.initial_busy or self.clock < self.resources_until else [],
                "available_ram_gb": 12, "gpus": [{"index": 0, "free_mb": 7500}]}

    def spawn(self, args, stdout, stderr):
        if self.child and self.clock < self.child_end + self.descendant_linger:
            raise AssertionError("Previous real descendant still alive")
        self.commands.append(args)
        request = json.loads(Path(args[args.index("--request") + 1]).read_text(encoding="utf-8"))
        self.child_start, self.child_end = self.clock, self.clock + self.child_duration
        self.child = FakeChild(self, request, 100 + 10 * len(self.commands))
        stdout.write(b"CPU lifecycle fixture\n")
        stdout.flush()
        return self.child


class StudySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("reports", "runs", "alignment_lab", "tests_v2", "scripts", "rlhf_lab",
                     "models/snapshot", "models/tokenizer", "data/audit", "reports/archive"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.root_patch = patch.object(ss, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        legacy = {"scripts/verify_alignlab_cpu.py", "rlhf_lab/runtime.py"} | {
            f"alignment_lab/history_{i:02}.py" for i in range(50)}
        for name in legacy | ss.NEW_SOURCE_FILES:
            (self.root / name).write_text("# frozen CPU fixture\n", encoding="utf-8")
        archived = {}
        for i, name in enumerate(sorted(legacy)):
            source = self.root / name
            target = self.root / f"reports/archive/source-{i}.py"
            target.write_bytes(source.read_bytes())
            archived[name] = {"snapshot_path": str(target), "sha256": ss.sha(source),
                              "size_bytes": source.stat().st_size}
        archive_manifest = self.root / "reports/archive/manifest.json"
        write(archive_manifest, {"kind": "alignlab-frozen-capacity-source-snapshot-v1",
                                 "file_count": 52, "files": archived})
        base = self.root / "models/snapshot/config.json"
        base.write_text("{}", encoding="utf-8")
        model = base.parent / "snapshot-manifest.json"
        write(model, {"files": {"config.json": {"sha256": ss.sha(base), "size_bytes": base.stat().st_size}}})
        token_dir = self.root / "models/tokenizer"
        (token_dir / "config.json").write_text("{}", encoding="utf-8")
        write(token_dir / "source-manifest.json",
              {"artifacts": {"config.json": {"sha256": ss.sha(token_dir / "config.json")}}})
        audit = self.root / "data/audit/manifest.json"
        artifacts = {}
        for stage in ss.STAGES:
            path = audit.parent / (ss.PARTITIONS[stage] + ".jsonl")
            path.write_text("".join(json.dumps({"record_id": f"{stage}-{i}"}) + "\n"
                                    for i in range(ss.COUNTS[stage])), encoding="utf-8")
            artifacts[path.name] = ss.sha(path)
        write(audit, {"data_ready": True, "artifacts": artifacts})
        capacity = self.root / "reports/capacity.json"
        write(capacity, {"status": "accepted_registered_full_capacity_workload_only", "all_checks_passed": True,
                         "capacity_verified": True, "formal_training": False,
                         "model_snapshot": {"sha256": ss.sha(model)},
                         "data": {"manifest": {"sha256": ss.sha(audit)}},
                         "source_snapshot": {"path": str(archive_manifest), "sha256": ss.sha(archive_manifest)}})
        priority = self.root / "reports/priority.json"
        write(priority, {"status": "paused"})
        sources = {name: ss.sha(self.root / name) for name in legacy | ss.NEW_SOURCE_FILES}
        cpu = self.root / "reports/cpu.json"
        write(cpu, {"status": "cpu_software_checks_passed", "source_unchanged": True,
                    "cuda_initialized": False, "device": "cpu", "tests_run": 300,
                    "errors": 0, "failures": 0, "skipped": 0, "source_sha256": sources})
        self.output, self.path = self.root / "runs/parents-1", self.root / "reports/plan.json"
        self.plan = {"schema_version": 1, "kind": ss.KIND, "output": str(self.output),
                     "wall_seconds": 43200, "poll_seconds": 10, "source_sha256": sources,
                     "model_snapshot": str(model.parent), "model_manifest_sha256": ss.sha(model),
                     "tokenizer_dir": str(token_dir), "audit_dir": str(audit.parent),
                     "audit_manifest_sha256": ss.sha(audit), "cpu_check_report": str(cpu),
                     "cpu_check_report_sha256": ss.sha(cpu), "capacity_acceptance_path": str(capacity),
                     "capacity_acceptance_sha256": ss.sha(capacity),
                     "prerequisites": [{"path": str(priority), "sha256": ss.sha(priority),
                                        "json_equals": {"status": "paused"}}]}
        self.save_plan()
        self.system = FakeSystem(self)
        mock = patch.object(ss, "verify_snapshot", return_value={"repo_id": ss.REPO_ID, "revision": ss.REVISION})
        self.snapshot = mock.start()
        self.addCleanup(mock.stop)
        self.result_changes, self.after_publish = {}, None

    def save_plan(self):
        write(self.path, self.plan)
        self.plan_sha = ss.sha(self.path)

    def run_campaign(self, execute=True):
        return ss.run_parents(self.path, self.plan_sha, execute=execute, system=self.system)

    def publish(self, request, pid):
        output, stage = Path(request["output"]), request["stage"]
        output.mkdir()
        write(output / "request.json", request)
        count, updates = ss.COUNTS[stage], ss.UPDATES[stage]
        records = [{"record_id": f"{stage}-{i}", "cluster_id": f"c-{i}", "prompt_id": f"p-{i}",
                    "prompt_tokens": 1, "chosen_tokens": 1, "rejected_tokens": 1 if stage == "reward" else None,
                    "token_sha256": "a" * 64} for i in range(count)]
        summary = {"scope": ss.SCOPE, "partition": ss.PARTITIONS[stage], "configured_rows": count,
                   "selected_rows": count, "real_rows": count, "synthetic_rows": 0,
                   "accepted_manifest_sha256": request["audit_manifest_sha256"],
                   "records_sha256": hashlib.sha256(ss._canonical(records)).hexdigest()}
        write(output / "selection.json", {"summary": summary, "records": records})
        selection = {**summary, "selection_manifest_path": str(output / "selection.json"),
                     "selection_manifest_sha256": ss.sha(output / "selection.json")}
        inputs = {"source/" + name: file_record(self.root / name) for name in request["source_sha256"]}
        inputs.update({"parent_plan": file_record(request["parent_plan_path"]),
                       "capacity_acceptance": file_record(request["capacity_acceptance_path"]),
                       "data/manifest": file_record(Path(request["audit_dir"]) / "manifest.json"),
                       "data/partition": file_record(Path(request["audit_dir"]) / (ss.PARTITIONS[stage] + ".jsonl")),
                       "data/selection": file_record(output / "selection.json"),
                       "base/config.json": file_record(Path(request["model_snapshot"]) / "config.json"),
                       "tokenizer/config.json": file_record(Path(request["tokenizer_dir"]) / "config.json"),
                       "tokenizer/source-manifest.json": file_record(Path(request["tokenizer_dir"]) / "source-manifest.json")})
        identity = {"inputs": inputs, "parents": {k: file_record(Path(v["artifact_dir"]) / "manifest.json")
                                                for k, v in request["parents"].items()},
                    "config": {"stage": stage, "purpose": ss.STAGE_KIND, "scope": ss.SCOPE,
                               "formal_training": True, "synthetic_rows": 0,
                               "expected_rows": count, "expected_updates": updates}}
        write(output / "identity.json", identity)
        tokens = count * (2 if stage == "reward" else 1)
        completion = {"scope": ss.SCOPE, "full_accepted_dataset_pass": True, "synthetic_rows": 0,
                      "stop_reason": "data_exhausted", "data_pass_complete": True, "protocol_complete": True,
                      "rows_processed": count, "optimizer_updates": updates, "response_tokens": tokens}
        artifact = output / "final-artifact"
        artifact_sha = bundle(artifact, identity, stage=stage, completion=completion)
        checkpoints = []
        for u in sorted({1, updates} | set(range(50, updates + 1, 50))):
            loc = output / f"update-{u:06d}"
            digest = bundle(loc, identity)
            checkpoints.append({"checkpoint_path": str(loc), "manifest_sha256": digest,
                                "optimizer_updates": u, "active_elapsed_seconds": 1, "reasons": ["fixture_boundary"]})
        boundaries = [{"event": "optimizer_boundary", "cursor": {"optimizer_updates": u}}
                      for u in range(1, updates + 1)]
        objectives = [{"event": "objective_after_engine_return", "update": u,
                       "rows": min(16, count - (u - 1) * 16), "objective": .5, "gradient_norm": .2}
                      for u in range(1, updates + 1)]
        (output / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in boundaries + objectives),
                                              encoding="utf-8")
        result = {"status": "formal_parent_stage_passed", "formal_training": True, "research_complete": False,
                  "stage": stage, "request": request, "pid": pid,
                  "started_at_epoch": self.system.child_start, "finished_at_epoch": self.system.child_end,
                  "deadline_epoch": request["deadline_epoch"], "actual_optimizer_updates": updates,
                  "selection": selection, "identity": identity, "completion": completion,
                  "cursor": {"offset": count, "optimizer_updates": updates, "completed_rollouts": 0,
                             "active_rollout": False, "response_tokens": tokens},
                  "outcome_stop_reason": "data_exhausted", "adapter_weights_changed": True,
                  "scalar_heads_changed": {"reward": True} if stage == "reward" else {},
                  "model": {"device": "cuda:0", "quantization": ss.NF4, "quantized_linear_count": 196,
                            "parameters_selected": 123, "snapshot_manifest_sha256": request["model_manifest_sha256"]},
                  "gpu_samples": [{"gpus": [{"index": 0}], "max_allocated_bytes": 100, "max_reserved_bytes": 200}],
                  "objective_logging": "after_engine_returns", "objective_records": updates,
                  "artifact": {"artifact_dir": str(artifact), "manifest_path": str(artifact / "manifest.json"),
                               "manifest_sha256": artifact_sha},
                  "checkpoints": checkpoints, "latest_safe_checkpoint": checkpoints[-1]}
        result.update(self.result_changes)
        write(output / "progress.json", result)
        if self.after_publish:
            self.after_publish(output, result)

    def test_plan_only_no_output_lock_or_spawn(self):
        result = self.run_campaign(False)
        self.assertEqual(result["status"], "plan_only")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.system.commands)
        self.assertEqual(self.snapshot.call_count, 0)

    def test_resources_busy_before_start_never_wait_or_spend_budget(self):
        self.system.initial_busy = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertIsNone(result["started_at_epoch"])
        self.assertFalse(self.system.sleeps)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "reports/.alignlab-parents.lock").exists())

    def test_preexisting_stop_never_starts(self):
        Path(str(self.path) + ".stop").write_text("user pause")
        result = self.run_campaign()
        self.assertEqual(result["status"], "stopped_before_start")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.system.commands)

    def test_two_full_real_stages_serial_and_final_reward_partial_batch(self):
        self.system.descendant_linger = 10
        result = self.run_campaign()
        self.assertEqual(result["status"], "parents_passed")
        self.assertTrue(result["parents_verified"])
        self.assertTrue(result["capacity_verified"])
        self.assertFalse(result["research_complete"])
        self.assertEqual(len(self.system.commands), 2)
        self.assertEqual(result["elapsed_seconds"], 60)
        sft, reward = result["stages"]
        self.assertEqual({r["request"]["deadline_epoch"] for r in result["stages"]}, {44200})
        self.assertEqual((sft["request"]["expected_rows"], sft["request"]["expected_updates"]), (4000, 250))
        self.assertEqual((reward["request"]["expected_rows"], reward["request"]["expected_updates"]), (13139, 822))
        self.assertEqual(reward["request"]["parents"]["sft"],
                         {k: sft["artifact"][k] for k in ("artifact_dir", "manifest_sha256")})
        self.assertTrue(all(len(r["known_processes"]) == 2 and r["active_processes"] == [] for r in result["stages"]))
        self.assertFalse((self.root / "reports/.alignlab-parents.lock").exists())
        with self.assertRaises(FileExistsError):
            self.run_campaign()

    def test_new_unregistered_eval_file_does_not_invalidate_frozen_dependencies(self):
        (self.root / "alignment_lab/new_eval.py").write_text("# independent later evaluation")
        self.assertEqual(self.run_campaign(False)["status"], "plan_only")

    def test_changed_registered_dependency_or_archive_rejected(self):
        target = self.root / "alignment_lab/history_00.py"
        target.write_text("# changed")
        with self.assertRaisesRegex(ValueError, "SHA changed"):
            self.run_campaign(False)
        self.assertFalse(self.output.exists())

    def test_missing_new_implementation_pin_and_stale_cpu_proof_rejected(self):
        del self.plan["source_sha256"]["alignment_lab/study_stage.py"]
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "Frozen sources"):
            self.run_campaign()
        self.assertFalse(self.system.commands)

    def test_cpu_proof_must_cover_current_formal_code(self):
        p = Path(self.plan["cpu_check_report"])
        value = json.loads(p.read_text())
        value["source_sha256"]["alignment_lab/study_stage.py"] = "b" * 64
        write(p, value)
        self.plan["cpu_check_report_sha256"] = ss.sha(p)
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "CPU source-before"):
            self.run_campaign()

    def test_stop_in_child_waits_for_real_descendant_and_no_next_stage(self):
        self.system.descendant_linger = 10
        self.system.on_sleep = lambda s: (self.output / "STOP").write_text("user STOP")
        result = self.run_campaign()
        self.assertEqual(result["status"], "paused")
        self.assertEqual(self.system.clock, 1030)
        self.assertEqual(len(self.system.commands), 1)

    def test_deadline_during_child_does_not_reset_budget_or_kill(self):
        self.system.child_duration = 43210
        self.system.on_sleep = lambda s: setattr(s, "clock", max(s.clock, 44200))
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(len(self.system.commands), 1)
        self.assertGreaterEqual(self.system.clock, 44210)

    def test_snapshot_time_and_poststart_resource_wait_are_charged(self):
        def delay(system):
            if len(system.commands) == 1 and system.clock == system.child_end:
                system.resources_until = system.clock + 20
        self.system.on_sleep = delay
        result = self.run_campaign()
        self.assertEqual(result["deadline_epoch"], 44200)
        self.assertEqual(result["elapsed_seconds"], 60)
        self.assertEqual(result["stages"][1]["launch_epoch"], 1040)

    def test_budget_exhaustion_before_first_child_has_no_gpu_dispatch(self):
        def slow(*args):
            self.system.clock += 43201
            return {"repo_id": ss.REPO_ID, "revision": ss.REVISION}
        self.snapshot.side_effect = slow
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(self.system.commands)

    def test_identity_change_during_child_signals_stop_waits_and_fails(self):
        self.system.on_sleep = lambda s: (self.root / "alignment_lab/study_stage.py").write_text("# changed")
        with self.assertRaisesRegex(RuntimeError, "SHA changed"):
            self.run_campaign()
        self.assertEqual(self.system.clock, 1020)
        self.assertTrue((self.output / "STOP").exists())
        self.assertEqual(len(self.system.commands), 1)

    def test_pid_reuse_is_not_accepted(self):
        self.system.on_sleep = lambda s: s.process_override.update(
            {s.child.pid + 1: {"created_at_utc": stamp(9999)}})
        with self.assertRaisesRegex(RuntimeError, "PID reused"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_child_failure_no_retry_no_artifact_acceptance(self):
        self.system.child_code = 2
        with self.assertRaisesRegex(RuntimeError, "exited 2"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_unknown_lock_and_other_training_preserved(self):
        path = self.root / "reports/.alignlab-capacity.lock"
        path.write_text("foreign")
        with self.assertRaises(FileExistsError):
            self.run_campaign()
        self.assertEqual(path.read_text(), "foreign")
        self.assertFalse(self.output.exists())

    def test_other_training_process_blocks_before_output(self):
        self.system.extra_processes = [{"pid": 77, "parent_pid": 1, "created_at_utc": stamp(950),
                                       "command_line": "python -m windows.train --updates 2000"}]
        with self.assertRaisesRegex(RuntimeError, "Other training"):
            self.run_campaign()
        self.assertFalse(self.output.exists())

    def test_capacity_completion_never_initializes_formal_reward(self):
        def replace_completion(output, result):
            path = output / "final-artifact/manifest.json"
            value = json.loads(path.read_text())
            value["completion"].update(scope="capacity_selection_data_pass", full_accepted_dataset_pass=False)
            write(path, value)
            result["artifact"]["manifest_sha256"] = ss.sha(path)
            write(output / "progress.json", result)
        self.after_publish = replace_completion
        with self.assertRaisesRegex(ValueError, "Capacity or incomplete"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_payload_tamper_prevents_reward_start(self):
        self.after_publish = lambda output, result: (output / "final-artifact/weights.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA changed"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_cpu_model_cannot_claim_formal_gpu_training(self):
        self.result_changes = {"model": {"device": "cpu"}}
        with self.assertRaisesRegex(RuntimeError, "CUDA NF4"):
            self.run_campaign()

    def test_missing_objective_update_or_final_partial_batch_rejected(self):
        def remove_metric(output, result):
            path = output / "metrics.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
        self.after_publish = remove_metric
        with self.assertRaisesRegex(RuntimeError, "full-partition objectives"):
            self.run_campaign()

    def test_wrong_report_pid_or_incomplete_training_rejected(self):
        self.result_changes = {"pid": 999999}
        with self.assertRaisesRegex(RuntimeError, "exact completed"):
            self.run_campaign()


if __name__ == "__main__":
    unittest.main()
