"""CPU-only fake-system lifecycle and on-disk evidence admission tests."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import capacity_supervisor as cs


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def file_record(path):
    return {"path": str(Path(path).resolve()), "sha256": cs.sha(path), "size_bytes": Path(path).stat().st_size}


def bundle(path, payload, identity, **fields):
    path.mkdir()
    (path / payload).write_bytes(b"fixture tensor payload; never loaded")
    record = file_record(path / payload)
    manifest = {"status": "complete", "identity": identity,
        "identity_sha256": hashlib.sha256(cs._canonical(identity)).hexdigest(),
        "artifacts": {payload: {k: record[k] for k in ("sha256", "size_bytes")}}, **fields}
    write(path / "manifest.json", manifest)
    return cs.sha(path / "manifest.json")


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
        self.child, self.child_code, self.child_duration, self.descendant_linger = None, 0, 2, 0
        self.initial_busy, self.resources_until, self.extra_processes = False, 0, []
        self.on_sleep, self.process_override = None, {}

    def now(self):
        return self.clock

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds); self.clock += seconds
        if self.on_sleep:
            self.on_sleep(self)

    def processes(self):
        command = f"python scripts/run_alignlab_capacity.py --plan {self.test.path} --execute"
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
            raise AssertionError("Previous child still alive")
        self.commands.append(args)
        request = json.loads(Path(args[args.index("--request") + 1]).read_text())
        self.child_start, self.child_end = self.clock, self.clock + self.child_duration
        self.child = FakeChild(self, request, 100 + 10 * len(self.commands))
        stdout.write(b"fixture child\n"); stdout.flush()
        return self.child


class CapacitySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("reports", "runs", "alignment_lab", "tests_v2", "scripts", "rlhf_lab",
                     "models/snapshot", "models/tokenizer", "data/audit"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.root_patch = patch.object(cs, "ROOT", self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        for name in cs.CLI_FILES | {"alignment_lab/capacity_supervisor.py", "tests_v2/test_capacity_supervisor.py"}:
            (self.root / name).write_text("# frozen fixture\n")
        base = self.root / "models/snapshot/config.json"; base.write_text("{}")
        model = base.parent / "snapshot-manifest.json"
        write(model, {"files": {"config.json": {"sha256": cs.sha(base), "size_bytes": base.stat().st_size}}})
        audit = self.root / "data/audit/manifest.json"; write(audit, {"data_ready": True})
        prior = self.root / "reports/priority.json"; write(prior, {"status": "paused"})
        cpu = self.root / "reports/cpu.json"
        sources = {name: cs.sha(self.root / name) for name in cs.source_inventory()}
        write(cpu, {"status": "cpu_software_checks_passed", "source_unchanged": True, "cuda_initialized": False,
            "device": "cpu", "tests_run": 270, "errors": 0, "failures": 0, "skipped": 0,
            "source_sha256": sources})
        initial = self.root / "reports/initial.json"
        write(initial, {"status": "accepted_initial_short_sft_probe_only", "all_checks_passed": True,
            "formal_training": False, "capacity_verified": False,
            "model_snapshot": {"manifest_sha256": cs.sha(model)}, "timing": {"probe_elapsed_seconds": 64.31412887573242}})
        self.output, self.path = self.root / "runs/capacity-1", self.root / "reports/plan.json"
        self.plan = {"schema_version": 1, "kind": cs.KIND, "formal_training": False,
            "output": str(self.output), "wall_seconds": 14335, "poll_seconds": 1, "source_sha256": sources,
            "model_snapshot": str(model.parent), "model_manifest_sha256": cs.sha(model),
            "tokenizer_dir": str(self.root / "models/tokenizer"), "audit_dir": str(audit.parent),
            "audit_manifest_sha256": cs.sha(audit), "cpu_check_report": str(cpu), "cpu_check_report_sha256": cs.sha(cpu),
            "initial_probe_acceptance": str(initial), "initial_probe_acceptance_sha256": cs.sha(initial),
            "prerequisites": [{"path": str(prior), "sha256": cs.sha(prior), "json_equals": {"status": "paused"}}]}
        self.save_plan()
        self.system = FakeSystem(self)
        self.snapshot = patch.object(cs, "verify_snapshot", return_value={"repo_id": cs.REPO_ID, "revision": cs.REVISION})
        self.snapshot_mock = self.snapshot.start(); self.addCleanup(self.snapshot.stop)
        self.result_changes, self.after_publish = {}, None

    def save_plan(self):
        write(self.path, self.plan); self.plan_sha = cs.sha(self.path)

    def run_campaign(self, execute=True):
        return cs.run_capacity(self.path, self.plan_sha, execute=execute, system=self.system)

    def publish(self, request, pid):
        output, stage = Path(request["output"]), request["stage"]
        output.mkdir(); write(output / "request.json", request)
        inputs = {"source/" + name: file_record(self.root / name) for name in request["source_sha256"]}
        inputs.update({"capacity_plan": file_record(request["capacity_plan_path"]),
            "data/manifest": file_record(Path(request["audit_dir"]) / "manifest.json"),
            "base/config.json": file_record(Path(request["model_snapshot"]) / "config.json")})
        identity = {"inputs": inputs, "parents": {k: file_record(Path(v["artifact_dir"]) / "manifest.json")
                    for k, v in request["parents"].items()}, "config": {"stage": stage}}
        count, updates, synthetic = cs.COUNTS[stage], cs.UPDATES[stage], int(stage in {"sft", "reward", "dpo"})
        result = {"status": "capacity_stage_passed", "stage_capacity_passed": True,
            "capacity_verified": False, "formal_training": False, "stage": stage, "request": request, "pid": pid,
            "started_at_epoch": self.system.child_start, "finished_at_epoch": self.system.child_end,
            "deadline_epoch": request["deadline_epoch"], "actual_optimizer_updates": updates,
            "recovery_optimizer_updates": int(stage == "ppo"), "identity": identity,
            "model": {"device": "cuda:0", "quantization": cs.NF4, "quantized_linear_count": 196,
                      "parameters_selected": 123, "snapshot_manifest_sha256": request["model_manifest_sha256"]},
            "gpu_samples": [{"gpus": [{"index": 0}], "max_allocated_bytes": 100, "max_reserved_bytes": 200}],
            "frozen_full_audits": [{"time_epoch": self.system.child_end,
                "roles": ["reference", "reward"] if stage in {"ppo", "eval"} else ["reference"] if stage == "dpo" else [],
                "cache_manifest_sha256": None, "reward_head_content_verified": stage in {"ppo", "eval"}}],
            "adapter_weights_changed": stage != "eval",
            "scalar_heads_changed": {"reward": True} if stage == "reward" else {"value": True} if stage == "ppo" else {},
            "selection": {"selected_rows": count, "configured_rows": count, "real_rows": count - synthetic,
                "synthetic_rows": synthetic, "accepted_manifest_sha256": request["audit_manifest_sha256"],
                "real_unique_prompts": count, "synthetic_stress": {"prompt_tokens": 1024, "chosen_tokens": 768,
                    "rejected_tokens": None if stage == "sft" else 768, "padding_tokens": 0, "attention_all_true": True}},
            "checkpoints": [], "cursor": {"offset": count, "optimizer_updates": updates, "active_rollout": False,
                "completed_rollouts": 8}, "changed_parameter_names": ["adapter"], "outcome_stop_reason": "data_exhausted"}
        if stage == "eval":
            normal = {"quality_evaluation": False, "capacity_only": True, "termination": "eos",
                      "final_action_flushed_to_kv": True}
            result["generation"] = {"eos_aware": normal,
                "forced_shape": {**normal, "generated_tokens": 2048, "prompt_tokens": 1024,
                    "max_cache_tokens": 3072, "ignore_eos": True, "termination": "forced_max_new_tokens"}}
        else:
            completion = {"stop_reason": "data_exhausted", "data_pass_complete": True,
                "protocol_complete": True, "optimizer_updates": updates, "rows_processed": count}
            artifact = output / "final-artifact"
            digest = bundle(artifact, "weights.pt", identity, format="alignlab-completed-stage-v1",
                            stage=stage, completion=completion)
            result["artifact"] = {"artifact_dir": str(artifact), "manifest_path": str(artifact / "manifest.json"),
                                  "manifest_sha256": digest}
            for update in sorted({1, 2, 256} if stage == "ppo" else {updates}):
                directory = output / f"update-{update:06d}"
                digest = bundle(directory, "state.pt", identity, format="alignlab-optimizer-checkpoint-v1")
                result["checkpoints"].append({"checkpoint_path": str(directory), "manifest_sha256": digest})
            (output / "metrics.jsonl").write_text("".join(json.dumps({"event": "objective", "update": n,
                "objective": .5, "gradient_norm": .2}) + "\n" for n in range(1, updates + 1)))
            if stage == "dpo":
                cache = output / "reference-cache"; cache.mkdir()
                write(cache / "identity.json", {}); (cache / "scores.jsonl").write_text("{}\n")
                write(cache / "manifest.json", {"status": "cache_complete", "format": "alignlab-reference-logprobs-v1",
                    "artifacts": {name: {k: file_record(cache / name)[k] for k in ("sha256", "size_bytes")}
                                  for name in ("identity.json", "scores.jsonl")}})
                result["reference_cache_manifest_sha256"] = cs.sha(cache / "manifest.json")
                result["frozen_full_audits"][-1]["cache_manifest_sha256"] = result["reference_cache_manifest_sha256"]
            if stage == "ppo":
                directory = output / "replayed-update-000002"
                digest = bundle(directory, "state.pt", identity, format="alignlab-optimizer-checkpoint-v1")
                checkpoints = result["checkpoints"]
                result["recovery"] = {"verified": True, "cursor_oldstats_rng_exact": True,
                    "final_artifact_exported_before_replay": True, "resume_checkpoint": checkpoints[0]["manifest_sha256"],
                    "continuous_checkpoint": checkpoints[1]["manifest_sha256"],
                    "parameters_optimizer_rtol": 2e-5, "parameters_optimizer_atol": 2e-6,
                    "replay_checkpoint_path": str(directory), "replay_checkpoint": digest}
                result["synthetic_backward"] = {"parameters_unchanged": True, "optimizer_updates": 0,
                                                "prompt_tokens": 1024, "action_tokens": 256}
        result.update(self.result_changes)
        write(output / "progress.json", result)
        if self.after_publish:
            self.after_publish(output, result)

    def test_plan_only_has_no_output_lock_resource_probe_or_spawn(self):
        result = self.run_campaign(False)
        self.assertEqual(result["status"], "plan_only")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.system.commands)
        self.assertEqual(self.snapshot_mock.call_count, 0)

    def test_busy_before_admission_has_no_output_lock_clock_or_budget(self):
        self.system.initial_busy = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertIsNone(result["started_at_epoch"])
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "reports/.alignlab-capacity.lock").exists())
        self.assertEqual(self.system.clock, 1000)
        self.assertFalse(self.system.commands)

    def test_five_stages_serial_shared_deadline_real_parent_pins_and_wrapper_exit(self):
        self.system.descendant_linger = 3
        result = self.run_campaign()
        self.assertEqual(result["status"], "capacity_passed")
        self.assertTrue(result["capacity_verified"])
        self.assertFalse(result["formal_training"])
        self.assertEqual([r["stage"] for r in result["stages"]], list(cs.STAGES))
        self.assertEqual(len(self.system.commands), 5)
        self.assertEqual({r["request"]["deadline_epoch"] for r in result["stages"]}, {15335})
        self.assertEqual(result["elapsed_seconds"], 25)
        self.assertTrue(all(len(r["known_processes"]) == 2 for r in result["stages"]))
        sft, reward, dpo, ppo, evaluation = result["stages"]
        self.assertEqual(dpo["request"]["parents"]["sft"], {k: sft["artifact"][k] for k in ("artifact_dir", "manifest_sha256")})
        self.assertEqual(ppo["request"]["parents"]["sft"], dpo["request"]["parents"]["sft"])
        self.assertEqual(evaluation["request"]["parents"], ppo["request"]["parents"])
        self.assertFalse((self.root / "reports/.alignlab-capacity.lock").exists())
        with self.assertRaises(FileExistsError):
            self.run_campaign()

    def test_resource_wait_after_start_consumes_same_budget(self):
        def delay(system):
            if system.child and len(system.commands) == 1 and system.clock == system.child_end:
                system.resources_until = system.clock + 7
        self.system.on_sleep = delay
        result = self.run_campaign()
        self.assertEqual(result["started_at_epoch"], 1000)
        self.assertEqual(result["deadline_epoch"], 15335)
        self.assertEqual(result["elapsed_seconds"], 17)
        self.assertEqual(result["stages"][1]["launch_epoch"], 1009)

    def test_budget_expires_during_first_snapshot_no_child_and_no_reset(self):
        def slow(*args):
            self.system.clock += 14336
            return {"repo_id": cs.REPO_ID, "revision": cs.REVISION}
        self.snapshot_mock.side_effect = slow
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(self.system.commands)
        with self.assertRaises(FileExistsError):
            self.run_campaign()

    def test_stop_during_child_signals_owned_campaign_and_waits_natural_exit(self):
        self.system.descendant_linger = 3
        self.system.on_sleep = lambda s: (self.output / "STOP").write_text("user STOP") if not (self.output / "STOP").exists() else None
        result = self.run_campaign()
        self.assertEqual(result["status"], "paused")
        self.assertEqual(self.system.clock, 1005)
        self.assertEqual((self.output / "STOP").read_text(), "user STOP")
        self.assertEqual(len(self.system.commands), 1)

    def test_deadline_during_child_writes_stop_no_kill_or_next_child(self):
        self.system.child_duration = 14340
        self.system.on_sleep = lambda s: setattr(s, "clock", max(s.clock, 15335))
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(len(self.system.commands), 1)
        self.assertGreaterEqual(self.system.clock, 15340)
        self.assertEqual(json.loads((self.output / "STOP").read_text())["reason"], "budget_exhausted")

    def test_source_change_during_child_stops_then_fails_after_exit(self):
        def change(system):
            (self.root / "alignment_lab/capacity_supervisor.py").write_text("# changed")
        self.system.on_sleep = change
        with self.assertRaisesRegex(RuntimeError, "SHA changed"):
            self.run_campaign()
        self.assertEqual(self.system.clock, 1002)
        self.assertEqual(len(self.system.commands), 1)
        self.assertTrue((self.output / "STOP").exists())

    def test_child_failure_never_retries_or_advances(self):
        self.system.child_code = 2
        with self.assertRaisesRegex(RuntimeError, "exited 2"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)
        with self.assertRaises(FileExistsError):
            self.run_campaign()

    def test_pid_reuse_cannot_count_as_original_alive_or_success(self):
        def reuse(system):
            system.process_override[system.child.pid + 1] = {"created_at_utc": stamp(9999)}
        self.system.on_sleep = reuse
        with self.assertRaisesRegex(RuntimeError, "PID reused"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)
        self.assertTrue((self.output / "STOP").exists())

    def test_unknown_existing_locks_preserved_without_campaign_output(self):
        lock = self.root / "reports/.alignlab-capacity.lock"; lock.write_text("foreign")
        with self.assertRaises(FileExistsError):
            self.run_campaign()
        self.assertEqual(lock.read_text(), "foreign")
        self.assertFalse(self.output.exists())

    def test_other_training_or_capacity_process_blocks_before_output(self):
        self.system.extra_processes = [{"pid": 77, "parent_pid": 1, "created_at_utc": stamp(950),
                                       "command_line": "python -m windows.train --updates 2000"}]
        with self.assertRaisesRegex(RuntimeError, "Other training"):
            self.run_campaign()
        self.assertFalse(self.output.exists())

    def test_bad_plan_pin_or_stale_cpu_source_rejected_before_start(self):
        with self.assertRaises(ValueError):
            cs.run_capacity(self.path, "f" * 64, execute=True, system=self.system)
        cpu = Path(self.plan["cpu_check_report"])
        value = json.loads(cpu.read_text()); value["source_sha256"]["alignment_lab/capacity_supervisor.py"] = "f" * 64
        write(cpu, value); self.plan["cpu_check_report_sha256"] = cs.sha(cpu); self.save_plan()
        with self.assertRaisesRegex(ValueError, "source-before"):
            self.run_campaign()
        self.assertFalse(self.output.exists())

    def test_artifact_tamper_or_extra_file_prevents_next_stage(self):
        self.after_publish = lambda output, result: (output / "final-artifact/weights.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA changed"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_wrong_report_pid_request_or_update_cannot_be_accepted(self):
        self.result_changes = {"pid": 44444}
        with self.assertRaisesRegex(RuntimeError, "exact completed"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_prior_probe_charge_cannot_be_forgotten(self):
        initial = Path(self.plan["initial_probe_acceptance"])
        value = json.loads(initial.read_text()); value["timing"]["probe_elapsed_seconds"] = 66
        write(initial, value); self.plan["initial_probe_acceptance_sha256"] = cs.sha(initial); self.save_plan()
        with self.assertRaisesRegex(ValueError, "four-hour"):
            self.run_campaign()
        self.assertFalse(self.output.exists())

    def test_cpu_or_missing_memory_cannot_claim_real_cuda_capacity(self):
        self.result_changes = {"model": {"device": "cpu"}}
        with self.assertRaisesRegex(RuntimeError, "CUDA NF4"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)

    def test_missing_full_frozen_audit_cannot_pass(self):
        self.result_changes = {"frozen_full_audits": []}
        with self.assertRaisesRegex(RuntimeError, "full frozen"):
            self.run_campaign()
        self.assertEqual(len(self.system.commands), 1)


if __name__ == "__main__":
    unittest.main()
