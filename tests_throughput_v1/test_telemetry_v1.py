"""Pending CPU tests for explicit PPO telemetry inheritance; never production data.

Tests are intentionally not run while the original GPU campaign is active.
"""
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests_v2"))

import torch

from alignment_lab import training_telemetry as old
from alignment_lab import throughput_telemetry_v1 as resumed
from alignment_lab.checkpointing import build_identity, load_checkpoint, save_checkpoint
from alignment_lab.alignment_supervisor_r3 import JsonWriter
from alignment_lab.stage_engine import parameter_mapping
from alignment_lab.stage_engine_observed import run_ppo
from test_shared_qwen import fixture
from test_stage_engine import examples, ppo


def append(path, record):
    with Path(path).open("ab") as stream:
        stream.write(old.canonical(record) + b"\n")


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": old.sha(path)}


class ResumedTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU fixture must not inherit initialized CUDA")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = ppo(max_updates=6, deadline_epoch=time.time() + 3600)
        self.identity_config = {
            "stage": "ppo", "scope": "registered_unique_real_prompt_subset", "synthetic_rows": 0,
            "work": asdict(self.config), "registered_workload": {
                "rows": 3, "optimizer_updates": 6, "scope": "registered_unique_real_prompt_subset"},
            "model": {"fixture": "tiny random CPU"}, "selection": {"fixture_rows": 3},
            "optimizer": {"name": "AdamW", "lr": 1e-3}, "execution_revision": "original-fixture"}
        self.files = {"engine": ROOT / "alignment_lab/stage_engine_observed.py",
                      "telemetry": ROOT / "alignment_lab/training_telemetry.py",
                      "test": Path(__file__).resolve()}
        self.identity = build_identity(self.files, self.identity_config)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def initialized(self):
        backend, *_ = fixture()
        mapping = parameter_mapping(backend, "policy", heads=("value",))
        optimizer = torch.optim.AdamW(mapping.values(), lr=1e-3)
        return backend, mapping, optimizer, torch.Generator().manual_seed(321)

    def checkpoint(self, output, writer, identity, mapping, optimizer, generator, cursor, extra):
        name = "update-%06d" % cursor["optimizer_updates"]
        before = time.time()
        sidecar = writer.checkpoint_sidecar(name, cursor, extra)
        saved = save_checkpoint(output / name, identity=identity, parameters=mapping,
            optimizer=optimizer, cursor={**cursor, "telemetry": sidecar},
            generators={"sampling": generator}, extra_state=extra)
        finished = time.time()
        binding = writer.bind_checkpoint(saved, sidecar, started_at_epoch=before, finished_at_epoch=finished)
        return {**saved, "telemetry_sidecar": sidecar, "telemetry_binding": binding,
                "cursor": copy.deepcopy(cursor), "optimizer_updates": cursor["optimizer_updates"]}

    def original(self, *, stop_update=3, stopped_inside_writer=False, hard_block_update=None):
        output = self.root / "original"
        output.mkdir()
        started = time.time()
        writer = old.TrainingTelemetry(output / "telemetry", stage="ppo", started_at_epoch=started)
        writer.set_identity(self.identity)
        backend, mapping, optimizer, generator = self.initialized()
        saved = []
        stopped_payload = {}
        metrics_path = output / "metrics.jsonl"
        stop_file = output / "STOP"
        json_writer = JsonWriter(self.config.deadline_epoch, stop_file.exists)
        guard_result = {}
        def checkpoint(cursor, extra):
            if cursor["optimizer_updates"] == stop_update:
                if stopped_inside_writer:
                    stop_file.write_text("fixture requested stop", encoding="utf-8")
                saved.append(self.checkpoint(output, writer, self.identity, mapping, optimizer, generator, cursor, extra))
                if stopped_inside_writer:
                    append(metrics_path, {"event": "requested_stop_boundary", "cursor": cursor,
                                          "active_rollout_saved": extra is not None})
                    stopped_payload.update(cursor=copy.deepcopy(cursor), extra_state=copy.deepcopy(extra))
                    # Actual R3 failure path: ordinary publication sees STOP and
                    # raises before run_ppo returns or its metrics can be flushed.
                    json_writer(output / "ordinary-progress.json", {"status": "requested_stop_boundary_saved"})
                    self.fail("Original JsonWriter ignored a real STOP marker")
            if cursor["optimizer_updates"] == hard_block_update:
                from alignment_lab.alignment_process_guard_r3 import GuardSession, POLICY
                own = {"pid": 101, "parent_pid": 100, "created_at_utc": "2026-10-21T08:00:00.1234567Z",
                       "command_line": "python scripts/run_alignlab_alignment_stage_r3.py --request fixture --execute"}
                competing = {**own, "pid": 202, "parent_pid": 200,
                             "command_line": "python scripts/run_alignlab_throughput_cpu_checks_v1.py --output fixture"}
                guard = GuardSession(lambda: [own, competing], owned_identities=[own],
                    deadline_epoch=self.config.deadline_epoch, policy=POLICY,
                    on_event=lambda event: append(output / "process-guard.jsonl", event))
                result = guard.check(boundary="optimizer_complete", optimizer_updates=hard_block_update,
                                     save_boundary=lambda: self.fail("Original hard block unexpectedly saved state"))
                self.assertEqual(result["reason"], "hard_blocker")
                self.assertIsNone(result["saved_boundary"])
                guard_result.update(result)
                stopped_payload.update(cursor=copy.deepcopy(cursor), extra_state=copy.deepcopy(extra))
                raise InterruptedError("Strict process boundary blocked: hard_blocker")
            append(metrics_path, {"event": "engine_boundary", "cursor": cursor})
        writer.warm_start()
        try:
            result = run_ppo(backend, "policy", optimizer, examples(), self.config, device="cpu",
                generator=generator, reference_role="reference", reward_role="reward",
                checkpoint_callback=checkpoint, stop_requested=lambda: bool(saved) and hard_block_update is None,
                event_callback=lambda event: writer.observe(event, backend=backend, role="policy"))
        except InterruptedError as exc:
            self.assertTrue(stopped_inside_writer or hard_block_update is not None)
            self.assertIn("Strict process boundary blocked" if hard_block_update else "JSON publication stopped", str(exc))
            self.assertFalse((output / "ordinary-progress.json").exists())
            # Fixture access to the exact callback state, not a fabricated
            # engine-return event or objective record in either ledger.
            result = SimpleNamespace(**stopped_payload, stop_reason="paused", metrics=())
        self.assertEqual(result.stop_reason, "paused")
        if not stopped_inside_writer and hard_block_update is None:
            writer.safe_engine_return(result.cursor, result.extra_state, result.stop_reason)
            for metric in result.metrics:
                append(metrics_path, {"event": "objective_after_engine_return", **metric})
        progress = {"kind": "alignlab-formal-alignment-stage-r3-v1", "stage": "ppo", "status": "paused",
            "safe_resume_available": True, "identity": self.identity, "cursor": result.cursor,
            "actual_optimizer_updates": result.cursor["optimizer_updates"], "latest_safe_checkpoint": saved[-1],
            "checkpoints": saved, "deadline_epoch": self.config.deadline_epoch,
            "request": {"deadline_epoch": self.config.deadline_epoch},
            "started_at_epoch": started, "finished_at_epoch": time.time()}
        if hard_block_update is not None:
            progress.update(error="InterruptedError: Strict process boundary blocked: hard_blocker",
                            process_guard={"last": guard_result})
        if not stopped_inside_writer and hard_block_update is None:
            progress["objective_records"] = len(result.metrics)
        if stopped_inside_writer:
            json_writer(output / "progress.json", progress, terminal=True)
            progress_pin = pin(output / "progress.json")
        else:
            progress_pin = old.write_new(output / "progress.json", progress)
        history = {"prior_progress": progress_pin, "metrics": pin(metrics_path),
            "cost_events": pin(output / "telemetry/cost-events.jsonl"),
            "checkpoint": {"path": saved[-1]["checkpoint_path"], "manifest_sha256": saved[-1]["manifest_sha256"],
                           "sidecar": saved[-1]["telemetry_sidecar"], "binding": saved[-1]["telemetry_binding"]}}
        return history, result

    def authorize_fixture_rollback(self, history):
        """Test-only trust anchors; production accepts only the real user SHA."""
        progress = old.read(history["prior_progress"]["path"])
        committed = progress["latest_safe_checkpoint"]["optimizer_updates"]
        observed = progress["actual_optimizer_updates"]
        incident = {"kind": "alignlab-r3-throughput-preparation-incident-v1", "campaign_status": "failed",
            "observed_optimizer_updates": observed, "last_saved_optimizer_updates": committed,
            "updates_after_last_saved_checkpoint": observed - committed, "saved_boundary_on_hard_block": None,
            "original_deadline_epoch": self.config.deadline_epoch, "checkpoint": history["checkpoint"],
            "pins": {"ppo42/progress.json": history["prior_progress"], "ppo42/metrics.jsonl": history["metrics"],
                     "ppo42/telemetry/cost-events.jsonl": history["cost_events"],
                     "ppo42/process-guard.jsonl": pin(self.root / "original/process-guard.jsonl")}}
        incident_pin = old.write_new(self.root / "incident.json", incident)
        auth = {"kind": "alignlab-explicit-rollback-replay-authorization-v1", "source": "direct_user_reply_in_this_chat",
            "incident": incident_pin, "prior_observed_optimizer_updates": observed, "restore_optimizer_updates": committed,
            "replay_first_update": committed + 1, "replay_last_update": observed, "replay_updates": observed - committed,
            "original_deadline_epoch": self.config.deadline_epoch,
            "original_started_at_epoch": self.config.deadline_epoch - 3600, "wall_seconds": 3600,
            "original_plan": old.write_new(self.root / "original-plan.json", {"wall_seconds": 3600}),
            "throughput_authorization": old.write_new(self.root / "throughput-authorization.json", {"fixture": True})}
        for flag in ("permit_original_backend_continuation",
                     "permit_optimized_backend_only_after_actual_cuda_equivalence_and_measured_speedup",
                     "preserve_failed_history", "replayed_updates_must_be_separately_accounted",
                     "require_new_entry_safe_boundary_save_before_guard_stop", "do_not_change_original_179_sources",
                     "no_new_seed_or_data_or_protocol", "no_parallel_gpu_training", "no_deadline_extension"):
            auth[flag] = True
        history["rollback_authorization"] = old.write_new(self.root / "rollback-authorization.json", auth)
        self.enterContext(patch.object(resumed, "ROLLBACK_AUTHORIZATION_SHA256", history["rollback_authorization"]["sha256"]))
        self.enterContext(patch.object(resumed, "ROLLBACK_INCIDENT_SHA256", incident_pin["sha256"]))
        return history

    def migration(self, history):
        config = copy.deepcopy(self.identity_config)
        config["execution_revision"] = "resumed-fixture"
        identity = build_identity({**self.files, "resume": ROOT / "alignment_lab/throughput_telemetry_v1.py"}, config)
        migration = old.write_new(self.root / "migration.json", {"kind": resumed.MIGRATION_KIND,
            "history_sha256": old.digest(history), "old_identity_sha256": old.digest(self.identity),
            "new_identity_sha256": old.digest(identity), "original_deadline_epoch": self.config.deadline_epoch})
        return identity, migration

    def test_active_rollout_resume_preserves_old_anchors_and_counts_each_update_once(self):
        history, partial = self.original(stopped_inside_writer=True)
        verified = resumed.verify_resume_evidence(history, expected_old_identity_sha256=old.digest(self.identity))
        self.assertTrue(verified["cursor"]["active_rollout"])
        self.assertNotIn("telemetry", verified["cursor"])
        self.assertFalse(verified["engine_return_observed"])
        self.assertEqual(verified["post_engine_metrics"], [])
        self.assertEqual([m["update"] for m in verified["metric_values"]], [1, 2, 3])
        self.assertEqual(verified["metrics_source"]["pin"], history["cost_events"])
        old_bytes = {p: p.read_bytes() for p in (self.root / "original").rglob("*") if p.is_file()}
        identity, migration = self.migration(history)
        output = self.root / "resumed"
        output.mkdir()
        writer = resumed.ResumedTrainingTelemetry(output / "telemetry", history=history,
                                                  migration_pin=migration, new_identity=identity)
        backend, mapping, optimizer, generator = self.initialized()
        loaded = load_checkpoint(history["checkpoint"]["path"], expected_identity=self.identity,
            parameters=mapping, optimizer=optimizer, generators={"sampling": generator},
            expected_manifest_sha256=history["checkpoint"]["manifest_sha256"])
        restored_cursor = dict(loaded["cursor"])
        restored_cursor.pop("telemetry")
        self.assertEqual(restored_cursor, partial.cursor)
        receipts = copy.deepcopy(verified["progress"]["checkpoints"])
        def checkpoint(cursor, extra):
            append(output / "metrics.jsonl", {"event": "engine_boundary", "cursor": cursor})
            if cursor["optimizer_updates"] == 6:
                receipts.append(self.checkpoint(output, writer, identity, mapping, optimizer, generator, cursor, extra))
        writer.warm_start()
        result = run_ppo(backend, "policy", optimizer, examples(), self.config, device="cpu", generator=generator,
            reference_role="reference", reward_role="reward", state=restored_cursor, extra_state=loaded["extra_state"],
            checkpoint_callback=checkpoint, event_callback=lambda e: writer.observe(e, backend=backend, role="policy"))
        writer.safe_engine_return(result.cursor, result.extra_state, result.stop_reason)
        for metric in result.metrics:
            append(output / "metrics.jsonl", {"event": "objective_after_engine_return", **metric})
        completion = {"rows_processed": 3, "optimizer_updates": 6, "response_tokens": result.cursor["response_tokens"],
            "completed_rollouts": result.cursor["completed_rollouts"], "stop_reason": "data_exhausted",
            "data_pass_complete": True, "protocol_complete": True, "full_selected_dataset_pass": True,
            "full_accepted_dataset_pass": False, "synthetic_rows": 0, "scope": "registered_unique_real_prompt_subset"}
        result_pin = writer.finish(cursor=result.cursor, completion=completion, checkpoints=receipts)
        manifest = resumed.verify_resumed_training_telemetry(writer.path,
            expected_manifest_sha256=result_pin["sha256"], expected_identity_sha256=old.digest(identity),
            expected_completion=completion, checkpoints=receipts, expected_metrics_pin=pin(output / "metrics.jsonl"))
        self.assertEqual(manifest["rollouts"][:1], verified["sidecar"]["rollouts"])
        self.assertEqual([r["update"] for r in resumed._jsonl(manifest["new_metrics"])
                          if r["event"] == "objective_after_engine_return"], [4, 5, 6])
        for path, content in old_bytes.items():
            self.assertEqual(path.read_bytes(), content)

    def test_old_post_return_metrics_are_cross_checked_against_actual_observations(self):
        history, _ = self.original()
        result = resumed.verify_resume_evidence(history, expected_old_identity_sha256=old.digest(self.identity))
        self.assertTrue(result["engine_return_observed"])
        self.assertEqual(len(result["post_engine_metrics"]), 3)
        records = resumed._jsonl(history["metrics"])
        next(r for r in records if r["event"] == "objective_after_engine_return")["objective"] += 1
        path = Path(history["metrics"]["path"])
        path.write_bytes(b"".join(old.canonical(r) + b"\n" for r in records))
        history["metrics"] = pin(path)
        with self.assertRaisesRegex(ValueError, "differs from its actual observed"):
            resumed.verify_resume_evidence(history, expected_old_identity_sha256=old.digest(self.identity))

    def test_real_hard_block_lag_requires_exact_authorization_and_retains_abandoned_costs(self):
        history, partial = self.original(stop_update=1, hard_block_update=3)
        old_bytes = {p: p.read_bytes() for p in (self.root / "original").rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "lags or differs"):
            resumed.verify_resume_evidence(history, expected_old_identity_sha256=old.digest(self.identity))
        self.authorize_fixture_rollback(history)
        evidence = resumed.verify_resume_evidence(history, expected_old_identity_sha256=old.digest(self.identity))
        self.assertEqual(evidence["cursor"]["optimizer_updates"], 1)
        self.assertEqual(evidence["progress"]["cursor"]["optimizer_updates"], 3)
        self.assertEqual(evidence["original_observed_optimizer_updates"], 3)
        self.assertEqual([m["update"] for m in evidence["metric_values"]], [1])
        self.assertEqual([m["update"] for m in evidence["replay_reference_metrics"]], [2, 3])
        self.assertEqual(evidence["abandoned_segment"]["updates"], 2)
        self.assertFalse(evidence["engine_return_observed"])
        self.assertEqual(evidence["post_engine_metrics"], [])
        self.assertEqual(len([p for p in evidence["all_points"] if p["event"] == "optimizer_end"]), 3)
        self.assertEqual(len([p for p in evidence["points"] if p["event"] == "optimizer_end"]), 1)
        measured = sum(p["seconds"] for p in evidence["all_points"]
                       if p["event"] == "measured_phase" and p["phase"] == "optimizer_start")
        self.assertEqual(evidence["costs"]["optimizer_active_seconds"], measured)
        accounting = resumed._update_accounting(evidence, 6)
        self.assertEqual(accounting["logical_optimizer_updates"], 6)
        self.assertEqual(accounting["total_physical_optimizer_updates"], 8)
        self.assertEqual(accounting["replayed_optimizer_updates"], 2)
        # Actual saved optimizer/RNG/active rollout replay, with no synthetic
        # observations and no old post-return metrics added to either ledger.
        backend, mapping, optimizer, generator = self.initialized()
        loaded = load_checkpoint(history["checkpoint"]["path"], expected_identity=self.identity,
            parameters=mapping, optimizer=optimizer, generators={"sampling": generator},
            expected_manifest_sha256=history["checkpoint"]["manifest_sha256"])
        cursor = dict(loaded["cursor"])
        cursor.pop("telemetry")
        result = run_ppo(backend, "policy", optimizer, examples(), self.config, device="cpu", generator=generator,
            reference_role="reference", reward_role="reward", state=cursor, extra_state=loaded["extra_state"])
        self.assertEqual(list(result.metrics[:2]), evidence["replay_reference_metrics"])
        for path, content in old_bytes.items():
            self.assertEqual(path.read_bytes(), content)
        changed = copy.deepcopy(history)
        changed["rollback_authorization"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Unrecognized rollback"):
            resumed.verify_resume_evidence(changed, expected_old_identity_sha256=old.digest(self.identity))

    def test_lagging_checkpoint_and_changed_identity_are_rejected(self):
        history, _ = self.original()
        with self.assertRaisesRegex(ValueError, "identities disagree"):
            resumed.verify_resume_evidence(history, expected_old_identity_sha256="0" * 64)
        progress_path = Path(history["prior_progress"]["path"])
        progress = old.read(progress_path)
        progress["actual_optimizer_updates"] += 1
        progress_path.write_bytes(old.canonical(progress))
        history["prior_progress"] = pin(progress_path)
        with self.assertRaisesRegex(ValueError, "lags or differs"):
            resumed.verify_resume_evidence(history, expected_old_identity_sha256=old.digest(self.identity))

    def test_migration_rejects_new_sampling_configuration_or_deadline(self):
        history, _ = self.original()
        identity, migration = self.migration(history)
        for key in ("rollout_batch", "deadline_epoch"):
            with self.subTest(key=key):
                changed = copy.deepcopy(identity)
                changed["config"]["work"][key] += 1
                document = old.read(migration["path"])
                document["new_identity_sha256"] = old.digest(changed)
                temporary_pin = old.write_new(self.root / (key + ".json"), document)
                with self.assertRaisesRegex(ValueError, "frozen numerical configuration"):
                    resumed._migration(temporary_pin, history, self.identity, changed)

    def test_missing_or_duplicate_real_optimizer_update_is_rejected(self):
        history, _ = self.original(stopped_inside_writer=True)
        points = resumed._jsonl(history["cost_events"])
        for broken in ([p for p in points if not (p["event"] == "optimizer_end" and p["update"] == 2)],
                       points + [next(p for p in points if p["event"] == "optimizer_end")]):
            with self.assertRaisesRegex(ValueError, "optimizer ends"):
                resumed._metrics([], 1, 3, broken, allow_observed_only=True)

    def test_open_actual_phase_is_rejected_without_inventing_a_close(self):
        history, _ = self.original(stopped_inside_writer=True)
        result = resumed.verify_resume_evidence(history, expected_old_identity_sha256=old.digest(self.identity))
        points = copy.deepcopy(result["points"])
        last = points[-1]["time_epoch"]
        points.append({"event": "optimizer_start", "time_epoch": last, "boundary_epoch": last, "target_update": 4})
        with self.assertRaisesRegex(ValueError, "Open observed phase"):
            resumed._cost_segment(points, result["bindings"], started=result["progress"]["started_at_epoch"],
                finished=result["progress"]["finished_at_epoch"], warm_started=result["warm_started_at_epoch"],
                allow_checkpoint_stop=True)


if __name__ == "__main__":
    unittest.main()
