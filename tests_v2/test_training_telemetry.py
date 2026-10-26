"""Real tiny random CPU rollout evidence; no production model or GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from alignment_lab import training_telemetry as t
from alignment_lab.alignment_stage import BoundaryCheckpoints
from alignment_lab.stage_engine_observed import run_ppo, event_to_plain
from test_stage_artifacts import backend_for
from test_stage_engine import examples, ppo


class TrainingTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.backend, _ = backend_for("ppo")
        self.identity = {"fixture": "tiny CPU observed evidence"}
        self.writer = t.TrainingTelemetry(self.root / "telemetry", stage="ppo", started_at_epoch=time.time())
        self.writer.set_identity(self.identity)
        self.optimizer = torch.optim.AdamW(self.backend.parameters_for("policy", heads=("value",)), lr=1e-3)
        self.generator = torch.Generator().manual_seed(51)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def run_tiny(self, before_event=None):
        def observe(event):
            if before_event:
                before_event(event_to_plain(event))
            self.writer.observe(event, backend=self.backend, role="policy")
        self.writer.warm_start()
        result = run_ppo(self.backend, "policy", self.optimizer, examples(), ppo(), device="cpu",
            generator=self.generator, reference_role="reference", reward_role="reward",
            clock=lambda: 100, event_callback=observe)
        self.writer.point("training_engine_return")
        return result

    def finish(self, result):
        completion = {"rows_processed": 3, "response_tokens": result.cursor["response_tokens"]}
        pin = self.writer.finish(cursor=result.cursor, completion=completion)
        return t.verify_training_telemetry(self.writer.path, expected_manifest_sha256=pin["sha256"],
            expected_identity_sha256=t.digest(self.identity), expected_completion=completion), pin

    def test_real_rollout_commits_before_update_with_exact_generation_hash_and_raw_stats(self):
        expected, starts, collections = {}, [], []
        def inspect(e):
            if e["event"] == "rollout_start":
                expected[e["rollout_index"]] = t.policy_anchor(self.backend, "policy",
                    update=e["cursor"]["optimizer_updates"], identity_sha256=t.digest(self.identity))
                starts.append(e["cursor"]["optimizer_updates"])
            if e["event"] == "rollout_collected":
                collections.append(copy.deepcopy(e))
            if e["event"] == "optimizer_start":
                # The event is before any backward; immutable evidence must already exist.
                self.assertTrue((self.writer.path / "rollouts" / f"rollout-{e['rollout_index']:06d}.json").is_file())
                self.assertTrue(all(p.grad is None for p in self.backend._all_parameters()))
        result = self.run_tiny(inspect)
        manifest, _ = self.finish(result)
        self.assertEqual(starts, [0, 4])
        self.assertEqual(len(manifest["rollouts"]), 2)
        for index, pin in enumerate(manifest["rollouts"]):
            raw = t.read(pin["path"])
            anchor = t.read(raw["generation_anchor"]["path"])
            self.assertEqual(anchor["adapter_sha256"], expected[index]["adapter_sha256"])
            self.assertEqual(raw["responses"], collections[index]["responses"])
            self.assertEqual(raw["active"]["reward_scores"], collections[index]["reward_scores"])
            self.assertEqual(raw["active"]["reference_logprobs"], collections[index]["reference_logprobs"])
            self.assertEqual(raw["active"]["observation_origin"], collections[index]["observation_origin"])
        self.assertNotEqual(expected[0]["adapter_sha256"], expected[1]["adapter_sha256"])
        self.assertGreater(manifest["costs"]["sampling_seconds"], 0)
        self.assertGreater(manifest["costs"]["frozen_scoring_seconds"], 0)
        self.assertGreater(manifest["costs"]["optimizer_active_seconds"], 0)
        self.assertEqual(sum(t.read(p["path"])["summary"]["responses"] for p in manifest["rollouts"]), 3)

    def test_rollout_tamper_untracked_file_and_manifest_pin_rejected(self):
        manifest, pin = self.finish(self.run_tiny())
        raw = Path(manifest["rollouts"][0]["path"])
        payload = raw.read_bytes()
        raw.write_bytes(payload + b" ")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            t.verify_training_telemetry(self.writer.path, expected_manifest_sha256=pin["sha256"],
                                        expected_identity_sha256=t.digest(self.identity))
        raw.write_bytes(payload)
        (self.writer.path / "untracked.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Untracked"):
            t.verify_training_telemetry(self.writer.path, expected_manifest_sha256=pin["sha256"],
                                        expected_identity_sha256=t.digest(self.identity))
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            t.verify_training_telemetry(self.writer.path, expected_manifest_sha256="0" * 64,
                                        expected_identity_sha256=t.digest(self.identity))

    def test_missing_active_generation_anchor_never_inferred_and_partial_never_complete(self):
        cursor = {"completed_rollouts": 0, "optimizer_updates": 0, "active_rollout": True}
        with self.assertRaisesRegex(ValueError, "original generation"):
            self.writer.checkpoint_sidecar("unsafe", cursor, {"responses": [], "record_ids": []})
        with self.assertRaises(ValueError):
            self.writer.finish(cursor=cursor, completion={})
        self.assertFalse((self.writer.path / "manifest.json").exists())
        with self.assertRaises(FileExistsError):
            t.TrainingTelemetry(self.writer.path, stage="ppo", started_at_epoch=time.time())

    def test_original_rng_metadata_and_nonfinite_summary_are_rejected(self):
        manifest, _ = self.finish(self.run_tiny())
        raw = t.read(manifest["rollouts"][0]["path"])
        broken = copy.deepcopy(raw)
        broken["active"]["reward_scores"][0] = float("nan")
        with self.assertRaises(ValueError):
            t.rollout_summary(broken)
        broken = copy.deepcopy(raw)
        broken["active"]["observation_origin"]["rng_before_hex"] = "0001"
        with self.assertRaisesRegex(ValueError, "collection identity"):
            t.verify_rollout_origin(broken, t.read(raw["generation_anchor"]["path"]))
        broken = copy.deepcopy(raw)
        broken["responses"][0]["action_ids"][0] = True
        with self.assertRaises(ValueError):
            t.rollout_summary(broken)

    def test_cost_proof_rejects_duplicate_missing_and_detached_phase_intervals(self):
        manifest, pin = self.finish(self.run_tiny())
        points = [__import__("json").loads(line) for line in
                  (self.writer.path / "cost-events.jsonl").read_text().splitlines()]
        kwargs = dict(started=manifest["started_at_epoch"], finished=manifest["finished_at_epoch"],
                      warm_started=manifest["warm_training_started_at_epoch"])
        index = next(i for i, p in enumerate(points) if p["event"] == "measured_phase")
        for mode in ("duplicate", "missing", "detached"):
            changed = copy.deepcopy(points)
            if mode == "duplicate":
                changed.insert(index, copy.deepcopy(changed[index]))
            elif mode == "missing":
                changed.pop(index)
            else:
                changed[index]["phase_id"] += 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                t.cost_summary(changed, [], **kwargs)
        changed = copy.deepcopy(manifest)
        changed["costs"]["optimizer_active_seconds"] += 3
        (self.writer.path / "manifest.json").write_bytes(t.canonical(changed))
        with self.assertRaisesRegex(ValueError, "cost summary"):
            t.verify_training_telemetry(self.writer.path,
                expected_manifest_sha256=t.sha(self.writer.path / "manifest.json"),
                expected_identity_sha256=t.digest(self.identity))

    def test_entire_sampling_or_scoring_interval_cannot_be_removed_and_rehashed(self):
        manifest, _ = self.finish(self.run_tiny())
        events_path = self.writer.path / "cost-events.jsonl"
        points = [json.loads(line) for line in events_path.read_text().splitlines()]
        for phase in ("sampling", "frozen_scoring"):
            start, end = phase + "_start", phase + "_end"
            def omitted(point):
                return ((point["event"] in (start, end) and point["rollout_index"] == 0)
                        or (point["event"] in ("phase_interval_start", "measured_phase")
                            and point["phase"] == start and point["phase_id"] == 0))
            changed = [point for point in points if not omitted(point)]
            self.assertEqual(len(points) - len(changed), 4)
            events_path.write_bytes(b"".join(t.canonical(point) + b"\n" for point in changed))
            forged = copy.deepcopy(manifest)
            forged["files"]["cost-events.jsonl"] = {"sha256": t.sha(events_path),
                                                    "size_bytes": events_path.stat().st_size}
            forged["costs"] = t.cost_summary(changed, [], started=manifest["started_at_epoch"],
                finished=manifest["finished_at_epoch"], warm_started=manifest["warm_training_started_at_epoch"])
            manifest_path = self.writer.path / "manifest.json"
            manifest_path.write_bytes(t.canonical(forged))
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "Rollout lifecycle"):
                t.verify_training_telemetry(self.writer.path, expected_manifest_sha256=t.sha(manifest_path),
                    expected_identity_sha256=t.digest(self.identity))
    def test_cold_cache_and_warm_bind_are_distinct_measured_intervals(self):
        now = [100.0]
        writer = t.TrainingTelemetry(self.root / "cost-fixture", stage="dpo",
                                     started_at_epoch=90, clock=lambda: now[0])
        writer.set_identity(self.identity)
        for at, name in ((100, "cold_cache_start"), (103, "cold_cache_prepared"),
                         (113, "cold_cache_scoring_complete"), (118, "cold_cache_bind_complete")):
            now[0] = at
            writer.point(name)
        writer.warm_start()
        now[0] = 121
        writer.point("training_engine_return")
        pin = writer.finish(cursor={"active_rollout": False, "completed_rollouts": 0}, completion={})
        costs = t.read(pin["path"])["costs"]
        self.assertEqual([costs[k] for k in ("cold_cache_prepare_seconds", "cold_cache_scoring_seconds",
            "cold_cache_commit_bind_seconds", "cold_cache_total_seconds", "engine_wall_seconds")], [3, 10, 5, 18, 3])
        self.assertNotIn("warm_cache_bind_seconds", costs)

    def test_same_cursor_time_thresholds_merge_without_faking_original_save_time(self):
        schedule = BoundaryCheckpoints(1000)
        cursor = {"optimizer_updates": 1}
        writer = Mock(return_value={"checkpoint_path": "/fixture", "manifest_sha256": "a" * 64})
        first = schedule.save(cursor, 20, warm_elapsed=10, writer=writer, clock=lambda: 100)
        later = schedule.save(cursor, 7202, warm_elapsed=3603, writer=writer, clock=lambda: 200)
        self.assertIs(first, later)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual((later["save_started_at_epoch"], later["save_finished_at_epoch"]), (100, 100))
        self.assertIn("cost_2h", later["reasons"])
        self.assertIn("warm_cost_1h", later["reasons"])
        self.assertEqual(later["cost_observations"][-1]["observed_at_epoch"], 200)
        self.assertEqual(later["cost_observations"][-1]["thresholds"][-1]["overshoot_seconds"], 3)
        self.assertTrue(later["cost_observations"][-1]["reused_same_cursor"])


if __name__ == "__main__":
    unittest.main()
