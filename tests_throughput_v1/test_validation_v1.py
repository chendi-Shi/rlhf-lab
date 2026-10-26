"""CPU validation of the disposable CUDA replay controller; no CUDA claim."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from alignment_lab.checkpointing import build_identity, load_checkpoint, save_checkpoint
from alignment_lab.stage_engine import PPOConfig, parameter_mapping
from alignment_lab.stage_engine_observed import run_ppo
from alignment_lab.throughput_sampling_v1 import make_sampling_backend
from alignment_lab.throughput_validation_v1 import (
    checked_guard_boundary, decide, equivalence_projection, numeric_active, replay_segment, replay_target,
    state_digest, validate_cpu_report, validate_rollback_authorization, STARTED, DEADLINE,
)
from tests_throughput_v1.test_sampling_v1 import fixture


def config():
    return PPOConfig(seed=42, rollout_batch=16, optimizer_batch=1, epochs=2,
        max_responses=48, max_updates=96, max_new_tokens=3,
        deadline_epoch=time.time() + 1800, pad_id=0, eos_id=2, chunk_tokens=2)


def examples():
    return [{"record_id": f"validation-fixture-{index}", "prompt_ids": (4, 8 + index % 5, 9)}
            for index in range(48)]


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("Only CPU software validation is allowed")

    def test_state_digest_is_bitwise_and_covers_tensor_dtype_shape_and_all_rng(self):
        self.assertNotEqual(state_digest(torch.tensor([0.])), state_digest(torch.tensor([-0.])))
        self.assertNotEqual(state_digest(0.), state_digest(-0.))
        self.assertNotEqual(state_digest(torch.tensor([1], dtype=torch.int32)), state_digest(torch.tensor([1], dtype=torch.int64)))
        self.assertNotEqual(state_digest(torch.ones(1, 2)), state_digest(torch.ones(2, 1)))
        self.assertNotEqual(state_digest([1]), state_digest((1,)))
        self.assertNotEqual(state_digest({1: "a"}), state_digest({"1": "a"}))
        self.assertEqual(state_digest(torch.tensor(7.)), state_digest(torch.tensor(7.)))
        with self.assertRaises(ValueError):
            state_digest(float("nan"))

    def test_cpu_gate_requires_actual_full_suite_and_exact_sources(self):
        sources = {"x.py": "a" * 64}
        report = {"kind": "alignlab-throughput-cpu-validation-v1", "status": "passed", "device": "cpu",
            "cuda_initialized": False, "tests_run": 20, "failures": 0, "errors": 0, "skipped": 0,
            "source_before": sources, "source_after": sources,
            "modules": ["tests_throughput_v1.test_" + name + "_v1" for name in ("stage", "telemetry", "sampling", "validation")]}
        self.assertTrue(validate_cpu_report(report, sources))
        for key, value in (("cuda_initialized", True), ("skipped", 1), ("tests_run", 0),
                           ("status", "failed"), ("modules", ["test_sampling_v1"]), ("source_after", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cpu_report({**report, key: value}, sources)

    def test_numeric_projection_only_excludes_ephemeral_origin_identity(self):
        old = {"responses": [{"old_values": [0.]}], "observation_origin": {
            "generation_model_versions": [["a", 1]], "model_versions": [[123, 1]],
            "origin_sha256": "x", "rng_after_hex": "0001", "record_ids": ["a"]}}
        new = copy.deepcopy(old)
        new["observation_origin"].update(generation_model_versions=[["a", 99]], model_versions=[[456, 99]], origin_sha256="y")
        self.assertEqual(state_digest(numeric_active(old)), state_digest(numeric_active(new)))
        new["observation_origin"]["rng_after_hex"] = "0002"
        self.assertNotEqual(state_digest(numeric_active(old)), state_digest(numeric_active(new)))
        self.assertEqual(old["observation_origin"]["origin_sha256"], "x")

    def test_target_is_next_exact_sixteen_real_prompts_not_active_replay(self):
        rows = examples()
        cursor = {"offset": 16, "order": list(reversed(range(48))), "completed_rollouts": 0, "active_rollout": True}
        target = replay_target(cursor, config(), rows)
        self.assertEqual(target["rollout_index"], 1)
        self.assertEqual(target["completed_rollouts"], 2)
        self.assertEqual(target["record_ids"], [rows[i]["record_id"] for i in cursor["order"][16:32]])
        with self.assertRaises(ValueError):
            replay_target({**cursor, "offset": 40}, config(), rows)

    def test_hard_guard_denial_or_error_saves_completed_validation_boundary(self):
        calls = []
        class Guard:
            def check(self, **kwargs):
                return {"admission_allowed": False, "reason": "competing_process"}
        with self.assertRaisesRegex(InterruptedError, "after saving"):
            checked_guard_boundary(Guard(), name="optimizer", update=630,
                save=lambda: calls.append(630) or {"complete": True})
        self.assertEqual(calls, [630])
        class ErrorGuard:
            def check(self, **kwargs):
                raise RuntimeError("snapshot unavailable")
        with self.assertRaisesRegex(RuntimeError, "snapshot unavailable"):
            checked_guard_boundary(ErrorGuard(), name="optimizer", update=631,
                save=lambda: calls.append(631) or {"complete": True})
        self.assertEqual(calls, [630, 631])
        class WaitThenDeny:
            def check(self, **kwargs):
                kwargs["save_boundary"]()
                return {"admission_allowed": False, "reason": "deadline_exhausted"}
        with self.assertRaises(InterruptedError):
            checked_guard_boundary(WaitThenDeny(), name="optimizer", update=632,
                save=lambda: calls.append(632) or {"complete": True})
        self.assertEqual(calls, [630, 631, 632])

    def test_explicit_rollback_preserves_observed_630_and_restores_only_609(self):
        record = lambda name: {"path": name, "sha256": "a" * 64}
        checkpoint = {"path": "checkpoint609", "manifest_sha256": "b" * 64,
                      "sidecar": record("sidecar"), "binding": record("binding")}
        history = {"rollback_authorization": record("auth"), "prior_progress": record("progress"),
                   "metrics": record("metrics"), "cost_events": record("events"), "checkpoint": checkpoint}
        old = {"alignment_plan_path": "old-plan", "alignment_plan_sha256": "a" * 64}
        throughput = record("throughput")
        auth = {"kind": "alignlab-explicit-rollback-replay-authorization-v1", "source": "direct_user_reply_in_this_chat",
            "user_selected_answer": "允许从609恢复并重算21步，验证后继续（推荐）", "original_plan": record("old-plan"),
            "throughput_authorization": throughput, "original_started_at_epoch": STARTED,
            "original_deadline_epoch": DEADLINE, "wall_seconds": 259200, "restore_optimizer_updates": 609,
            "prior_observed_optimizer_updates": 630, "replay_first_update": 610, "replay_last_update": 630,
            "replay_updates": 21, "incident": record("incident")}
        auth.update({name: True for name in ("permit_original_backend_continuation",
            "permit_optimized_backend_only_after_actual_cuda_equivalence_and_measured_speedup",
            "preserve_failed_history", "replayed_updates_must_be_separately_accounted",
            "require_new_entry_safe_boundary_save_before_guard_stop", "do_not_change_original_179_sources",
            "no_new_seed_or_data_or_protocol", "no_parallel_gpu_training", "no_deadline_extension")})
        prior = {"actual_optimizer_updates": 630, "cursor": {"optimizer_updates": 630},
                 "latest_safe_checkpoint": {"optimizer_updates": 609, "cursor": {"optimizer_updates": 609}}}
        incident = {"observed_optimizer_updates": 630, "last_saved_optimizer_updates": 609,
            "campaign_status": "failed", "updates_after_last_saved_checkpoint": 21, "checkpoint": checkpoint,
            "pins": {"ppo42/progress.json": history["prior_progress"], "ppo42/metrics.jsonl": history["metrics"],
                     "ppo42/telemetry/cost-events.jsonl": history["cost_events"]}}
        objects = {"auth": auth, "incident": incident}
        with patch("alignment_lab.throughput_validation_v1.pin", side_effect=lambda value: value["path"]), \
             patch("alignment_lab.throughput_validation_v1.read", side_effect=lambda path: objects[path]):
            self.assertEqual(validate_rollback_authorization(history, prior, old, throughput), auth)
            self.assertEqual(prior["cursor"]["optimizer_updates"], 630)
            auth["restore_optimizer_updates"] = 610
            with self.assertRaises(ValueError):
                validate_rollback_authorization(history, prior, old, throughput)

    def test_real_tiny_qwen_full_checkpoint_replay_covers_active_and_next_rollout(self):
        backend, _, _ = fixture()
        mapping = parameter_mapping(backend, "policy", heads=("value",))
        optimizer = torch.optim.AdamW(mapping.values(), lr=1e-6)
        generator = torch.Generator().manual_seed(42)
        cfg, rows = config(), examples()
        boundaries = []
        partial = run_ppo(backend, "policy", optimizer, rows, cfg, device="cpu", generator=generator,
            reference_role="reference", reward_role="reward",
            checkpoint_callback=lambda c, a: boundaries.append(c), stop_requested=lambda: len(boundaries) == 3)
        self.assertEqual(partial.stop_reason, "paused")
        self.assertTrue(partial.cursor["active_rollout"])
        self.assertEqual(partial.extra_state["offset"], 3)
        root = Path(__file__).resolve().parents[1]
        identity = build_identity({"validation": root / "alignment_lab/throughput_validation_v1.py",
                                   "test": Path(__file__).resolve()}, {"cpu_fixture": True, "config": asdict(cfg)})
        with tempfile.TemporaryDirectory() as directory:
            saved = save_checkpoint(Path(directory) / "boundary", identity=identity, parameters=mapping,
                optimizer=optimizer, cursor=partial.cursor, generators={"sampling": generator}, extra_state=partial.extra_state)
            candidate = make_sampling_backend(backend)
            target = replay_target(partial.cursor, cfg, rows)
            results = []
            origin = {"parameters": state_digest(mapping), "adam": state_digest(optimizer.state_dict())}
            for owner in (backend, candidate):
                restored = load_checkpoint(saved["checkpoint_path"], expected_identity=identity, parameters=mapping,
                    optimizer=optimizer, generators={"sampling": generator}, expected_manifest_sha256=saved["manifest_sha256"])
                self.assertEqual(origin, {"parameters": state_digest(mapping), "adam": state_digest(optimizer.state_dict())})
                result = replay_segment(owner, "policy", optimizer, rows, cfg, device="cpu", generator=generator,
                    state=restored["cursor"], extra_state=restored["extra_state"], target=target)
                results.append(result)
            self.assertEqual(state_digest(equivalence_projection(results[0])), state_digest(equivalence_projection(results[1])))
            self.assertEqual(results[0]["trial_optimizer_updates"], 61)
            self.assertEqual(results[0]["final_cursor"]["optimizer_updates"], 64)
            self.assertEqual(results[0]["formal_optimizer_updates_committed"], 0)
            self.assertEqual(len(results[0]["rollouts"][0]["evidence"]["responses"]), 16)
            load_checkpoint(saved["checkpoint_path"], expected_identity=identity, parameters=mapping,
                optimizer=optimizer, generators={"sampling": generator}, expected_manifest_sha256=saved["manifest_sha256"])
            self.assertEqual(origin, {"parameters": state_digest(mapping), "adam": state_digest(optimizer.state_dict())})
        self.assertFalse(torch.cuda.is_initialized())

    def test_gate_rejects_unequal_trajectory_wait_contamination_and_slower_candidate(self):
        equal = {"boundaries": [], "rollouts": [], "metrics": [], "final_cursor": {},
            "final_parameters_sha256": "a", "final_adam_sha256": "b", "final_all_rng_sha256": "c",
            "trial_optimizer_updates": 32, "sampling_seconds": 8.0, "wall_seconds": 10.0}
        trials = [{"backend": name, "result": {**copy.deepcopy(equal), "wall_seconds": seconds}}
            for name, seconds in (("original", 10), ("deferred_scalar", 9), ("deferred_scalar", 9), ("original", 10))]
        self.assertEqual(decide(trials, ["deferred_scalar"], 1.05)["selected_backend"], "deferred_scalar")
        for key, value in (("final_adam_sha256", "changed"), ("process_wait_seconds", 1), ("wall_seconds", 30)):
            altered = copy.deepcopy(trials)
            altered[1]["result"][key] = value
            self.assertEqual(decide(altered, ["deferred_scalar"], 1.05)["selected_backend"], "original")
        altered = copy.deepcopy(trials)
        altered[-1]["result"]["final_all_rng_sha256"] = "nonrepeatable-original"
        self.assertFalse(decide(altered, ["deferred_scalar"], 1.05)["baseline_repeatability_passed"])


if __name__ == "__main__":
    unittest.main()
