"""CPU proofs for explicit state restoration and non-replayed update accounting.

These fixtures are not production CUDA-equivalence or speedup evidence.  The
parent coordinates when to run them; importing this file launches no training.
"""
import copy
from dataclasses import asdict
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests_v2"))
from test_shared_qwen import fixture
from test_stage_engine import examples, ppo

from alignment_lab.checkpointing import build_identity, save_checkpoint, _capture_rng
from alignment_lab.stage_engine import parameter_mapping
from alignment_lab.stage_engine_observed import run_ppo
from alignment_lab.training_telemetry import write_new, digest
from alignment_lab.throughput_stage_v1 import (joined_metrics, restore_boundary, _owned, validate_request,
    verify_migration_payload, require_saved_guard_boundary, update_accounting, replay_comparison)


def assert_tree(test, left, right):
    test.assertIs(type(left), type(right))
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(list(left), list(right))
        for k in left:
            assert_tree(test, left[k], right[k])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree(test, a, b)
    else:
        test.assertEqual(left, right)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU proof must not initialize CUDA")

    def initialized(self):
        backend, _, _, _, _, _ = fixture()
        random.seed(1729)
        np.random.seed(1730)
        mapping = parameter_mapping(backend, "policy", heads=("value",))
        return backend, mapping, torch.optim.AdamW(mapping.values(), lr=1e-3), torch.Generator().manual_seed(321)

    def run_engine(self, backend, optimizer, generator, config, **kw):
        return run_ppo(backend, "policy", optimizer, examples(), config, device="cpu", generator=generator,
            reference_role="reference", reward_role="reward", clock=lambda: 100.0, **kw)

    def identity(self, config, directory):
        artifact = Path(directory) / "input.txt"
        artifact.write_text("frozen tiny CPU experiment", encoding="utf-8")
        return build_identity({"fixture": artifact, "runner": ROOT / "alignment_lab/throughput_stage_v1.py",
            "engine": ROOT / "alignment_lab/stage_engine_observed.py"}, {"fixture_only": True, "work": asdict(config)})

    def test_active_rollout_restores_all_rng_adam_and_exact_remaining_steps(self):
        config = ppo()
        baseline, base_map, base_opt, base_gen = self.initialized()
        full = self.run_engine(baseline, base_opt, base_gen, config)
        baseline_rng = _capture_rng({"sampling": base_gen})
        part, part_map, part_opt, part_gen = self.initialized()
        boundaries = []
        partial = self.run_engine(part, part_opt, part_gen, config,
            checkpoint_callback=lambda c, e: boundaries.append(c), stop_requested=lambda: len(boundaries) == 3)
        self.assertEqual(partial.stop_reason, "paused")
        self.assertTrue(partial.cursor["active_rollout"])
        self.assertEqual((partial.extra_state["epoch"], partial.extra_state["offset"]), (1, 1))
        checkpoint_rng = _capture_rng({"sampling": part_gen})
        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity(config, directory)
            sidecar = write_new(Path(directory) / "sidecar.json", {"cursor": partial.cursor})
            saved = save_checkpoint(Path(directory) / "checkpoint", identity=identity, parameters=part_map,
                optimizer=part_opt, cursor={**partial.cursor, "telemetry": sidecar}, generators={"sampling": part_gen},
                extra_state=partial.extra_state)
            checkpoint = {"path": saved["checkpoint_path"], "manifest_sha256": saved["manifest_sha256"], "sidecar": sidecar}
            target, mapping, optimizer, generator = self.initialized()
            torch.rand(9)
            random.random()
            np.random.random(9)
            generator.manual_seed(999)
            restored = restore_boundary(checkpoint, identity, parameters=mapping, optimizer=optimizer, generator=generator)
            assert_tree(self, _capture_rng({"sampling": generator}), checkpoint_rng)
            self.assertNotIn("telemetry", restored["cursor"])
            assert_tree(self, restored["extra_state"], partial.extra_state)
            self.assertEqual(restored["cursor"], partial.cursor)
            final = self.run_engine(target, optimizer, generator, config, state=restored["cursor"], extra_state=restored["extra_state"])
        self.assertEqual(final.metrics[0]["update"], 4)
        self.assertEqual(final.metrics, full.metrics[3:])
        self.assertEqual(final.cursor, full.cursor)
        self.assertEqual(joined_metrics(partial.metrics, final.metrics, 3, final.cursor["optimizer_updates"]), full.metrics)
        for name in mapping:
            torch.testing.assert_close(mapping[name], base_map[name], rtol=0, atol=0)
        assert_tree(self, optimizer.state_dict(), base_opt.state_dict())
        assert_tree(self, _capture_rng({"sampling": generator}), baseline_rng)

    def test_checkpoint_identity_relabeling_is_rejected_without_target_mutation(self):
        config = ppo()
        source, mapping, optimizer, generator = self.initialized()
        stopped = []
        partial = self.run_engine(source, optimizer, generator, config,
            checkpoint_callback=lambda c, e: stopped.append(c), stop_requested=lambda: bool(stopped))
        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity(config, directory)
            sidecar = write_new(Path(directory) / "sidecar.json", {"cursor": partial.cursor})
            saved = save_checkpoint(Path(directory) / "checkpoint", identity=identity, parameters=mapping,
                optimizer=optimizer, cursor={**partial.cursor, "telemetry": sidecar}, generators={"sampling": generator}, extra_state=partial.extra_state)
            target, targets, target_opt, target_gen = self.initialized()
            parameters_before = {k: p.detach().clone() for k, p in targets.items()}
            rng_before = _capture_rng({"sampling": target_gen})
            incorrect = copy.deepcopy(identity)
            incorrect["config"]["purpose"] = "new identity cannot replace old validation"
            with self.assertRaisesRegex(ValueError, "identity/config/parent"):
                restore_boundary({"path": saved["checkpoint_path"], "manifest_sha256": saved["manifest_sha256"], "sidecar": sidecar},
                    incorrect, parameters=targets, optimizer=target_opt, generator=target_gen)
            for name in targets:
                torch.testing.assert_close(targets[name], parameters_before[name], rtol=0, atol=0)
            self.assertEqual(len(target_opt.state), 0)
            assert_tree(self, _capture_rng({"sampling": target_gen}), rng_before)

    def test_accounting_rejects_replayed_and_skipped_updates(self):
        previous = ({"update": 1}, {"update": 2})
        self.assertEqual(joined_metrics(previous, ({"update": 3},), 2, 3), previous + ({"update": 3},))
        for wrong in (({"update": 2}, {"update": 3}), ({"update": 4},), ()):
            with self.assertRaisesRegex(ValueError, "duplicated, skipped or replayed"):
                joined_metrics(previous, wrong, 2, 3)

    def test_rollback_logical_and_physical_work_are_separate(self):
        before = update_accounting(609, 630, 609)
        self.assertEqual((before["logical_optimizer_updates"], before["physical_optimizer_updates"],
                          before["replayed_optimizer_updates"]), (609, 630, 0))
        replayed = update_accounting(609, 630, 630)
        self.assertEqual((replayed["physical_optimizer_updates"], replayed["replayed_optimizer_updates"]), (651, 21))
        final = update_accounting(609, 630, 2048)
        self.assertEqual((final["logical_optimizer_updates"], final["physical_optimizer_updates"],
                          final["new_optimizer_updates"], final["replayed_optimizer_updates"]), (2048, 2069, 1439, 21))
        for counters in ((609, 608, 609), (609, 630, 608), (True, 630, 630)):
            with self.assertRaises(ValueError):
                update_accounting(*counters)

    def test_guard_refusal_saves_once_and_never_swallows_save_failure(self):
        class Guard:
            def __init__(self, reason, wait=False):
                self.reason, self.wait = reason, wait

            def check(self, *, boundary, optimizer_updates, save_boundary):
                if self.wait:
                    try:
                        save_boundary()
                    except Exception:
                        return {"admission_allowed": False, "reason": "callback_failed"}
                return {"admission_allowed": self.reason == "cleared", "reason": self.reason}

        for reason in ("hard_blocker", "stop_requested", "deadline_exhausted", "identity_changed"):
            for wait in (False, True):
                calls, stopped = [], []
                def save():
                    calls.append("saved")
                    return {"complete": True, "optimizer_updates": 630}
                error = TimeoutError if reason == "deadline_exhausted" else InterruptedError
                with self.assertRaises(error):
                    require_saved_guard_boundary(Guard(reason, wait), "optimizer_complete", 630, save,
                        on_stop=lambda result, receipt: stopped.append((result["reason"], receipt)))
                self.assertEqual(calls, ["saved"])
                self.assertEqual(stopped[0][0], reason)
        for wait in (False, True):
            calls = []
            def failed_save():
                calls.append("attempt")
                raise OSError("real checkpoint write failed")
            with self.assertRaisesRegex(OSError, "real checkpoint write failed"):
                require_saved_guard_boundary(Guard("hard_blocker", wait), "optimizer_complete", 630, failed_save)
            self.assertEqual(calls, ["attempt"])
        calls = []
        require_saved_guard_boundary(Guard("cleared"), "optimizer_complete", 630, lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_hard_guard_stop_commits_current_real_weights_adam_rng_and_rollout(self):
        config = ppo()
        backend, mapping, optimizer, generator = self.initialized()
        seen, saved = [], []
        class HardGuard:
            def check(self, **kwargs):
                return {"admission_allowed": False, "reason": "hard_blocker"}
        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity(config, directory)
            def checkpoint(cursor, active):
                seen.append(cursor["optimizer_updates"])
                def save():
                    sidecar = write_new(Path(directory) / "sidecar.json", {"cursor": cursor})
                    receipt = save_checkpoint(Path(directory) / "checkpoint", identity=identity,
                        parameters=mapping, optimizer=optimizer, cursor={**cursor, "telemetry": sidecar},
                        generators={"sampling": generator}, extra_state=active)
                    saved.append((receipt, sidecar, copy.deepcopy(cursor), copy.deepcopy(active)))
                    return {"complete": True, "optimizer_updates": cursor["optimizer_updates"], **receipt}
                require_saved_guard_boundary(HardGuard(), "optimizer_complete", cursor["optimizer_updates"], save)
            with self.assertRaises(InterruptedError):
                self.run_engine(backend, optimizer, generator, config, checkpoint_callback=checkpoint)
            self.assertEqual(seen, [1])
            self.assertEqual(len(saved), 1)
            prior_rng = _capture_rng({"sampling": generator})
            expected_parameters = {name: p.detach().clone() for name, p in mapping.items()}
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            receipt, sidecar, cursor, active = saved[0]
            target, targets, target_optimizer, target_generator = self.initialized()
            restored = restore_boundary({"path": receipt["checkpoint_path"],
                "manifest_sha256": receipt["manifest_sha256"], "sidecar": sidecar}, identity,
                parameters=targets, optimizer=target_optimizer, generator=target_generator)
            self.assertEqual(restored["cursor"], cursor)
            assert_tree(self, restored["extra_state"], active)
            assert_tree(self, target_optimizer.state_dict(), expected_optimizer)
            assert_tree(self, _capture_rng({"sampling": target_generator}), prior_rng)
            for name in targets:
                torch.testing.assert_close(targets[name], expected_parameters[name], rtol=0, atol=0)

    def test_replay_matches_only_surviving_observations_and_rejects_drift(self):
        event = {"update": 610, "responses": 1, "action_tokens": 2, "objective": 0.125,
            "gradient_norm": 0.75, "target_update": 610, "rollout_index": 19,
            "record_ids": ["real"], "response_indices": [1], "epoch": 0, "rollout_offset": 1,
            "cursor": {"order": [1, 0], "optimizer_updates": 610}}
        values = {key: event[key] for key in ("update", "responses", "action_tokens", "objective", "gradient_norm")}
        reference = {**event, "cursor_sha256": digest(event["cursor"]), "order_sha256": digest(event["cursor"]["order"])}
        result = replay_comparison(reference, values, event, sampling_rng_hex="0123", expected_sampling_rng_hex="0123")
        self.assertTrue(result["preserved_observations_equal"])
        self.assertFalse(result["prior_final_state_bitwise_verified"])
        for key, value in (("objective", 0.126), ("gradient_norm", 0.8), ("record_ids", ["other"]),
                           ("response_indices", [0]), ("cursor", {"order": [0, 1], "optimizer_updates": 610})):
            with self.assertRaisesRegex(ValueError, "differs from preserved"):
                replay_comparison(reference, values, {**event, key: value}, sampling_rng_hex="0123", expected_sampling_rng_hex="0123")
        with self.assertRaises(ValueError):
            replay_comparison(reference, values, event, sampling_rng_hex="4567", expected_sampling_rng_hex="0123")

    def test_migration_changes_only_telemetry_identity_binding(self):
        old = {"parameters": {"p": torch.tensor([1.0])},
            "optimizer": {"state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.tensor([0.2])}}},
            "cursor": {"order": [1, 0], "optimizer_updates": 3, "telemetry": {"path": "old", "sha256": "a"}},
            "rng": {"sampling": torch.tensor([1, 2], dtype=torch.uint8)},
            "extra_state": {"epoch": 1, "offset": 1, "advantages": torch.tensor([0.4])}}
        fresh = copy.deepcopy(old)
        fresh["cursor"]["telemetry"] = {"path": "new", "sha256": "b"}
        self.assertTrue(verify_migration_payload(old, fresh))
        for section, mutate in (("parameters", lambda value: value["p"].add_(1)),
                ("optimizer", lambda value: value["state"][0]["step"].add_(1)),
                ("rng", lambda value: value["sampling"].add_(1)),
                ("cursor", lambda value: value["order"].reverse()),
                ("extra_state", lambda value: value["advantages"].add_(1))):
            bad = copy.deepcopy(fresh)
            mutate(bad[section])
            with self.assertRaisesRegex(ValueError, "Migration changed"):
                verify_migration_payload(old, bad)

    def test_budget_or_kind_change_fails_before_original_validation(self):
        with patch("alignment_lab.throughput_stage_v1.original.validate_request") as old:
            with self.assertRaises(ValueError):
                validate_request({"kind": "alignlab-ppo-throughput-resume-v1", "original_deadline_epoch": 1792894046.0})
            old.assert_not_called()

    def test_owned_processes_require_new_script_exact_request_hash_and_parent(self):
        path = ROOT / "runs/new/ppo42.request.json"
        digest = "a" * 64
        parent = {"pid": 100, "parent_pid": 90, "created_at_utc": "2026-10-23T07:00:00Z", "command_line": "supervisor"}
        child = {"pid": 101, "parent_pid": 100, "created_at_utc": "2026-10-23T07:00:01Z", "command_line":
            f'python scripts/run_alignlab_throughput_stage_v1.py --request "{path}" --request-sha256 {digest} --execute'}
        self.assertEqual(_owned([parent, child], {"supervisor_processes": [parent]}, path, digest, pid=101), [child])
        for bad in ({**child, "parent_pid": 200}, {**child, "command_line": child["command_line"].replace(digest, "b" * 64)}):
            with self.assertRaises(RuntimeError):
                _owned([parent, bad], {"supervisor_processes": [parent]}, path, digest, pid=101)


if __name__ == "__main__":
    unittest.main()
