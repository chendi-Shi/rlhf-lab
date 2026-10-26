"""Tiny CPU equality/recovery tests; no pretrained-model or GPU evidence."""
from collections.abc import Mapping
import copy
from pathlib import Path
import random
import tempfile
import unittest

import torch

from alignment_lab import checkpointing
from alignment_lab import stage_engine as old
from alignment_lab import stage_engine_observed as observed
from alignment_lab.shared_qwen import SharedQwen3
from alignment_lab.training_steps import reference_pair_scores
from test_shared_qwen import fixture
from test_stage_engine import examples, supervised, ppo


def initialized(kind="ppo"):
    backend, model, heads, _, _, _ = fixture()
    role, selected_heads = "policy", ("value",) if kind == "ppo" else ()
    if kind == "reward":
        role, selected_heads = "reward", ("reward",)
        backend = SharedQwen3(model, trainable_roles={role}, heads=heads,
                             trainable_heads={role: selected_heads})
    optimizer = torch.optim.AdamW(backend.parameters_for(role, heads=selected_heads), lr=1e-3)
    return backend, optimizer, role, torch.Generator().manual_seed(321)


def run_ppo(module, backend, optimizer, generator, **kwargs):
    return module.run_ppo(backend, "policy", optimizer, examples(), ppo(), device="cpu",
        generator=generator, reference_role="reference", reward_role="reward",
        clock=lambda: 100.0, **kwargs)


class ObservedEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def assert_tree_equal(self, actual, expected):
        self.assertEqual(type(actual), type(expected))
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def assert_models_equal(self, first, second):
        for a, b in zip(first._all_parameters(), second._all_parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in second._all_parameters()))

    def assert_plain_immutable(self, event):
        self.assertNotIsInstance(event, torch.Tensor)
        if isinstance(event, Mapping):
            with self.assertRaises(TypeError):
                event["cannot_mutate"] = True
            for value in event.values():
                self.assert_plain_immutable(value)
        elif isinstance(event, tuple):
            for value in event:
                self.assert_plain_immutable(value)
        else:
            self.assertIn(type(event), (type(None), str, bool, int, float))

    def test_supervised_all_kinds_match_original_parameters_optimizer_cursor(self):
        for kind in ("sft", "reward", "dpo"):
            with self.subTest(kind=kind):
                baseline, base_optimizer, role, _ = initialized(kind)
                provider = lambda chosen, rejected: reference_pair_scores(
                    baseline, "reference", chosen, rejected, chunk_tokens=2)
                original = old.run_supervised(baseline, role, base_optimizer, examples(), supervised(kind),
                    device="cpu", reference_provider=provider, clock=lambda: 100)
                backend, optimizer, role, _ = initialized(kind)
                provider = lambda chosen, rejected: reference_pair_scores(
                    backend, "reference", chosen, rejected, chunk_tokens=2)
                events, sequence = [], []

                def observe(event):
                    self.assert_plain_immutable(event)
                    plain = observed.event_to_plain(event)
                    events.append(plain)
                    sequence.append(("event", plain["event"], plain["cursor"]["optimizer_updates"]))
                    # A mutable logger copy must not alias the engine cursor.
                    plain["cursor"]["offset"] = -999

                def checkpoint(cursor, extra):
                    self.assertIsNone(extra)
                    sequence.append(("checkpoint", cursor["optimizer_updates"]))

                result = observed.run_supervised(backend, role, optimizer, examples(), supervised(kind),
                    device="cpu", reference_provider=provider, clock=lambda: 100,
                    event_callback=observe, checkpoint_callback=checkpoint)
                self.assertEqual(result, original)
                self.assert_models_equal(baseline, backend)
                self.assert_tree_equal(optimizer.state_dict(), base_optimizer.state_dict())
                self.assertEqual([item["event"] for item in events],
                    ["preparation_start", "preparation_end", "optimizer_start", "optimizer_end"] * 2)
                for index, entry in enumerate(sequence):
                    if entry[0] == "checkpoint":
                        self.assertEqual(sequence[index - 1], ("event", "optimizer_end", entry[1]))

    def test_ppo_matches_original_sampling_parameters_optimizer_and_final_cursor(self):
        baseline, base_optimizer, _, base_generator = initialized()
        original_checkpoints = []
        full = run_ppo(old, baseline, base_optimizer, base_generator,
            checkpoint_callback=lambda cursor, extra: original_checkpoints.append((cursor, extra)))
        backend, optimizer, _, generator = initialized()
        events, checkpoints, sequence = [], [], []

        def observer(event):
            self.assert_plain_immutable(event)
            value = observed.event_to_plain(event)
            events.append(value)
            sequence.append(value["event"])

        def checkpoint(cursor, extra):
            checkpoints.append((cursor, extra))
            sequence.append("checkpoint")

        actual = run_ppo(observed, backend, optimizer, generator,
                         event_callback=observer, checkpoint_callback=checkpoint)
        self.assertEqual(actual, full)
        self.assert_models_equal(baseline, backend)
        self.assert_tree_equal(optimizer.state_dict(), base_optimizer.state_dict())
        self.assertTrue(torch.equal(generator.get_state(), base_generator.get_state()))
        collected = [e for e in events if e["event"] == "rollout_collected"]
        self.assertEqual(len(collected), 2)
        self.assertEqual([e["rollout_index"] for e in collected], [0, 1])
        for current, source in zip(collected, (original_checkpoints[0][1], original_checkpoints[4][1])):
            self.assertEqual(current["responses"], observed.event_to_plain(observed._immutable(source["responses"])))
            for key in ("advantages", "returns", "reference_logprobs", "reward_scores"):
                self.assertEqual(current[key], source[key].tolist())
            self.assertEqual(current["cursor"]["optimizer_updates"], current["cursor_before"]["optimizer_updates"])
            self.assertTrue(current["observation_origin"]["rng_before_hex"])
            self.assertTrue(current["observation_origin"]["rng_after_hex"])
        for (original_cursor, original_extra), (cursor, extra) in zip(original_checkpoints, checkpoints):
            self.assertEqual(cursor, original_cursor)
            if extra is None:
                self.assertIsNone(original_extra)
            else:
                stripped = {k: v for k, v in extra.items() if k != "observation_origin"}
                self.assert_tree_equal(stripped, original_extra)
        self.assertEqual(sequence.count("optimizer_end"), 6)
        self.assertEqual(sequence.count("checkpoint"), 6)
        self.assertEqual(sequence.count("rollout_finished"), 2)

    def test_active_disk_resume_matches_continuous_and_keeps_original_anchors(self):
        baseline, base_optimizer, _, base_generator = initialized()
        full = run_ppo(observed, baseline, base_optimizer, base_generator)
        backend, optimizer, _, generator = initialized()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.txt"
            source.write_text("actual tiny CPU checkpoint identity", encoding="utf-8")
            identity = checkpointing.build_identity({"fixture": source}, {"observed": "cpu-only"})
            saved = []

            def checkpoint(cursor, extra):
                if cursor["optimizer_updates"] == 3:
                    saved.append(checkpointing.save_checkpoint(root / "active", identity=identity,
                        parameters=old.parameter_mapping(backend, "policy", heads=("value",)), optimizer=optimizer,
                        cursor=cursor, generators={"sampling": generator}, extra_state=extra))

            partial = run_ppo(observed, backend, optimizer, generator,
                checkpoint_callback=checkpoint, stop_requested=lambda: bool(saved))
            self.assertTrue(partial.cursor["active_rollout"])
            self.assertEqual(partial.cursor["optimizer_updates"], 3)
            origin = copy.deepcopy(partial.extra_state["observation_origin"])
            restored_backend, restored_optimizer, _, restored_generator = initialized()
            state = checkpointing.load_checkpoint(root / "active", expected_identity=identity,
                parameters=old.parameter_mapping(restored_backend, "policy", heads=("value",)),
                optimizer=restored_optimizer, generators={"sampling": restored_generator},
                expected_manifest_sha256=saved[0]["manifest_sha256"])
            self.assertEqual(state["extra_state"]["observation_origin"], origin)
            resumed_events = []
            result = run_ppo(observed, restored_backend, restored_optimizer, restored_generator,
                state=state["cursor"], extra_state=state["extra_state"],
                event_callback=lambda event: resumed_events.append(observed.event_to_plain(event)))
            self.assertEqual(result.cursor, full.cursor)
            self.assertEqual(result.metrics, full.metrics[3:])
            self.assert_models_equal(baseline, restored_backend)
            self.assert_tree_equal(restored_optimizer.state_dict(), base_optimizer.state_dict())
            self.assertTrue(torch.equal(restored_generator.get_state(), base_generator.get_state()))
            self.assertEqual(state["extra_state"]["observation_origin"], origin)
            self.assertEqual([e["rollout_index"] for e in resumed_events if e["event"] == "rollout_collected"], [1])

    def test_missing_or_changed_active_origin_and_rng_refused(self):
        backend, optimizer, _, generator = initialized()
        saved = []
        partial = run_ppo(observed, backend, optimizer, generator,
            checkpoint_callback=lambda cursor, extra: saved.append(cursor), stop_requested=lambda: len(saved) == 1)
        for change in ("missing", "origin_sha", "rng"):
            with self.subTest(change=change):
                extra = copy.deepcopy(partial.extra_state)
                if change == "missing":
                    del extra["observation_origin"]
                elif change == "origin_sha":
                    extra["observation_origin"]["rng_before_hex"] = "00"
                else:
                    generator.manual_seed(1000)
                with self.assertRaises(ValueError):
                    run_ppo(observed, backend, optimizer, generator, state=partial.cursor, extra_state=extra)

    def test_stop_from_collection_prevents_first_update_and_checkpoints_original_rollout(self):
        backend, optimizer, _, generator = initialized()
        events, saved = [], []

        def observe(event):
            events.append(event["event"])

        result = run_ppo(observed, backend, optimizer, generator, event_callback=observe,
            stop_requested=lambda: "rollout_collected" in events,
            checkpoint_callback=lambda cursor, extra: saved.append((cursor, extra)))
        self.assertEqual(result.stop_reason, "paused")
        self.assertEqual(result.cursor["optimizer_updates"], 0)
        self.assertNotIn("optimizer_start", events)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][1]["observation_origin"], result.extra_state["observation_origin"])

    def test_observer_overhead_deadline_and_stop_prevent_new_gradient_dispatch(self):
        for kind in ("sft", "dpo"):
            backend, optimizer, role, _ = initialized(kind)
            done = []
            provider = lambda c, r: reference_pair_scores(backend, "reference", c, r, chunk_tokens=2)

            def observe(event):
                if event["event"] == "optimizer_start":
                    done.append(True)

            result = observed.run_supervised(backend, role, optimizer, examples(), supervised(kind), device="cpu",
                reference_provider=provider, event_callback=observe,
                clock=lambda: 1000 if done else 100)
            self.assertEqual(result.stop_reason, "budget_exhausted")
            self.assertEqual(result.cursor["optimizer_updates"], 0)
            self.assertEqual(len(optimizer.state), 0)
            self.assertTrue(all(p.grad is None for p in backend._all_parameters()))

    def test_callback_consuming_rng_or_mutating_parameters_is_rejected(self):
        for mode in ("torch_rng", "python_rng", "sampling_rng", "parameter"):
            with self.subTest(mode=mode):
                backend, optimizer, _, generator = initialized()

                def observe(event):
                    if mode == "torch_rng":
                        torch.rand(1)
                    elif mode == "python_rng":
                        random.random()
                    elif mode == "sampling_rng":
                        torch.rand(1, generator=generator)
                    else:
                        with torch.no_grad():
                            backend.parameters_for("policy", heads=("value",))[0].add_(1)

                with self.assertRaisesRegex(RuntimeError, "Observer changed"):
                    run_ppo(observed, backend, optimizer, generator, event_callback=observe)
                self.assertEqual(len(optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
