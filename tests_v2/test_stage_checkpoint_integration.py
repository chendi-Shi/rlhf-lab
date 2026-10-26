"""Tiny CPU engines restored into fresh model/optimizer objects from disk."""
import dataclasses
from pathlib import Path
import tempfile
import unittest

import torch

from alignment_lab.checkpointing import build_identity, load_checkpoint, save_checkpoint
from alignment_lab.stage_engine import parameter_mapping, run_supervised
from test_shared_qwen import fixture
from test_stage_engine import examples, ppo, ppo_run, supervised


class StageCheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def assert_tree_equal(self, left, right):
        self.assertIs(type(left), type(right))
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(list(left), list(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def identity(self, config):
        root = Path(__file__).resolve().parents[1]
        files = [root / "alignment_lab" / (name + ".py") for name in
                 ("stage_engine", "shared_qwen", "rollout", "ppo_objective",
                  "training_steps", "token_logprobs", "checkpointing")]
        files += [root / "tests_v2" / "test_shared_qwen.py",
                  root / "tests_v2" / "test_stage_engine.py", Path(__file__).resolve()]
        return build_identity(
            files={str(path.relative_to(root)): path for path in sorted(files)},
            config={"fixture_only": True, "stage": dataclasses.asdict(config), "examples": examples()},
        )

    def initialized(self, *, value=False):
        backend, _, _, _, _, _ = fixture()
        mapping = parameter_mapping(backend, "policy", heads=("value",) if value else ())
        optimizer = torch.optim.AdamW(mapping.values(), lr=1e-3)
        return backend, mapping, optimizer, torch.Generator().manual_seed(321)

    def test_ppo_disk_restore_mid_epoch_equals_continuous_training(self):
        config = ppo()
        baseline, _, baseline_optimizer, baseline_generator = self.initialized(value=True)
        full = ppo_run(baseline, baseline_optimizer, baseline_generator, config)
        partial_backend, mapping, optimizer, generator = self.initialized(value=True)
        saved = []
        partial = ppo_run(partial_backend, optimizer, generator, config,
                          checkpoint_callback=lambda cursor, extra: saved.append(cursor),
                          stop_requested=lambda: len(saved) == 3)
        self.assertEqual(partial.stop_reason, "paused")
        self.assertEqual(partial.extra_state["epoch"], 1)
        self.assertEqual(partial.extra_state["offset"], 1)
        with tempfile.TemporaryDirectory() as folder:
            identity = self.identity(config)
            receipt = save_checkpoint(Path(folder) / "update-000003", identity=identity,
                parameters=mapping, optimizer=optimizer, cursor=partial.cursor,
                generators={"sampling": generator}, extra_state=partial.extra_state)
            resumed, target_mapping, target_optimizer, target_generator = self.initialized(value=True)
            restored = load_checkpoint(receipt["checkpoint_path"], expected_identity=identity,
                parameters=target_mapping, optimizer=target_optimizer,
                generators={"sampling": target_generator},
                expected_manifest_sha256=receipt["manifest_sha256"])
            self.assert_tree_equal(restored["extra_state"], partial.extra_state)
            final = ppo_run(resumed, target_optimizer, target_generator, config,
                           state=restored["cursor"], extra_state=restored["extra_state"])
        self.assertEqual(final.cursor, full.cursor)
        self.assertEqual(final.stop_reason, full.stop_reason)
        self.assertEqual(final.metrics, full.metrics[3:])
        for a, b in zip(baseline._all_parameters(), resumed._all_parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assert_tree_equal(baseline_optimizer.state_dict(), target_optimizer.state_dict())
        self.assertTrue(torch.equal(baseline_generator.get_state(), target_generator.get_state()))

    def test_sft_disk_restore_keeps_batch_cursor_and_optimizer_moments(self):
        config = supervised()
        baseline, _, baseline_optimizer, _ = self.initialized()
        full = run_supervised(baseline, "policy", baseline_optimizer, examples(), config,
                              device="cpu", clock=lambda: 100)
        backend, mapping, optimizer, generator = self.initialized()
        saved = []
        partial = run_supervised(backend, "policy", optimizer, examples(), config, device="cpu",
            clock=lambda: 100, checkpoint_callback=lambda c, e: saved.append(c),
            stop_requested=lambda: bool(saved))
        self.assertEqual(partial.cursor["offset"], 2)
        with tempfile.TemporaryDirectory() as folder:
            identity = self.identity(config)
            receipt = save_checkpoint(Path(folder) / "update-000001", identity=identity,
                parameters=mapping, optimizer=optimizer, cursor=partial.cursor,
                generators={"sampling": generator})
            resumed, target_mapping, target_optimizer, target_generator = self.initialized()
            restored = load_checkpoint(receipt["checkpoint_path"], expected_identity=identity,
                parameters=target_mapping, optimizer=target_optimizer,
                generators={"sampling": target_generator},
                expected_manifest_sha256=receipt["manifest_sha256"])
            final = run_supervised(resumed, "policy", target_optimizer, examples(), config,
                                   device="cpu", clock=lambda: 100, state=restored["cursor"])
        self.assertEqual(final.cursor, full.cursor)
        self.assertEqual(final.metrics, full.metrics[1:])
        for a, b in zip(baseline._all_parameters(), resumed._all_parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assert_tree_equal(baseline_optimizer.state_dict(), target_optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()
