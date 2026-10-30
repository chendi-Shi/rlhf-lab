"""V5 RM integration fixtures; completed-training evidence remains separately pinned."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

import torch
from alignment_lab import evaluation_model_v3 as original
from alignment_lab import evaluation_model_v5 as evaluation
from alignment_lab.evaluation_attention_receipt_v5 import validate_attention_execution_receipt
import test_evaluation_model_v3 as fixtures


class EvaluationModelV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.fixture = fixtures.EvaluationModelTests()
        self.fixture.setUp()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.fixture.tearDown()

    def bind(self, stage):
        return evaluation.bind_evaluation_model(self.fixture._loaded(stage),
            self.fixture.records[stage], input_files=self.fixture.files)

    def test_reward_scores_match_original_weights_and_full_audit_remains_identical(self):
        old = original.bind_evaluation_model(self.fixture._loaded("reward"),
            self.fixture.records["reward"], input_files=self.fixture.files)
        new = self.bind("reward")
        before = new.frozen_audit(full=True)
        metadata = new.metadata
        for prompt, completion in (((4, 8, 9), (11, 2)), ((4, 8), (11, 0, 2)),
                                   ((4,) * 73, (11,) * 40 + (2,))):
            self.assertAlmostEqual(new.reward_score(prompt, completion), old.reward_score(prompt, completion), places=5)
        self.assertEqual(before, new.frozen_audit(full=True))
        self.assertEqual(metadata, new.metadata)
        receipt = validate_attention_execution_receipt(new.attention_execution_receipt(), enabled=True, minimum_calls=6)
        self.assertEqual(receipt["blocked_math_executed"], 6)
        self.assertEqual(receipt["efficient_executed"], 0)
        self.assertEqual(new._backend._base.config._attn_implementation,
                         old._backend._base.config._attn_implementation)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in new._backend._all_parameters()))

    def test_every_policy_stage_keeps_original_forward_and_has_no_rm_execution(self):
        for stage in ("base", "sft", "dpo", "ppo"):
            loaded = self.fixture._loaded(stage)
            methods = [layer.self_attn.forward for layer in loaded.backend._base.model.layers]
            model = evaluation.bind_evaluation_model(loaded, self.fixture.records[stage], input_files=self.fixture.files)
            self.assertEqual(methods, [layer.self_attn.forward for layer in model._backend._base.model.layers])
            self.assertIsNone(model.metadata["reward_attention_contract"])
            receipt = model.attention_execution_receipt()
            validate_attention_execution_receipt(receipt, enabled=False)
            model.greedy_generate((4, 8), eos_id=2, max_new_tokens=2)
            self.assertEqual(receipt, model.attention_execution_receipt())
            self.assertTrue(model.frozen_audit(full=True)["unchanged"])

    def test_rm_binding_tamper_is_rejected_by_existing_frozen_audit_entrypoint(self):
        model = self.bind("reward")
        layer = model._backend._base.model.layers[0].self_attn
        layer.forward = lambda *_args, **_kwargs: None
        with self.assertRaisesRegex(RuntimeError, "binding"):
            model.frozen_audit(full=False)

    def test_receipt_is_detached_and_execution_is_required_for_acceptance(self):
        model = self.bind("reward")
        with self.assertRaisesRegex(ValueError, "counters"):
            validate_attention_execution_receipt(model.attention_execution_receipt(), enabled=True)
        model.reward_score((4, 8), (2,))
        first = model.attention_execution_receipt()
        changed = copy.deepcopy(first)
        changed["observations"][0]["calls"] += 900
        self.assertEqual(first, model.attention_execution_receipt())
        with self.assertRaises(ValueError):
            validate_attention_execution_receipt(changed, enabled=True)


if __name__ == "__main__":
    unittest.main()
