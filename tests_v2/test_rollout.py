"""Tiny random Qwen3 rollout checks: CPU software, not pretrained/GPU results."""
import dataclasses
import unittest
from unittest.mock import patch

import torch

from alignment_lab.rollout import sample_rollout, pack_rollout, compact_actions, score_frozen_rollout
from test_shared_qwen import fixture


def sample(backend, prompts=((4, 8, 9), (5, 7)), seed=19, limit=4):
    return sample_rollout(backend, "policy", prompts, eos_id=2, pad_id=0,
        max_new_tokens=limit, generator=torch.Generator().manual_seed(seed), device="cpu")


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def test_cached_sampling_matches_full_teacher_forced_action_probabilities_and_values(self):
        backend, model, _, _, _, _ = fixture()
        head_rows, decoder_lengths, caches = [], [], []
        hh = model.get_base_model().lm_head.register_forward_pre_hook(
            lambda module, args: head_rows.append(tuple(args[0].shape)))
        def observe(module, args, kwargs):
            decoder_lengths.append(kwargs["input_ids"].shape[1])
            caches.append(kwargs["past_key_values"] is not None)
        bh = model.get_base_model().model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            rollout = sample(backend, prompts=((4, 8, 9),))
        finally:
            hh.remove(); bh.remove()
        n = len(rollout.responses[0].action_ids)
        self.assertEqual(decoder_lengths, [3] + [1] * (n - 1))
        self.assertEqual(caches, [False] + [True] * (n - 1))
        self.assertTrue(all(shape == (1, 32) for shape in head_rows))
        packed = pack_rollout(rollout, device="cpu")
        inputs = {k: packed[k] for k in ("input_ids", "attention_mask", "labels")}
        scores, values = backend.read("policy", lambda view: view.token_scores_and_values(**inputs, chunk_tokens=2))
        torch.testing.assert_close(compact_actions(scores.logprobs, rollout), packed["old_logprobs"], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(compact_actions(values, rollout), packed["old_values"], rtol=1e-5, atol=2e-6)
        self.assertFalse(scores.logprobs.requires_grad)
        self.assertFalse(torch.cuda.is_initialized())

    def test_rng_state_replay_and_new_response_cache_is_fresh(self):
        backend, _, _, _, _, _ = fixture()
        generator = torch.Generator().manual_seed(77)
        options = dict(eos_id=2, pad_id=0, max_new_tokens=5, generator=generator, device="cpu")
        first = sample_rollout(backend, "policy", ((4, 8), (5, 7, 9)), **options)
        generator.set_state(torch.tensor(list(first.rng_before), dtype=torch.uint8))
        second = sample_rollout(backend, "policy", ((4, 8), (5, 7, 9)), **options)
        self.assertEqual(first.responses, second.responses)
        self.assertEqual(first.rng_after, second.rng_after)
        self.assertNotEqual(first.rng_before, first.rng_after)

    def test_sampled_pad_then_eos_remain_real_actions_and_prompt_eos_does_not_stop(self):
        backend, _, _, _, _, _ = fixture()
        with patch("torch.multinomial", side_effect=[torch.tensor([[0]]), torch.tensor([[2]])]):
            rollout = sample(backend, prompts=((4, 2, 9),))
        response = rollout.responses[0]
        self.assertEqual(response.action_ids, (0, 2))
        self.assertEqual(response.termination, "eos")
        packed = pack_rollout(rollout, device="cpu")
        self.assertEqual(packed["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(packed["labels"].tolist(), [[-100, -100, -100, 0, 2]])
        self.assertEqual(packed["action_mask"].tolist(), [[True, True]])
        inputs = {k: packed[k] for k in ("input_ids", "attention_mask", "labels")}
        scored = backend.read("policy", lambda view: view.token_scores(**inputs))
        torch.testing.assert_close(compact_actions(scored.logprobs, rollout), packed["old_logprobs"])

    def test_eos_only_and_horizon_are_distinct_without_synthetic_eos(self):
        backend, _, _, _, _, _ = fixture()
        with patch("torch.multinomial", side_effect=[torch.tensor([[2]]), torch.tensor([[13]]), torch.tensor([[14]])]):
            rollout = sample(backend, prompts=((4, 8, 9), (5, 7)), limit=2)
        self.assertEqual(rollout.responses[0].action_ids, (2,))
        self.assertEqual(rollout.responses[0].termination, "eos")
        self.assertEqual(rollout.responses[1].action_ids, (13, 14))
        self.assertEqual(rollout.responses[1].termination, "max_new_tokens")
        packed = pack_rollout(rollout, device="cpu")
        self.assertEqual(packed["action_mask"].tolist(), [[True, False], [True, True]])
        invalid = dataclasses.replace(rollout.responses[1], termination="eos")
        with self.assertRaisesRegex(ValueError, "termination"):
            pack_rollout(dataclasses.replace(rollout, responses=(invalid,)), device="cpu")

    def test_frozen_scoring_does_not_change_any_parameter_or_old_statistics(self):
        backend, model, heads, _, _, _ = fixture()
        rollout = sample(backend)
        before = [p.detach().clone() for p in backend._all_parameters()]
        original = rollout.responses
        reference, reward = score_frozen_rollout(backend, rollout, reference_role="reference", reward_role="reward", device="cpu", chunk_tokens=2)
        packed = pack_rollout(rollout, device="cpu")
        # Fixture policy and reference weights are initially identical.
        torch.testing.assert_close(reference, packed["old_logprobs"], rtol=1e-5, atol=2e-6)
        self.assertEqual(reward.shape, (2,))
        self.assertFalse(reference.requires_grad or reward.requires_grad)
        self.assertEqual(original, rollout.responses)
        for previous, current in zip(before, backend._all_parameters()):
            torch.testing.assert_close(previous, current, rtol=0, atol=0)
            self.assertIsNone(current.grad)
        with torch.no_grad():
            heads["value"].bias.add_(0.1)
        with self.assertRaisesRegex(RuntimeError, "unchanged"):
            score_frozen_rollout(backend, rollout, reference_role="reference", reward_role="reward", device="cpu")

    def test_invalid_role_inputs_budget_generator_and_context_fail(self):
        backend, _, _, _, _, _ = fixture()
        for prompts in ((), ((),), ((-1,),), ((97,),), ((True,),)):
            with self.assertRaises(ValueError):
                sample(backend, prompts=prompts)
        for limit in (0, True, 126):
            with self.assertRaises(ValueError):
                sample(backend, prompts=((4, 8, 9),), limit=limit)
        with self.assertRaises(ValueError):
            sample_rollout(backend, "reference", ((4,),), eos_id=2, pad_id=0, max_new_tokens=2,
                           generator=torch.Generator(), device="cpu")
        with self.assertRaises(ValueError):
            sample_rollout(backend, "policy", ((4,),), eos_id=2, pad_id=0, max_new_tokens=2,
                           generator=None, device="cpu")
        with self.assertRaisesRegex(ValueError, "value head"):
            backend.read("policy", lambda view: view.sample_response(torch.tensor([4, 8]),
                         eos_id=2, max_new_tokens=2, generator=torch.Generator(), head="reward"))

    def test_sampling_rejects_pending_gradients_nonfinite_scores_and_external_head_mode_change(self):
        backend, model, heads, _, _, _ = fixture()
        backend.parameters_for("policy")[0].grad = torch.zeros_like(backend.parameters_for("policy")[0])
        with self.assertRaisesRegex(RuntimeError, "Clear"):
            sample(backend)
        backend.parameters_for("policy")[0].grad = None
        with torch.no_grad():
            heads["value"].bias.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            sample(backend)
        with torch.no_grad():
            heads["value"].bias.zero_()
        def corrupt(view):
            heads["value"].train()
            return view.sample_response(torch.tensor([4, 8]), eos_id=2, max_new_tokens=2,
                                        generator=torch.Generator())
        with self.assertRaisesRegex(RuntimeError, "head mode"):
            backend.read("policy", corrupt)

    def test_packing_rejects_silent_id_coercion_and_corrupt_metadata(self):
        backend, _, _, _, _, _ = fixture()
        with patch("torch.multinomial", return_value=torch.tensor([[13]])):
            rollout = sample(backend, prompts=((4, 8),), limit=1)
        for invalid in (3.9, True, -1, 97):
            response = dataclasses.replace(rollout.responses[0], action_ids=(invalid,))
            with self.assertRaisesRegex(ValueError, "exact"):
                pack_rollout(dataclasses.replace(rollout, responses=(response,)), device="cpu")
        for corrupt in ({"vocab_size": True}, {"max_new_tokens": 1.5}, {"eos_id": 97}, {"pad_id": -1}):
            with self.assertRaises(ValueError):
                pack_rollout(dataclasses.replace(rollout, **corrupt), device="cpu")


if __name__ == "__main__":
    unittest.main()
