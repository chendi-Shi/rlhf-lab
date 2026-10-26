"""Independent numeric checks of token alignment, RLHF losses and boundaries."""

import math
import unittest

import torch

from rlhf_lab.objectives import (
    categorical_entropy,
    completion_logprobs,
    dpo_loss,
    generalized_advantage_estimate,
    masked_mean,
    ppo_loss,
    response_token_mask,
    reward_model_loss,
    sampled_kl_rewards,
    sequence_logprobs,
    whiten_advantages,
)


class CompletionLogprobTests(unittest.TestCase):
    def test_shift_counts_real_eos_but_not_same_id_padding(self):
        # IDs 4,1 are prompt; ID 2 is the response, ID 3 its EOS AND padding ID.
        ids = torch.tensor([[4, 1, 2, 3, 3]])
        response = torch.tensor([[0, 0, 1, 1, 1]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        logits = torch.zeros(1, 5, 5, dtype=torch.double)
        logits[0, 1, 2] = math.log(9)
        logits[0, 2, 3] = math.log(6)
        logits.requires_grad_()
        values = completion_logprobs(logits, ids, response, attention)
        expected = torch.tensor([[0., math.log(9 / 13), math.log(6 / 10), 0.]], dtype=torch.double)
        torch.testing.assert_close(values, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(response_token_mask(ids, response, attention).tolist(), [[False, True, True, False]])
        (-values.sum()).backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 3:].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 1:3].abs().sum()), 0)

    def test_left_padding_and_sum_instead_of_length_average(self):
        ids = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 3]])
        attention = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        response = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
        logits = torch.zeros(2, 5, 4, dtype=torch.double)
        result = sequence_logprobs(logits, ids, response, attention)
        torch.testing.assert_close(result, torch.tensor([-2 * math.log(4), -3 * math.log(4)], dtype=torch.double))

    def test_padding_does_not_change_completion_probability(self):
        torch.manual_seed(6)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[0, 1, 1]])
        logits = torch.randn(1, 3, 5, dtype=torch.double)
        short = sequence_logprobs(logits, ids, mask)
        padded_ids = torch.cat([ids, torch.tensor([[3, 3]])], dim=1)
        padded_logits = torch.cat([logits, torch.randn(1, 2, 5, dtype=torch.double) * 100], dim=1)
        long = sequence_logprobs(padded_logits, padded_ids, torch.tensor([[0, 1, 1, 1, 1]]),
                                 torch.tensor([[1, 1, 1, 0, 0]]))
        torch.testing.assert_close(short, long, rtol=0, atol=0)

    def test_unpredictable_position_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "position zero"):
            response_token_mask(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))


class PreferenceLossTests(unittest.TestCase):
    def test_dpo_numeric_reference_adjustment_and_gradient_direction(self):
        chosen = torch.tensor([-2.], dtype=torch.double, requires_grad=True)
        rejected = torch.tensor([-3.], dtype=torch.double, requires_grad=True)
        ref_chosen = torch.tensor([-2.], dtype=torch.double, requires_grad=True)
        ref_rejected = torch.tensor([-2.], dtype=torch.double, requires_grad=True)
        loss, stats = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.2)
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(-0.2)), places=12)
        self.assertAlmostEqual(stats["reward_margin"].item(), 0.2, places=12)
        loss.backward()
        expected_gradient = 0.2 / (1 + math.exp(0.2))
        self.assertAlmostEqual(chosen.grad.item(), -expected_gradient, places=12)
        self.assertAlmostEqual(rejected.grad.item(), expected_gradient, places=12)
        self.assertIsNone(ref_chosen.grad)
        self.assertIsNone(ref_rejected.grad)
        # Equal policy/reference preference ratios imply a log(2) loss.
        same, _ = dpo_loss(chosen.detach(), rejected.detach(), torch.tensor([-4.]), torch.tensor([-5.]), beta=0.2)
        self.assertAlmostEqual(same.item(), math.log(2), places=7)

    def test_bradley_terry_and_explicit_pair_mean_centering(self):
        chosen, rejected = torch.tensor([3.], dtype=torch.double), torch.tensor([1.], dtype=torch.double)
        loss, stats = reward_model_loss(chosen, rejected, centering_coefficient=0.25)
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(-2)) + 1, places=12)
        self.assertEqual(stats["centering_loss"].item(), 4)
        uncentered, _ = reward_model_loss(chosen, rejected)
        shifted, _ = reward_model_loss(chosen + 17, rejected + 17)
        self.assertEqual(uncentered.item(), shifted.item())


class RolloutTests(unittest.TestCase):
    def test_gae_hand_numeric_variable_terminals_and_no_bootstrap(self):
        rewards = torch.tensor([[999., 1., 2., 999.], [3., 999., 999., 999.]], dtype=torch.double, requires_grad=True)
        values = torch.tensor([[999., .5, 1., 999.], [1., 999., 999., 999.]], dtype=torch.double, requires_grad=True)
        mask = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0]])
        advantage, returns = generalized_advantage_estimate(rewards, values, mask, gamma=.9, lam=.8)
        torch.testing.assert_close(advantage, torch.tensor([[0., 2.12, 1., 0.], [2., 0., 0., 0.]], dtype=torch.double))
        torch.testing.assert_close(returns, torch.tensor([[0., 2.62, 2., 0.], [3., 0., 0., 0.]], dtype=torch.double))
        self.assertFalse(advantage.requires_grad)
        self.assertFalse(returns.requires_grad)
        whitened, same_returns = generalized_advantage_estimate(rewards, values, mask, gamma=.9, lam=.8, whiten=True)
        torch.testing.assert_close(returns, same_returns, rtol=0, atol=0)
        self.assertAlmostEqual(whitened[mask.bool()].mean().item(), 0, places=12)
        self.assertAlmostEqual(whitened[mask.bool()].var(unbiased=False).item(), 1, places=6)

    def test_gae_mask_gap_resets_recursion(self):
        advantage, returns = generalized_advantage_estimate(
            torch.tensor([[1., 999., 2.]]), torch.tensor([[.2, 999., .3]]), torch.tensor([[1, 0, 1]]), gamma=1, lam=1)
        torch.testing.assert_close(advantage, torch.tensor([[.8, 0., 1.7]]))
        torch.testing.assert_close(returns, torch.tensor([[1., 0., 2.]]))

    def test_single_token_whitening_is_finite_and_keeps_signal(self):
        values = torch.tensor([[float("nan"), 2., float("nan")]])
        result = whiten_advantages(values, torch.tensor([[0, 1, 0]]))
        torch.testing.assert_close(result, torch.tensor([[0., 2., 0.]]))
        self.assertTrue(torch.isnan(values[0, 0]))  # No in-place mutation.

    def test_reference_kl_reward_sign_and_terminal_score_once(self):
        log_two = math.log(2)
        policy = torch.tensor([[0., -log_two, -2 * log_two, 0.], [-1., -2., 0., 0.]], dtype=torch.double, requires_grad=True)
        reference = torch.tensor([[0., -2 * log_two, -log_two, 0.], [-1., -2., 0., 0.]], dtype=torch.double, requires_grad=True)
        mask = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])
        score = torch.tensor([2., 1.], dtype=torch.double, requires_grad=True)
        rewards, stats = sampled_kl_rewards(policy, reference, score, mask, kl_coefficient=.2)
        expected = torch.tensor([[0., -.2 * log_two, 2 + .2 * log_two, 0.], [0., 1., 0., 0.]], dtype=torch.double)
        torch.testing.assert_close(rewards, expected, rtol=1e-12, atol=1e-12)
        self.assertFalse(rewards.requires_grad)
        self.assertAlmostEqual(stats["kl_mean"].item(), 0, places=12)
        with self.assertRaisesRegex(ValueError, "terminal score"):
            sampled_kl_rewards(policy, reference, score, torch.zeros_like(mask))


class PPOLossTests(unittest.TestCase):
    def test_policy_clipping_numeric_and_old_rollout_is_detached(self):
        old = torch.tensor([[-.7, -1.2, -1.7, -2.3]], dtype=torch.double, requires_grad=True)
        ratios = torch.tensor([[1.5, .5, 1.5, .5]], dtype=torch.double)
        new = (old.detach() + ratios.log()).requires_grad_()
        advantages = torch.tensor([[1., 1., -1., -1.]], dtype=torch.double, requires_grad=True)
        old_values = torch.zeros_like(old, requires_grad=True)
        targets = torch.zeros_like(old, requires_grad=True)
        new_values = torch.zeros_like(old, requires_grad=True)
        loss, stats = ppo_loss(new, old, advantages, new_values, old_values, targets, torch.ones_like(old), value_coefficient=0)
        # Per-token negative clipped surrogates: -1.2, -.5, 1.5, .8.
        self.assertAlmostEqual(loss.item(), .15, places=12)
        self.assertEqual(stats["clipfrac"].item(), 1)
        self.assertAlmostEqual(stats["ratio_mean"].item(), 1, places=12)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([[0., -.125, .375, 0.]], dtype=torch.double), rtol=1e-12, atol=1e-12)
        for frozen in (old, advantages, old_values, targets):
            self.assertIsNone(frozen.grad)

    def test_clipped_value_loss_uses_old_values_and_max_of_squared_errors(self):
        zero = torch.zeros(1, 2, dtype=torch.double)
        values = torch.tensor([[.8, 0.]], dtype=torch.double, requires_grad=True)
        loss, stats = ppo_loss(zero, zero, zero, values, torch.tensor([[0., 2.]], dtype=torch.double),
                               torch.ones_like(zero), torch.ones_like(zero), value_coefficient=1)
        # Direct squared errors .04,1; clipped errors .64,.64 -> .5*mean(.64,1).
        self.assertAlmostEqual(stats["value_loss"].item(), .41, places=12)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([[0., -.5]], dtype=torch.double), rtol=1e-12, atol=1e-12)

    def test_ppo_padding_invariance_including_nan_padding(self):
        new = torch.tensor([[-1.2, -2.]], dtype=torch.double)
        old = torch.tensor([[-1.3, -1.9]], dtype=torch.double)
        adv = torch.tensor([[1., -2.]], dtype=torch.double)
        value, old_value, target = torch.tensor([[.2, .4]], dtype=torch.double), torch.zeros_like(new), torch.ones_like(new)
        mask = torch.ones_like(new)
        short_loss, short_stats = ppo_loss(new, old, adv, value, old_value, target, mask)
        def pad(array):
            return torch.cat([array, torch.full((1, 3), float("nan"), dtype=torch.double)], dim=-1)
        long_loss, long_stats = ppo_loss(*map(pad, (new, old, adv, value, old_value, target)), torch.tensor([[1, 1, 0, 0, 0]]))
        torch.testing.assert_close(short_loss, long_loss, rtol=0, atol=0)
        for key in short_stats:
            torch.testing.assert_close(short_stats[key], long_stats[key], rtol=0, atol=0)

    def test_entropy_bonus_and_empty_mask_guard(self):
        logits = torch.zeros(1, 2, 4, dtype=torch.double, requires_grad=True)
        entropies = categorical_entropy(logits)
        torch.testing.assert_close(entropies, torch.full((1, 2), math.log(4), dtype=torch.double))
        zero = torch.zeros(1, 2, dtype=torch.double)
        loss, stats = ppo_loss(zero, zero, zero, zero, zero, zero, torch.ones_like(zero),
                               entropy=entropies, entropy_coefficient=.01)
        self.assertAlmostEqual(loss.item(), -.01 * math.log(4), places=12)
        self.assertAlmostEqual(stats["entropy"].item(), math.log(4), places=12)
        with self.assertRaisesRegex(ValueError, "valid token"):
            masked_mean(zero, torch.zeros_like(zero))
        with self.assertRaisesRegex(ValueError, "requires per-token entropy"):
            ppo_loss(zero, zero, zero, zero, zero, zero, torch.ones_like(zero), entropy_coefficient=.01)


if __name__ == "__main__":
    unittest.main()
