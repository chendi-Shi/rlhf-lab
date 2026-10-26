"""Hand calculations and dense-autograd CPU oracles; not a PPO training run."""
import math
import unittest

import torch

from alignment_lab.ppo_objective import (
    clipped_ppo_loss, masked_gae, shaped_rewards, whiten_advantages,
)


def tensor(values, *, grad=False, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=grad)


def inputs():
    return {
        "new_logprobs": tensor([[-0.2, -0.9, -1.8], [-1.2, -0.5, 0.0]], grad=True),
        "new_values": tensor([[0.9, 0.1, -0.2], [0.8, -0.8, 0.0]], grad=True),
        "old_logprobs": tensor([[-0.7, -0.3, -1.6], [-1.1, -0.1, 0.0]]),
        "old_values": tensor([[0.2, 0.0, 0.1], [0.1, -0.1, 0.0]]),
        "advantages": tensor([[1.1, -0.8, 0.4], [-1.3, 0.7, 0.0]]),
        "returns": tensor([[0.6, 0.8, -0.6], [-0.4, 0.4, 0.0]]),
        "mask": torch.tensor([[True, True, True], [True, True, False]]),
    }


def calculate(batch, *, denominator=5, policy_clip=0.2, value_clip=0.3, value_coef=0.4):
    return clipped_ppo_loss(**batch, policy_clip=policy_clip, value_clip=value_clip,
                            value_coef=value_coef, token_denominator=denominator)


def dense_scalar_oracle(batch, policy_clip, value_clip, value_coef, denominator):
    """Independent scalar positive-surrogate/min and branch-based critic loss."""
    policy, critic = [], []
    for row in range(batch["mask"].shape[0]):
        for column in range(batch["mask"].shape[1]):
            if not batch["mask"][row, column]:
                continue
            index = (row, column)
            ratio = torch.exp(batch["new_logprobs"][index] - batch["old_logprobs"][index])
            advantage = batch["advantages"][index]
            unclipped = ratio * advantage
            bounded = torch.clamp(ratio, 1 - policy_clip, 1 + policy_clip) * advantage
            policy.append(-torch.minimum(unclipped, bounded))
            value, old, target = (batch[key][index] for key in ("new_values", "old_values", "returns"))
            bounded_value = old + torch.clamp(value - old, -value_clip, value_clip)
            raw_error, bounded_error = (value - target) ** 2, (bounded_value - target) ** 2
            critic.append(0.5 * torch.where(raw_error >= bounded_error, raw_error, bounded_error))
    policy_sum, value_sum = torch.stack(policy).sum(), torch.stack(critic).sum()
    return policy_sum, value_sum, (policy_sum + value_coef * value_sum) / denominator


class PPOObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shaped_rewards_hand_calculation_sign_and_terminal_score_once(self):
        mask = torch.tensor([[True, True, False], [True, False, False]])
        old = tensor([[-0.2, -0.3, float("nan")], [-0.7, float("inf"), 22]])
        reference = tensor([[-0.4, -0.1, float("nan")], [-0.2, -999, float("inf")]])
        result = shaped_rewards(old, reference, tensor([2, -1]), mask, 0.5)
        torch.testing.assert_close(result, tensor([[-0.1, 2.1, 0], [-0.75, 0, 0]]))
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)
        terminal_only = shaped_rewards(old, reference, tensor([2, -1]), mask, 0)
        torch.testing.assert_close(terminal_only, tensor([[0, 2, 0], [-1, 0, 0]]))

    def test_gae_hand_calculation_and_zero_bootstrap_despite_garbage_padding(self):
        mask = torch.tensor([[True, True, False], [True, False, False]])
        rewards = tensor([[1, 2, float("nan")], [3, float("nan"), 999]])
        values = tensor([[0.5, 0.25, 999], [1, float("inf"), float("nan")]])
        advantage, returns = masked_gae(rewards, values, mask, gamma=0.9, lam=0.8)
        torch.testing.assert_close(advantage, tensor([[1.985, 1.75, 0], [2, 0, 0]]))
        torch.testing.assert_close(returns, tensor([[2.485, 2, 0], [3, 0, 0]]))
        self.assertFalse(advantage.requires_grad or returns.requires_grad)

    def test_eos_only_and_length_cap_both_terminate_without_synthetic_eos(self):
        # Row 0 generated only EOS. Row 1 reached its two-action length cap.
        # Termination reasons remain caller metadata; the math adds no action.
        mask = torch.tensor([[True, False], [True, True]])
        rewards = shaped_rewards(tensor([[0, 100], [0, 0]]), tensor([[0, -100], [0, 0]]),
                                 tensor([4, 2]), mask, kl_coef=0.1)
        advantages, returns = masked_gae(rewards, tensor([[0.6, 999], [0.4, 7]]), mask, gamma=1, lam=1)
        torch.testing.assert_close(rewards, tensor([[4, 0], [0, 2]]))
        torch.testing.assert_close(advantages, tensor([[3.4, 0], [1.6, -5]]))
        torch.testing.assert_close(returns, tensor([[4, 0], [2, 2]]))
        self.assertEqual(returns.shape, (2, 2))

    def test_gae_zero_gamma_zero_lambda_and_monte_carlo_limits(self):
        mask = torch.tensor([[True, True, True]])
        rewards, values = tensor([[1, 2, 3]]), tensor([[0.4, 0.5, 0.6]])
        _, returns = masked_gae(rewards, values, mask, gamma=0, lam=0.8)
        torch.testing.assert_close(returns, rewards)
        _, returns = masked_gae(rewards, values, mask, gamma=0.9, lam=0)
        torch.testing.assert_close(returns, tensor([[1.45, 2.54, 3]]))
        _, returns = masked_gae(rewards, values, mask, gamma=0.9, lam=1)
        torch.testing.assert_close(returns, tensor([[5.23, 4.7, 3]]))

    def test_whitening_global_population_variance_without_modifying_targets(self):
        mask = torch.tensor([[True, True], [True, False]])
        advantages = tensor([[1, 3], [7, float("nan")]])
        untouched = advantages.clone()
        result = whiten_advantages(advantages, mask, epsilon=1e-8)
        expected = tensor([[-8 / 3, -2 / 3], [10 / 3, 0]]) / math.sqrt(56 / 9 + 1e-8)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(advantages, untouched, equal_nan=True)
        self.assertAlmostEqual(result[mask].mean().item(), 0, places=12)
        self.assertFalse(result.requires_grad)
        raw_advantages, targets = masked_gae(tensor([[1, 3], [7, 0]]), torch.zeros_like(advantages), mask, 1, 1)
        before = targets.clone()
        whiten_advantages(raw_advantages, mask)
        torch.testing.assert_close(targets, before, rtol=0, atol=0)

    def test_singleton_and_constant_advantages_whiten_to_finite_zero(self):
        for values, mask in ((tensor([[9]]), torch.tensor([[True]])),
                             (tensor([[4, 4], [4, float("nan")]]), torch.tensor([[True, True], [True, False]]))):
            result = whiten_advantages(values, mask)
            torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)

    def test_policy_clip_hand_calculation_and_gradient_directions(self):
        new_logprobs = tensor([[math.log(1.5), math.log(0.5), math.log(1.5), math.log(0.5)]], grad=True)
        new_values = tensor([[0, 0, 0, 0]], grad=True)
        zero = torch.zeros_like(new_values, requires_grad=False)
        result = clipped_ppo_loss(new_logprobs, new_values, zero, zero, tensor([[1, 1, -1, -1]]), zero,
                                  torch.ones((1, 4), dtype=torch.bool), 0.2, 0.2, 0.5, 4)
        self.assertAlmostEqual(result.policy_sum.item(), 0.6, places=12)
        self.assertAlmostEqual(result.objective_contribution.item(), 0.15, places=12)
        result.objective_contribution.backward()
        torch.testing.assert_close(new_logprobs.grad, tensor([[0, -0.125, 0.375, 0]]))
        torch.testing.assert_close(new_values.grad, zero)
        self.assertEqual(result.diagnostics["policy_clip_fraction"].item(), 1)
        self.assertAlmostEqual(result.diagnostics["ratio_mean"].item(), 1, places=12)

    def test_value_clipping_half_factor_and_coefficient_are_each_applied_once(self):
        new_values = tensor([[0.9, 0.1]], grad=True)
        new_logprobs = tensor([[0, 0]], grad=True)
        zero = tensor([[0, 0]])
        result = clipped_ppo_loss(new_logprobs, new_values, zero, zero, zero, tensor([[1, 1]]),
                                  torch.tensor([[True, True]]), 0.2, 0.2, 0.5, 2)
        self.assertAlmostEqual(result.value_sum.item(), 0.725, places=12)
        self.assertAlmostEqual(result.objective_contribution.item(), 0.18125, places=12)
        result.objective_contribution.backward()
        torch.testing.assert_close(new_values.grad, tensor([[0, -0.225]]))
        torch.testing.assert_close(new_logprobs.grad, zero)

    def test_vectorized_loss_and_both_gradients_match_independent_dense_oracle(self):
        batch = inputs()
        oracle = {key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in batch.items()}
        result = calculate(batch, denominator=7)
        psum, vsum, expected = dense_scalar_oracle(oracle, 0.2, 0.3, 0.4, 7)
        torch.testing.assert_close(result.policy_sum, psum)
        torch.testing.assert_close(result.value_sum, vsum)
        torch.testing.assert_close(result.objective_contribution, expected)
        result.objective_contribution.backward()
        expected.backward()
        for key in ("new_logprobs", "new_values"):
            torch.testing.assert_close(batch[key].grad, oracle[key].grad, rtol=1e-12, atol=1e-12)
        self.assertEqual(result.valid_tokens, 5)
        self.assertEqual(result.token_denominator, 7)
        self.assertTrue(all(not value.requires_grad for value in result.diagnostics.values()))
        self.assertTrue(all(batch[key].grad is None for key in ("old_logprobs", "old_values", "advantages", "returns")))

    def test_effective_batch_denominator_reproduces_dense_gradient_accumulation(self):
        full, accumulated = inputs(), inputs()
        # Whiten once globally, not once on each optimizer microbatch.
        full["advantages"] = whiten_advantages(full["advantages"], full["mask"])
        accumulated["advantages"] = full["advantages"].clone()
        dense = calculate(full)
        dense.objective_contribution.backward()
        parts = []
        for row in range(2):
            micro = {key: value[row:row + 1] for key, value in accumulated.items()}
            part = calculate(micro, denominator=5)
            part.objective_contribution.backward()
            parts.append(part)
        torch.testing.assert_close(sum(part.objective_contribution for part in parts), dense.objective_contribution)
        torch.testing.assert_close(sum(part.policy_sum for part in parts), dense.policy_sum)
        torch.testing.assert_close(sum(part.value_sum for part in parts), dense.value_sum)
        for key in ("new_logprobs", "new_values"):
            torch.testing.assert_close(accumulated[key].grad, full[key].grad, rtol=1e-12, atol=1e-12)
        self.assertEqual([part.valid_tokens for part in parts], [3, 2])

    def test_nan_padding_does_not_change_loss_diagnostics_or_valid_gradients(self):
        baseline, padded = inputs(), {}
        for key, value in baseline.items():
            if key == "mask":
                padded[key] = torch.cat((value, torch.zeros((2, 3), dtype=torch.bool)), dim=1)
            else:
                extended = torch.cat((value.detach(), torch.full((2, 3), float("nan"), dtype=value.dtype)), dim=1)
                extended[~torch.cat((baseline["mask"], torch.zeros((2, 3), dtype=torch.bool)), dim=1)] = float("nan")
                padded[key] = extended.requires_grad_(value.requires_grad)
        original, actual = calculate(baseline), calculate(padded)
        torch.testing.assert_close(original.objective_contribution, actual.objective_contribution, rtol=0, atol=0)
        for key in original.diagnostics:
            torch.testing.assert_close(original.diagnostics[key], actual.diagnostics[key], rtol=0, atol=0)
        original.objective_contribution.backward()
        actual.objective_contribution.backward()
        for key in ("new_logprobs", "new_values"):
            torch.testing.assert_close(padded[key].grad[:, :3], baseline[key].grad, rtol=0, atol=0)
            self.assertTrue(torch.equal(padded[key].grad[~padded["mask"]], torch.zeros_like(padded[key].grad[~padded["mask"]])))

    def test_old_statistics_must_be_detached_and_new_statistics_differentiable(self):
        for key in ("old_logprobs", "old_values", "advantages", "returns"):
            batch = inputs()
            batch[key].requires_grad_(True)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "detached"):
                calculate(batch)
        for key in ("new_logprobs", "new_values"):
            batch = inputs()
            batch[key] = batch[key].detach()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "require gradients"):
                calculate(batch)
        mask = torch.tensor([[True]])
        z = tensor([[0]])
        for which in (0, 1, 2):
            arguments = [z.clone(), z.clone(), tensor([1])]
            arguments[which].requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "detached"):
                shaped_rewards(*arguments, mask, kl_coef=0.1)
        with self.assertRaisesRegex(ValueError, "detached"):
            masked_gae(z, z.clone().requires_grad_(True), mask, 1, 1)
        with self.assertRaisesRegex(ValueError, "detached"):
            whiten_advantages(z.clone().requires_grad_(True), mask)

    def test_mask_requires_nonempty_left_aligned_contiguous_bool_rows(self):
        malformed = (torch.tensor([[1, 1]]), torch.tensor([[False, True]]),
                     torch.tensor([[True, False, True]]), torch.tensor([[False, False]]),
                     torch.ones((0, 3), dtype=torch.bool), torch.ones((1, 0), dtype=torch.bool))
        for mask in malformed:
            with self.subTest(mask=mask.tolist()), self.assertRaises(ValueError):
                whiten_advantages(torch.zeros(mask.shape), mask)
        batch = inputs()
        batch["old_values"] = torch.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "shape and device"):
            calculate(batch)

    def test_nonfinite_valid_values_and_arithmetic_overflow_are_rejected(self):
        for key in ("new_logprobs", "new_values", "old_logprobs", "old_values", "advantages", "returns"):
            batch = inputs()
            with torch.no_grad():
                batch[key][0, 0] = float("nan")
            with self.subTest(key=key), self.assertRaises(FloatingPointError):
                calculate(batch)
        batch = inputs()
        with torch.no_grad():
            batch["new_logprobs"][0, 0] = 1000
        with self.assertRaisesRegex(FloatingPointError, "No|no implicit clipping"):
            calculate(batch)
        batch = inputs()
        with torch.no_grad():
            batch["new_values"][0, 0] = 1e200
        with self.assertRaises(FloatingPointError):
            calculate(batch)

    def test_hyperparameters_and_global_token_count_are_validated(self):
        batch = inputs()
        for denominator in (0, -1, 4, 5.0, True, float("inf"), float("nan")):
            with self.subTest(denominator=denominator), self.assertRaises(ValueError):
                calculate(batch, denominator=denominator)
        for key in ("policy_clip", "value_clip", "value_coef"):
            for value in (-0.1, float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    calculate(batch, **{key: value})
        mask, z = torch.tensor([[True]]), tensor([[0]])
        for gamma, lam in ((-0.1, 1), (1.1, 1), (1, -0.1), (1, 1.1), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                masked_gae(z, z, mask, gamma, lam)
        for coefficient in (-0.1, float("inf"), True):
            with self.assertRaises(ValueError):
                shaped_rewards(z, z, tensor([1]), mask, coefficient)
        for epsilon in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                whiten_advantages(z, mask, epsilon)

    def test_half_precision_inputs_are_promoted_for_loss_and_rewards(self):
        batch = inputs()
        for key, value in batch.items():
            if key != "mask":
                batch[key] = value.detach().half().requires_grad_(value.requires_grad)
        result = calculate(batch)
        self.assertEqual(result.objective_contribution.dtype, torch.float32)
        result.objective_contribution.backward()
        self.assertTrue(bool(torch.isfinite(batch["new_logprobs"].grad).all()))
        rewards = shaped_rewards(batch["old_logprobs"], batch["old_logprobs"],
                                 tensor([1, 2], dtype=torch.float16), batch["mask"], 0.1)
        self.assertEqual(rewards.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
