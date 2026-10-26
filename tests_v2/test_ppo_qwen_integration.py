"""One tiny random CPU PPO update with an independent dense gradient oracle."""
import copy
import unittest

import torch
from torch.nn import functional as F

from alignment_lab.ppo_objective import shaped_rewards, masked_gae, whiten_advantages, clipped_ppo_loss
from alignment_lab.rollout import sample_rollout, pack_rollout, compact_actions, score_frozen_rollout
from test_qwen_training_integration import dense_model
import test_qwen_training_integration as integration
from test_shared_qwen import fixture


class PPOQwenIntegrationTests(unittest.TestCase):
    def test_cached_rollout_frozen_scores_and_ppo_update_match_dense_gradients(self):
        torch.set_num_threads(2)
        backend, model, heads, _, _, _ = fixture()
        rollout = sample_rollout(backend, "policy", ((4, 8, 9), (5, 7)), eos_id=2, pad_id=0,
                    max_new_tokens=4, generator=torch.Generator().manual_seed(19), device="cpu")
        packed = pack_rollout(rollout, device="cpu")
        inputs = {key: packed[key] for key in ("input_ids", "attention_mask", "labels")}
        mask = packed["action_mask"]
        ref, reward = score_frozen_rollout(backend, rollout, reference_role="reference", reward_role="reward", device="cpu")
        rewards = shaped_rewards(packed["old_logprobs"], ref, reward, mask, 0.05)
        advantages, returns = masked_gae(rewards, packed["old_values"], mask, 1.0, 0.95)
        advantages = whiten_advantages(advantages, mask)
        old_snapshots = [tensor.clone() for tensor in (packed["old_logprobs"], packed["old_values"], advantages, returns, ref, reward)]
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if ".policy." not in name}
        rm_head_before = {name: p.detach().clone() for name, p in heads["reward"].named_parameters()}
        dense = dense_model(model, "policy")
        dense_head = copy.deepcopy(heads["value"])
        dense_head.requires_grad_(True)
        optimizer = torch.optim.AdamW(backend.parameters_for("policy", heads=("value",)), lr=1e-3)
        denominator = int(mask.sum())
        results = []

        def loss(view):
            scores, values = view.token_scores_and_values(**inputs, chunk_tokens=2)
            results.append(clipped_ppo_loss(compact_actions(scores.logprobs, rollout),
                compact_actions(values, rollout), packed["old_logprobs"], packed["old_values"],
                advantages, returns, mask, 0.2, 0.2, 0.5, denominator))
            return results[-1].objective_contribution

        actual = backend.backward("policy", optimizer, loss, heads=("value",))
        hidden = dense.get_base_model().model(input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], use_cache=False).last_hidden_state
        logits = dense.get_base_model().lm_head(hidden[:, :-1]).float()
        labels = inputs["labels"][:, 1:]
        full_lp = -F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                    ignore_index=-100, reduction="none").view_as(labels)
        lp = compact_actions(full_lp, rollout)
        values = compact_actions(dense_head(hidden[:, :-1]).squeeze(-1), rollout)
        ratio = (lp - packed["old_logprobs"]).exp()
        policy = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
        vc = packed["old_values"] + (values - packed["old_values"]).clamp(-0.2, 0.2)
        critic = 0.5 * torch.maximum((values - returns).square(), (vc - returns).square())
        expected = (policy[mask].sum() + 0.5 * critic[mask].sum()) / denominator
        expected.backward()
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-4, atol=2e-6)
        integration.QwenTrainingIntegrationTests.assert_role_gradients(self, model, dense, "policy")
        for param, oracle in zip(heads["value"].parameters(), dense_head.parameters()):
            torch.testing.assert_close(param.grad, oracle.grad, rtol=5e-4, atol=2e-7)
        torch.testing.assert_close(results[0].diagnostics["ratio_mean"], torch.tensor(1.0), rtol=1e-5, atol=2e-6)
        policy_before = [p.detach().clone() for p in backend.parameters_for("policy", heads=("value",))]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(policy_before, backend.parameters_for("policy", heads=("value",)))))
        for name, param in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(param, frozen[name], rtol=0, atol=0)
        for name, param in heads["reward"].named_parameters():
            torch.testing.assert_close(param, rm_head_before[name], rtol=0, atol=0)
        for previous, tensor in zip(old_snapshots, (packed["old_logprobs"], packed["old_values"], advantages, returns, ref, reward)):
            torch.testing.assert_close(previous, tensor, rtol=0, atol=0)
        ref_after = backend.read("reference", lambda view: view.token_scores(**inputs))
        reward_after = backend.read("reward", lambda view: view.rewards(**inputs))
        torch.testing.assert_close(compact_actions(ref_after.logprobs, rollout), ref, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(reward_after, reward, rtol=1e-5, atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
