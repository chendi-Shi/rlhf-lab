"""Real tiny Qwen3/PEFT integration; CPU random weights, never capacity evidence."""
import copy
import unittest

import torch
from torch.nn import functional as F

from alignment_lab.shared_qwen import SharedQwen3
from alignment_lab.training_steps import (
    dpo_pair_backward, reference_pair_scores, reward_pair_backward, sft_backward,
)
from test_shared_qwen import fixture


def batches(ids, attention, labels):
    chosen = dict(input_ids=ids, attention_mask=attention, labels=labels)
    rejected = {key: value.clone() for key, value in chosen.items()}
    rejected["input_ids"][:, 3] = torch.tensor([21, 22])
    rejected["labels"][:, 3] = torch.tensor([21, 22])
    # Unequal completion lengths exercise raw sums and correct denominator use.
    rejected["input_ids"][1, 5:] = torch.tensor([2, 0])
    rejected["labels"][1, 5:] = torch.tensor([2, -100])
    rejected["attention_mask"][1, 6] = 0
    return chosen, rejected


def row(batch, index):
    return {key: value[index:index + 1] for key, value in batch.items()}


def dense_model(model, role):
    dense = copy.deepcopy(model)
    dense.gradient_checkpointing_disable()
    dense.set_adapter(role)
    dense.requires_grad_(False)
    for name, param in dense.named_parameters():
        if f".{role}." in name:
            param.requires_grad_(True)
    dense.train()
    return dense


def dense_sums(model, batch):
    # Independent full-vocabulary oracle; no chunked production scorer.
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False).logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                             ignore_index=-100, reduction="none").view_as(labels)
    return -losses.sum(-1)


def dense_rewards(model, head, batch):
    hidden = model.get_base_model().model(input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    last = batch["attention_mask"].sum(-1) - 1
    return head(hidden[torch.arange(len(last)), last]).squeeze(-1)


class QwenTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def assert_role_gradients(self, model, oracle, role):
        expected = dict(oracle.named_parameters())
        active = 0.0
        for name, param in model.named_parameters():
            if f".{role}." in name:
                self.assertIsNotNone(param.grad)
                torch.testing.assert_close(param.grad, expected[name].grad,
                                           rtol=5e-4, atol=2e-7)
                active += param.grad.abs().sum().item()
            else:
                self.assertIsNone(param.grad)
        self.assertGreater(active, 0.0)

    def test_sft_accumulation_on_checkpointed_qwen_matches_dense_effective_batch(self):
        backend, model, _, ids, attention, labels = fixture()
        chosen, _ = batches(ids, attention, labels)
        dense = dense_model(model, "policy")
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        denominator = int(labels.ne(-100).sum())
        actual = [sft_backward(backend, "policy", optimizer, row(chosen, index),
                    token_denominator=denominator, chunk_tokens=2) for index in range(2)]
        expected = -dense_sums(dense, chosen).sum() / denominator
        expected.backward()
        torch.testing.assert_close(sum(r.objective_contribution for r in actual), expected.detach())
        self.assert_role_gradients(model, dense, "policy")

    def test_dpo_serial_replay_accumulates_real_qwen_and_reference_survives_policy_step(self):
        backend, model, _, ids, attention, labels = fixture()
        chosen, rejected = batches(ids, attention, labels)
        reference = [reference_pair_scores(backend, "reference", row(chosen, index),
                        row(rejected, index), chunk_tokens=2) for index in range(2)]
        # Change only the policy after caching; policy/reference no longer coincide.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B.policy." in name:
                    param.add_(0.02)
        dense = dense_model(model, "policy")
        ref_oracle = dense_model(model, "reference")
        with torch.no_grad():
            ref_margin = dense_sums(ref_oracle, chosen) - dense_sums(ref_oracle, rejected)
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        beta = 0.3
        actual = [dpo_pair_backward(backend, "policy", optimizer, row(chosen, index),
                  row(rejected, index), reference[index], beta=beta, pair_denominator=2,
                  chunk_tokens=2) for index in range(2)]
        margin = beta * (dense_sums(dense, chosen) - dense_sums(dense, rejected) - ref_margin)
        expected = F.softplus(-margin).mean()
        expected.backward()
        torch.testing.assert_close(sum(r.objective_contribution for r in actual), expected.detach())
        self.assert_role_gradients(model, dense, "policy")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # Same frozen reference inputs remain valid across a real optimizer step.
        dpo_pair_backward(backend, "policy", optimizer, row(chosen, 0), row(rejected, 0),
                          reference[0], beta=beta, pair_denominator=1, chunk_tokens=2)

    def test_reward_serial_replay_matches_dense_reward_adapter_and_scalar_head(self):
        _, model, heads, ids, attention, labels = fixture()
        backend = SharedQwen3(model, trainable_roles={"reward"}, heads=heads,
                              trainable_heads={"reward": ("reward",)})
        chosen, rejected = batches(ids, attention, labels)
        dense = dense_model(model, "reward")
        dense_head = copy.deepcopy(heads["reward"])
        dense_head.requires_grad_(True)
        optimizer = torch.optim.AdamW(backend.parameters_for("reward", heads=("reward",)), lr=1e-3)
        actual = [reward_pair_backward(backend, "reward", optimizer, row(chosen, index),
                  row(rejected, index), pair_denominator=2) for index in range(2)]
        expected = F.softplus(-(dense_rewards(dense, dense_head, chosen)
                                - dense_rewards(dense, dense_head, rejected))).mean()
        expected.backward()
        torch.testing.assert_close(sum(r.objective_contribution for r in actual), expected.detach())
        self.assert_role_gradients(model, dense, "reward")
        for param, oracle in zip(heads["reward"].parameters(), dense_head.parameters()):
            torch.testing.assert_close(param.grad, oracle.grad, rtol=5e-4, atol=2e-7)
        self.assertIsNone(heads["value"].weight.grad)


if __name__ == "__main__":
    unittest.main()
