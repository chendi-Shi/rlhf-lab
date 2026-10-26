import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from alignment_lab.token_logprobs import completion_logprobs, completion_token_logprobs


class CompletionLogProbTests(unittest.TestCase):
    def test_token_scores_and_ppo_ratio_gradient_match_dense_reference(self):
        torch.manual_seed(62)
        hidden = torch.randn(2, 8, 5, requires_grad=True)
        dense_hidden = hidden.detach().clone().requires_grad_()
        head = nn.Linear(5, 29)
        dense_head = copy.deepcopy(head)
        labels = torch.randint(0, 29, (2, 8))
        labels[:, :3] = -100
        labels[0, -2:] = -100
        result = completion_token_logprobs(hidden, head, labels, chunk_tokens=3)
        dense_logits = dense_head(dense_hidden[:, :-1]).float()
        dense = -F.cross_entropy(dense_logits.flatten(0, 1), labels[:, 1:].flatten(),
                                ignore_index=-100, reduction="none").reshape(2, 7)
        mask = labels[:, 1:].ne(-100)
        torch.testing.assert_close(result.logprobs, dense)
        self.assertTrue(torch.equal(result.mask, mask))
        self.assertTrue(torch.equal(result.counts, mask.sum(-1)))
        self.assertTrue(torch.equal(result.logprobs[~mask], torch.zeros_like(result.logprobs[~mask])))
        # Frozen rollout values: both signs and both clipping branches contribute.
        old = dense.detach() + torch.linspace(-0.4, 0.4, 14).reshape(2, 7)
        advantage = torch.tensor([[1., -1., 2., -2., 1., -1., 3.], [-1., 2., -3., 4., -2., 1., -1.]])
        def loss(new):
            ratio = (new - old).exp()
            clipped = ratio.clamp(0.8, 1.2)
            return torch.maximum(-ratio * advantage, -clipped * advantage)[mask].mean()
        loss(result.logprobs).backward()
        loss(dense).backward()
        torch.testing.assert_close(hidden.grad, dense_hidden.grad, rtol=2e-5, atol=1e-7)
        for got, want in zip(head.parameters(), dense_head.parameters()):
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-5, atol=1e-7)

    def test_eos_only_token_score_uses_last_prompt_state_and_masks_history(self):
        hidden = torch.randn(1, 6, 4, requires_grad=True)
        head = nn.Linear(4, 11)
        labels = torch.tensor([[-100, -100, -100, -100, 2, -100]])
        result = completion_token_logprobs(hidden, head, labels, chunk_tokens=1)
        self.assertEqual(result.mask.tolist(), [[False, False, False, True, False]])
        expected = head(hidden[0, 3]).log_softmax(-1)[2]
        torch.testing.assert_close(result.logprobs[0, 3], expected)
        (-result.logprobs.sum()).backward()
        self.assertGreater(hidden.grad[0, 3].abs().sum().item(), 0)
        self.assertEqual(hidden.grad[0, [0, 1, 2, 4, 5]].abs().sum().item(), 0)

    def test_dpo_loss_and_gradients_match_full_vocabulary_reference(self):
        torch.manual_seed(9)
        hidden = torch.randn(4, 9, 7, requires_grad=True)
        head = nn.Linear(7, 31, bias=True)
        labels = torch.randint(0, 31, (4, 9), dtype=torch.long)
        labels[:, :3] = -100
        labels[0, -2:] = -100
        labels[2, -1] = -100
        full_hidden = hidden.detach().clone().requires_grad_()
        full_head = copy.deepcopy(head)
        logits = full_head(full_hidden[:, :-1]).float()
        per_token = -F.cross_entropy(logits.reshape(-1, 31), labels[:, 1:].reshape(-1),
                                     ignore_index=-100, reduction="none").reshape(4, -1)
        expected = per_token.sum(-1)
        actual = completion_logprobs(hidden, head, labels, chunk_tokens=2)
        torch.testing.assert_close(actual.sums, expected, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(actual.counts, labels[:, 1:].ne(-100).sum(-1))
        # A paired nonlinear objective checks that both chosen/rejected graphs survive.
        ref_margins = torch.tensor([0.2, -0.4])
        def dpo_loss(logps):
            return -F.logsigmoid(0.1 * (logps[:2] - logps[2:] - ref_margins)).mean()
        dpo_loss(actual.sums).backward()
        dpo_loss(expected).backward()
        torch.testing.assert_close(hidden.grad, full_hidden.grad, rtol=2e-5, atol=1e-7)
        for got, want in zip(head.parameters(), full_head.parameters()):
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-5, atol=1e-7)

    def test_only_completion_positions_reach_head_and_eos_is_scored(self):
        class RecordingHead(nn.Linear):
            def __init__(self):
                super().__init__(5, 13)
                self.rows = []
            def forward(self, x):
                self.rows.append(len(x))
                return super().forward(x)
        head = RecordingHead()
        hidden = torch.randn(2, 10, 5)
        labels = torch.full((2, 10), -100, dtype=torch.long)
        labels[0, 7:9] = torch.tensor([4, 2])  # 2 is a real EOS in this fixture.
        labels[1, 8] = 2
        with torch.no_grad():
            result = completion_logprobs(hidden, head, labels, chunk_tokens=2)
        self.assertEqual(head.rows, [2, 1])
        self.assertEqual(result.counts.tolist(), [2, 1])

    def test_rejects_sequences_with_no_predicted_answer_tokens(self):
        hidden = torch.randn(2, 4, 3)
        labels = torch.full((2, 4), -100, dtype=torch.long)
        labels[0, 2] = 1
        labels[1, 0] = 1  # First token has no preceding hidden state to predict it.
        with self.assertRaisesRegex(ValueError, "completion"):
            completion_logprobs(hidden, nn.Linear(3, 7), labels)

    def test_frozen_head_still_backpropagates_to_hidden_states(self):
        hidden = torch.randn(1, 5, 4, requires_grad=True)
        head = nn.Linear(4, 17).requires_grad_(False)
        labels = torch.tensor([[-100, -100, 3, 4, 2]])
        result = completion_logprobs(hidden, head, labels, chunk_tokens=1)
        (-result.sums.mean()).backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_trainable_head_gets_gradients_with_detached_hidden_states(self):
        torch.manual_seed(21)
        hidden = torch.randn(2, 7, 4)
        head = nn.Linear(4, 19)
        reference = copy.deepcopy(head)
        labels = torch.randint(0, 19, (2, 7))
        labels[:, :2] = -100
        result = completion_logprobs(hidden, head, labels, chunk_tokens=3)
        logits = reference(hidden[:, :-1]).float()
        expected = F.cross_entropy(logits.reshape(-1, 19), labels[:, 1:].reshape(-1),
                                   ignore_index=-100, reduction="sum")
        (-result.sums.sum()).backward()
        expected.backward()
        for got, want in zip(head.parameters(), reference.parameters()):
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-5, atol=2e-6)

    def test_recomputation_matches_retained_graph_with_partial_last_chunk(self):
        torch.manual_seed(37)
        x = torch.randn(2, 6, 5, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        head = nn.Linear(5, 23)
        other = copy.deepcopy(head)
        labels = torch.randint(0, 23, (2, 6))
        labels[:, :2] = -100
        labels[1, -1] = -100  # Seven valid tokens, chunked as 3, 3, 1.
        a = completion_logprobs(x, head, labels, chunk_tokens=3, recompute_logits=True)
        b = completion_logprobs(y, other, labels, chunk_tokens=3, recompute_logits=False)
        torch.testing.assert_close(a.sums, b.sums)
        a.sums.square().sum().backward()
        b.sums.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad)
        for got, want in zip(head.parameters(), other.parameters()):
            torch.testing.assert_close(got.grad, want.grad)


if __name__ == "__main__":
    unittest.main()
