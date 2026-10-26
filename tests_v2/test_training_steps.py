"""Tiny float64 CPU protocol fixture; not pretrained or CUDA capacity evidence."""
import copy
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from alignment_lab.training_steps import (
    dpo_pair_backward, reference_pair_scores, reward_pair_backward, sft_backward,
)


class ToyBackend:
    """Small nonlinear model implementing the shared read/backward contract.

    Dense comparisons below use the same weights directly, bypassing replay and
    its analytic derivative implementation. No Transformers/PEFT dependency.
    """

    def __init__(self, trainable="policy"):
        torch.manual_seed(701)
        self.embedding = nn.Embedding(19, 5).double()
        self.lm_head = nn.Linear(5, 19).double()
        self.adapters = nn.ModuleDict({name: nn.Linear(5, 5, bias=False).double()
                                       for name in ("policy", "reference", "reward")})
        self.heads = nn.ModuleDict({name: nn.Linear(5, 1).double() for name in ("reward", "value")})
        self.trainable = {trainable}
        self.active_role = None
        self.events = []
        self.live_graphs = 0
        self.max_live_graphs = 0
        self.replay_drift = False
        self.bad_reference_grad = False
        self.mutate_after_backward = False
        for p in self._all_parameters():
            p.requires_grad_(False)

    def _all_parameters(self):
        for module in (self.embedding, self.lm_head, self.adapters, self.heads):
            yield from module.parameters()

    def parameters_for(self, role, *, heads=()):
        if role not in self.trainable:
            raise ValueError("Frozen/unknown training role")
        if heads and (role != "reward" or heads != ("reward",)):
            raise ValueError("Head is not trainable for this role")
        return list(self.adapters[role].parameters()) + [p for name in heads for p in self.heads[name].parameters()]

    def require_frozen_role(self, role):
        if role not in self.adapters or role in self.trainable:
            raise ValueError("Reference must be a known frozen role")

    def frozen_role_versions(self, role):
        self.require_frozen_role(role)
        params = list(self.embedding.parameters()) + list(self.lm_head.parameters()) + list(self.adapters[role].parameters())
        return tuple((id(p), p._version) for p in params)

    def _activate(self, role, training, heads=()):
        if self.active_role != role and any(p.grad is not None for p in self._all_parameters()):
            raise RuntimeError("Clear gradients before switching adapter")
        selected = self.parameters_for(role, heads=heads) if training else []
        for p in self._all_parameters():
            p.requires_grad_(False)
        for p in selected:
            p.requires_grad_(True)
        self.active_role = role

    def hidden(self, role, input_ids):
        embedding = self.embedding(input_ids)
        # Cumulative context gives earlier prompt tokens a real influence.
        positions = torch.arange(1, input_ids.shape[1] + 1, dtype=torch.float64)
        context = embedding.cumsum(1) / positions[None, :, None]
        return torch.tanh(context + self.adapters[role](context))

    def sequence(self, role, batch):
        hidden = self.hidden(role, batch["input_ids"])
        logits = self.lm_head(hidden[:, :-1])
        labels = batch["labels"][:, 1:]
        mask = labels.ne(-100)
        gathered = logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return SimpleNamespace(sums=gathered.masked_fill(~mask, 0).sum(-1), counts=mask.sum(-1))

    def rewards(self, role, batch, head="reward"):
        hidden = self.hidden(role, batch["input_ids"])
        last = batch["attention_mask"].sum(-1) - 1
        return self.heads[head](hidden[torch.arange(len(last)), last]).squeeze(-1)

    def read(self, role, callback):
        self._activate(role, False)
        self.events.append(("read", role))
        with torch.no_grad():
            return callback(ToyView(self, role, False))

    def backward(self, role, optimizer, callback, *, heads=()):
        self._activate(role, True, heads)
        selected = self.parameters_for(role, heads=heads)
        assert {id(p) for p in selected} == {id(p) for g in optimizer.param_groups for p in g["params"]}
        self.events.append(("backward", role))
        loss = callback(ToyView(self, role, True))
        loss.backward()
        assert all(p.grad is None for p in self._all_parameters() if id(p) not in set(map(id, selected)))
        if self.mutate_after_backward:
            with torch.no_grad():
                selected[0].add_(0.1)
        return loss.detach()


class ToyView:
    def __init__(self, backend, role, training):
        self.backend, self.role, self.training = backend, role, training

    def _track(self, value):
        if self.training:
            if self.backend.replay_drift:
                value = value + 0.1
            self.backend.live_graphs += 1
            self.backend.max_live_graphs = max(self.backend.max_live_graphs, self.backend.live_graphs)
            value.register_hook(lambda gradient: self._consumed(gradient))
        elif self.backend.bad_reference_grad:
            value.requires_grad_(True)
        return value

    def _consumed(self, gradient):
        self.backend.live_graphs -= 1
        return gradient

    def sequence_scores(self, input_ids, attention_mask, labels, *, chunk_tokens):
        batch = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        result = self.backend.sequence(self.role, batch)
        return SimpleNamespace(sums=self._track(result.sums), counts=result.counts)

    def rewards(self, input_ids, attention_mask, labels, *, head):
        batch = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return self._track(self.backend.rewards(self.role, batch, head))


def batches():
    def make(rows):
        ids = torch.tensor(rows, dtype=torch.long)
        attention = ids.ne(0).long()
        labels = ids.clone()
        labels[:, :2] = -100
        labels[~attention.bool()] = -100
        return dict(input_ids=ids, attention_mask=attention, labels=labels)
    return (make([[1, 3, 5, 6, 2, 0], [4, 3, 7, 2, 0, 0]]),
            make([[1, 3, 8, 2, 0, 0], [4, 3, 9, 10, 11, 2]]))


def sliced(batch, index):
    return {key: value[index:index + 1] for key, value in batch.items()}


def optimizer(backend, role="policy"):
    return torch.optim.SGD(backend.parameters_for(role, heads=("reward",) if role == "reward" else ()), lr=0.03)


def dense_parameters(backend, role):
    selected = backend.parameters_for(role, heads=("reward",) if role == "reward" else ())
    for p in selected:
        p.requires_grad_(True)
    return selected


class TrainingStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_gradients(self, actual, expected, role):
        left = actual.parameters_for(role, heads=("reward",) if role == "reward" else ())
        right = expected.parameters_for(role, heads=("reward",) if role == "reward" else ())
        for a, b in zip(left, right):
            self.assertIsNotNone(a.grad)
            self.assertIsNotNone(b.grad)
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-12)
        selected = set(map(id, left))
        self.assertTrue(all(p.grad is None for p in actual._all_parameters() if id(p) not in selected))

    def test_sft_accumulation_matches_token_weighted_dense_ce_without_stepping(self):
        backend, chosen = ToyBackend(), batches()[0]
        dense = copy.deepcopy(backend)
        dense_parameters(dense, "policy")
        before = [p.detach().clone() for p in backend._all_parameters()]
        opt = optimizer(backend)
        # Unequal completion lengths (3 and 2) expose accidental mean-of-means.
        results = [sft_backward(backend, "policy", opt, sliced(chosen, i), token_denominator=5) for i in range(2)]
        logits = dense.lm_head(dense.hidden("policy", chosen["input_ids"])[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), chosen["labels"][:, 1:].flatten(), ignore_index=-100)
        loss.backward()
        torch.testing.assert_close(sum(r.objective_contribution for r in results), loss.detach())
        self.assertEqual([r.units for r in results], [3, 2])
        self.assertTrue(all(r.backward_surrogate is None for r in results))
        self.assert_gradients(backend, dense, "policy")
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, backend._all_parameters())))

    def test_bradley_terry_serial_replay_matches_dense_joint_graph_and_head_gradients(self):
        backend, (chosen, rejected) = ToyBackend("reward"), batches()
        dense = copy.deepcopy(backend)
        dense_parameters(dense, "reward")
        result = reward_pair_backward(backend, "reward", optimizer(backend, "reward"), chosen, rejected, pair_denominator=2)
        margin = dense.rewards("reward", chosen) - dense.rewards("reward", rejected)
        loss = -F.logsigmoid(margin).mean()
        loss.backward()
        torch.testing.assert_close(result.objective_mean, loss.detach())
        self.assertFalse(torch.isclose(result.backward_surrogate, result.objective_contribution).item())
        self.assert_gradients(backend, dense, "reward")
        self.assertEqual(backend.events, [("read", "reward")] * 2 + [("backward", "reward")] * 2)
        self.assertEqual((backend.max_live_graphs, backend.live_graphs), (1, 0))

    def test_reward_microbatch_accumulation_matches_dense(self):
        backend, (chosen, rejected) = ToyBackend("reward"), batches()
        dense = copy.deepcopy(backend)
        dense_parameters(dense, "reward")
        opt = optimizer(backend, "reward")
        results = [reward_pair_backward(backend, "reward", opt, sliced(chosen, i), sliced(rejected, i),
                                       pair_denominator=2) for i in range(2)]
        loss = F.softplus(-(dense.rewards("reward", chosen) - dense.rewards("reward", rejected))).mean()
        loss.backward()
        torch.testing.assert_close(sum(r.objective_contribution for r in results), loss.detach())
        self.assert_gradients(backend, dense, "reward")

    def test_dpo_sum_scores_loss_and_gradients_match_dense_not_length_means(self):
        backend, (chosen, rejected) = ToyBackend(), batches()
        dense = copy.deepcopy(backend)
        dense_parameters(dense, "policy")
        reference = reference_pair_scores(backend, "reference", chosen, rejected, chunk_tokens=2)
        result = dpo_pair_backward(backend, "policy", optimizer(backend), chosen, rejected, reference,
                                  beta=0.37, pair_denominator=2, chunk_tokens=2)
        pc, pr = dense.sequence("policy", chosen), dense.sequence("policy", rejected)
        with torch.no_grad():
            rc, rr = dense.sequence("reference", chosen), dense.sequence("reference", rejected)
        expected = -F.logsigmoid(0.37 * ((pc.sums - pr.sums) - (rc.sums - rr.sums))).mean()
        wrong = F.softplus(-0.37 * ((pc.sums / pc.counts - pr.sums / pr.counts) -
                                   (rc.sums / rc.counts - rr.sums / rr.counts))).mean()
        expected.backward()
        torch.testing.assert_close(result.objective_mean, expected.detach())
        self.assertFalse(torch.isclose(result.objective_mean, wrong.detach(), rtol=1e-5, atol=1e-6).item())
        self.assert_gradients(backend, dense, "policy")
        self.assertEqual((backend.max_live_graphs, backend.live_graphs), (1, 0))
        self.assertFalse(reference.chosen.requires_grad)
        self.assertFalse(reference.rejected.requires_grad)

    def test_dpo_cached_reference_allows_microbatch_accumulation_without_role_switch(self):
        backend, (chosen, rejected) = ToyBackend(), batches()
        dense = copy.deepcopy(backend)
        dense_parameters(dense, "policy")
        refs = [reference_pair_scores(backend, "reference", sliced(chosen, i), sliced(rejected, i)) for i in range(2)]
        opt = optimizer(backend)
        results = [dpo_pair_backward(backend, "policy", opt, sliced(chosen, i), sliced(rejected, i), refs[i],
                                    beta=0.2, pair_denominator=2) for i in range(2)]
        with torch.no_grad():
            ref_margin = dense.sequence("reference", chosen).sums - dense.sequence("reference", rejected).sums
        loss = F.softplus(-0.2 * (dense.sequence("policy", chosen).sums - dense.sequence("policy", rejected).sums - ref_margin)).mean()
        loss.backward()
        torch.testing.assert_close(sum(r.objective_contribution for r in results), loss.detach())
        self.assert_gradients(backend, dense, "policy")
        self.assertTrue(all(role == "policy" for _, role in backend.events[4:]))
        with self.assertRaisesRegex(RuntimeError, "Clear gradients"):
            reference_pair_scores(backend, "reference", chosen, rejected)

    def test_identical_policy_and_reference_have_hand_calculated_log_two_loss(self):
        backend, (chosen, rejected) = ToyBackend(), batches()
        backend.adapters["reference"].load_state_dict(backend.adapters["policy"].state_dict())
        reference = reference_pair_scores(backend, "reference", chosen, rejected)
        result = dpo_pair_backward(backend, "policy", optimizer(backend), chosen, rejected, reference,
                                  beta=0.1, pair_denominator=2)
        self.assertAlmostEqual(result.objective_mean.item(), math.log(2), places=13)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in backend.parameters_for("policy")), 0)

    def test_reference_cache_survives_policy_step_but_not_base_or_reference_updates(self):
        backend, (chosen, rejected) = ToyBackend(), batches()
        reference = reference_pair_scores(backend, "reference", chosen, rejected)
        opt = optimizer(backend)
        dpo_pair_backward(backend, "policy", opt, chosen, rejected, reference, beta=0.2, pair_denominator=2)
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.assertEqual(reference.frozen_versions, backend.frozen_role_versions("reference"))
        dpo_pair_backward(backend, "policy", opt, chosen, rejected, reference, beta=0.2, pair_denominator=2)
        for mutate in ("base", "reference"):
            with self.subTest(mutate=mutate):
                changed = ToyBackend()
                cached = reference_pair_scores(changed, "reference", chosen, rejected)
                target = changed.embedding.weight if mutate == "base" else changed.adapters["reference"].weight
                with torch.no_grad():
                    target.add_(0.01)
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    dpo_pair_backward(changed, "policy", optimizer(changed), chosen, rejected, cached,
                                      beta=0.2, pair_denominator=2)
                self.assertTrue(all(p.grad is None for p in changed._all_parameters()))

    def test_frozen_reference_required_even_if_a_record_claims_it_is_frozen(self):
        backend, (chosen, rejected) = ToyBackend(), batches()
        for role in ("policy", "absent"):
            with self.assertRaisesRegex(ValueError, "known frozen"):
                reference_pair_scores(backend, role, chosen, rejected)
        reference = reference_pair_scores(backend, "reference", chosen, rejected)
        backend.trainable.add("reference")
        with self.assertRaisesRegex(ValueError, "known frozen"):
            dpo_pair_backward(backend, "policy", optimizer(backend), chosen, rejected, reference,
                              beta=0.2, pair_denominator=2)
        backend = ToyBackend()
        backend.bad_reference_grad = True
        with self.assertRaisesRegex(ValueError, "detached"):
            reference_pair_scores(backend, "reference", chosen, rejected)

    def test_wrong_optimizer_and_frozen_roles_rejected_before_forward(self):
        backend, (chosen, rejected) = ToyBackend("reward"), batches()
        incomplete = torch.optim.SGD(backend.adapters["reward"].parameters(), lr=0.01)
        with self.assertRaisesRegex(ValueError, "exactly"):
            reward_pair_backward(backend, "reward", incomplete, chosen, rejected, pair_denominator=2)
        self.assertEqual(backend.events, [])
        backend = ToyBackend()
        wrong = torch.optim.SGD(list(backend.parameters_for("policy")) + list(backend.embedding.parameters()), lr=0.01)
        with self.assertRaisesRegex(ValueError, "exactly"):
            sft_backward(backend, "policy", wrong, chosen, token_denominator=5)
        with self.assertRaisesRegex(ValueError, "Frozen"):
            sft_backward(backend, "reference", wrong, chosen, token_denominator=5)
        self.assertEqual(backend.events, [])

    def test_replay_drift_and_parameter_mutation_raise_instead_of_silently_updating(self):
        chosen, rejected = batches()
        for algorithm in ("reward", "dpo"):
            with self.subTest(algorithm=algorithm):
                backend = ToyBackend("reward" if algorithm == "reward" else "policy")
                ref = None if algorithm == "reward" else reference_pair_scores(backend, "reference", chosen, rejected)
                backend.replay_drift = True
                with self.assertRaisesRegex(RuntimeError, "Replay score differs"):
                    if algorithm == "reward":
                        reward_pair_backward(backend, "reward", optimizer(backend, "reward"), chosen, rejected, pair_denominator=2)
                    else:
                        dpo_pair_backward(backend, "policy", optimizer(backend), chosen, rejected, ref, beta=0.1, pair_denominator=2)
                self.assertTrue(all(p.grad is None for p in backend._all_parameters()))
        backend = ToyBackend("reward")
        backend.mutate_after_backward = True
        with self.assertRaisesRegex(RuntimeError, "Parameters changed"):
            reward_pair_backward(backend, "reward", optimizer(backend, "reward"), chosen, rejected, pair_denominator=2)
        self.assertEqual(sum(event == "backward" for event, _ in backend.events), 1)

    def test_reference_input_owner_and_tensor_integrity(self):
        chosen, rejected = batches()
        backend = ToyBackend()
        reference = reference_pair_scores(backend, "reference", chosen, rejected)
        changed = {key: value.clone() for key, value in chosen.items()}
        changed["input_ids"][0, 2] = changed["labels"][0, 2] = 12
        with self.assertRaisesRegex(ValueError, "exact chosen/rejected"):
            dpo_pair_backward(backend, "policy", optimizer(backend), changed, rejected, reference, beta=0.2, pair_denominator=2)
        other = ToyBackend()
        with self.assertRaisesRegex(ValueError, "different backend"):
            dpo_pair_backward(other, "policy", optimizer(other), chosen, rejected, reference, beta=0.2, pair_denominator=2)
        differentiable = replace(reference, chosen=reference.chosen.clone().requires_grad_(True))
        with self.assertRaisesRegex(ValueError, "detached"):
            dpo_pair_backward(backend, "policy", optimizer(backend), chosen, rejected, differentiable,
                              beta=0.2, pair_denominator=2)
        reference.chosen.add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "modified"):
            dpo_pair_backward(backend, "policy", optimizer(backend), chosen, rejected, reference, beta=0.2, pair_denominator=2)

    def test_beta_and_normalization_must_be_finite_positive_valid_counts(self):
        backend, (chosen, rejected) = ToyBackend(), batches()
        reference = reference_pair_scores(backend, "reference", chosen, rejected)
        opt = optimizer(backend)
        for beta in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                dpo_pair_backward(backend, "policy", opt, chosen, rejected, reference, beta=beta, pair_denominator=2)
        for denominator in (0, -1, float("nan"), float("inf"), True, 1, 2.5):
            with self.subTest(denominator=denominator), self.assertRaises(ValueError):
                dpo_pair_backward(backend, "policy", opt, chosen, rejected, reference, beta=0.2, pair_denominator=denominator)
        for keyword in ("length_normalize", "clip_range", "reference_free"):
            with self.subTest(keyword=keyword), self.assertRaises(TypeError):
                dpo_pair_backward(backend, "policy", opt, chosen, rejected, reference, beta=0.2,
                                  pair_denominator=2, **{keyword: True})

    def test_finite_inputs_that_overflow_dpo_margin_are_rejected_before_backward(self):
        backend, (chosen, rejected) = ToyBackend(), batches()
        reference = reference_pair_scores(backend, "reference", chosen, rejected)
        # A finite positive beta is insufficient if its products overflow.
        huge = replace(reference, chosen=torch.full_like(reference.chosen, 1e308),
                       rejected=torch.full_like(reference.rejected, -1e308))
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite DPO"):
            dpo_pair_backward(backend, "policy", optimizer(backend), chosen, rejected, huge,
                              beta=10, pair_denominator=2)
        self.assertTrue(all(p.grad is None for p in backend._all_parameters()))

    def test_padding_invariance_and_identical_prompts_are_required(self):
        chosen, rejected = batches()
        baseline, padded = ToyBackend("reward"), ToyBackend("reward")
        a = reward_pair_backward(baseline, "reward", optimizer(baseline, "reward"), chosen, rejected, pair_denominator=2)
        def pad(batch):
            return {key: F.pad(value, (0, 3), value=-100 if key == "labels" else 0) for key, value in batch.items()}
        b = reward_pair_backward(padded, "reward", optimizer(padded, "reward"), pad(chosen), pad(rejected), pair_denominator=2)
        torch.testing.assert_close(a.objective_sum, b.objective_sum, rtol=0, atol=0)
        self.assert_gradients(baseline, padded, "reward")
        invalid = {key: value.clone() for key, value in rejected.items()}
        invalid["input_ids"][0, 0] = 6
        with self.assertRaisesRegex(ValueError, "identical prompt"):
            reference_pair_scores(ToyBackend(), "reference", chosen, invalid)
        invalid = {key: value.clone() for key, value in chosen.items()}
        invalid["labels"][0, 3] = -100
        with self.assertRaisesRegex(ValueError, "contiguous"):
            sft_backward(ToyBackend(), "policy", optimizer(ToyBackend()), invalid, token_denominator=5)


if __name__ == "__main__":
    unittest.main()
