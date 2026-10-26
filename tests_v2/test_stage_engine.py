"""Bounded tiny-Qwen3 CPU loops; no pretrained weights or GPU campaign."""
import copy
import dataclasses
import unittest

import torch

from alignment_lab.shared_qwen import SharedQwen3
from alignment_lab.stage_engine import SupervisedConfig, PPOConfig, run_supervised, run_ppo, parameter_mapping
from alignment_lab.training_steps import reference_pair_scores
from test_shared_qwen import fixture


def examples():
    return [dict(record_id=f"row-{i}", prompt_ids=(4 + i, 8),
                 chosen_ids=(11 + i, 2), rejected_ids=(16 + i, 18, 2)) for i in range(3)]


def supervised(kind="sft", **kwargs):
    values = dict(kind=kind, seed=42, effective_batch=2, microbatch=1, max_updates=20,
                  deadline_epoch=1000.0, pad_id=0, eos_id=2, chunk_tokens=2)
    values.update(kwargs)
    return SupervisedConfig(**values)


def ppo(**kwargs):
    values = dict(seed=42, rollout_batch=2, optimizer_batch=1, epochs=2, max_responses=3,
                  max_updates=20, max_new_tokens=3, deadline_epoch=1000.0, pad_id=0, eos_id=2, chunk_tokens=2)
    values.update(kwargs)
    return PPOConfig(**values)


def ppo_run(backend, optimizer, generator, config, **kwargs):
    return run_ppo(backend, "policy", optimizer, examples(), config, device="cpu", generator=generator,
                   reference_role="reference", reward_role="reward", clock=lambda: 100.0, **kwargs)


class StageEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def test_all_supervised_stages_finish_one_pass_and_clear_gradients(self):
        for kind in ("sft", "reward", "dpo"):
            with self.subTest(kind=kind):
                backend, model, heads, _, _, _ = fixture()
                role, selected_heads = "policy", ()
                if kind == "reward":
                    role, selected_heads = "reward", ("reward",)
                    backend = SharedQwen3(model, trainable_roles={role}, heads=heads, trainable_heads={role: selected_heads})
                optimizer = torch.optim.AdamW(backend.parameters_for(role, heads=selected_heads), lr=1e-3)
                before = [p.detach().clone() for p in backend.parameters_for(role, heads=selected_heads)]
                calls = []
                provider = lambda c, r: reference_pair_scores(backend, "reference", c, r, chunk_tokens=2)
                outcome = run_supervised(backend, role, optimizer, examples(), supervised(kind), device="cpu",
                            reference_provider=provider, checkpoint_callback=lambda c, e: calls.append(c), clock=lambda: 100)
                self.assertEqual(outcome.stop_reason, "data_exhausted")
                self.assertEqual(outcome.cursor["optimizer_updates"], 2)
                self.assertEqual(outcome.cursor["offset"], 3)
                self.assertEqual(len(calls), 2)
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, backend.parameters_for(role, heads=selected_heads))))
                self.assertTrue(all(p.grad is None for p in backend._all_parameters()))

    def test_supervised_pause_resume_does_not_repeat_rows_or_reset_limits(self):
        backend, _, _, _, _, _ = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        saved = []
        partial = run_supervised(backend, "policy", optimizer, examples(), supervised(), device="cpu",
            checkpoint_callback=lambda c, e: saved.append(c), stop_requested=lambda: bool(saved), clock=lambda: 100)
        self.assertEqual(partial.stop_reason, "paused")
        self.assertEqual(partial.cursor["offset"], 2)
        invalid = copy.deepcopy(partial.cursor)
        invalid["offset"] = 1
        invalid["response_tokens"] = len(examples()[invalid["order"][0]]["chosen_ids"])
        with self.assertRaisesRegex(ValueError, "boundary"):
            run_supervised(backend, "policy", optimizer, examples(), supervised(), device="cpu",
                           state=invalid, clock=lambda: 100)
        for changed in (supervised(deadline_epoch=2000), supervised(seed=43), supervised(max_updates=21)):
            with self.assertRaisesRegex(ValueError, "identity"):
                run_supervised(backend, "policy", optimizer, examples(), changed, device="cpu", state=partial.cursor)
        finished = run_supervised(backend, "policy", optimizer, examples(), supervised(), device="cpu",
                                  state=partial.cursor, clock=lambda: 100)
        self.assertEqual(finished.cursor["offset"], 3)
        self.assertEqual(finished.cursor["optimizer_updates"], 2)
        self.assertEqual(len(finished.metrics), 1)

    def test_expired_deadline_or_stop_dispatches_no_optimizer_update(self):
        for reason in ("paused", "budget_exhausted"):
            backend, _, _, _, _, _ = fixture()
            optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
            out = run_supervised(backend, "policy", optimizer, examples(), supervised(), device="cpu",
                      stop_requested=lambda: reason == "paused", clock=lambda: 1000)
            self.assertEqual(out.stop_reason, reason)
            self.assertEqual(out.cursor["optimizer_updates"], 0)
            self.assertEqual(len(optimizer.state), 0)

    def test_supervised_partial_work_limit_is_not_data_completion(self):
        backend, _, _, _, _, _ = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        out = run_supervised(backend, "policy", optimizer, examples(), supervised(max_updates=1), device="cpu", clock=lambda: 100)
        self.assertEqual(out.stop_reason, "update_limit")
        self.assertEqual(out.cursor["offset"], 2)

    def test_dpo_stop_or_deadline_during_reference_preparation_dispatches_no_update(self):
        for reason in ("paused", "budget_exhausted"):
            with self.subTest(reason=reason):
                backend, _, _, _, _, _ = fixture()
                optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
                cached = []
                def provider(chosen, rejected):
                    result = reference_pair_scores(backend, "reference", chosen, rejected, chunk_tokens=2)
                    cached.append(True)
                    return result
                out = run_supervised(backend, "policy", optimizer, examples(), supervised("dpo"),
                    device="cpu", reference_provider=provider,
                    stop_requested=lambda: reason == "paused" and bool(cached),
                    clock=lambda: 1000 if reason == "budget_exhausted" and cached else 100)
                self.assertEqual(out.stop_reason, reason)
                self.assertEqual(out.cursor["optimizer_updates"], 0)
                self.assertEqual(len(optimizer.state), 0)
                self.assertTrue(all(p.grad is None for p in backend._all_parameters()))

    def test_ppo_mid_epoch_pause_resume_equals_continuous_run(self):
        def initialized():
            backend, _, _, _, _, _ = fixture()
            optimizer = torch.optim.AdamW(backend.parameters_for("policy", heads=("value",)), lr=1e-3)
            return backend, optimizer, torch.Generator().manual_seed(321)
        baseline, opt, gen = initialized()
        full = ppo_run(baseline, opt, gen, ppo())
        self.assertEqual(full.stop_reason, "data_exhausted")
        self.assertEqual(full.cursor["optimizer_updates"], 6)
        backend, optimizer, generator = initialized()
        saved = []
        partial = ppo_run(backend, optimizer, generator, ppo(), checkpoint_callback=lambda c, e: saved.append((c, e)),
                          stop_requested=lambda: len(saved) == 3)
        self.assertEqual(partial.stop_reason, "paused")
        self.assertTrue(partial.cursor["active_rollout"])
        self.assertEqual(partial.extra_state["epoch"], 1)
        frozen_old = copy.deepcopy(partial.extra_state["responses"])
        final = ppo_run(backend, optimizer, generator, ppo(), state=partial.cursor, extra_state=partial.extra_state)
        self.assertEqual(final.cursor, full.cursor)
        self.assertEqual(partial.extra_state["responses"], frozen_old)
        self.assertEqual([m["objective"] for m in full.metrics[3:]], [m["objective"] for m in final.metrics])
        for a, b in zip(baseline._all_parameters(), backend._all_parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(gen.get_state(), generator.get_state()))

    def test_ppo_resume_requires_frozen_active_statistics_and_consistent_cursor(self):
        backend, _, _, _, _, _ = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy", heads=("value",)), lr=1e-3)
        generator, saved = torch.Generator().manual_seed(321), []
        out = ppo_run(backend, optimizer, generator, ppo(), checkpoint_callback=lambda c, e: saved.append(c),
                      stop_requested=lambda: len(saved) == 1)
        with self.assertRaisesRegex(ValueError, "missing"):
            ppo_run(backend, optimizer, generator, ppo(), state=out.cursor)
        bad = copy.deepcopy(out.cursor);bad["optimizer_updates"] += 1
        with self.assertRaisesRegex(ValueError, "counters"):
            ppo_run(backend, optimizer, generator, ppo(), state=bad, extra_state=out.extra_state)
        bad_extra = copy.deepcopy(out.extra_state);bad_extra["record_ids"] = ["wrong", "wrong"]
        with self.assertRaisesRegex(ValueError, "dataset cursor"):
            ppo_run(backend, optimizer, generator, ppo(), state=out.cursor, extra_state=bad_extra)
        bad_extra = copy.deepcopy(out.extra_state)
        bad_extra["responses"][0]["prompt_ids"] = (45, 46)
        with self.assertRaisesRegex(ValueError, "dataset cursor"):
            ppo_run(backend, optimizer, generator, ppo(), state=out.cursor, extra_state=bad_extra)
        bad_extra = copy.deepcopy(out.extra_state)
        bad_extra["responses"] = bad_extra["responses"][:1]
        bad_extra["record_ids"] = bad_extra["record_ids"][:1]
        bad_extra["offset"] = 0
        for key in ("advantages", "returns", "reference_logprobs"):
            bad_extra[key] = bad_extra[key][:1, :len(bad_extra["responses"][0]["action_ids"])]
        bad_extra["reward_scores"] = bad_extra["reward_scores"][:1]
        invalid = copy.deepcopy(out.cursor)
        invalid["offset"] = 1
        invalid["optimizer_updates"] = 0
        with self.assertRaisesRegex(ValueError, "rollout size"):
            ppo_run(backend, optimizer, generator, ppo(), state=invalid, extra_state=bad_extra)

    def test_parameter_mapping_matches_optimizer_order_without_base_weights(self):
        backend, _, _, _, _, _ = fixture()
        selected = backend.parameters_for("policy", heads=("value",))
        mapping = parameter_mapping(backend, "policy", heads=("value",))
        self.assertEqual(list(map(id, selected)), list(map(id, mapping.values())))
        self.assertTrue(all(".policy." in name or name.startswith("scalar_heads.value.") for name in mapping))


if __name__ == "__main__":
    unittest.main()
