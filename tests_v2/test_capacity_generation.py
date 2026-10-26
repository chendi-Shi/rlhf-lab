"""Tiny random CPU Qwen/KV checks, never pretrained/GPU capacity evidence."""
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from alignment_lab.capacity_generation import greedy_capacity_probe
from alignment_lab.shared_qwen import SharedQwen3
from test_shared_qwen import fixture


def probe(backend, **kwargs):
    options = dict(eos_id=2, max_new_tokens=4, device="cpu")
    options.update(kwargs)
    return greedy_capacity_probe(backend, "policy", (4, 8, 9), **options)


class CapacityGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def test_cached_greedy_matches_dense_and_projects_only_one_position(self):
        backend, model, _, _, _, _ = fixture()
        projected, inputs, caches, masks = [], [], [], []
        head = model.get_base_model().lm_head.register_forward_pre_hook(
            lambda m, args: projected.append(tuple(args[0].shape)))
        def observe(m, args, kwargs):
            inputs.append(kwargs["input_ids"].shape[1])
            caches.append(kwargs["past_key_values"] is not None)
            masks.append(bool(kwargs["attention_mask"].bool().all()))
        decoder = model.get_base_model().model.register_forward_pre_hook(observe, with_kwargs=True)
        before = [p.detach().clone() for p in backend._all_parameters()]
        rng = torch.get_rng_state().clone()
        try:
            out = probe(backend, ignore_eos=True)
        finally:
            head.remove(); decoder.remove()
        self.assertEqual(inputs, [3, 1, 1, 1, 1])
        self.assertEqual(caches, [False, True, True, True, True])
        self.assertTrue(all(masks))
        self.assertEqual(projected, [(1, 32)] * 4)
        self.assertEqual(out["max_cache_tokens"], 7)
        self.assertTrue(out["final_action_flushed_to_kv"])
        def dense(view):
            ids, actions = (4, 8, 9), []
            for _ in range(4):
                tensor = torch.tensor([ids])
                hidden = view.hidden(tensor, torch.ones_like(tensor))
                token = int(backend._base.lm_head(hidden[:, -1]).argmax(-1).item())
                actions.append(token); ids += (token,)
            return tuple(actions)
        self.assertEqual(out["action_ids"], backend.read("policy", dense))
        for old, new in zip(before, backend._all_parameters()):
            self.assertTrue(torch.equal(old, new))
            self.assertIsNone(new.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_eos_aware_stops_while_forced_keeps_original_eos_ids(self):
        backend, _, _, _, _, _ = fixture()
        logits = torch.zeros(1, 97); logits[0, 2] = 9
        with patch.object(backend._base.lm_head, "forward", return_value=logits):
            normal = probe(backend)
            forced = probe(backend, ignore_eos=True)
        self.assertEqual(normal["action_ids"], (2,))
        self.assertEqual(normal["termination"], "eos")
        self.assertEqual(forced["action_ids"], (2, 2, 2, 2))
        self.assertEqual(forced["termination"], "forced_max_new_tokens")
        self.assertEqual(forced["generated_eos_count"], 4)
        self.assertEqual(forced["max_cache_tokens"], 7)
        self.assertFalse(forced["synthetic_eos_added"])
        self.assertFalse(forced["quality_evaluation"])

    def test_stop_and_deadline_checked_per_token_without_fake_completion(self):
        backend, _, _, _, _, _ = fixture()
        with patch.object(backend._base.model, "forward", wraps=backend._base.model.forward) as forward:
            out = probe(backend, stop_requested=lambda: True)
            self.assertEqual(forward.call_count, 0)
        self.assertEqual(out["action_ids"], ())
        self.assertEqual(out["termination"], "stop_requested")
        ticks = iter((1.0, 2.0, 3.0))
        out = probe(backend, ignore_eos=True, deadline_epoch=2.0, clock=lambda: next(ticks))
        self.assertEqual(len(out["action_ids"]), 1)
        self.assertEqual(out["termination"], "deadline")
        self.assertFalse(out["final_action_flushed_to_kv"])
        self.assertEqual(out["max_cache_tokens"], 3)

    def test_strict_inputs_pending_grad_and_nonfinite_logits_fail(self):
        backend, _, _, _, _, _ = fixture()
        for args in (dict(max_new_tokens=True), dict(max_new_tokens=126),
                     dict(eos_id=97), dict(ignore_eos=1), dict(deadline_epoch=float("nan")),
                     dict(device="cuda"), dict(device="meta")):
            with self.assertRaises(ValueError):
                probe(backend, **args)
        p = backend.parameters_for("policy")[0]
        p.grad = torch.zeros_like(p)
        with self.assertRaisesRegex(RuntimeError, "cleared"):
            probe(backend)
        p.grad = None
        with patch.object(backend._base.lm_head, "forward", return_value=torch.full((1, 97), float("nan"))):
            with self.assertRaises(FloatingPointError):
                probe(backend)
        self.assertEqual(backend._phase, "idle")

    def test_actual_cpu_1024_prompt_2048_actions_reaches_3072_cached_positions(self):
        # Small hidden size/layer count keeps this a CPU software contract test.
        config = Qwen3Config(vocab_size=19, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=3072, attention_dropout=0.0,
            pad_token_id=0, eos_token_id=2, tie_word_embeddings=False)
        config._attn_implementation = "eager"
        base = Qwen3ForCausalLM(config)
        model = get_peft_model(base, LoraConfig(r=1, lora_alpha=1,
            target_modules=["q_proj", "v_proj"], lora_dropout=0, bias="none",
            task_type="CAUSAL_LM"), adapter_name="policy")
        backend = SharedQwen3(model, trainable_roles={"policy"})
        out = greedy_capacity_probe(backend, "policy", (4,) * 1024, eos_id=2,
            max_new_tokens=2048, ignore_eos=True, device="cpu")
        self.assertEqual(out["generated_tokens"], 2048)
        self.assertEqual(out["max_cache_tokens"], 3072)
        self.assertEqual(out["decoder_calls"], 2049)
        self.assertTrue(out["final_action_flushed_to_kv"])
        self.assertEqual(out["termination"], "forced_max_new_tokens")
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
