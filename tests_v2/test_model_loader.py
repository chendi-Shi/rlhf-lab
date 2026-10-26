"""Offline tiny random CPU construction checks; no CUDA or pretrained loading."""
import dataclasses
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from alignment_lab.model_loader import ModelLoadConfig, _attach_roles, validate_config


def tiny():
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, eos_token_id=2, pad_token_id=0,
        tie_word_embeddings=True, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg)


class ModelLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def test_stage_roles_heads_and_frozen_base_are_explicit(self):
        for stage in ("sft", "reward", "dpo", "ppo"):
            with self.subTest(stage=stage):
                base = tiny()
                backend, role, heads = _attach_roles(base, ModelLoadConfig(stage), torch.device("cpu"))
                selected = backend.parameters_for(role, heads=heads)
                self.assertTrue(selected)
                self.assertTrue(all(p.dtype == torch.float32 for p in selected))
                self.assertTrue(all(not p.requires_grad for p in backend._all_parameters()))
                self.assertTrue(base.is_gradient_checkpointing)
                if stage in ("dpo", "ppo"):
                    backend.require_frozen_role("reference")
                if stage == "ppo":
                    backend.require_frozen_role("reward")
                    self.assertTrue(torch.equal(backend._heads["value"].weight,
                                                torch.zeros_like(backend._heads["value"].weight)))
                    with self.assertRaises(ValueError):
                        backend.parameters_for("reward", heads=("reward",))

    def test_seeded_construction_is_repeatable_and_restores_cpu_rng(self):
        base = tiny()
        rng = torch.get_rng_state().clone()
        first, _, _ = _attach_roles(base, ModelLoadConfig("ppo", seed=123), torch.device("cpu"))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        second, _, _ = _attach_roles(tiny(), ModelLoadConfig("ppo", seed=123), torch.device("cpu"))
        for role in first._roles:
            for a, b in zip(first._parameters[role], second._parameters[role]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for head in first._heads:
            for a, b in zip(first._heads[head].parameters(), second._heads[head].parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_no_implicit_cpu_auto_or_unindexed_gpu_configuration(self):
        for device in ("cpu", "auto", "cuda", "cuda:-1"):
            with self.assertRaises(ValueError):
                validate_config(ModelLoadConfig("sft", device=device))
        for kwargs in ({"stage": "unknown"}, {"lora_r": 0}, {"seed": True},
                       {"target_modules": ("q_proj", "q_proj")}, {"attention": "flash_attention_2"}):
            with self.assertRaises(ValueError):
                validate_config(dataclasses.replace(ModelLoadConfig("sft"), **kwargs))


if __name__ == "__main__":
    unittest.main()
