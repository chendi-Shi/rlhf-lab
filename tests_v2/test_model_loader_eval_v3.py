"""Tiny local CPU/meta regression checks; never CUDA or production model loading."""
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from alignment_lab.model_loader_eval_v3 import (
    EvaluationQwen3ForCausalLM, _validate_loading_info,
)


def config():
    result = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, eos_token_id=2, pad_token_id=0,
        tie_word_embeddings=True, attention_dropout=0.0)
    result._attn_implementation = "eager"
    return result


def diagnostics(**changes):
    fields = dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=set(),
                  error_msgs=[], conversion_errors={})
    fields.update(changes)
    return SimpleNamespace(**fields)


class EvaluationLoaderV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise AssertionError("These regression checks require CUDA to remain uninitialized")

    def test_initialized_full_vocab_linear_never_casts_or_draws_random_values(self):
        model = EvaluationQwen3ForCausalLM(config())
        # Exact production matrix dimensions without allocating its storage.
        linear = nn.Linear(2048, 151936, bias=False, device="meta", dtype=torch.bfloat16)
        linear.weight._is_hf_initialized = True
        rng = torch.get_rng_state().clone()
        with patch.object(torch.Tensor, "float", side_effect=AssertionError("FP32 temporary forbidden")):
            model._init_weights(linear)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_initialized_weight_and_bias_are_bytewise_unchanged(self):
        model = EvaluationQwen3ForCausalLM(config())
        linear = nn.Linear(8, 5, dtype=torch.bfloat16)
        before = [p.detach().clone() for p in linear.parameters()]
        for parameter in linear.parameters():
            parameter._is_hf_initialized = True
        with patch.object(torch.Tensor, "float", side_effect=AssertionError("No cast allowed")):
            model._init_weights(linear)
        for parameter, expected in zip(linear.parameters(), before):
            self.assertTrue(torch.equal(parameter, expected))

    def test_uninitialized_or_partial_linear_still_uses_original_initializer(self):
        model = EvaluationQwen3ForCausalLM(config())
        for initialized_weight in (False, True):
            with self.subTest(initialized_weight=initialized_weight):
                linear = nn.Linear(8, 5)
                if initialized_weight:
                    linear.weight._is_hf_initialized = True
                with torch.no_grad():
                    linear.bias.fill_(9)
                original = Qwen3ForCausalLM._init_weights
                with patch.object(Qwen3ForCausalLM, "_init_weights", autospec=True,
                                  side_effect=original) as invoked:
                    model._init_weights(linear)
                invoked.assert_called_once()
                self.assertTrue(torch.equal(linear.bias, torch.zeros_like(linear.bias)))

    def test_nonpersistent_rope_buffers_still_get_real_initialization(self):
        model = EvaluationQwen3ForCausalLM(config())
        rope = model.model.rotary_emb
        expected, _ = rope.compute_default_rope_parameters(rope.config)
        with torch.no_grad():
            rope.inv_freq.zero_()
            rope.original_inv_freq.zero_()
        for buffer in rope.buffers():
            if hasattr(buffer, "_is_hf_initialized"):
                delattr(buffer, "_is_hf_initialized")
        model._init_weights(rope)
        torch.testing.assert_close(rope.inv_freq, expected, rtol=0, atol=0)
        torch.testing.assert_close(rope.original_inv_freq, expected, rtol=0, atol=0)

    def test_raw_load_diagnostics_fail_closed(self):
        model = EvaluationQwen3ForCausalLM(config())
        for parameter in model.parameters():
            parameter._is_hf_initialized = True
        bad = [dict(missing_keys={"model.layers.0.self_attn.q_proj.weight"}),
               dict(missing_keys={"model.embed_tokens.weight", "lm_head.weight"}),
               dict(unexpected_keys={"extra.weight"}),
               dict(mismatched_keys={("model.norm.weight", (33,), (32,))}),
               dict(error_msgs=["load failed"]),
               dict(conversion_errors={"model.layers.0.self_attn.q_proj.weight": "failed"})]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _validate_loading_info(model, diagnostics(**changes), allow_tied_missing=True)

    def test_unloaded_parameter_is_rejected_even_if_diagnostics_claim_complete(self):
        model = EvaluationQwen3ForCausalLM(config())
        for parameter in model.parameters():
            parameter._is_hf_initialized = True
        delattr(model.model.norm.weight, "_is_hf_initialized")
        with self.assertRaisesRegex(ValueError, "not actually loaded"):
            _validate_loading_info(model, diagnostics(), allow_tied_missing=True)

    def test_real_local_roundtrip_preserves_all_weights_ties_and_rope(self):
        source = Qwen3ForCausalLM(config()).to(dtype=torch.bfloat16)
        expected = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
        with TemporaryDirectory() as directory:
            source.save_pretrained(directory, safe_serialization=True)
            loaded, info = EvaluationQwen3ForCausalLM.from_pretrained(
                directory, local_files_only=True, use_safetensors=True,
                dtype=torch.bfloat16, device_map={"": "cpu"},
                attn_implementation="eager", output_loading_info=True)
            try:
                self.assertTrue(all(not value for value in info.values()))
                self.assertEqual(set(loaded.state_dict()), set(expected))
                for name, tensor in loaded.state_dict().items():
                    self.assertTrue(torch.equal(tensor, expected[name]), name)
                self.assertEqual(loaded.lm_head.weight.data_ptr(), loaded.model.embed_tokens.weight.data_ptr())
                self.assertTrue(torch.isfinite(loaded.model.rotary_emb.inv_freq).all())
                self.assertTrue(torch.count_nonzero(loaded.model.rotary_emb.inv_freq))
                self.assertFalse(torch.cuda.is_initialized())
            finally:
                # Windows cannot remove a safetensors file while CPU tensors map it.
                tensor = None
                del loaded
                gc.collect()

    def test_real_local_missing_checkpoint_weight_cannot_be_randomly_initialized(self):
        source = Qwen3ForCausalLM(config())
        with TemporaryDirectory() as directory:
            source.save_pretrained(directory, safe_serialization=True)
            checkpoint = Path(directory) / "model.safetensors"
            tensors = load_file(str(checkpoint))
            del tensors["model.layers.0.self_attn.q_proj.weight"]
            save_file(tensors, str(checkpoint), metadata={"format": "pt"})
            with self.assertRaisesRegex(ValueError, "missing accepted weights"):
                EvaluationQwen3ForCausalLM.from_pretrained(
                    directory, local_files_only=True, use_safetensors=True,
                    dtype=torch.bfloat16, device_map={"": "cpu"}, attn_implementation="eager")


if __name__ == "__main__":
    unittest.main()
