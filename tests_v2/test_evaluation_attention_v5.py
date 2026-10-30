"""Focused CPU-only numeric and dispatch tests; never GPU capacity evidence."""
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm
from alignment_lab import evaluation_attention_v5 as attention
from alignment_lab.evaluation_attention_receipt_v5 import validate_attention_execution_receipt


def tiny_qwen(dtype=torch.float32):
    with torch.random.fork_rng():
        torch.manual_seed(73)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8,
                            max_position_embeddings=512, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    model.requires_grad_(False)
    return model


class RewardAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_qwen_forward_matches_for_fp32_bf16_causal_and_right_padding(self):
        for dtype in (torch.float32, torch.bfloat16):
            for padded in (False, True):
                with self.subTest(dtype=dtype, padded=padded):
                    model = tiny_qwen(dtype)
                    original_methods = [layer.self_attn.forward.__func__ for layer in model.model.layers]
                    ids = (torch.arange(2 * 259).reshape(2, 259) % 91) + 3
                    mask = torch.ones_like(ids)
                    if padded:
                        mask[1, -7:] = 0
                    positions = mask.cumsum(-1).sub(1).clamp_min(0)
                    weights = {k: v.clone() for k, v in model.state_dict().items()}
                    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                        original = model.model(input_ids=ids, attention_mask=mask,
                                               position_ids=positions, use_cache=False).last_hidden_state
                    self.assertTrue(all(fn is Qwen3Attention.forward for fn in original_methods))
                    owner = attention.install_reward_attention(model, stage="reward", cpu_fixture=True)
                    self.assertEqual(model.config._attn_implementation, "sdpa")
                    with torch.no_grad():
                        actual = model.model(input_ids=ids, attention_mask=mask,
                                             position_ids=positions, use_cache=False).last_hidden_state
                    tolerance = {torch.float32: (1e-5, 3e-6), torch.bfloat16: (2e-2, 2e-2)}[dtype]
                    torch.testing.assert_close(actual, original, rtol=tolerance[0], atol=tolerance[1])
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
                    receipt = validate_attention_execution_receipt(owner.receipt(), enabled=True, minimum_calls=2)
                    self.assertEqual(receipt["max_block_rows"], 128)
                    self.assertEqual(receipt["observations"][0]["query_tokens"], 259)
                    self.assertEqual(receipt["observations"][0]["key_tokens"], 259)

    def test_explicit_gqa_repeat_matches_original_torch_gqa_with_full_softmax(self):
        generator = torch.Generator().manual_seed(9)
        q = torch.randn(1, 4, 259, 8, generator=generator)
        k = torch.randn(1, 2, 259, 8, generator=generator)
        v = torch.randn(1, 2, 259, 8, generator=generator)
        owner = attention.RewardAttention((), cpu_fixture=True)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                scale=8 ** -0.5, enable_gqa=True, dropout_p=0.0)
            actual = owner.attend(q, k, v, None, scaling=8 ** -0.5)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        observation = owner.receipt()["observations"][0]
        self.assertEqual((observation["query_heads"], observation["original_kv_heads"]), (4, 2))

    def test_real_loader_style_fp32_norms_and_bf16_projections_preserve_autocast_boundary(self):
        model = tiny_qwen(torch.bfloat16)
        for module in model.modules():
            if isinstance(module, Qwen3RMSNorm):
                module.to(dtype=torch.float32)
        ids = (torch.arange(259)[None, :] % 91) + 3
        mask = torch.ones_like(ids); mask[:, -9:] = 0
        positions = mask.cumsum(-1).sub(1).clamp_min(0)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        actual_dtypes = []
        original_sdpa = F.scaled_dot_product_attention

        def capture(query, key, value, **kwargs):
            actual_dtypes.append((query.dtype, key.dtype, value.dtype))
            return original_sdpa(query, key, value, **kwargs)

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), \
                sdpa_kernel(SDPBackend.MATH), \
                patch.object(F, "scaled_dot_product_attention", side_effect=capture):
            expected = model.model(input_ids=ids, attention_mask=mask,
                                   position_ids=positions, use_cache=False).last_hidden_state
        self.assertTrue(actual_dtypes)
        self.assertTrue(all(row == (torch.float32, torch.float32, torch.bfloat16) for row in actual_dtypes))
        owner = attention.install_reward_attention(model, stage="reward", cpu_fixture=True)
        actual_dtypes.clear()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), \
                patch.object(F, "scaled_dot_product_attention", side_effect=capture):
            actual = model.model(input_ids=ids, attention_mask=mask,
                                 position_ids=positions, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(all(row == (torch.bfloat16,) * 3 for row in actual_dtypes))
        self.assertEqual(model.config._attn_implementation, "sdpa")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        receipt = validate_attention_execution_receipt(owner.receipt(), enabled=True, minimum_calls=2)
        self.assertTrue(all(row["dtype"] == "torch.bfloat16" for row in receipt["observations"]))

    def test_later_query_blocks_use_absolute_causality_and_retain_last_key(self):
        length = 259
        q = k = torch.zeros(1, 1, length, 1)
        v = torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1)
        owner = attention.RewardAttention((), cpu_fixture=True)
        observed = []
        original = F.scaled_dot_product_attention

        def record(query, key, value, **kwargs):
            observed.append((query.shape[2], key.shape[2], kwargs["is_causal"]))
            return original(query, key, value, **kwargs)

        with torch.no_grad(), patch.object(attention.F, "scaled_dot_product_attention", side_effect=record):
            result = owner.attend(q, k, v, None, scaling=1.0)
        # Uniform causal attention averages 0..absolute_position, including row 258.
        torch.testing.assert_close(result.flatten(), torch.arange(length) / 2, rtol=1e-6, atol=2e-5)
        self.assertEqual(observed, [(128, 259, False), (128, 259, False), (3, 259, False)])

    def test_boolean_and_additive_masks_preserved_against_original_sdpa(self):
        generator = torch.Generator().manual_seed(12)
        q, k, v = [torch.randn(1, 2, 259, 8, generator=generator) for _ in range(3)]
        boolean = torch.ones(259, 259, dtype=torch.bool).tril()[None, None]
        boolean[..., -11:] = False
        additive = torch.zeros_like(boolean, dtype=torch.float32).masked_fill(~boolean, float("-inf"))
        additive[..., 0] -= .35  # Preserve real additive bias, not merely masked positions.
        for mask in (boolean, additive):
            owner = attention.RewardAttention((), cpu_fixture=True)
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=.37)
                actual = owner.attend(q, k, v, mask, scaling=.37)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    def test_policy_instances_and_global_attention_interfaces_unchanged(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        functions, masks = dict(ALL_ATTENTION_FUNCTIONS), dict(ALL_MASK_ATTENTION_FUNCTIONS)
        for stage in ("base", "sft", "dpo", "ppo"):
            model = tiny_qwen()
            before = [layer.self_attn.forward for layer in model.model.layers]
            self.assertIsNone(attention.install_reward_attention(model, stage=stage, cpu_fixture=True))
            self.assertEqual(before, [layer.self_attn.forward for layer in model.model.layers])
            self.assertEqual(model.config._attn_implementation, "sdpa")
        attention.install_reward_attention(tiny_qwen(), stage="reward", cpu_fixture=True)
        self.assertEqual(functions, dict(ALL_ATTENTION_FUNCTIONS))
        self.assertEqual(masks, dict(ALL_MASK_ATTENTION_FUNCTIONS))

    def test_efficient_execution_error_is_not_retried_as_math(self):
        owner = attention.RewardAttention((), cpu_fixture=True)
        q = torch.zeros(1, 2, 3, 8)
        with torch.no_grad(), patch.object(owner, "_supports_efficient", return_value=True), \
                patch.object(attention, "sdpa_kernel", return_value=nullcontext()) as kernel, \
                patch.object(attention.F, "scaled_dot_product_attention", side_effect=torch.OutOfMemoryError("fixture")) as call, \
                patch.object(attention, "full_kv_query_blocked_math", side_effect=AssertionError("must not retry")):
            with self.assertRaises(torch.OutOfMemoryError):
                owner.attend(q, q, q, None, scaling=8 ** -.5)
        kernel.assert_called_once_with(SDPBackend.EFFICIENT_ATTENTION)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(owner.receipt()["blocked_math_executed"], 0)
        self.assertEqual(owner.receipt()["efficient_executed"], 0)

    def test_production_cpu_and_incomplete_inputs_fail_closed(self):
        q = torch.zeros(1, 2, 3, 8)
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError, "Production"):
                attention.RewardAttention((), cpu_fixture=False).attend(q, q, q, None, scaling=1)
            with self.assertRaisesRegex(ValueError, "equal-length"):
                attention.RewardAttention((), cpu_fixture=True).attend(q, q[:, :, :-1], q[:, :, :-1], None, scaling=1)
            with self.assertRaisesRegex(ValueError, "every original"):
                attention.RewardAttention((), cpu_fixture=True).attend(q, q, q, torch.ones(3, 2, dtype=torch.bool), scaling=1)
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            attention.RewardAttention((), cpu_fixture=True).attend(q, q, q, None, scaling=1)

    def test_receipt_rejects_eligibility_only_truncation_bad_precision_and_forged_counts(self):
        owner = attention.RewardAttention((), cpu_fixture=True)
        q = torch.zeros(1, 2, 3, 8)
        with torch.no_grad():
            owner.attend(q, q, q, None, scaling=1)
        receipt = owner.receipt()
        validate_attention_execution_receipt(receipt, enabled=True)
        for mutate in (
            lambda r: r.update(efficient_supported=1),
            lambda r: r.update(blocked_math_executed=0),
            lambda r: r.update(input_truncated=True),
            lambda r: r["observations"][0].update(key_tokens=2),
            lambda r: r["observations"][0].update(dtype="torch.float16"),
            lambda r: r.update(max_block_rows=129),
        ):
            changed = copy.deepcopy(receipt); mutate(changed)
            with self.assertRaises(ValueError):
                validate_attention_execution_receipt(changed, enabled=True)


if __name__ == "__main__":
    unittest.main()
