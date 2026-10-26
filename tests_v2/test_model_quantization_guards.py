"""Exercise the production guard with CPU/meta stand-ins, never real quantization.

Only bitsandbytes' type boundary is mocked; _check_quantized, parameter/buffer
enumeration, tensor dtypes and tied/untied storage comparisons execute unchanged.
CPU is the expected device; meta tensors represent an incorrect/offloaded device.
These tests establish rejection behavior, not CUDA/NF4 kernel compatibility.
"""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.model_loader import _check_quantized


class MockParams4bit(nn.Parameter):
    pass


def quantized_weight(device="cpu"):
    weight = MockParams4bit(torch.zeros((8, 1), device=device, dtype=torch.uint8),
                            requires_grad=False)
    weight.bnb_quantized = True
    weight.quant_state = SimpleNamespace(quant_type="nf4", nested=True)
    weight.quant_type = "nf4"
    weight.compress_statistics = True
    return weight


class MockLinear4bit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = quantized_weight()
        self.compute_dtype = torch.bfloat16


class MockBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_loaded_in_4bit = True
        self.config = SimpleNamespace(model_type="qwen3", tie_word_embeddings=True)
        self.projections = nn.ModuleList([MockLinear4bit(), MockLinear4bit()])
        self.embedding = nn.Embedding(11, 4, dtype=torch.bfloat16)
        self.output = nn.Linear(4, 11, bias=False, dtype=torch.bfloat16)
        self.output.weight = self.embedding.weight
        self.register_buffer("rope_fixture", torch.zeros(2))

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


class ModelQuantizationGuardTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        bnb = ModuleType("bitsandbytes")
        bnb.__path__ = []
        bnb_nn = ModuleType("bitsandbytes.nn")
        bnb_nn.Linear4bit = MockLinear4bit
        bnb_nn.Params4bit = MockParams4bit
        bnb.nn = bnb_nn
        mocked = patch.dict(sys.modules, {"bitsandbytes": bnb, "bitsandbytes.nn": bnb_nn})
        mocked.start()
        self.addCleanup(mocked.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def check(self, base):
        return _check_quantized(base, torch.device("cpu"))

    def test_valid_mock_counts_real_module_instances_without_forward(self):
        base = MockBase()
        with patch.object(MockLinear4bit, "forward", side_effect=AssertionError("No forward allowed")):
            self.assertEqual(self.check(base), 2)
        self.assertEqual(base.embedding.weight.data_ptr(), base.output.weight.data_ptr())

    def test_unloaded_missing_or_nonboolean_flag_rejected(self):
        for flag in (False, 1, "true", None):
            with self.subTest(flag=flag):
                base = MockBase()
                base.is_loaded_in_4bit = flag
                with self.assertRaisesRegex(ValueError, "real 4-bit"):
                    self.check(base)
        base = MockBase()
        del base.is_loaded_in_4bit
        with self.assertRaisesRegex(ValueError, "real 4-bit"):
            self.check(base)

    def test_unquantized_model_cannot_pass_using_loaded_flag_only(self):
        base = MockBase()
        base.projections = nn.ModuleList([nn.Linear(4, 4)])
        with self.assertRaisesRegex(ValueError, "Linear4bit"):
            self.check(base)

    def test_plain_parameter_is_not_accepted_as_params4bit(self):
        base = MockBase()
        base.projections[0].weight = nn.Parameter(torch.zeros((8, 1)), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "NF4"):
            self.check(base)

    def test_missing_quant_state_rejected_before_first_forward(self):
        base = MockBase()
        base.projections[0].weight.quant_state = None
        with self.assertRaisesRegex(ValueError, "incomplete NF4"):
            self.check(base)

    def test_not_yet_quantized_weight_is_rejected(self):
        base = MockBase()
        base.projections[0].weight.bnb_quantized = False
        with self.assertRaisesRegex(ValueError, "NF4"):
            self.check(base)

    def test_wrong_quantization_type_or_missing_double_quant_rejected(self):
        for field, value in (("quant_type", "fp4"), ("compress_statistics", False)):
            with self.subTest(field=field):
                base = MockBase()
                setattr(base.projections[0].weight, field, value)
                with self.assertRaisesRegex(ValueError, "NF4"):
                    self.check(base)

    def test_wrong_or_unset_compute_dtype_rejected(self):
        for dtype in (None, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                base = MockBase()
                base.projections[0].compute_dtype = dtype
                with self.assertRaisesRegex(ValueError, "NF4"):
                    self.check(base)

    def test_parameter_on_another_device_rejected(self):
        base = MockBase()
        base.projections[0].weight = quantized_weight("meta")
        with self.assertRaisesRegex(ValueError, "offloaded|another device"):
            self.check(base)

    def test_buffer_on_another_device_rejected(self):
        base = MockBase()
        base.rope_fixture = torch.zeros(2, device="meta")
        with self.assertRaisesRegex(ValueError, "offloaded|another device"):
            self.check(base)

    def test_input_and_output_embeddings_must_both_be_bfloat16(self):
        for name in ("embedding", "output"):
            with self.subTest(name=name):
                base = MockBase()
                module = getattr(base, name)
                module.weight = nn.Parameter(module.weight.detach().to(torch.float32))
                with self.assertRaisesRegex(ValueError, "remain BF16"):
                    self.check(base)

    def test_tie_flag_without_shared_storage_is_rejected(self):
        base = MockBase()
        base.output.weight = nn.Parameter(base.embedding.weight.detach().clone())
        self.assertTrue(torch.equal(base.embedding.weight, base.output.weight))
        self.assertNotEqual(base.embedding.weight.data_ptr(), base.output.weight.data_ptr())
        with self.assertRaisesRegex(ValueError, "tied Qwen3"):
            self.check(base)

    def test_shared_storage_without_tie_config_is_rejected(self):
        base = MockBase()
        base.config.tie_word_embeddings = False
        with self.assertRaisesRegex(ValueError, "tied Qwen3"):
            self.check(base)


if __name__ == "__main__":
    unittest.main()
