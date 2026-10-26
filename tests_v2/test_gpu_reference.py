"""CPU receipt/cache and mocked quant-state tests, NOT CUDA/BNB kernel evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from alignment_lab import gpu_reference as gpu
from alignment_lab.training_steps import dpo_pair_backward, reference_pair_scores
from tests_v2.test_reference_bridge import identity_inputs, paired_batches, optimizer, slice_batch
from tests_v2.test_shared_qwen import fixture


class MockQuantState:
    def __init__(self, *, nested=False, inner=False):
        self.absmax = torch.ones(1, dtype=torch.uint8 if nested else torch.float32)
        self.shape = None if inner else torch.Size((8, 8))
        self.code = torch.linspace(-1, 1, 256 if inner else 16)
        self.dtype = torch.float32
        self.blocksize = 256 if inner else 64
        self.quant_type = None if inner else "nf4"
        self.offset = torch.tensor(0.25) if nested else None
        self.state2 = MockQuantState(inner=True) if nested else None
        self.nested = nested


class MockParams4bit(torch.nn.Parameter):
    pass


class MockLinear4bit:
    pass


def quant_fixture(nested=True):
    """Byte/state layout only. Never used as a working quantized forward model."""
    parameter = MockParams4bit(torch.arange(32, dtype=torch.uint8).reshape(32, 1), requires_grad=False)
    parameter.quant_state = MockQuantState(nested=nested)
    parameter.bnb_quantized = True
    parameter.quant_storage = torch.uint8
    parameter.quant_type = "nf4"
    parameter.compress_statistics = nested
    parameter.blocksize = 64
    module = MockLinear4bit()
    module.quant_state = parameter.quant_state
    module.quant_storage = torch.uint8
    module.compute_dtype = torch.bfloat16
    module.compute_type_is_set = True
    module.in_features = module.out_features = 8
    config = {"method": "bitsandbytes", "bits": 4, "quant_type": "nf4",
              "double_quant": nested, "compute_dtype": "bfloat16"}
    return parameter, module, config


class GPUReferenceCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parts = fixture()
        self.backend = self.parts[0]
        self.chosen, self.rejected = paired_batches(self.parts)
        inputs = identity_inputs(self.root / "inputs", self.backend)
        self.accepted_sha = inputs.pop("accepted_dataset_manifest_sha256")
        self.scoring_config = inputs.pop("scoring_config")
        self.artifacts = inputs
        self.output = self.root / "cache"

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def capture(self, backend=None, **overrides):
        args = dict(backend=backend or self.backend, reference_role="reference", artifact_files=self.artifacts,
                    accepted_data_manifest_sha256=self.accepted_sha, precision="float32",
                    quantization="none", hash_chunk_bytes=128)
        args.update(overrides)
        return gpu.capture_frozen_reference(**args)

    def create(self, receipt=None, **overrides):
        args = dict(receipt=receipt or self.capture(), output_dir=self.output,
                    pairs=[(self.chosen, self.rejected)], scoring_config=self.scoring_config, chunk_tokens=2)
        args.update(overrides)
        return gpu.create_reference_cache(**args)

    def bind(self, receipt, binding, **overrides):
        args = dict(receipt=receipt, output_dir=self.output, expected_identity=binding.identity,
                    expected_manifest_sha256=binding.manifest_sha256)
        args.update(overrides)
        return gpu.bind_reference_cache(**args)

    def test_content_hash_transfers_are_bounded_and_hash_exact_raw_tensor_bytes(self):
        tensor = torch.arange(103, dtype=torch.float32)
        copied_sizes = []
        original = gpu._copy_chunk_to_cpu

        def copy_chunk(chunk):
            copied_sizes.append(chunk.numel() * chunk.element_size())
            return original(chunk)

        with patch.object(gpu, "_copy_chunk_to_cpu", side_effect=copy_chunk):
            actual = gpu._tensor_hash(tensor, 32)
        expected = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        self.assertEqual(actual["sha256"], expected)
        self.assertGreater(len(copied_sizes), 1)
        self.assertLessEqual(max(copied_sizes), 32)
        self.assertEqual(sum(copied_sizes), tensor.numel() * tensor.element_size())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            gpu._tensor_hash(torch.ones(5, 7).t(), 32)
        with self.assertRaises(FloatingPointError):
            gpu._tensor_hash(torch.tensor([float("nan")]), 32)

    def test_bfloat16_bytes_are_hashed_without_numpy_bfloat16_conversion(self):
        tensor = torch.tensor([0.5, 1, -2], dtype=torch.bfloat16)
        expected = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
        self.assertEqual(gpu._tensor_hash(tensor, 16)["sha256"], expected)

    def test_receipt_reports_full_versus_version_guards_honestly_and_rejects_tracked_updates(self):
        receipt = self.capture()
        self.assertEqual(receipt.verify(), {"verified": True, "mode": "identity_versions_structure_file_stat",
                                           "untracked_content_writes_checked": False, "file_contents_rehashed": False})
        self.assertEqual(receipt.verify(full=True)["mode"], "full_content")
        with torch.no_grad():
            self.backend._parameters["reference"][0].add_(0.1)
        with self.assertRaisesRegex(RuntimeError, "identity/version"):
            receipt.verify()

    def test_untracked_data_write_requires_full_audit_and_is_detected_there(self):
        receipt = self.capture()
        parameter = self.backend._parameters["reference"][0]
        before = parameter._version
        parameter.data.add_(0.1)
        self.assertEqual(parameter._version, before)
        self.assertFalse(receipt.verify()["untracked_content_writes_checked"])
        with self.assertRaisesRegex(RuntimeError, "contents changed"):
            receipt.verify(full=True)

    def test_reference_storage_replacement_and_configuration_changes_are_rejected(self):
        receipt = self.capture()
        parameter = self.backend._parameters["reference"][0]
        parameter.data = parameter.data.clone()
        with self.assertRaises(RuntimeError):
            receipt.verify()
        receipt = self.capture()
        self.backend._base.config.max_position_embeddings *= 2
        with self.assertRaises(RuntimeError):
            receipt.verify()

    def test_cache_creation_really_reads_reference_and_lookup_does_not_forward(self):
        receipt = self.capture()
        with patch.object(self.backend, "read", wraps=self.backend.read) as reads:
            binding = self.create(receipt, pairs=[(self.chosen, self.rejected)] * 2)
        self.assertEqual([call.args[0] for call in reads.call_args_list], ["reference"] * 4)
        with patch.object(self.backend, "read", side_effect=AssertionError("lookup must not forward")):
            cached = binding.pair_scores(self.chosen, self.rejected)
        online = reference_pair_scores(self.backend, "reference", self.chosen, self.rejected, chunk_tokens=2)
        torch.testing.assert_close(cached.chosen, online.chosen, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(cached.rejected, online.rejected, rtol=1e-6, atol=2e-6)
        self.assertEqual(cached.chosen_counts.tolist(), [2, 4])
        self.assertEqual(binding.identity["scoring_config"][gpu.ANCHOR_KEY], receipt.anchor)

    def test_fresh_backend_disk_reload_matches_online_dpo_loss_and_gradients(self):
        binding = self.create()
        fresh, online_backend = fixture()[0], fixture()[0]
        loaded = self.bind(self.capture(fresh), binding)
        cached = loaded.pair_scores(self.chosen, self.rejected)
        online = reference_pair_scores(online_backend, "reference", self.chosen, self.rejected, chunk_tokens=2)
        actual = dpo_pair_backward(fresh, "policy", optimizer(fresh), self.chosen, self.rejected,
                                   cached, beta=0.1, pair_denominator=2, chunk_tokens=2)
        expected = dpo_pair_backward(online_backend, "policy", optimizer(online_backend), self.chosen, self.rejected,
                                     online, beta=0.1, pair_denominator=2, chunk_tokens=2)
        torch.testing.assert_close(actual.objective_sum, expected.objective_sum, rtol=1e-6, atol=1e-6)
        for a, b in zip(fresh.parameters_for("policy"), online_backend.parameters_for("policy")):
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=1e-7)
        loaded.receipt.verify(full=True)

    def test_policy_accumulation_and_update_preserve_receipt_without_adapter_switch(self):
        receipt = self.capture()
        binding = self.create(receipt)
        opt = optimizer(self.backend)
        first = tuple(slice_batch(batch, 0) for batch in (self.chosen, self.rejected))
        second = tuple(slice_batch(batch, 1) for batch in (self.chosen, self.rejected))
        dpo_pair_backward(self.backend, "policy", opt, *first, binding.pair_scores(*first),
                          beta=0.1, pair_denominator=2, chunk_tokens=2)
        with patch.object(self.backend, "read", side_effect=AssertionError("no switch/forward")):
            second_reference = binding.pair_scores(*second)
        dpo_pair_backward(self.backend, "policy", opt, *second, second_reference,
                          beta=0.1, pair_denominator=2, chunk_tokens=2)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for head in self.parts[2].values():
                head.weight.add_(0.25)
        receipt.verify(full=True)
        binding.pair_scores(self.chosen, self.rejected)

    def test_manifest_pin_identity_and_exact_tokens_are_enforced(self):
        receipt = self.capture()
        binding = self.create(receipt)
        with self.assertRaises(ValueError):
            self.bind(receipt, binding, expected_manifest_sha256=None)
        identity = binding.identity
        identity["scoring_config"]["template"] = "different"
        with self.assertRaises(ValueError):
            self.bind(receipt, binding, expected_identity=identity)
        changed = {name: value.clone() for name, value in self.chosen.items()}
        changed["input_ids"][0, 3] = 40
        changed["labels"][0, 3] = 40
        with self.assertRaisesRegex(ValueError, "Exact prompt/completion"):
            binding.pair_scores(changed, self.rejected)
        manifest = self.output / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaises(RuntimeError):
            binding.pair_scores(self.chosen, self.rejected)

    def test_declared_artifact_files_and_accepted_dataset_are_rehashed(self):
        receipt = self.capture()
        for path in (self.artifacts["base_files"]["weights"], self.artifacts["adapter_files"]["reference"],
                     self.artifacts["dataset_manifest"]):
            original = path.read_bytes()
            path.write_bytes(original + b"tamper")
            with self.assertRaises(RuntimeError):
                receipt.verify()
            path.write_bytes(original)
            # Stat snapshots intentionally reject even restored contents until
            # a new receipt is captured and fully verified.
            receipt = self.capture()
        with self.assertRaises(ValueError):
            self.capture(accepted_data_manifest_sha256="0" * 64)

    def test_existing_cache_no_overwrite_unknown_config_and_ambient_autocast_rejected(self):
        receipt = self.capture()
        binding = self.create(receipt)
        original = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with patch.object(self.backend, "read", side_effect=AssertionError("no rescore")):
            with self.assertRaises(FileExistsError):
                self.create(receipt)
            self.bind(receipt, binding).pair_scores(self.chosen, self.rejected)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.output.iterdir()})
        with self.assertRaisesRegex(ValueError, "Only explicit"):
            self.capture(quantization={"method": "gptq"})
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "Ambient autocast"):
                self.create(receipt, output_dir=self.root / "other")
        self.assertFalse((self.root / "other").exists())

    def test_does_not_accept_caller_anchor_or_quantization_claim_without_actual_cuda_weights(self):
        receipt = self.capture()
        config = copy.deepcopy(self.scoring_config)
        config[gpu.ANCHOR_KEY] = {"asserted": "trusted"}
        with self.assertRaisesRegex(ValueError, "not caller assertions"):
            self.create(receipt, scoring_config=config)
        quant = {"method": "bitsandbytes", "bits": 4, "quant_type": "nf4",
                 "double_quant": True, "compute_dtype": "bfloat16"}
        with self.assertRaisesRegex(ValueError, "CPU reference receipt"):
            self.capture(precision="bfloat16", quantization=quant)
        self.backend._model.is_loaded_in_4bit = True
        with self.assertRaisesRegex(ValueError, "quantization flags"):
            self.capture()

    def test_fast_lookup_uses_in_memory_cache_and_no_full_tensor_or_file_hashing(self):
        binding = self.create()
        with patch.object(gpu, "load_cache", side_effect=AssertionError("no cache reload")), \
                patch.object(gpu, "validate_identity", side_effect=AssertionError("no full identity hash")), \
                patch.object(gpu, "_tensor_hash", side_effect=AssertionError("no full tensor hash")), \
                patch("alignment_lab.reference_cache._file_record", side_effect=AssertionError("no file hash")):
            binding.pair_scores(self.chosen, self.rejected)
        with (self.output / "scores.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(RuntimeError, "cache file stat"):
            binding.pair_scores(self.chosen, self.rejected)

    def test_same_stat_source_and_cache_tamper_are_detected_by_explicit_full_boundary_audit(self):
        receipt = self.capture()
        binding = self.create(receipt)
        for path, audit in (
                (self.artifacts["source_files"]["fixture_protocol"], receipt.verify),
                (self.output / "scores.jsonl", binding.verify)):
            original, before = path.read_bytes(), path.stat()
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertFalse(audit(full=False)["file_contents_rehashed"])
            with self.assertRaises(ValueError):
                audit(full=True)
            path.write_bytes(original)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            audit(full=True)


class QuantStateMockTests(unittest.TestCase):
    """Validate schema/guard semantics without importing BNB or running kernels."""

    def inspect(self, parameter, module, config):
        tensors, objects = {}, {}
        metadata = gpu._inspect_quantized_weight(parameter, module, config, name="weight",
                                                 tensors=tensors, guards=objects,
                                                 types=(MockParams4bit, MockLinear4bit, MockQuantState))
        return metadata, tensors, objects

    def test_mock_nested_state_captures_all_tensor_contents_and_object_guards(self):
        parameter, module, config = quant_fixture()
        metadata, tensors, objects = self.inspect(parameter, module, config)
        self.assertTrue(metadata["state"]["nested"])
        self.assertEqual(len(tensors), 5)
        self.assertEqual(len(objects), 2)
        before = {name: gpu._tensor_hash(value, 32) for name, value in tensors.items()}
        state = parameter.quant_state.state2
        before_guard = gpu._tensor_guard(state.code)
        state.code.add_(0.01)
        self.assertNotEqual(before_guard, gpu._tensor_guard(state.code))
        after = {name: gpu._tensor_hash(value, 32) for name, value in tensors.items()}
        self.assertNotEqual(before, after)

    def test_mock_non_nested_state_and_metadata_changes(self):
        parameter, module, config = quant_fixture(nested=False)
        metadata, tensors, objects = self.inspect(parameter, module, config)
        self.assertFalse(metadata["state"]["nested"])
        self.assertEqual(len(tensors), 2)
        self.assertEqual(len(objects), 1)
        parameter.quant_state.blocksize = 128
        with self.assertRaisesRegex(ValueError, "Packed weight shape"):
            self.inspect(parameter, module, config)

    def test_unknown_state_schema_missing_state_and_cpu_packing_are_rejected(self):
        for change in (
                lambda p, m: setattr(p, "quant_state", None),
                lambda p, m: setattr(p.quant_state, "packing_format_for_cpu", True),
                lambda p, m: setattr(p.quant_state, "absmax", None),
                lambda p, m: setattr(p.quant_state.state2, "state2", MockQuantState(inner=True)),
                lambda p, m: setattr(m, "compute_type_is_set", False),
                lambda p, m: setattr(m, "compute_dtype", torch.float16),
                lambda p, m: setattr(p, "bnb_quantized", False),
        ):
            with self.subTest(change=change):
                parameter, module, config = quant_fixture()
                change(parameter, module)
                with self.assertRaises(ValueError):
                    self.inspect(parameter, module, config)

    def test_quantization_storage_counts_and_nested_content_are_verified(self):
        for change in (
                lambda p: setattr(p.quant_state, "shape", torch.Size((16, 8))),
                lambda p: setattr(p.quant_state, "quant_type", "unknown"),
                lambda p: setattr(p.quant_state, "absmax", torch.ones(2, dtype=torch.uint8)),
                lambda p: setattr(p.quant_state.state2, "code", torch.ones(16)),
                lambda p: setattr(p.quant_state, "offset", torch.ones(2)),
        ):
            with self.subTest(change=change):
                parameter, module, config = quant_fixture()
                change(parameter)
                with self.assertRaises(ValueError):
                    self.inspect(parameter, module, config)

    def test_mock_quant_state_untracked_write_needs_content_hash(self):
        parameter, module, config = quant_fixture()
        _, tensors, _ = self.inspect(parameter, module, config)
        code = parameter.quant_state.code
        before_guard = gpu._tensor_guard(code)
        before_hash = gpu._tensor_hash(code, 32)
        code.data.add_(0.1)
        self.assertEqual(before_guard, gpu._tensor_guard(code))
        self.assertNotEqual(before_hash, gpu._tensor_hash(code, 32))
        self.assertTrue(tensors)


if __name__ == "__main__":
    unittest.main()
