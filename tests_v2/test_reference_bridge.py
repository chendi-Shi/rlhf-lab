"""Actual tiny random Qwen CPU scoring; no pretrained/GPU capacity evidence."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.metadata
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
from safetensors.torch import save_file

from alignment_lab.reference_bridge import (
    ANCHOR_KEY, PROTOCOL_KEY, bind_reference_cache, create_reference_cache,
)
from alignment_lab.reference_cache import load_cache
from alignment_lab.training_steps import dpo_pair_backward, reference_pair_scores
from tests_v2.test_shared_qwen import fixture


def paired_batches(parts):
    _, _, _, ids, attention, labels = parts
    chosen = {"input_ids": ids.clone(), "attention_mask": attention.clone(), "labels": labels.clone()}
    rejected = {name: tensor.clone() for name, tensor in chosen.items()}
    # Same prompt, differing completion, unequal completion lengths, and one real
    # pad-ID action. Padding is determined by attention/labels, never token ID.
    rejected["input_ids"][0, 3] = 0
    rejected["labels"][0, 3] = 0
    rejected["input_ids"][1, 3:6] = torch.tensor([20, 21, 22])
    rejected["labels"][1, 3:6] = torch.tensor([20, 21, 22])
    return chosen, rejected


def identity_inputs(directory, backend):
    """Persist actual tiny base/reference weights and explicit fixture metadata."""
    directory.mkdir()
    all_adapters = {id(p) for params in backend._parameters.values() for p in params}
    reference = {id(p) for p in backend._parameters["reference"]}
    base_path, adapter_path = directory / "base.safetensors", directory / "reference.safetensors"
    base_tensors = {name: p.detach().clone().contiguous() for name, p in backend._model.named_parameters()
                    if id(p) not in all_adapters}
    base_tensors.update({name: b.detach().clone().contiguous() for name, b in backend._model.named_buffers()})
    save_file(base_tensors, str(base_path))
    save_file({name: p.detach().clone().contiguous() for name, p in backend._model.named_parameters()
               if id(p) in reference}, str(adapter_path))
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_text(json.dumps({"fixture": "exact_integer_ids", "vocab_size": 97}), encoding="utf-8")
    scoring_source = directory / "fixture_protocol.py"
    scoring_source.write_text("# Exact fixture IDs; no tokenizer transformation.\n", encoding="utf-8")
    manifest = directory / "accepted-data.json"
    manifest.write_text(json.dumps({
        "status": "audit_complete", "data_ready": True,
        "readiness_blockers": [], "training_started": False,
        "fixture_only": True,
    }), encoding="utf-8")
    return {
        "base_files": {"weights": base_path},
        "adapter_files": {"reference": adapter_path},
        "tokenizer_files": {"tokenizer": tokenizer_path},
        "source_files": {"fixture_protocol": scoring_source, "fixture": Path(__file__).resolve()},
        "dataset_manifest": manifest,
        "accepted_dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "scoring_config": {
            "template": "tiny_fixture_prompt_then_completion",
            "serialization": "exact_int64_ids_no_tokenizer_invocation",
            "precision": "float32", "quantization": "none",
            "library_versions": {
                name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
            },
        },
    }


def optimizer(backend):
    return torch.optim.SGD(backend.parameters_for("policy"), lr=0.05)


def slice_batch(batch, index):
    return {key: value[index:index + 1] for key, value in batch.items()}


class ReferenceBridgeTests(unittest.TestCase):
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
        self.identity_kwargs = identity_inputs(self.root / "inputs", self.backend)
        self.output = self.root / "cache"

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def create(self, **overrides):
        arguments = dict(backend=self.backend, reference_role="reference",
                         output_dir=self.output, pairs=[(self.chosen, self.rejected)],
                         identity_kwargs=self.identity_kwargs, chunk_tokens=2)
        arguments.update(overrides)
        return create_reference_cache(**arguments)

    def reload(self, binding, backend=None, **overrides):
        arguments = dict(backend=backend or self.backend, reference_role="reference",
                         output_dir=self.output, expected_identity=binding.identity,
                         expected_manifest_sha256=binding.manifest_sha256)
        arguments.update(overrides)
        return bind_reference_cache(**arguments)

    def assert_scores(self, actual, expected):
        torch.testing.assert_close(actual.chosen, expected.chosen, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(actual.rejected, expected.rejected, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(actual.chosen_counts, expected.chosen_counts, rtol=0, atol=0)
        torch.testing.assert_close(actual.rejected_counts, expected.rejected_counts, rtol=0, atol=0)
        self.assertTrue(all(not value.requires_grad for value in (
            actual.chosen, actual.rejected, actual.chosen_counts, actual.rejected_counts)))

    def test_creation_really_scores_frozen_reference_exact_sequences_and_deduplicates(self):
        with patch.object(self.backend, "read", wraps=self.backend.read) as reads:
            binding = self.create(pairs=[(self.chosen, self.rejected)] * 2)
        self.assertEqual(reads.call_count, 4)
        self.assertEqual([call.args[0] for call in reads.call_args_list], ["reference"] * 4)
        online = reference_pair_scores(self.backend, "reference", self.chosen, self.rejected, chunk_tokens=2)
        self.assert_scores(binding.pair_scores(self.chosen, self.rejected), online)
        self.assertEqual(online.chosen_counts.tolist(), [2, 4])
        cache = load_cache(binding.path, binding.identity, expected_manifest_sha256=binding.manifest_sha256)
        self.assertEqual(len(cache.entries), 4)
        self.assertEqual(cache.lookup([4, 8, 9], [0, 2]).token_count, 2)
        self.assertIn(ANCHOR_KEY, binding.identity["scoring_config"])
        self.assertEqual(binding.identity["scoring_config"][PROTOCOL_KEY]["chunk_tokens"], 2)

    def test_disk_reload_into_identical_backend_matches_online_dpo_loss_and_gradients(self):
        binding = self.create()
        online_backend, cached_backend = fixture()[0], fixture()[0]
        online = reference_pair_scores(online_backend, "reference", self.chosen, self.rejected, chunk_tokens=2)
        loaded = self.reload(binding, cached_backend)
        with patch.object(cached_backend, "read", side_effect=AssertionError("lookup must not forward")):
            cached = loaded.pair_scores(self.chosen, self.rejected)
        self.assert_scores(cached, online)
        online_result = dpo_pair_backward(
            online_backend, "policy", optimizer(online_backend), self.chosen, self.rejected,
            online, beta=0.1, pair_denominator=2, chunk_tokens=2)
        cached_result = dpo_pair_backward(
            cached_backend, "policy", optimizer(cached_backend), self.chosen, self.rejected,
            cached, beta=0.1, pair_denominator=2, chunk_tokens=2)
        torch.testing.assert_close(cached_result.objective_sum, online_result.objective_sum,
                                   rtol=1e-6, atol=1e-6)
        for left, right in zip(cached_backend.parameters_for("policy"), online_backend.parameters_for("policy")):
            self.assertIsNotNone(left.grad)
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=1e-7)
        selected = {id(p) for p in cached_backend.parameters_for("policy")}
        self.assertTrue(all(p.grad is None for p in cached_backend._all_parameters() if id(p) not in selected))

    def test_policy_update_and_independent_heads_do_not_invalidate_reference_binding(self):
        binding = self.create()
        cached = binding.pair_scores(self.chosen, self.rejected)
        opt = optimizer(self.backend)
        before = [p.detach().clone() for p in self.backend.parameters_for("policy")]
        dpo_pair_backward(self.backend, "policy", opt, self.chosen, self.rejected, cached,
                          beta=0.1, pair_denominator=2, chunk_tokens=2)
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.assertTrue(any(not torch.equal(p, prior) for p, prior in zip(self.backend.parameters_for("policy"), before)))
        with torch.no_grad():
            for head in self.parts[2].values():
                head.weight.add_(0.25)
        self.assert_scores(binding.pair_scores(self.chosen, self.rejected), cached)
        self.assert_scores(self.reload(binding).pair_scores(self.chosen, self.rejected), cached)

    def test_lookup_during_policy_gradient_accumulation_never_switches_or_forwards(self):
        binding = self.create()
        opt = optimizer(self.backend)
        first = tuple(slice_batch(batch, 0) for batch in (self.chosen, self.rejected))
        second = tuple(slice_batch(batch, 1) for batch in (self.chosen, self.rejected))
        dpo_pair_backward(self.backend, "policy", opt, *first, binding.pair_scores(*first),
                          beta=0.1, pair_denominator=2, chunk_tokens=2)
        gradients = [p.grad.clone() for p in self.backend.parameters_for("policy")]
        with patch.object(self.backend, "read", side_effect=AssertionError("lookup must not forward")):
            second_reference = binding.pair_scores(*second)
        for p, before in zip(self.backend.parameters_for("policy"), gradients):
            torch.testing.assert_close(p.grad, before, rtol=0, atol=0)
        dpo_pair_backward(self.backend, "policy", opt, *second, second_reference,
                          beta=0.1, pair_denominator=2, chunk_tokens=2)
        self.assertTrue(any(not torch.equal(p.grad, before)
                            for p, before in zip(self.backend.parameters_for("policy"), gradients)))
        with self.assertRaisesRegex(RuntimeError, "Clear"):
            self.create(output_dir=self.root / "new-cache")
        self.assertFalse((self.root / "new-cache").exists())

    def test_actual_reference_parameter_mutation_rejects_binding_and_existing_lookup(self):
        binding = self.create()
        with torch.no_grad():
            self.backend._parameters["reference"][0].add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "versions"):
            binding.pair_scores(self.chosen, self.rejected)
        with self.assertRaisesRegex(ValueError, "contents or metadata"):
            self.reload(binding)

    def test_actual_base_parameter_mutation_rejects_binding(self):
        binding = self.create()
        adapter_ids = {id(p) for params in self.backend._parameters.values() for p in params}
        base = next(p for p in self.backend._model.parameters() if id(p) not in adapter_ids)
        with torch.no_grad():
            base.add_(0.01)
        with self.assertRaisesRegex(ValueError, "contents or metadata"):
            self.reload(binding)

    def test_data_write_without_version_increment_is_detected_by_actual_tensor_hash(self):
        binding = self.create()
        ref = self.backend._parameters["reference"][0]
        version = ref._version
        ref.data.add_(0.01)  # Intentionally exercise unsupported PyTorch bypass.
        self.assertEqual(ref._version, version)
        with self.assertRaisesRegex(ValueError, "contents or metadata"):
            binding.pair_scores(self.chosen, self.rejected)

    def test_reference_layer_and_base_config_metadata_changes_are_rejected(self):
        binding = self.create()
        layer = next(module for module in self.backend._model.modules()
                     if hasattr(module, "lora_A") and "reference" in module.lora_A)
        original = layer.scaling["reference"]
        layer.scaling["reference"] *= 2
        with self.assertRaisesRegex(ValueError, "contents or metadata"):
            self.reload(binding)
        layer.scaling["reference"] = original
        self.backend._base.config.max_position_embeddings *= 2
        with self.assertRaisesRegex(ValueError, "contents or metadata"):
            binding.pair_scores(self.chosen, self.rejected)

    def test_frozen_buffer_content_changes_are_rejected(self):
        binding = self.create()
        buffer = next(self.backend._model.buffers())
        buffer.data.add_(0.01)
        with self.assertRaisesRegex(ValueError, "contents or metadata"):
            self.reload(binding)

    def test_manifest_pin_is_mandatory_immutable_and_detects_byte_changes(self):
        binding = self.create()
        with self.assertRaises(ValueError):
            self.reload(binding, expected_manifest_sha256=None)
        with self.assertRaises(ValueError):
            self.reload(binding, expected_manifest_sha256="0" * 64)
        with self.assertRaises(FrozenInstanceError):
            binding.manifest_sha256 = "0" * 64
        manifest = self.output / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.reload(binding)

    def test_cache_identity_scores_and_untracked_file_tampering_are_rejected(self):
        binding = self.create()
        for filename in ("identity.json", "scores.jsonl"):
            path = self.output / filename
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                binding.pair_scores(self.chosen, self.rejected)
            path.write_bytes(original)
        (self.output / "untracked.txt").write_text("not part of cache", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.reload(binding)

    def test_all_original_identity_file_groups_and_accepted_manifest_are_revalidated(self):
        binding = self.create()
        files = [self.identity_kwargs[group][next(iter(self.identity_kwargs[group]))]
                 for group in ("base_files", "adapter_files", "tokenizer_files", "source_files")]
        files.append(self.identity_kwargs["dataset_manifest"])
        for path in files:
            path = Path(path)
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.reload(binding)
            path.write_bytes(original)
        # Verify a source path is actually tracked without editing project code.
        source_identity = binding.identity
        sources = source_identity["files"]["source"]
        self.assertTrue(any(record["path"] == str(Path(__file__).resolve()) for record in sources.values()))

    def test_identity_configuration_change_or_unaccepted_manifest_digest_is_rejected(self):
        binding = self.create()
        changed = binding.identity
        changed["scoring_config"]["template"] = "different-template"
        with self.assertRaises(ValueError):
            self.reload(binding, expected_identity=changed)
        changed = copy.deepcopy(self.identity_kwargs)
        changed["accepted_dataset_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Dataset manifest SHA"):
            self.create(output_dir=self.root / "other", identity_kwargs=changed)
        self.assertFalse((self.root / "other").exists())

    def test_exact_token_mismatch_and_pair_prompt_mismatch_are_rejected(self):
        binding = self.create()
        changed = {key: value.clone() for key, value in self.chosen.items()}
        changed["input_ids"][0, 3] = 80
        changed["labels"][0, 3] = 80
        with self.assertRaisesRegex(ValueError, "Exact prompt/completion"):
            binding.pair_scores(changed, self.rejected)
        changed = {key: value.clone() for key, value in self.chosen.items()}
        changed["input_ids"][0, 0] = 80
        with self.assertRaisesRegex(ValueError, "identical prompt"):
            binding.pair_scores(changed, self.rejected)

    def test_reference_record_owner_inputs_and_tensor_versions_are_enforced_by_dpo(self):
        binding = self.create()
        record = binding.pair_scores(self.chosen, self.rejected)
        other = fixture()[0]
        with self.assertRaisesRegex(ValueError, "different backend"):
            dpo_pair_backward(other, "policy", optimizer(other), self.chosen, self.rejected, record,
                              beta=0.1, pair_denominator=2)
        changed = {key: value.clone() for key, value in self.chosen.items()}
        # Padding changes do not change cache sequence keys but do change the
        # exact DPO input fingerprint recorded at lookup.
        changed["input_ids"][0, -1] = 50
        with self.assertRaisesRegex(ValueError, "exact chosen/rejected"):
            dpo_pair_backward(self.backend, "policy", optimizer(self.backend), changed, self.rejected,
                              record, beta=0.1, pair_denominator=2)
        record.chosen.add_(0.1)
        with self.assertRaisesRegex(RuntimeError, "score tensors were modified"):
            dpo_pair_backward(self.backend, "policy", optimizer(self.backend), self.chosen, self.rejected,
                              record, beta=0.1, pair_denominator=2)

    def test_existing_output_is_not_rescored_and_reuse_is_completely_read_only(self):
        binding = self.create()
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.output.iterdir()}
        with patch.object(self.backend, "read", side_effect=AssertionError("no rescore")):
            with self.assertRaises(FileExistsError):
                self.create()
            self.reload(binding).pair_scores(self.chosen, self.rejected)
        after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.output.iterdir()}
        self.assertEqual(before, after)

    def test_empty_inputs_invalid_ids_and_caller_supplied_anchor_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.create(pairs=[])
        self.assertFalse(self.output.exists())
        changed = {key: value.clone() for key, value in self.chosen.items()}
        changed["input_ids"][0, 3] = 97
        changed["labels"][0, 3] = 97
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.create(pairs=[(changed, self.rejected)])
        for key in (ANCHOR_KEY, PROTOCOL_KEY):
            args = copy.deepcopy(self.identity_kwargs)
            args["scoring_config"][key] = {"asserted": "trusted"}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "never accepted"):
                self.create(identity_kwargs=args)

    def test_library_version_precision_and_quantization_must_be_actual_supported_values(self):
        args = copy.deepcopy(self.identity_kwargs)
        args["scoring_config"]["library_versions"]["torch"] = "self-asserted"
        with self.assertRaisesRegex(ValueError, "actual installed torch"):
            self.create(identity_kwargs=args)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast disabled"):
                self.create()
        self.backend._model.is_loaded_in_4bit = True
        with self.assertRaisesRegex(ValueError, "Quantized"):
            self.create()
        self.backend._model.is_loaded_in_4bit = False
        self.backend._model.double()
        with self.assertRaisesRegex(ValueError, "float32"):
            self.create()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
