"""CPU-only full-context selection and honest capacity receipt tests."""
import hashlib
import json
from types import SimpleNamespace
import unittest

from alignment_lab import evaluation_capacity_v5 as cap
from alignment_lab.evaluation_protocol_v2 import EvaluationDataset, canonical, digest


def dataset(name, field, rows, lengths):
    identity = {"record_count": len(rows), "records_sha256": digest(rows),
                "ordered_ids_sha256": digest([r[field] for r in rows])}
    return EvaluationDataset(name, field, canonical(rows), canonical(identity), canonical(lengths))


class CapacityV2Tests(unittest.TestCase):
    def test_nine_models_and_maximum_real_contexts_are_retained(self):
        ifeval = dataset("ifeval", "key", [{"key": 1}, {"key": 2}], {"1": 50, "2": 373})
        held = dataset("helpsteer3_final_holdout", "evaluation_id", [{"evaluation_id": "a"}, {"evaluation_id": "b"}],
                       {"a": [100, 3000], "b": [2000, 5481]})
        rm = dataset("rm_bench", "id", [{"id": "x"}], {"x": {"chosen": [10, 20, 2363], "rejected": [30, 40, 50]}})
        context = SimpleNamespace(assets=SimpleNamespace(ifeval=ifeval, rm_bench=rm), dataset=held)
        selection = cap.select_maximum_inputs(context)
        self.assertEqual(len(cap.MODEL_IDS), 9)
        self.assertEqual({k: v["input_tokens"] for k, v in selection["cases"].items()},
                         {"ifeval": 373, "helpsteer_rm": 5481, "rm_bench": 2363})
        self.assertFalse(selection["quality_selection_allowed"])
        self.assertFalse(selection["input_truncation"])

    def test_forced_capacity_requires_every_one_of_2048_actions(self):
        prompt, actions = (1, 2), [33] * 2048
        receipt = {"action_ids": actions, "prompt_ids": prompt, "prompt_tokens": 2,
            "generated_tokens": 2048, "max_new_tokens": 2048, "eos_id": cap.EOS,
            "ignore_eos": True, "attention_all_true": True, "synthetic_eos_added": False,
            "final_action_flushed_to_kv": True, "max_cache_tokens": 2050,
            "decoder_calls": 2049, "generated_eos_count": 0,
            "tokens_sha256": hashlib.sha256(json.dumps([list(prompt), actions], separators=(",", ":")).encode()).hexdigest(),
            "termination": "forced_max_new_tokens", "mode": "forced_continuation_kv_stress",
            "capacity_only": True, "quality_evaluation": False}
        cap._generation(receipt, prompt, forced=True)
        receipt["generated_tokens"] = 2047
        with self.assertRaises(ValueError):
            cap._generation(receipt, prompt, forced=True)

    def test_cpu_fixture_metadata_cannot_prove_cuda_capacity(self):
        with self.assertRaisesRegex(ValueError, "initialized CUDA"):
            cap._memory({"device": "cpu", "cuda_initialized": False})
        with self.assertRaises(ValueError):
            cap._memory({"device": "cuda:0", "cuda_initialized": True, "device_name": "fixture",
                "total_memory_bytes": 8192, "max_allocated_bytes": 0, "max_reserved_bytes": 1,
                "allocated_bytes": 0, "reserved_bytes": 1})


if __name__ == "__main__":
    unittest.main()
