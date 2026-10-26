import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from rlhf_lab import runtime
from rlhf_lab.__main__ import dispatch, parser
from rlhf_lab.tokenization import encode_pair, encode_sft


class Tokenizer:
    eos_token_id = 0

    def encode(self, value, add_special_tokens=False):
        return [ord(c) for c in value]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = runtime.load_config("configs/pilot.json")

    def test_busy_hardware_blocks_before_importing_torch(self):
        busy = {"available_ram_gb": 2.1, "gpus": [{"index": 0, "driver": "472.80",
            "free_mb": 71, "utilization_percent": 100}]}
        busy["blockers"] = runtime.resource_blockers(busy, self.config)
        self.assertGreaterEqual(len(busy["blockers"]), 4)
        with patch.object(runtime, "doctor", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "not started"):
                runtime.assert_resources(self.config)

    def test_train_cli_defaults_to_plan_and_no_heavy_imports(self):
        program = "import sys; sys.path.insert(0, sys.argv[1]); from rlhf_lab.__main__ import main; main(['train','--stage','ppo']); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", program, str(runtime.ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "plan_only")

    def test_generate_defaults_to_validation_and_plan(self):
        args = parser().parse_args(["generate", "--model", "runs/pilot/ppo", "--label", "ppo", "--output", "runs/test.jsonl"])
        result = dispatch(args)
        self.assertEqual(result["split"], "valid")
        self.assertFalse(result["generation_started"])

    def test_sft_masks_prompt_and_truncates_answer_with_eos(self):
        config = {"max_prompt_tokens": 3, "max_response_tokens": 3}
        row = encode_sft(Tokenizer(), {"prompt": "abcde", "response": "wxyz"}, config)
        self.assertEqual(row["input_ids"], [68, 82, 58, 32, 119, 0])
        self.assertEqual(row["labels"], [-100, -100, -100, 32, 119, 0])
        self.assertEqual(row["attention_mask"], [1] * 6)

    def test_summary_prefix_is_shared_without_double_marker(self):
        config = {"max_prompt_tokens": 100, "max_response_tokens": 100}
        row = encode_sft(Tokenizer(), {"prompt": "An article.", "response": "TL;DR: A summary."}, config)
        rendered = "".join(map(chr, row["input_ids"][:-1]))
        self.assertEqual(rendered, "An article.\n\nTL;DR: A summary.")

    def test_identical_truncated_preference_is_dropped(self):
        config = {"max_prompt_tokens": 3, "max_response_tokens": 3}
        pair = {"prompt": "abc", "chosen": "xyright", "rejected": "xywrong"}
        self.assertIsNone(encode_pair(Tokenizer(), pair, config))

    def test_ready_hardware_has_no_resource_blockers(self):
        info = {"available_ram_gb": 10, "gpus": [{"index": 0, "driver": "580.100",
                 "free_mb": 7000, "utilization_percent": 0}]}
        self.assertEqual(runtime.resource_blockers(info, self.config), [])

    def test_missing_gpu_or_memory_is_not_treated_as_idle(self):
        self.assertTrue(runtime.resource_blockers({"available_ram_gb": None, "gpus": []}, self.config))


if __name__ == "__main__":
    unittest.main()
