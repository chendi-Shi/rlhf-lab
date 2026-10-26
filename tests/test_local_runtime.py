import json
import subprocess
import sys
import unittest

from rlhf_lab.local_runtime import load_local_config, local_blockers
from rlhf_lab.runtime import ROOT
from rlhf_lab.tokenization import encode_response


class Tokenizer:
    eos_token_id = 0

    def encode(self, value, add_special_tokens=False):
        return [ord(c) for c in value]


class LocalRuntimeTests(unittest.TestCase):
    def test_cpu_is_explicit_and_busy_gpu_does_not_block_cpu(self):
        config = load_local_config("configs/local-smoke.json")
        self.assertEqual(local_blockers({"available_ram_gb": 12, "gpus": []}, config), [])
        self.assertTrue(local_blockers({"available_ram_gb": 1, "gpus": []}, config))

    def test_cuda_respects_busy_gpu_without_wrong_driver_floor(self):
        config = load_local_config("configs/local-gpu.json")
        info = {"available_ram_gb": 12, "gpus": [{"index": 0, "driver": "472.80", "free_mb": 7500, "utilization_percent": 0}]}
        self.assertEqual(local_blockers(info, config), [])
        info["gpus"][0]["free_mb"] = 1000
        self.assertTrue(local_blockers(info, config))

    def test_local_plan_does_not_import_training_runtime(self):
        program = "import sys; sys.path.insert(0,sys.argv[1]); from rlhf_lab.__main__ import main; assert main(['local-train'])==0; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", program, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "plan_only")

    def test_truncation_is_not_a_natural_eos(self):
        config = {"max_response_tokens": 3, "completion_eos_policy": "natural"}
        self.assertEqual(encode_response(Tokenizer(), "abcd", config), [32, 97, 98])
        self.assertEqual(encode_response(Tokenizer(), "a", config), [32, 97, 0])
