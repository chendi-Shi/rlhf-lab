"""Stdlib-only guard/aggregation tests; fixture scores are never training results."""

import builtins
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rlhf_lab.reward_eval import _aggregate, score_reward


class RewardEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {"output_dir": str(self.root / "runs"), "data_dir": str(self.root / "data"),
                       "eval_limit": 8, "gpu_index": 0}

    def tearDown(self):
        self.temporary.cleanup()

    def test_busy_machine_fails_before_heavy_imports(self):
        original_import = builtins.__import__
        attempted = []
        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "datasets", "trl", "accelerate"}:
                attempted.append(name)
                raise AssertionError(f"Premature heavy import: {name}")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=checked_import), patch(
                "rlhf_lab.reward_eval.assert_resources", side_effect=RuntimeError("GPU busy")):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                score_reward(self.config, self.root / "reward")
        self.assertEqual(attempted, [])
        self.assertFalse((self.root / "runs").exists())

    def test_existing_report_is_not_overwritten(self):
        path = self.root / "report.json"
        path.write_text("existing result", encoding="utf-8")
        with patch("rlhf_lab.reward_eval.assert_resources") as guard:
            with self.assertRaises(FileExistsError):
                score_reward(self.config, self.root / "reward", output_path=path)
        guard.assert_not_called()
        self.assertEqual(path.read_text(encoding="utf-8"), "existing result")

    def test_only_heldout_splits_and_positive_limits(self):
        for kwargs in ({"split": "train"}, {"limit": 0}, {"limit": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                score_reward(self.config, self.root / "reward", **kwargs)

    def test_prompt_macro_does_not_overweight_repeated_articles(self):
        # Synthetic numerical fixture: three wins for A, one loss for B.
        rows = [{"prompt_id": "A", "chosen_score": 2.0, "rejected_score": 1.0} for _ in range(3)]
        rows.append({"prompt_id": "B", "chosen_score": 0.0, "rejected_score": 1.0})
        result = _aggregate(rows)
        self.assertEqual(result["pairwise_accuracy"], 0.75)
        self.assertEqual(result["prompt_macro_accuracy"], 0.5)
        self.assertEqual(result["unique_prompts"], 2)
        self.assertAlmostEqual(result["prompt_macro_loss"], (math.log1p(math.exp(-1)) + math.log1p(math.exp(1))) / 2)

    def test_ties_and_large_negative_margin_remain_finite(self):
        result = _aggregate([{"prompt_id": "A", "chosen_score": 0.0, "rejected_score": 0.0},
                             {"prompt_id": "B", "chosen_score": -1000.0, "rejected_score": 0.0}])
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["pairwise_accuracy"], 0.0)
        self.assertTrue(math.isfinite(result["pairwise_loss"]))

    def test_empty_or_nonfinite_scores_cannot_be_reported(self):
        for rows in ([], [{"prompt_id": "A", "chosen_score": float("nan"), "rejected_score": 0.0}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                _aggregate(rows)


if __name__ == "__main__":
    unittest.main()
