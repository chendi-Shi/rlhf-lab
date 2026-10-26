"""Small synthetic fixtures verify evaluation bookkeeping, never model quality."""

import csv
import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlhf_lab.evaluate import (
    BLIND_FIELDS,
    export_blind_pairs,
    load_generations,
    response_proxies,
    score_judgments,
    summarize_generations,
    wilson_interval,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.left = self.root / "left.jsonl"
        self.right = self.root / "right.jsonl"
        self.csv = self.root / "blind.csv"
        self.mapping = self.root / "private.json"
        self.left_rows = self.generations("sft")
        self.right_rows = self.generations("ppo")
        self.write_jsonl(self.left, self.left_rows)
        self.write_jsonl(self.right, self.right_rows)

    @staticmethod
    def generations(label):
        return [
            {"id": str(index), "prompt": f"Article {index}\nTL;DR:",
             "response": f"A summary {index}.", "model_label": label,
             "model_path": f"runs/{label}", "seed": 42}
            for index in range(4)
        ]

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def export(self):
        return export_blind_pairs(self.left, self.right, self.csv, self.mapping, seed=19)

    def prepared_data(self):
        from rlhf_lab.data import prepare
        source = self.root / "synthetic-comparisons.jsonl"
        self.write_jsonl(source, [
            {"prompt": f"Synthetic article {index}", "chosen": "Preferred fixture", "rejected": "Rejected fixture"}
            for index in range(30)
        ])
        destination = self.root / "data"
        prepare(source, destination, train_limit=20, valid_limit=3, test_limit=3)
        return destination

    def read_csv(self):
        with self.csv.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def write_csv(self, rows):
        with self.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BLIND_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_blind_export_has_separate_identity_mapping_and_no_judgments(self):
        result = self.export()
        self.assertEqual(result["pairs"], 4)
        rows = self.read_csv()
        self.assertEqual(list(rows[0]), BLIND_FIELDS)
        self.assertTrue(all(row["judgment"] == "" for row in rows))
        self.assertNotIn("model_label", self.csv.read_text(encoding="utf-8"))
        mapping = json.loads(self.mapping.read_text(encoding="utf-8"))
        self.assertEqual(mapping["left_model"]["model_label"], "sft")
        self.assertEqual(mapping["right_model"]["model_label"], "ppo")

    def test_order_is_reproducible_and_independent_of_input_row_order(self):
        self.export()
        self.write_jsonl(self.left, self.left_rows[::-1])
        self.write_jsonl(self.right, self.right_rows[::-1])
        other_csv, other_map = self.root / "second.csv", self.root / "second.json"
        export_blind_pairs(self.left, self.right, other_csv, other_map, seed=19)
        self.assertEqual(self.csv.read_bytes(), other_csv.read_bytes())
        self.assertEqual(self.mapping.read_bytes(), other_map.read_bytes())

    def test_missing_ids_and_prompt_mismatch_are_rejected_before_write(self):
        for bad_rows in (self.right_rows[:-1], [{**self.right_rows[0], "prompt": "different"}, *self.right_rows[1:]]):
            with self.subTest(rows=bad_rows):
                self.write_jsonl(self.right, bad_rows)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.csv.exists())
                self.assertFalse(self.mapping.exists())

    def test_duplicate_and_malformed_generation_rows_are_rejected(self):
        bad_inputs = [
            self.left_rows + [self.left_rows[0]],
            [{**self.left_rows[0], "seed": True}],
            [{**self.left_rows[0], "response": None}],
            [{**self.left_rows[0], "id": 1}],
            [{"id": "missing"}],
            [self.left_rows[0], {**self.left_rows[1], "model_label": "another"}],
            [],
        ]
        for bad_rows in bad_inputs:
            with self.subTest(rows=bad_rows):
                self.write_jsonl(self.left, bad_rows)
                with self.assertRaises(ValueError):
                    load_generations(self.left)
        self.left.write_text("not-json\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_generations(self.left)

    def test_no_overwrite_and_no_shared_output_path(self):
        self.export()
        original = self.csv.read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.csv.read_bytes(), original)
        with self.assertRaises(ValueError):
            export_blind_pairs(self.left, self.right, self.root / "same", self.root / "same")

    def test_unjudged_is_not_a_win_and_has_no_rate_or_interval(self):
        self.export()
        result = score_judgments(self.csv, self.mapping)
        self.assertEqual(result["unjudged"], 4)
        self.assertEqual(result["judged"], 0)
        self.assertEqual(result["wins"], 0)
        self.assertIsNone(result["decisive_win_rate"])
        self.assertIsNone(result["decisive_win_rate_wilson_95"])
        self.assertIsNone(result["tie_adjusted_score"])

    def test_actual_judgments_map_to_wins_ties_losses_and_partial_coverage(self):
        self.export()
        mapping = json.loads(self.mapping.read_text(encoding="utf-8"))
        by_pair = {item["pair_id"]: item for item in mapping["pairs"]}
        rows = self.read_csv()
        for index, row in enumerate(rows):
            pair = by_pair[row["pair_id"]]
            if index == 0:
                row["judgment"] = "A" if pair["a_side"] == "left" else "B"
            elif index == 1:
                row["judgment"] = "B" if pair["a_side"] == "left" else "A"
            elif index == 2:
                row["judgment"] = " tie "
        self.write_csv(rows)
        report_path = self.root / "score.json"
        result = score_judgments(self.csv, self.mapping, report_path)
        self.assertEqual([result[key] for key in ("wins", "ties", "losses", "unjudged")], [1, 1, 1, 1])
        self.assertEqual(result["judged"], 3)
        self.assertEqual(result["decisive_judgments"], 2)
        self.assertEqual(result["decisive_win_rate"], 0.5)
        self.assertEqual(result["tie_adjusted_score"], 0.5)
        self.assertEqual(result["decisive_win_rate_wilson_95"], wilson_interval(1, 2))
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), result)
        with self.assertRaises(FileExistsError):
            score_judgments(self.csv, self.mapping, report_path)

    def test_all_ties_have_no_decisive_win_rate(self):
        self.export()
        rows = self.read_csv()
        for row in rows:
            row["judgment"] = "tie"
        self.write_csv(rows)
        result = score_judgments(self.csv, self.mapping)
        self.assertEqual(result["tie_adjusted_score"], 0.5)
        self.assertEqual(result["ties"], 4)
        self.assertIsNone(result["decisive_win_rate_wilson_95"])

    def test_judgment_corruption_missing_rows_and_duplicate_rows_rejected(self):
        self.export()
        original = self.read_csv()
        invalid = [
            original[:-1],
            original + [original[0]],
            [{**original[0], "judgment": "sft"}, *original[1:]],
            [{**original[0], "response_a": "edited text"}, *original[1:]],
            [{**original[0], "pair_id": "unknown"}, *original[1:]],
        ]
        for rows in invalid:
            with self.subTest(rows=rows):
                self.write_csv(rows)
                with self.assertRaises(ValueError):
                    score_judgments(self.csv, self.mapping)

    def test_multiline_unicode_and_csv_quotes_roundtrip(self):
        self.left_rows[0]["prompt"] = '这是文本,含有"引号"\nTL;DR:'
        self.right_rows[0]["prompt"] = self.left_rows[0]["prompt"]
        self.write_jsonl(self.left, self.left_rows)
        self.write_jsonl(self.right, self.right_rows)
        self.export()
        self.assertEqual(score_judgments(self.csv, self.mapping)["unjudged"], 4)

    def test_wilson_reference_values_and_boundary_counts(self):
        interval = wilson_interval(5, 10)
        self.assertAlmostEqual(interval[0], 0.236593090512564, places=12)
        self.assertAlmostEqual(interval[1], 0.763406909487436, places=12)
        self.assertIsNone(wilson_interval(0, 0))
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0.0)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1.0)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)

    def test_repetition_diagnostic_is_explicitly_a_proxy(self):
        repeated = response_proxies("a b c a b c")
        self.assertEqual(repeated["repeated_trigram_fraction_proxy"], 0.25)
        self.assertEqual(response_proxies("")["repeated_trigram_fraction_proxy"], 0.0)
        self.left_rows[0]["response"] = ""
        self.write_jsonl(self.left, self.left_rows)
        report = summarize_generations(self.left)
        self.assertEqual(report["empty_responses"], 1)
        self.assertIn("mean_repeated_trigram_fraction_proxy", report)
        self.assertNotIn("quality", report)

    def test_generation_imports_no_training_library_and_guard_precedes_loading(self):
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            generation_module = importlib.import_module("rlhf_lab.generate")
            data_dir = self.prepared_data()
            config = {"data_dir": str(data_dir), "model_id": "openai-community/gpt2",
                      "model_revision": "test-only-revision", "max_prompt_tokens": 16,
                      "max_response_tokens": 8}
            with patch("rlhf_lab.runtime.assert_resources", side_effect=RuntimeError("test GPU busy")) as guard:
                with self.assertRaisesRegex(RuntimeError, "test GPU busy"):
                    generation_module.generate(config, config["model_id"], "base", self.root / "generated.jsonl")
                guard.assert_called_once_with(config, require_cuda=True)
            self.assertFalse((self.root / "generated.jsonl").exists())

    def test_validation_and_test_prompt_references_remain_separate(self):
        from rlhf_lab.generate import _read_prompts
        path = self.root / "prompts.jsonl"
        self.write_jsonl(path, [{"id": "valid", "prompt": "Text TL;DR:", "chosen": "human preference"}])
        result = _read_prompts(path, 1, "chosen")
        self.assertEqual(result[0]["reference"], "human preference")
        self.assertNotIn("chosen", result[0])
        self.write_jsonl(path, [{"id": "test", "prompt": "Text TL;DR:", "reference": "held out"}])
        self.assertEqual(_read_prompts(path, 1)[0]["reference"], "held out")

    def test_incomplete_local_checkpoint_is_rejected_before_gpu_loading(self):
        from rlhf_lab.generate import generate
        data_dir, checkpoint = self.prepared_data(), self.root / "checkpoint"
        checkpoint.mkdir()
        config = {"data_dir": str(data_dir), "model_id": "openai-community/gpt2", "model_revision": "test-only",
                  "max_prompt_tokens": 16, "max_response_tokens": 8, "seed": 42}
        with patch("rlhf_lab.runtime.assert_resources") as guard:
            with self.assertRaisesRegex(ValueError, "missing its completed"):
                generate(config, str(checkpoint), "sft", self.root / "generated.jsonl")
            (checkpoint / "run_manifest.json").write_text(json.dumps({"status": "failed", "stage": "sft"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed SFT or PPO"):
                generate(config, str(checkpoint), "sft", self.root / "generated.jsonl")
            manifest = {"status": "completed", "stage": "reward", "model_id": config["model_id"], "model_revision": config["model_revision"]}
            (checkpoint / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed SFT or PPO"):
                generate(config, str(checkpoint), "sft", self.root / "generated.jsonl")
            manifest["stage"] = "sft"
            (checkpoint / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset manifest"):
                generate(config, str(checkpoint), "sft", self.root / "generated.jsonl")
            manifest["dataset_manifest_sha256"] = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
            manifest["config"] = {**config, "max_prompt_tokens": 17}
            (checkpoint / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_prompt_tokens"):
                generate(config, str(checkpoint), "sft", self.root / "generated.jsonl")
            manifest["config"] = config
            (checkpoint / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no supported policy weights"):
                generate(config, str(checkpoint), "sft", self.root / "generated.jsonl")
            guard.assert_not_called()

    def test_generation_deduplicates_articles_before_applying_limit(self):
        from rlhf_lab.generate import _read_prompts
        path = self.root / "prompts.jsonl"
        self.write_jsonl(path, [
            {"id": "first-pair", "prompt": "Same article", "chosen": "first reference"},
            {"id": "second-pair", "prompt": " SAME  ARTICLE ", "chosen": "other reference"},
            {"id": "third-pair", "prompt": "Different article", "chosen": "third reference"},
        ])
        selected = _read_prompts(path, 2, "chosen")
        self.assertEqual([row["id"] for row in selected], ["first-pair", "third-pair"])
        self.assertEqual(selected[0]["reference"], "first reference")

    def test_generation_sidecars_enforce_matching_settings_and_unchanged_outputs(self):
        common = {
            "status": "completed", "model_id": "fixture", "model_revision": "fixture-only",
            "eval_split": "valid", "prompts_sha256": "fixture", "dataset_manifest_sha256": "fixture",
            "decoding": "greedy", "inference_dtype": "bfloat16",
            "config": {"max_prompt_tokens": 16, "max_response_tokens": 8, "seed": 42},
        }
        manifests = []
        for path in (self.left, self.right):
            manifest = {**common, "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            sidecar = Path(str(path) + ".manifest.json")
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            manifests.append((sidecar, manifest))
        sidecar, manifest = manifests[1]
        sidecar.write_text(json.dumps({**manifest, "eval_split": "test"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "eval_split"):
            self.export()
        sidecar.write_text(json.dumps(manifest), encoding="utf-8")
        self.right_rows[0]["response"] = "Changed after generation"
        self.write_jsonl(self.right, self.right_rows)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.export()
        self.write_jsonl(self.right, self.generations("ppo"))
        self.assertEqual(self.export()["pairs"], 4)


if __name__ == "__main__":
    unittest.main()
