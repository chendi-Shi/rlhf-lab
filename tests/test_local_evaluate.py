"""Evaluation orchestration with tiny in-memory stand-ins, never model downloads."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rlhf_lab import local_evaluate
from rlhf_lab.evaluate import load_generations, score_judgments
from rlhf_lab.runtime import sha256_file


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<eos>"
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [1 + ord(character) % 7 for character in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "<script>raw model text</script> " + " ".join(map(str, tokens))


class _Policy:
    def __init__(self):
        self.config = SimpleNamespace(pad_token_id=0)

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 8))

    def generate(self, input_ids, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 4
        return torch.cat([input_ids, torch.tensor([[3, 0]])], dim=-1)


class LocalEvaluationTests(unittest.TestCase):
    def test_prompt_selection_deduplicates_before_applying_limit(self):
        rows = [
            {"id": "one", "prompt": " Article A ", "chosen": "yes", "rejected": "no"},
            {"id": "two", "prompt": "article   a", "chosen": "yes", "rejected": "no"},
            {"id": "three", "prompt": "Article B", "chosen": "yes", "rejected": "no"},
        ]
        selected = local_evaluate._select_validation_rows(rows, 2)
        self.assertEqual([row["id"] for row in selected], ["one", "three"])
        with self.assertRaisesRegex(ValueError, "Duplicate validation ID"):
            local_evaluate._select_validation_rows(rows + [rows[0]], 2)

    def test_resource_guard_and_test_split_precede_loading(self):
        with patch.object(local_evaluate, "assert_local_resources", side_effect=RuntimeError("busy")) as guard:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                local_evaluate.evaluate_local({"device": "cuda"})
            guard.assert_called_once()
        with patch.object(local_evaluate, "assert_local_resources") as guard:
            with self.assertRaisesRegex(ValueError, "restricted to validation"):
                local_evaluate.evaluate_local({"eval_split": "test"})
            guard.assert_not_called()

    def test_mocked_end_to_end_preserves_validation_scope_and_empty_ballots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data, runs = root / "base", root / "data", root / "runs"
            for directory in (source, data, runs):
                directory.mkdir()
            (source / "model.safetensors").write_bytes(b"in-memory loader stand-in; not a real checkpoint")
            base_manifest = {"model_id": "openai-community/gpt2", "revision": "a" * 40,
                             "files": {"model.safetensors": {"sha256": sha256_file(source / "model.safetensors")}}}
            (source / "local-manifest.json").write_text(json.dumps(base_manifest), encoding="utf-8")
            rows = [{"id": str(index), "prompt": f"article {index}", "chosen": "good summary", "rejected": "bad summary"}
                    for index in range(3)]
            valid = data / "preferences_valid.jsonl"
            valid.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (data / "manifest.json").write_text(json.dumps({"source": {"human_preferences": True},
                 "files": {valid.name: {"sha256": sha256_file(valid)}}}), encoding="utf-8")
            for stage in ("sft", "reward", "dpo", "ppo"):
                (runs / stage).mkdir()
                (runs / stage / "run_manifest.json").write_text(json.dumps({"status": "completed", "stage": stage}), encoding="utf-8")
            config = {"model_path": str(source), "data_dir": str(data), "output_dir": str(runs),
                      "model_id": "openai-community/gpt2", "model_revision": "a" * 40,
                      "device": "cpu", "max_prompt_tokens": 8, "max_response_tokens": 4,
                      "completion_eos_policy": "natural", "eval_limit": 2, "eval_preference_limit": 3,
                      "num_threads": 1, "seed": 9}
            actual_read = local_evaluate.read_jsonl
            paths_read = []
            def tracked_read(path):
                paths_read.append(Path(path).name)
                return actual_read(path)
            def rewards(model, batch):
                return (batch["input_ids"] * batch["attention_mask"]).float().sum(-1)
            with patch.object(local_evaluate, "assert_local_resources", return_value={"device": "cpu"}), \
                    patch.object(local_evaluate, "read_jsonl", side_effect=tracked_read), \
                    patch("rlhf_lab.local_training.checkpoint_ready_local", return_value={"status": "completed"}), \
                    patch("rlhf_lab.local_training._load_model", side_effect=lambda *args, **kwargs: _Policy()) as loader, \
                    patch("rlhf_lab.local_training.reward_scores", side_effect=rewards), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()):
                result = local_evaluate.evaluate_local(config)
                self.assertEqual(loader.call_count, 5)  # Four policies, then one reward model.
                with self.assertRaises(FileExistsError):
                    local_evaluate.evaluate_local(config)
            self.assertEqual(paths_read, ["preferences_valid.jsonl"])
            output = runs / "evaluation"
            self.assertEqual(result["human_judgments"], 0)
            self.assertFalse(result["test_labels_decoded"])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reward_model"]["pairs"], 3)
            self.assertIsNone(summary["human_win_rate"])
            ids = []
            for label in ("base", "sft", "dpo", "ppo"):
                generated = load_generations(output / f"{label}.jsonl")
                ids.append([row["id"] for row in generated])
                self.assertTrue(all(row["seed"] == 9 and row["generated_tokens"] == 2 for row in generated))
            self.assertTrue(all(group == ids[0] for group in ids))
            for label in ("dpo", "ppo"):
                ballot = output / f"blind-{label}-vs-sft.csv"
                with ballot.open(encoding="utf-8", newline="") as stream:
                    self.assertTrue(all(row["judgment"] == "" for row in csv.DictReader(stream)))
                score = score_judgments(ballot, output / f"private-{label}-vs-sft.json")
                self.assertEqual(score["unjudged"], 2)
                self.assertIsNone(score["decisive_win_rate"])
            document = (output / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", document)
            self.assertIn("&lt;script&gt;", document)
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["artifacts"]["summary.json"], sha256_file(output / "summary.json"))


if __name__ == "__main__":
    unittest.main()
