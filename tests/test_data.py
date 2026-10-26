"""Synthetic fixtures below test preprocessing only; they are never training assets."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types

from rlhf_lab.data import prepare, prepare_hub, read_jsonl, validate_dataset


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "synthetic_test_fixture.jsonl"
        self.rows = [{"prompt": f"Synthetic fixture prompt {i}", "chosen": f"Preferred test {i}",
                      "rejected": f"Rejected test {i}"} for i in range(60)]

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, rows=None):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in
                                       (self.rows if rows is None else rows)), encoding="utf-8")
        Path(str(self.source) + ".provenance.json").write_text(json.dumps({
            "synthetic": True, "human_preferences": False, "purpose": "unit_test_fixture"}), encoding="utf-8")

    def test_reproducible_and_group_isolated(self):
        self.write(self.rows + [{**self.rows[0], "prompt": " SYNTHETIC  fixture prompt 0 ",
                                "chosen": "Another preference"}])
        first = prepare(self.source, self.root / "first")
        second = prepare(self.source, self.root / "second")
        self.assertEqual(first, second)
        report = validate_dataset(self.root / "first")
        self.assertTrue(report["valid"])
        self.assertTrue(report["source"]["synthetic"])
        self.assertEqual(sum(report["unique_prompts"].values()), 60)

    def test_invalid_duplicate_identical_and_conflicting_pairs_removed(self):
        extras = [{**self.rows[0]}, {"prompt": "missing"}, {"prompt": "equal", "chosen": " Same ",
                  "rejected": "same"}, {"prompt": "contradiction", "chosen": "A", "rejected": "B"},
                  {"prompt": "contradiction", "chosen": "B", "rejected": "A"}]
        self.write(self.rows + extras)
        manifest = prepare(self.source, self.root / "output")
        counts = manifest["counts"]["all"]
        self.assertEqual((counts["invalid"], counts["duplicate"], counts["identical"],
                          counts["contradictory"], counts["accepted"]), (1, 1, 1, 2, 60))

    def test_input_order_does_not_change_export(self):
        self.write()
        first = prepare(self.source, self.root / "first")
        self.write(list(reversed(self.rows)))
        second = prepare(self.source, self.root / "second")
        self.assertEqual(first["files"], second["files"])
        self.assertNotEqual(first["source"]["sha256"], second["source"]["sha256"])

    def test_limits_and_derived_data(self):
        self.write()
        manifest = prepare(self.source, self.root / "output", train_limit=10, valid_limit=3, test_limit=4)
        self.assertEqual(manifest["samples"]["preferences_train"], 10)
        self.assertEqual(manifest["samples"]["preferences_valid"], 3)
        self.assertEqual(manifest["samples"]["eval_prompts"], 4)
        chosen = {row["chosen"] for row in read_jsonl(self.root / "output" / "preferences_train.jsonl")}
        self.assertEqual({row["response"] for row in read_jsonl(self.root / "output" / "sft_train.jsonl")}, chosen)

    def test_tampered_file_fails_checksum(self):
        self.write()
        prepare(self.source, self.root / "output")
        with (self.root / "output" / "ppo_train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            validate_dataset(self.root / "output")

    def test_leakage_detected_even_with_updated_hash(self):
        self.write()
        manifest = prepare(self.source, self.root / "output")
        train = read_jsonl(self.root / "output" / "preferences_train.jsonl")
        path = self.root / "output" / "preferences_test.jsonl"
        rows = read_jsonl(path)
        rows[0]["prompt"] = train[0]["prompt"]
        content = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(content)
        manifest["files"][path.name]["sha256"] = hashlib.sha256(content).hexdigest()
        (self.root / "output" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_dataset(self.root / "output")

    def test_small_source_and_bad_limits_fail(self):
        self.write(self.rows[:2])
        with self.assertRaisesRegex(ValueError, "three"):
            prepare(self.source, self.root / "output")
        self.write()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            prepare(self.source, self.root / "output", train_limit=0)

    def test_unverified_local_source_does_not_claim_human_feedback(self):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        source = prepare(self.source, self.root / "output")["source"]
        self.assertFalse(source["human_preferences"])
        self.assertEqual(source["provenance_status"], "unverified")

    def test_json_errors_include_line(self):
        self.source.write_text('{}\nnot-json\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, ":2: invalid JSON"):
            read_jsonl(self.source)

    def test_hub_adapter_preserves_splits_and_drops_overlap_without_network(self):
        seen = []
        shared = {"prompt": "shared synthetic test prompt", "chosen": "A", "rejected": "B"}
        def fake_load(name, *, split, streaming, revision):
            seen.append((split, streaming, revision))
            return iter([shared] + [{**row, "prompt": f"{split}: {row['prompt']}"} for row in self.rows])
        module = types.ModuleType("datasets")
        module.load_dataset = fake_load
        with patch.dict("sys.modules", {"datasets": module}):
            manifest = prepare_hub(self.root / "output", train_limit=80, valid_limit=80, test_limit=80)
        self.assertEqual([item[0] for item in seen], ["test", "valid1", "train"])
        self.assertTrue(all(item[1] and len(item[2]) == 40 for item in seen))
        self.assertEqual(manifest["counts"]["train"]["overlap_pairs_removed"], 1)
        self.assertEqual(manifest["counts"]["valid"]["overlap_pairs_removed"], 1)
        self.assertTrue(validate_dataset(self.root / "output")["valid"])


if __name__ == "__main__":
    unittest.main()
