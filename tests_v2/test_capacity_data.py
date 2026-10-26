"""Selection/shape checks, not training or dataset quality evidence."""
import dataclasses
import json
import unittest

from alignment_lab.capacity_data import select_capacity_rows, make_capacity_stress_row
from alignment_lab.training_data import TrainingPartition, TrainingRow


def partition(stage, count=400):
    name = {"sft": "sft_train", "reward": "preference_train",
            "dpo": "preference_train", "ppo": "ppo_prompts"}[stage]
    rows = tuple(TrainingRow(f"record-{i}", f"cluster-{i}", f"prompt-{i}",
        (4 + i % 20,) * (1 + i % 101),
        None if stage == "ppo" else (24 + i % 20,) * (1 + i % 73) + (2,),
        (45 + i % 20,) * (1 + i % 91) + (2,) if stage in {"reward", "dpo"} else None)
        for i in range(count))
    return TrainingPartition(rows, name, "a" * 64, "b" * 64, "fixture", b"{}")


def select(data, stage, **kwargs):
    values = dict(stage=stage, eos_id=2, special_token_ids=(0, 1, 2, 3), vocab_size=97)
    values.update(kwargs)
    return select_capacity_rows(data, **values)


class CapacityDataTests(unittest.TestCase):
    def test_exact_default_budgets_and_truthful_synthetic_counts(self):
        for stage, count in (("sft", 16), ("reward", 16), ("dpo", 320), ("ppo", 128)):
            with self.subTest(stage=stage):
                data = partition(stage)
                out = select(data, stage)
                synthetic = int(stage != "ppo")
                self.assertEqual(len(out.rows), count)
                self.assertEqual(out.metadata["real_rows"], count - synthetic)
                self.assertEqual(out.metadata["synthetic_rows"], synthetic)
                self.assertEqual(out.metadata["real_unique_clusters"], count - synthetic)
                self.assertTrue(out.metadata["capacity_only"])
                self.assertFalse(out.metadata["quality_evaluation"])
                originals = {r.record_id: r for r in data.rows}
                real = [r for r in out.rows if r.record_id in originals]
                self.assertEqual(len(real), count - synthetic)
                self.assertTrue(all(row == originals[row.record_id] for row in real))
                self.assertEqual(out.metadata["accepted_manifest_sha256"], "a" * 64)

    def test_stress_exact_1024_768_shape_no_padding_one_eos_and_distinct_pair(self):
        for stage in ("sft", "reward", "dpo"):
            out = select(partition(stage), stage)
            row = out.rows[0]
            self.assertEqual(len(row.prompt_ids), 1024)
            self.assertFalse(set(row.prompt_ids) & {0, 1, 2, 3})
            self.assertEqual(len(row.chosen_ids), 768)
            self.assertEqual(row.chosen_ids.count(2), 1)
            self.assertEqual(row.chosen_ids[-1], 2)
            if stage != "sft":
                self.assertEqual(len(row.rejected_ids), 768)
                self.assertEqual(row.rejected_ids.count(2), 1)
                self.assertEqual(row.rejected_ids[-1], 2)
                self.assertNotEqual(row.chosen_ids, row.rejected_ids)
            self.assertEqual(out.metadata["synthetic_stress"]["padding_tokens"], 0)
            self.assertTrue(out.metadata["synthetic_stress"]["attention_all_true"])
            self.assertTrue(row.record_id.startswith("synthetic-capacity-"))

    def test_selection_is_seeded_order_independent_and_copies_metadata(self):
        data = partition("dpo")
        a, b = select(data, "dpo"), select(dataclasses.replace(data, rows=data.rows[::-1]), "dpo")
        self.assertEqual(a.rows, b.rows)
        self.assertEqual(a.metadata, b.metadata)
        self.assertNotEqual(a.rows, select(data, "dpo", seed=43).rows)
        copy = a.metadata
        copy["rows"].clear()
        self.assertEqual(len(a.metadata["rows"]), 320)
        json.dumps(a.metadata, allow_nan=False)

    def test_long_dimension_coverage_and_cluster_priority_before_relaxing(self):
        data = partition("reward", 50)
        out = select(data, "reward", count=8, include_synthetic=False)
        self.assertEqual(len(out.rows[0].prompt_ids), max(len(r.prompt_ids) for r in data.rows))
        # Longest chosen/rejected among candidates not consumed earlier.
        self.assertEqual(len(out.rows[1].chosen_ids),
                         max(len(r.chosen_ids) for r in data.rows if r != out.rows[0]))
        clustered = dataclasses.replace(data, rows=tuple(dataclasses.replace(r, cluster_id=f"cluster-{i % 3}")
                                                         for i, r in enumerate(data.rows)))
        selection = select(clustered, "reward", count=8, include_synthetic=False)
        self.assertEqual(len({r.cluster_id for r in selection.rows[:3]}), 3)
        self.assertEqual(selection.metadata["real_unique_clusters"], 3)
        self.assertEqual(len({r.record_id for r in selection.rows}), 8)

    def test_ppo_distinct_prompts_even_with_multiple_records_and_no_synthetic(self):
        data = partition("ppo", 8)
        data = dataclasses.replace(data, rows=tuple(dataclasses.replace(r, prompt_id=f"prompt-{i // 2}")
                                                    for i, r in enumerate(data.rows)))
        out = select(data, "ppo", count=4)
        self.assertEqual(len({r.prompt_id for r in out.rows}), 4)
        self.assertEqual(out.metadata["synthetic_rows"], 0)
        self.assertTrue(all(r.chosen_ids is None and r.rejected_ids is None for r in out.rows))
        with self.assertRaisesRegex(ValueError, "distinct PPO"):
            select(data, "ppo", count=5)

    def test_reject_wrong_partition_mutation_truncation_and_invalid_tokens(self):
        data = partition("sft", 20)
        variants = [
            dataclasses.replace(data, partition="final_holdout"),
            dataclasses.replace(data, rows=data.rows + (data.rows[0],)),
            dataclasses.replace(data, rows=(dataclasses.replace(data.rows[0], prompt_ids=(True,)),) + data.rows[1:]),
            dataclasses.replace(data, rows=(dataclasses.replace(data.rows[0], chosen_ids=(4, 2, 2)),) + data.rows[1:]),
            dataclasses.replace(data, rows=(dataclasses.replace(data.rows[0], prompt_ids=(4,) * 1025),) + data.rows[1:]),
        ]
        for bad in variants:
            with self.assertRaises(ValueError):
                select(bad, "sft")
        for options in (dict(seed=True), dict(count=1), dict(vocab_size=2),
                        dict(eos_id=True), dict(include_synthetic="yes")):
            with self.assertRaises(ValueError):
                select(data, "sft", **options)

    def test_stress_requires_actual_non_special_source_ids(self):
        row = TrainingRow("r", "c", "p", (4,), (4, 2), (4, 2))
        with self.assertRaisesRegex(ValueError, "distinct non-special"):
            make_capacity_stress_row((row,), stage="dpo", eos_id=2,
                special_token_ids=(0, 2), vocab_size=5)
        with self.assertRaisesRegex(ValueError, "include EOS"):
            make_capacity_stress_row((row,), stage="sft", eos_id=2,
                special_token_ids=(0,), vocab_size=5)


if __name__ == "__main__":
    unittest.main()
