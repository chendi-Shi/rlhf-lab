import copy
import unittest

from alignment_lab.research_inputs import select_development


def rows():
    result = []
    for domain, count in (("general", 9), ("stem", 6), ("code", 3), ("multilingual", 2)):
        for i in range(count):
            for repeat in range(2 if i == 0 else 1):
                result.append({"domain": domain, "record_id": f"{domain}-{i}-{repeat}",
                    "cluster_id": f"{domain}-{i}", "source_split": "train", "local_split": "tuning_dev",
                    "tokens": {"prompt": i * 100}, "overall_preference": repeat})
    return result


class ResearchSelectionTests(unittest.TestCase):
    def test_input_order_and_nonselection_fields_do_not_affect_selection(self):
        original = rows()
        selected, info = select_development(original, 11)
        changed = copy.deepcopy(list(reversed(original)))
        for row in changed:
            row["tokens"] = {"prompt": 99999}
            row["overall_preference"] = -3
            row["untrusted_model_score"] = 1e9
        again, other = select_development(changed, 11)
        self.assertEqual([r["record_id"] for r in selected], [r["record_id"] for r in again])
        self.assertEqual(info, other)
        self.assertEqual(len({r["cluster_id"] for r in selected}), 11)
        self.assertEqual(info["selected_domain_counts"], {"general": 5, "stem": 3, "code": 2, "multilingual": 1})

    def test_wrong_partition_duplicate_record_or_too_few_clusters_rejected(self):
        original = rows()
        for key, value in (("local_split", "final_holdout"), ("source_split", "validation")):
            bad = copy.deepcopy(original)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                select_development(bad, 11)
        with self.assertRaises(ValueError):
            select_development(original + [original[0]], 11)
        with self.assertRaises(ValueError):
            select_development(original, 21)

    def test_exact_population_and_single_prompt_are_well_defined(self):
        all_selected, info = select_development(rows(), 20)
        self.assertEqual(len(all_selected), 20)
        self.assertEqual(info["selected_domain_counts"], {"general": 9, "stem": 6, "code": 3, "multilingual": 2})
        one, _ = select_development(rows(), 1)
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0]["domain"], "general")


if __name__ == "__main__":
    unittest.main()
