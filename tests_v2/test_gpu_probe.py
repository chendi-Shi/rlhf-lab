"""Pure CPU admission/selection tests; never start the real GPU probe."""
import unittest
from alignment_lab.gpu_probe import admission_blockers, select_examples, require_active_boundary
from alignment_lab.training_data import TrainingRow


class GPUProbeTests(unittest.TestCase):
    def test_completion_cannot_pass_after_budget_or_stop(self):
        require_active_boundary(10, lambda: False, clock=lambda: 9)
        with self.assertRaises(TimeoutError):
            require_active_boundary(10, lambda: False, clock=lambda: 10)
        with self.assertRaises(TimeoutError):
            require_active_boundary(10, lambda: False, clock=lambda: 11)
        with self.assertRaises(InterruptedError):
            require_active_boundary(10, lambda: True, clock=lambda: 9)

    def test_resources_and_other_training_block_dispatch(self):
        gpu = [{"index": 0, "free_mb": 7400, "utilization_percent": 3}]
        self.assertEqual(admission_blockers(gpu, 12, []), [])
        for devices, ram, processes in (
            ([], 12, []), (gpu, 4, []),
            ([{"index": 0, "free_mb": 6000, "utilization_percent": 3}], 12, []),
            ([{"index": 0, "free_mb": 7400, "utilization_percent": 85}], 12, []),
            (gpu, 12, [{"ProcessId": 123, "CommandLine": "python -m windows.train"}]),
            (gpu, 12, [{"ProcessId": 124, "CommandLine": None}]),
        ):
            self.assertTrue(admission_blockers(devices, ram, processes))

    def test_selection_is_seeded_and_never_truncates_or_uses_ineligible_rows(self):
        rows = [TrainingRow(str(i), str(i), str(i), tuple(range(i + 1)), (7, 2))
                for i in range(10)]
        chosen, count = select_examples(rows, seed=42, count=4, prompt_limit=6, response_limit=2)
        second, _ = select_examples(rows, seed=42, count=4, prompt_limit=6, response_limit=2)
        self.assertEqual(chosen, second)
        self.assertEqual(count, 6)
        self.assertEqual(len({x.record_id for x in chosen}), 4)
        self.assertTrue(all(x in rows and len(x.prompt_ids) <= 6 for x in chosen))
        with self.assertRaises(ValueError):
            select_examples(rows, seed=42, count=4, prompt_limit=2, response_limit=2)


if __name__ == "__main__":
    unittest.main()
