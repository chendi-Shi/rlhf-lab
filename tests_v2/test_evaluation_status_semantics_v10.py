"""CPU source-only proof that the V10 repair cannot hide inference changes."""
from pathlib import Path
import unittest

from alignment_lab.evaluation_status_semantics_v10 import validate_v10_execution_semantics


class StatusSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.prior = (root / "alignment_lab/evaluation_execution_v9.py").read_bytes()
        cls.current = (root / "alignment_lab/evaluation_execution_v10.py").read_text(encoding="utf-8")

    def test_complete_module_proves_only_enumerated_status_io_change(self):
        receipt = validate_v10_execution_semantics(self.prior, self.current)
        self.assertTrue(receipt["full_module_ast_equal_after_exact_normalization"])
        self.assertFalse(receipt["model_or_tensor_imported"])

    def test_model_input_metric_manifest_and_unrelated_changes_are_rejected(self):
        changes = (
            ("max_new_tokens=2048", "max_new_tokens=1024"),
            ('eos_id=151645', 'eos_id=0'),
            ('for source in data.rows:', 'for source in data.rows[:2]:'),
            ('row.update(prompt=source["prompt"]', 'row.update(prompt="changed"'),
            ('_validate_rows(context, request, rows, decode_tokens=decode, codec=codec)', 'pass'),
            ('_new(output / "manifest.json", manifest)', 'pass'),
            ('"records": count, "prompts": count, "instructions": 834',
             '"records": count, "prompts": count, "instructions": 833'),
            ('import uuid', 'import uuid\nimport torch'),
            ('_boundary(request, system)\n                guard()', 'guard()'),
        )
        for before, after in changes:
            with self.subTest(change=before):
                self.assertIn(before, self.current)
                with self.assertRaises(ValueError):
                    validate_v10_execution_semantics(self.prior, self.current.replace(before, after, 1))

    def test_status_error_cleanup_and_original_deadline_contracts_are_exact(self):
        changes = (
            ('deadline_epoch=request["deadline_epoch"], serialize=canonical',
             'deadline_epoch=system.now() + 5, serialize=canonical'),
            ('primary_error = error', 'primary_error = None'),
            ('_progress(publisher, status, terminal=True)', '_progress(publisher, status)'),
            ('status["status_publication_failure"] = publisher.failure', 'pass'),
            ('if lock.is_file() and parse(lock.read_bytes()) == owner:', 'if lock.is_file():'),
            ('sleep_fn=system.sleep)', 'sleep_fn=system.sleep)\n    pass'),
            ('return publisher.publish(value, terminal=terminal)', 'return None'),
            ('"alignment_lab/evaluation_resources_v9.py",', '"invented.py",'),
        )
        for before, after in changes:
            with self.subTest(change=before):
                self.assertIn(before, self.current)
                with self.assertRaises(ValueError):
                    validate_v10_execution_semantics(self.prior, self.current.replace(before, after, 1))


if __name__ == "__main__":
    unittest.main()
