"""Explicit external file pins increase full-content coverage without execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import throughput_supervisor_v3_inputs2 as subject


class ExternalInputTests(unittest.TestCase):
    def tearDown(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_external_file_pin_included_and_content_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "project"
            root.mkdir()
            external = base / "dependency.py"
            external.write_text("original bytes")
            unrelated = base / "unrelated.py"
            unrelated.write_text("not a dependency")
            plan = root / "plan.json"
            plan.write_text(json.dumps({"runtime": {"path": str(external), "sha256": subject.sha(external)}}))
            live = subject.frozen.legacy._LiveIdentity({str(plan): subject.sha(plan)})
            with patch.object(subject, "ROOT", root), patch.object(subject, "_historical_map", return_value=({}, {})):
                subject._add_document_dependencies(live)
            self.assertEqual(set(live.files), {str(plan), str(external)})
            self.assertEqual(live.trees, {})
            proof = subject._ContentIdentity(live)
            external.write_text("modified bytes")
            with patch.object(live, "check", return_value=True):
                with self.assertRaisesRegex(ValueError, "content changed"): proof.check()

    def test_external_wrong_hash_or_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "project"
            root.mkdir()
            external = base / "dependency.py"
            external.write_text("current")
            plan = root / "plan.json"
            plan.write_text(json.dumps({"runtime": {"path": str(external), "sha256": "a" * 64}}))
            with patch.object(subject, "ROOT", root), patch.object(subject, "_historical_map", return_value=({}, {})):
                live = subject.frozen.legacy._LiveIdentity({str(plan): subject.sha(plan)})
                with self.assertRaises(ValueError): subject._add_document_dependencies(live)
                external.unlink()
                live = subject.frozen.legacy._LiveIdentity({str(plan): subject.sha(plan)})
                with self.assertRaises(FileNotFoundError): subject._add_document_dependencies(live)

    def test_original_dispatch_and_semantic_acceptance_remain_identical(self):
        for name in ("run_alignment", "validate_plan", "_metadata_plan"):
            self.assertIs(getattr(subject, name).__code__, getattr(subject.frozen, name).__code__)
        self.assertIs(subject.run_alignment.__globals__["_verify_stage"], subject.frozen._verify_stage)
        self.assertEqual(subject.frozen.REMAINING_SEQUENCE, (("dpo", 44), ("ppo", 44)))

    def test_cpu_proof_requires_new_tests_and_sources(self):
        plan = {"source_sha256": dict.fromkeys(subject.ADDED_SOURCES, "a" * 64), "cpu_report": {}}
        with patch.object(subject.previous, "_cpu_proof"), patch.object(subject.frozen, "_pin_record"), \
                patch.object(subject, "_parse", return_value={"modules": ["tests.test_throughput_supervisor_v3_inputs2"]}):
            subject._cpu_proof(plan)
            for name in subject.ADDED_SOURCES:
                changed = {**plan, "source_sha256": {k: v for k, v in plan["source_sha256"].items() if k != name}}
                with self.assertRaises(ValueError): subject._cpu_proof(changed)
        with patch.object(subject.previous, "_cpu_proof"), patch.object(subject.frozen, "_pin_record"), \
                patch.object(subject, "_parse", return_value={"modules": []}):
            with self.assertRaises(ValueError): subject._cpu_proof(plan)


if __name__ == "__main__":
    unittest.main()
