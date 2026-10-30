"""V6 saved-source scoring fixtures; no model loading or measured quality claims."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import score_alignlab_evaluation_v6 as cli
from tests_v2 import test_evaluation_scoring_cli_v2 as legacy

ROOT = Path(__file__).resolve().parents[1]


class ScorerV6Tests(unittest.TestCase):
    # Reuse fixture construction only, not the old test suite or its conclusions.
    @classmethod
    def setUpClass(cls):
        legacy.SavedScoringTests.setUpClass.__func__(cls)

    bind_roster = legacy.SavedScoringTests.bind_roster

    def setUp(self):
        self.scoring_sources = {name: legacy.sha(ROOT / name) for name in (
            "scripts/score_alignlab_evaluation_v6.py", "scripts/decode_alignlab_evaluation_v2.py",
            "alignment_lab/evaluation_protocol_v2.py", "alignment_lab/official_evaluation_v6.py")}
        legacy.SavedScoringTests.setUp(self)
        self.inference["source_sha256"] = {
            **self.inference["source_sha256"],
            **{name: legacy.sha(ROOT / name) for name in (
                "alignment_lab/evaluation_execution_v5.py", "scripts/run_alignlab_evaluation_stage_v5.py")}}
        self.refresh()

    def refresh(self, **kwargs):
        legacy.SavedScoringTests.refresh(self, **kwargs)
        self.request["source_sha256"] = dict(self.scoring_sources)
        legacy.save(self.request_path, self.request)

    def run_score(self):
        return cli.score_request(self.request_path, legacy.sha(self.request_path),
                                 output=self.directory / "scores")

    def test_explicit_old_inference_source_is_scored_without_mutation_or_deadline_reset(self):
        original_files = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        self.assertNotEqual(self.inference["source_sha256"], self.scoring_sources)
        result = self.run_score()
        self.assertEqual(result["status"], "official_scoring_passed")
        self.assertEqual(result["scores"]["denominators"], {"all": 2017, "non_ties": 1920, "ties": 97})
        self.assertEqual(result["deadline_epoch"], self.deadline)
        self.assertEqual(result["inference_manifest"], {
            "path": self.request["inference_manifest"], "sha256": self.request["inference_manifest_sha256"]})
        self.assertFalse(result["gpu_operations_performed"])
        self.assertFalse(result["research_complete"])
        self.assertEqual(result["human_labels"], 0)
        self.assertIsNone(result["human_win_rate"])
        for path, payload in original_files.items():
            self.assertEqual(path.read_bytes(), payload, str(path))

    def test_current_version_inference_and_scoring_closures_are_also_supported(self):
        self.inference["source_sha256"] = dict(self.scoring_sources)
        self.refresh()
        self.assertEqual(self.run_score()["status"], "official_scoring_passed")

    def test_original_deadline_cannot_be_extended_or_expired(self):
        self.request["deadline_epoch"] += 1
        legacy.save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "reset or extend"):
            self.run_score()
        self.refresh()
        with patch.object(cli.time, "time", return_value=self.deadline + 1):
            with self.assertRaises(TimeoutError):
                self.run_score()
        self.assertFalse((self.directory / "scores").exists())

    def test_v6_source_pins_are_mandatory_and_checked(self):
        self.request["source_sha256"].pop("alignment_lab/official_evaluation_v6.py")
        legacy.save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "source identity is incomplete"):
            self.run_score()
        self.refresh()
        self.request["source_sha256"]["alignment_lab/official_evaluation_v6.py"] = "0" * 64
        legacy.save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_score()
        self.assertFalse((self.directory / "scores").exists())

    def test_old_inference_source_hash_is_still_verified_independently(self):
        self.inference["source_sha256"] = copy.deepcopy(self.inference["source_sha256"])
        self.inference["source_sha256"]["alignment_lab/evaluation_execution_v5.py"] = "0" * 64
        self.refresh()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_score()
        self.assertFalse((self.directory / "scores").exists())

    def test_tampered_or_incomplete_saved_outputs_do_not_publish_scores(self):
        output = self.directory / "outputs.jsonl"
        output.write_bytes(output.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_score()
        self.refresh(rows=self.rows[:-1])
        with self.assertRaises(ValueError):
            self.run_score()
        self.assertFalse((self.directory / "scores").exists())

    def test_partial_parameter_audit_is_not_an_accepted_source(self):
        self.manifest["frozen_audits"][0]["full"] = False
        binding = legacy.save(self.directory / "manifest.json", self.manifest)
        self.request["inference_manifest_sha256"] = binding["sha256"]
        legacy.save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "Full before/after"):
            self.run_score()
        self.assertFalse((self.directory / "scores").exists())


if __name__ == "__main__":
    unittest.main()
